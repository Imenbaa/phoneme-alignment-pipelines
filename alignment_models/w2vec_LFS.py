import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import editdistance
from transformers import Wav2Vec2ForPreTraining
os.environ["HF_HOME"] = "/vol/experiments/cache_imbenamor"
os.environ["HF_DATASETS_CACHE"] = "/vol/experiments/cache_imbenamor/datasets"
os.environ["TMPDIR"] = "/vol/experiments/cache_imbenamor/tmp"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from datasets import load_dataset, load_from_disk, Audio
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer,
)
from transformers.trainer_pt_utils import LengthGroupedSampler

import numpy as np
import logging
 
logger = logging.getLogger(__name__)
# =========================
# Step 0: Config
# =========================
print("Step 0: Config")
exp_name   = "w2vCTC_FS_stage2"
model_name = "results/w2vCTC_FS_normalized/checkpoint-7811"  # best sharpness checkpoint
output_dir = f"results/{exp_name}"
#####stage 1#########
#exp_name      = "w2vCTC_FS_normalized"
#model_name    = "results/w2vCTC/checkpoint-23430"
#output_dir    = f"results/{exp_name}"
#############################
vocab_name    = "vocab_w2vCTC.json"
preprocessed_path = "results/w2vCTC_FS/preprocessed"

os.makedirs(output_dir, exist_ok=True)

# =========================
# Steps 1–6: Preprocessing (cached)
# =========================

if os.path.exists(preprocessed_path):
    print("Loading preprocessed dataset from disk...")
    dataset = load_from_disk(preprocessed_path)

else:
    print("Preprocessing from scratch...")

    print("Step 1: Loading dataset")
    dataset = load_dataset(
        "audiofolder",
        data_dir="/vol/experiments2/cbrazier/transcription/datasets/ester1_ester2_epac"
    )
    dataset = dataset.remove_columns(["transcription"])
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    print("Step 2: Filtering UNK and very short seq")

    def is_valid(example):
        has_unk     = "|ø|n|k|" in example["phonemes"]
        too_short   = len(example["audio"]["array"]) < 24000  # < 1.5s at 16kHz
        few_phones  = len(example["phoneme_single_v1"].split()) < 5
        return not has_unk and not too_short and not few_phones
    
    for split in ["train", "validation", "test"]:
        before = len(dataset[split])
        dataset[split] = dataset[split].filter(is_valid, num_proc=6)
        after  = len(dataset[split])
        print(f"  {split}: {before} → {after} ({before - after} removed)")

    print("Step 4: Build vocab")
    vocab_set = set()
    for split in ["train", "validation", "test"]:
        for seq in dataset[split]["phoneme_single_v1"]:
            vocab_set.update(seq.split())

    vocab_list = sorted(vocab_set)
    vocab_dict = {"[PAD]": 0, "[UNK]": 1}
    vocab_dict.update({v: i + 2 for i, v in enumerate(vocab_list)})

    with open(vocab_name, "w") as f:
        json.dump(vocab_dict, f)
    print(f"Vocab size: {len(vocab_dict)}")

    print("Step 5: Create processor")
    #convert phonemes to integer ids
    _tokenizer = Wav2Vec2CTCTokenizer(
        vocab_file=vocab_name,
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="", 
    )
    #audio-->numerical features
    _feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=16000,
        padding_value=0.0, do_normalize=True,
        return_attention_mask=True,  #produces a mask, telling model which parts are real audio vs padding
    )
    #awrapper that combines both
    _processor = Wav2Vec2Processor(
        feature_extractor=_feature_extractor, tokenizer=_tokenizer
    )

    print("Step 6: Preprocessing")
    def preprocess_batch(batch):
        audio_arrays   = [a["array"] for a in batch["audio"]]
        sampling_rates = [a["sampling_rate"] for a in batch["audio"]]
        assert all(sr == 16000 for sr in sampling_rates)
    
        inputs = _feature_extractor(
            audio_arrays, sampling_rate=16000,
            return_tensors="np", padding=False,
            return_attention_mask=True,
        )
        with _processor.as_target_processor():
            labels = _processor(batch["phoneme_single_v1"]).input_ids
    
        return {
            "input_values":   inputs.input_values,
            "attention_mask": inputs.attention_mask,
            "labels":         labels,
            "length":         [len(x) for x in inputs.input_values],  # ← from extracted features
        }
    
    for split in ["train", "validation", "test"]:
        dataset[split] = dataset[split].map(
            preprocess_batch,
            batched=True, batch_size=32, num_proc=6,
            remove_columns=dataset[split].column_names,
            cache_file_name=f"/vol/experiments/cache_imbenamor/map_FS_{split}.arrow",
        )

    print("Saving preprocessed dataset...")
    dataset.save_to_disk(preprocessed_path)

# =========================
#  load processor
# =========================
print("Loading processor...")
tokenizer = Wav2Vec2CTCTokenizer(
    vocab_file=vocab_name,
    unk_token="[UNK]",
    pad_token="[PAD]",
    word_delimiter_token="",
)
feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1,
    sampling_rate=16000,
    padding_value=0.0,
    do_normalize=True,
    return_attention_mask=True,
)
processor = Wav2Vec2Processor(
    feature_extractor=feature_extractor, tokenizer=tokenizer
)

# =========================
# Step 7: Data collator
# =========================
print("Step 7: Data collator")

@dataclass
class DataCollatorCTCWithPadding:
    #Build a custom batch
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        #executed at each batch
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]
        #make all audio sequences same length
        batch = self.processor.pad(
            input_features, padding=self.padding, return_tensors="pt"
        )
        # ensure attention_mask is in the batch
        if "attention_mask" not in batch:
            batch["attention_mask"] = (batch["input_values"] != 0).long()

        #pad labels
        labels_batch = self.processor.tokenizer.pad(
            label_features, padding=self.padding, return_tensors="pt"
        )
        #replace padding with -100 because ctc ignores it
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels
        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

# =========================
# Step 8: Metrics (PER)
# =========================
print("Step 8: Metrics")
#make ctc output sequence level by removing repetition and PAD
def ctc_collapse(token_list, pad_token="[PAD]"):
    collapsed, prev = [], None
    for t in token_list:
        if t != prev and t != pad_token:
            collapsed.append(t)
        prev = t
    return collapsed

def compute_metrics(pred):
    # pred.predictions is (N, T, vocab) 
    #take most probable token for each frame
    pred_ids   = np.argmax(pred.predictions, axis=-1)
    label_ids  = pred.label_ids

    all_per = []
    for pred_id_seq, label_id_seq in zip(pred_ids, label_ids):
        pred_tokens  = ctc_collapse([processor.tokenizer.convert_ids_to_tokens(int(i)) for i in pred_id_seq] )
        label_tokens = [processor.tokenizer.convert_ids_to_tokens(int(i)) for i in label_id_seq if i not in [-100,processor.tokenizer.pad_token_id]]
        per = editdistance.eval(pred_tokens, label_tokens) / max(1, len(label_tokens))
        all_per.append(per)

    return {"PER": float(np.mean(all_per))}

def get_loss_weights(global_step: int, warmup: int = 800):
    w = min(1.0, global_step / max(warmup, 1))
    return 0.0, 0.5 * w, 1.0 * w    # λ_ctc, λ_fs, λ_m

##############stage 2#######
def initialize_quantizer(model, dataloader, device, n_batches=200):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= n_batches:
                break
            input_values   = batch["input_values"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            outputs = model.wav2vec2(input_values, attention_mask=attention_mask)
            X = outputs.last_hidden_state
            B, T, D = X.shape
            idx    = torch.randint(0, T, (B,))
            frames = X[torch.arange(B), idx]
            embeddings.append(frames.cpu())

    embeddings = torch.cat(embeddings, dim=0)
    num_vars   = model.quantizer.num_vars
    perm       = torch.randperm(embeddings.shape[0])[:num_vars]
    model.quantizer.codevectors.data = embeddings[perm].to(device)
    print(f"Quantizer initialized from {embeddings.shape[0]} real frames")
    model.train()
##################################
#convert continuous audio features to discrete-like representations
#create targets for contrastive learning
class GumbelQuantizer(nn.Module):
    def __init__(self, input_dim: int, num_vars: int = 320, temp: float = 2.0):
        super().__init__()
        self.num_vars    = num_vars
        self.temp        = temp
        self.weight_proj = nn.Linear(input_dim, num_vars)
        self.codevectors = nn.Parameter(torch.randn(num_vars, input_dim))
        nn.init.uniform_(self.codevectors, -1.0, 1.0)

    def forward(self, X: torch.Tensor):
        logits  = self.weight_proj(X)
        if self.training:
            probs = F.gumbel_softmax(logits, tau=self.temp, hard=True)
        else:
            indices = logits.argmax(dim=-1)
            probs   = F.one_hot(indices, self.num_vars).float()
        return torch.matmul(probs, self.codevectors), probs
# =========================
# Step 9: Model
# =========================

class Wav2Vec2ForCTC_FS_REC(nn.Module):
    """
    Wav2Vec2 + Forward-Sum alignment + Gumbel-contrastive reconstruction.

    Loss = L_ctc + λ_fs * L_fs + λ_m * L_m

      L_ctc  : standard CTC on phoneme predictions
      L_fs   : forward-sum loss enforcing monotonic alignment
      L_m    : contrastive loss (eq. 4) between reconstruction head output h_t
               and quantized acoustic target q_t, with n=50 negatives per step

    The alignment A = softmax(D, dim=0) maps each frame to a distribution over
    phonemes (each column of D sums to 1). H = [X̂, YA] concatenates the masked
    frame with its phoneme-aligned embedding; the reconstruction head maps H → h_t
    which is then contrasted against q_t from the Gumbel quantizer.
    """

    def __init__(
        self,
        base_model: Wav2Vec2ForCTC,
        vocab_size: int,
        hidden_dim: int,
        #pretrained_quantizer=None,
        mask_prob:    float = 0.065,
        n_negatives:  int   = 50,
        temperature:  float = 0.1,

    ):
        super().__init__()

        self.wav2vec2 = base_model.wav2vec2 #extracts features X
        self.ctc_head = base_model.lm_head #predicts phonemes

        self.mask_prob   = mask_prob
        self.n_negatives = n_negatives #for contrastive loss
        self.temperature = temperature


        # Alignment module
        self.phoneme_embed = nn.Embedding(vocab_size, hidden_dim) # Y_embedding of all phonemes
        self.fx = nn.Linear(hidden_dim, hidden_dim) # projection of X
        self.fy = nn.Linear(hidden_dim, hidden_dim) #Projection of Y

        # maps unmasked X → quantized targets q_t
        self.quantizer = GumbelQuantizer(input_dim=hidden_dim)

            # Reconstruction head: [X̂; YA] (2*D) → h_t (D)
        self.reconstruction_head = nn.Linear(2 * hidden_dim, hidden_dim)
        
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.wav2vec2.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.wav2vec2.gradient_checkpointing_disable()
    #converts logits to phoneme sequence
    def _ctc_decode_batch(self, logits: torch.Tensor) -> torch.Tensor:
        pred_ids = logits.argmax(dim=-1)   # (B, T) — stays on GPU
    
        sequences = []
        for seq in pred_ids:               # loop over B samples only, not B*T tokens
            # Remove consecutive duplicates on GPU
            unique_mask = torch.cat([
                torch.tensor([True], device=seq.device),
                seq[1:] != seq[:-1]
            ])
            collapsed = seq[unique_mask]
            collapsed = collapsed[collapsed != 0]    # remove blank
    
            if collapsed.numel() == 0:
                collapsed = torch.tensor([1], device=seq.device)
            sequences.append(collapsed)
    
        return torch.nn.utils.rnn.pad_sequence(
            sequences, batch_first=True, padding_value=0
        )
    # ------------------------------------------------------------------
    # Masking
    # ------------------------------------------------------------------
    def mask_features(self, X: torch.Tensor):
        """Returns X_masked (B, T, D) and boolean mask (B, T)."""
        B, T, _ = X.shape
        mask = torch.rand(B, T, device=X.device) < self.mask_prob
        X_masked = X.clone()
        X_masked[mask] = 0.0
        return X_masked, mask

    # ------------------------------------------------------------------
    # Alignment
    # ------------------------------------------------------------------
    def compute_alignment(self, X: torch.Tensor, Y_emb: torch.Tensor):
        """
        D_ij = f_y(y_i)^T f_x(x̂_j)   — eq. (1), D ∈ (B, N, T)
        A    = softmax(D, dim=1)        — eq. (2): each frame column sums to 1
                                          (softmax over phoneme axis N)
        """
        D = torch.matmul(self.fy(Y_emb), self.fx(X).transpose(1, 2))  # (B, N, T)
        A = torch.softmax(D, dim=1)                                     # (B, N, T)
        return A

    # ------------------------------------------------------------------
    # Forward-sum loss  (monotonic CTC on the alignment matrix)
    # ------------------------------------------------------------------
    def forward_sum_loss(self, A, label_lengths, input_lengths):
        """
        Forward-sum loss via the CTC trick .
        
        Augment A with a blank column at index 0 set to near-zero probability,
        then call F.ctc_loss with labels = [1, 2, ..., N] (position indices).
        With blank~0, CTC cannot skip phonemes so only monotonic paths survive
        — equivalent to the forward-sum DP.
        
        A:             (B, N, T)
        label_lengths: (B,)
        input_lengths: (B,)
        """
        B, N, T = A.shape
    
        blank_prob    = torch.full((B, 1, T), 1e-8, device=A.device, dtype=torch.float32)
        A_with_blank  = torch.cat([blank_prob, A.float()], dim=1)
        log_probs     = torch.log(A_with_blank).permute(2, 0, 1)
    
        labels = torch.zeros(B, N, dtype=torch.long, device=A.device)
        for b in range(B):
            n = label_lengths[b].item()
            labels[b, :n] = torch.arange(1, n + 1, device=A.device)
    
        loss = F.ctc_loss(
            log_probs, labels, input_lengths, label_lengths,
            blank=0, reduction="none",   # ← per-sample loss, not mean
            zero_infinity=True,
        )
    
        # Normalize by label length so loss is comparable across sequence lengths
        loss = (loss / label_lengths.float().clamp(min=1)).mean()
        return loss

    # ------------------------------------------------------------------
    # Contrastive reconstruction loss  — eq. (4)
    # ------------------------------------------------------------------
    def contrastive_loss(
        self,
        H:           torch.Tensor,  # (B, T, D) — reconstruction head output
        X_quantized: torch.Tensor,  # (B, T, D) — Gumbel quantizer targets
        mask:        torch.Tensor,  # (B, T) bool — True at masked positions
        temperature=None
    ) -> torch.Tensor:
        """
        For each masked frame t, maximise similarity between h_t and q_t
        (true quantized feature) relative to n=50 negatives sampled from
        all other quantized frames in the batch.
        """
        B, T, D = H.shape
        temp = temperature if temperature is not None else self.temperature
        # Flatten to (B*T, D) for efficient sampling
        H_flat   = H.reshape(B * T, D)
        Q_flat   = X_quantized.reshape(B * T, D)  # all quantized vectors
        mask_flat = mask.reshape(B * T)            # (B*T,)

        if mask_flat.sum() == 0:
            return H_flat.sum() * 0.0  # maintains gradient graph

        # L2-normalise for cosine similarity
        H_norm = F.normalize(H_flat, dim=-1)       # (B*T, D)
        Q_norm = F.normalize(Q_flat, dim=-1)        # (B*T, D)

        # Select masked positions
        masked_H = H_norm[mask_flat]                # (M, D)
        masked_Q = Q_norm[mask_flat]                # (M, D) — positives
        M = masked_H.shape[0]

        # Sample n negatives per masked frame from all quantized positions
        # (including non-masked; mimics the original wav2vec2 setup)
        neg_idx  = torch.randint(0, B * T, (M, self.n_negatives), device=H.device)
        Q_neg    = Q_norm[neg_idx]                  # (M, n_negatives, D)

        # Positive similarities: (M,)
        sim_pos = (masked_H * masked_Q).sum(dim=-1) / temp

        # Negative similarities: (M, n_negatives)
        sim_neg = torch.bmm(
            Q_neg, masked_H.unsqueeze(-1)           # (M, n_negatives, D) × (M, D, 1)
        ).squeeze(-1) / temp

        # eq. (4): -log [exp(sim_pos) / (exp(sim_pos) + Σ exp(sim_neg))]
        all_sims = torch.cat([sim_pos.unsqueeze(1), sim_neg], dim=1)  # (M, 1+n_neg)
        loss = -sim_pos + torch.logsumexp(all_sims, dim=1)

        return loss.mean()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, input_values, attention_mask=None, labels=None,global_step=0):

        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        X       = outputs.last_hidden_state
        logits  = self.ctc_head(X)
        
        # Always decode from CTC — no ground truth needed at any point
        with torch.no_grad():
            labels_clean = self._ctc_decode_batch(logits)
        
        Y_emb = self.phoneme_embed(labels_clean)
        A     = self.compute_alignment(X, Y_emb)
        
        if labels is None:
            return {"loss": None, "logits": logits, "alignment": A}
        
        # Compute losses — L_fs uses predicted sequence, not ground truth
        if attention_mask is not None:
            #convert audio length to feature length
            input_lengths = self.wav2vec2._get_feat_extract_output_lengths(
                attention_mask.sum(dim=-1)
            ).long().clamp(min=1, max=X.shape[1])
        else:
            input_lengths = torch.full(
                (X.shape[0],), X.shape[1], dtype=torch.long, device=X.device
            )
        

        # Also enforce CTC requirement: frames must be >= labels
        label_lengths = (labels != -100).sum(dim=-1)
        input_lengths = torch.maximum(input_lengths, label_lengths).clamp(max=X.shape[1])
        
        
        loss_ctc = F.ctc_loss(logits.log_softmax(dim=-1).transpose(0, 1),labels,input_lengths,(labels != -100).sum(dim=-1),blank=0, reduction="mean", zero_infinity=True,)
        label_lengths_pred = (labels_clean != 0).sum(dim=-1)
        # In forward():
        loss_fs = self.forward_sum_loss(A, label_lengths_pred, input_lengths)

        X_quantized, _ = self.quantizer(X.detach())#no gradient flows into X
        X_masked, mask = self.mask_features(X)#hide some frames
        YA    = torch.matmul(A.transpose(1, 2), Y_emb)#each frame gets its aligned phoneme representation
        H     = self.reconstruction_head(torch.cat([X_masked, YA], dim=-1)) #it takes the X_masked with the phoneme infor and tries to reconstruct H

        current_temp = max(0.1, 1.0 - 0.9 * min(1.0, global_step / 10000))
        self.quantizer.temp = current_temp
        loss_m = self.contrastive_loss(H, X_quantized, mask, temperature=current_temp)
        #loss_m = torch.tensor(0.0)
        #stage1
        #def get_loss_weights(global_step, warmup=500):
            #w = min(1.0, global_step / max(warmup, 1))
            #return 1.0, 1.0 * w, 0.0   # matches paper exactly  
        #stage2
        #loss_fs = torch.tensor(0.0, device=X.device)
        loss_ctc =torch.tensor(0.0, device=X.device)

        
        λ_ctc, λ_fs, λ_m = get_loss_weights(global_step)
        loss = λ_ctc*loss_ctc + λ_fs * loss_fs + λ_m * loss_m
        
        return {
            "loss":     loss,
            "logits":   logits,
            "alignment": A,
            "loss_ctc": loss_ctc.detach(),
            "loss_fs":  loss_fs.detach(),
            "loss_m":   loss_m.detach(),
        }
    


# =========================
# Instantiate model
# =========================
print("Step 9: Building model")
"""
base_model = Wav2Vec2ForCTC.from_pretrained(
    model_name,
    ctc_loss_reduction="mean",
    ctc_zero_infinity=True,
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer),
)

model = Wav2Vec2ForCTC_FS_REC(
    base_model,
    vocab_size  = len(processor.tokenizer),
    hidden_dim  = base_model.config.hidden_size,
    mask_prob   = 0.065,
    n_negatives = 50,
    temperature = 0.1,

)
model.wav2vec2.feature_extractor._freeze_parameters()
"""
# Load the base architecture first from the original pretrained CTC model
base_model = Wav2Vec2ForCTC.from_pretrained(
    "results/w2vCTC/checkpoint-23430",   # original pretrained CTC — always has config.json
    ctc_loss_reduction="mean",
    ctc_zero_infinity=True,
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer),
)

# Build the full model wrapper
model = Wav2Vec2ForCTC_FS_REC(
    base_model,
    vocab_size  = len(processor.tokenizer),
    hidden_dim  = base_model.config.hidden_size,
    mask_prob   = 0.065,
    n_negatives = 50,
    temperature = 0.1,
    #pretrained_quantizer  = pretrained_quantizer,
)
model.wav2vec2.feature_extractor._freeze_parameters()

# Load Stage 1 weights on top
checkpoint_path = "results/w2vCTC_FS_normalized/checkpoint-7811/pytorch_model.bin"
# For newer transformers versions the file may be sharded:
if os.path.exists(checkpoint_path):
    state_dict = torch.load(checkpoint_path, map_location="cpu")
else:
    # Try safetensors format
    from safetensors.torch import load_file
    checkpoint_path = "results/w2vCTC_FS_normalized/checkpoint-7811/model.safetensors"
    state_dict = load_file(checkpoint_path)

model.load_state_dict(state_dict)
print("Stage 1 checkpoint loaded successfully")


# =========================
# Step 10: Trainer
# =========================
print("Step 10: Trainer")

training_args = TrainingArguments(
    output_dir                  = output_dir,
    group_by_length             = True,
    length_column_name          = "length",
    per_device_train_batch_size = 16,
    gradient_accumulation_steps = 16,
    evaluation_strategy         = "epoch",
    save_strategy               = "epoch",
    warmup_steps                = 200,
    weight_decay                = 0.005,
    learning_rate             = 1e-6,        # 10× lower than Stage 1
    num_train_epochs          = 10,
    metric_for_best_model     = "alignment_sharpness",
    greater_is_better         = True,
    logging_steps               = 1000,
    gradient_checkpointing      = False,
    max_grad_norm               = 2.0,
    fp16 = False,
    report_to      = "tensorboard",
    logging_dir    = f"{output_dir}/tensorboard",
    bf16 = True,
    load_best_model_at_end = True,
     dataloader_num_workers     = 8,
    dataloader_pin_memory      = True,
    dataloader_prefetch_factor = 2,
    remove_unused_columns       = False,   # we have attention_mask as extra field
)


class FSCTCTrainer(Trainer):
    """
    Trainer with debugging hooks for monitoring:
      - individual loss components (L_ctc, L_fs, L_m)
      - alignment sharpness (is A collapsing or diffuse?)
      - gradient norms per module
      - input_lengths sanity check (catches padding bugs early)
    """
 
    def __init__(self, *args, log_every: int = 100, **kwargs):
        super().__init__(*args, **kwargs)
        self._log_every  = log_every
        self._step_count = 0
        

    # ------------------------------------------------------------------
    # Sampler (unchanged from original)
    # ------------------------------------------------------------------
    def _get_train_sampler(self):
        return LengthGroupedSampler(
            batch_size=self.args.train_batch_size,
            dataset=self.train_dataset,
            lengths=self.train_dataset["length"],
        )
 
    def _get_eval_sampler(self, eval_dataset):
        return None
 
    # ------------------------------------------------------------------
    # compute_loss  — logs individual losses + alignment stats
    # ------------------------------------------------------------------
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(
            input_values   = inputs["input_values"],
            attention_mask = inputs.get("attention_mask"),
            labels         = inputs["labels"],
             global_step    = self.state.global_step
        )
 
        loss      = outputs["loss"]
        loss_ctc  = outputs.get("loss_ctc")
        loss_fs   = outputs.get("loss_fs")
        loss_m    = outputs.get("loss_m")
 
        self._step_count += 1
        should_log = (self._step_count % self._log_every == 0)
 
        if should_log and loss_ctc is not None:
            self._log_loss_components(loss, loss_ctc, loss_fs, loss_m)
            self._log_input_lengths(inputs, model)
 
        return (loss, outputs) if return_outputs else loss
 
    # ------------------------------------------------------------------
    # Log loss components
    # ------------------------------------------------------------------
    def _log_loss_components(self, loss, loss_ctc, loss_fs, loss_m):
        msg = (
            f"[step {self._step_count}] "
            f"loss={loss.item():.4f}  "
            f"L_ctc={loss_ctc.item():.4f}  "
            f"L_fs={loss_fs.item():.4f}  "
            f"L_m={loss_m.item():.4f}"
        )
        logger.warning(msg)   # warning level so it prints even without verbose logging
        print(msg)
 
        # Push to Trainer log history so it appears in logs.json
        self.log({
            "train/loss_ctc": loss_ctc.item(),
            "train/loss_fs":  loss_fs.item(),
            "train/loss_m":   loss_m.item(),
        })
 
        # Sanity checks — warn if any component looks degenerate
        if loss_ctc.item() < 0.01:
            logger.warning("  ⚠  L_ctc is near zero — possible label/blank collapse")
        if loss_fs.item() > 50.0:
            logger.warning("  ⚠  L_fs is very large — alignment may not have started learning")
        if torch.isnan(loss):
            logger.error("  ✗  NaN detected in total loss — stopping recommended")
 
    # ------------------------------------------------------------------
    # Log input_lengths to catch padding / conv-stride bugs
    # ------------------------------------------------------------------
    def _log_input_lengths(self, inputs, model):
        with torch.no_grad():
            attn = inputs.get("attention_mask")
            if attn is None:
                return
            raw_len  = attn.sum(dim=-1).float()
            feat_len = model.wav2vec2._get_feat_extract_output_lengths(
                attn.sum(dim=-1)
            ).float()
            label_len = (inputs["labels"] != -100).sum(dim=-1).float()
 
        msg = (
            f"  lengths — "
            f"raw: {raw_len.min().int().item()}–{raw_len.max().int().item()}  "
            f"frames: {feat_len.min().int().item()}–{feat_len.max().int().item()}  "
            f"labels: {label_len.min().int().item()}–{label_len.max().int().item()}"
        )
        print(msg)
 
        # Frame length must always exceed label length (CTC requirement)
        violations = (feat_len < label_len).sum().item()
        if violations > 0:
            logger.error(
                f"  ✗  {violations} samples where frame_len < label_len "
                f"— CTC will produce -inf loss for those samples"
            )
 
    # ------------------------------------------------------------------
    # Gradient norms — logged once per epoch via on_epoch_end
    # ------------------------------------------------------------------
    #stage1
    """def create_optimizer(self):
        self.optimizer = torch.optim.AdamW([
        {"params": self.model.wav2vec2.parameters(), "lr": 1e-5},
        {"params": self.model.ctc_head.parameters(), "lr": 1e-5},
        {"params": list(self.model.fx.parameters()) +
                   list(self.model.fy.parameters()) +
                   list(self.model.phoneme_embed.parameters()),
         "lr": 1e-4},   # 10× — randomly initialized
        ], weight_decay=self.args.weight_decay)
        return self.optimizer"""
    #stage2
    def create_optimizer(self):
        self.optimizer = torch.optim.AdamW([
            {
                "params": list(self.model.fx.parameters()) +
                          list(self.model.fy.parameters()) +
                          list(self.model.phoneme_embed.parameters()),
                "lr": 1e-6,
            },
            {
                "params": self.model.reconstruction_head.parameters(),
                "lr": 1e-4,
            },
            {
                "params": self.model.quantizer.codevectors,
                "lr": 1e-4,
            },
            {
                "params": self.model.quantizer.weight_proj.parameters(),
                "lr": 1e-3,  # highest — needs to learn fast from scratch
            },
        ], weight_decay=self.args.weight_decay)
        return self.optimizer
    def training_step(self, model, inputs):
        loss = super().training_step(model, inputs)
        
        if self._step_count % self._log_every == 0:
            norms = {}
            for name, module in [
                ("encoder",      model.wav2vec2.encoder),
                ("ctc_head",     model.ctc_head),
                ("fx",           model.fx),
                ("fy",           model.fy),
                ("ph_embed",     model.phoneme_embed),
                ("quantizer",    model.quantizer),
                ("recon_head",   model.reconstruction_head),
            ]:
                total = 0.0
                count = 0
                for p in module.parameters():
                    if p.grad is not None:
                        total += p.grad.norm(2).item() ** 2
                        count += 1
                norms[name] = total ** 0.5 if count > 0 else 0.0
 
            norm_str = "  grad norms — " + "  ".join(
                f"{k}: {v:.3f}" for k, v in norms.items()
            )
            print(norm_str)
 
            # Watch for dead modules (zero grad = not learning)
            dead = [k for k, v in norms.items() if v < 1e-8]
            if dead:
                logger.warning(f"  ⚠  zero gradients in: {dead}")
 
            # Watch for exploding grads in alignment modules
            for k in ("fx", "fy", "ph_embed"):
                if norms.get(k, 0) > 10.0:
                    logger.warning(f"  ⚠  large gradient in {k}: {norms[k]:.2f}")
 
        return loss
 
    # ------------------------------------------------------------------
    # Alignment sharpness — logged at evaluation
    # (sharp A means the model has learned to commit frames to phonemes)
    # ------------------------------------------------------------------
    def evaluate(self, *args, **kwargs):
        sharpness = self._log_alignment_sharpness()        # return the value
        metrics   = super().evaluate(*args, **kwargs)
        if sharpness is not None:
            metrics["eval_alignment_sharpness"] = sharpness  # inject into metrics dict
        return metrics

    def _log_alignment_sharpness(self):
        model = self.model
        model.eval()
        sharpness = None
        try:
            dl    = self.get_eval_dataloader()
            batch = next(iter(dl))
            batch = self._prepare_inputs(batch)
    
            with torch.no_grad():
                outputs = model.wav2vec2(
                    batch["input_values"],
                    attention_mask=batch.get("attention_mask")
                )
                X      = outputs.last_hidden_state
                logits = model.ctc_head(X)
    
                labels_clean = model._ctc_decode_batch(logits)
                Y_emb = model.phoneme_embed(labels_clean)
                A     = model.compute_alignment(X, Y_emb)
    
                eps       = 1e-8
                entropy   = -(A * (A + eps).log()).sum(dim=1)
                mean_H    = entropy.mean().item()
                max_H     = np.log(labels_clean.shape[1] + eps)
                sharpness = 100 * (1 - mean_H / max_H)
    
            print(
                f"\n[alignment @ step {self._step_count}] "
                f"entropy: {mean_H:.3f}/{max_H:.3f}  "
                f"sharpness: {sharpness:.1f}%\n"
            )
            self.log({"eval_alignment_sharpness": sharpness})
    
        except Exception as e:
            logger.warning(f"Alignment sharpness logging failed: {e}")
        finally:
            model.train()
    
        return sharpness   # ← return the value so evaluate() can inject it

 
    # ------------------------------------------------------------------
    # prediction_step (unchanged)
    # ------------------------------------------------------------------
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            outputs = model(
                input_values   = inputs["input_values"],
                attention_mask = inputs.get("attention_mask"),
                labels         = inputs["labels"],
            )
        loss   = outputs["loss"]
        logits = outputs["logits"]
        labels = inputs["labels"]
 
        if prediction_loss_only:
            return (loss, None, None)
        return (loss, logits, labels)


trainer = FSCTCTrainer(
    model           = model,
    data_collator   = data_collator,
    args            = training_args,
    compute_metrics = compute_metrics,
    train_dataset   = dataset["train"],
    eval_dataset    = dataset["validation"],
    log_every=1000,
    tokenizer       = processor.feature_extractor
)
# ── Stage 2 only ──────────────────────────────────────────────────

# After loading the model, before trainer.train()
for p in model.wav2vec2.parameters():
    p.requires_grad_(False)
for p in model.ctc_head.parameters():
    p.requires_grad_(False)
initialize_quantizer(
    model      = model,
    dataloader = trainer.get_train_dataloader(),
    device     = next(model.parameters()).device,
    n_batches  = 200,
)
trainer.train()
trainer.save_model(output_dir)

# =========================
# Save logs
# =========================
logs_path = os.path.join(output_dir, "logs.json")
with open(logs_path, "w") as f:
    json.dump(trainer.state.log_history, f, indent=4)
print(f"Training complete. Logs saved to {logs_path}")