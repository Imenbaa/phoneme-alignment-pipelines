"""Whisper encoder with a CTC head, for frame-level phoneme recognition.

The decoder is never instantiated: only the encoder runs, and a fresh linear
head maps its hidden states onto the phoneme vocabulary. ``config.vocab_size``
must be overridden at load time to match that vocabulary.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WhisperConfig, WhisperPreTrainedModel
from transformers.modeling_outputs import CausalLMOutput
from transformers.models.whisper.modeling_whisper import WhisperEncoder


class WhisperEncoderForCTC(WhisperPreTrainedModel):
    """Whisper encoder + linear CTC head."""

    config_class = WhisperConfig
    main_input_name = "input_features"

    def __init__(self, config):
        super().__init__(config)
        self.encoder = WhisperEncoder(config)
        self.dropout = nn.Dropout(getattr(config, "final_dropout", 0.0))
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)
        self.post_init()

    def freeze_conv_frontend(self):
        """Freeze the two convolutional layers below the transformer blocks."""
        for layer in (self.encoder.conv1, self.encoder.conv2):
            for p in layer.parameters():
                p.requires_grad = False

    def forward(self, input_features=None, labels=None, attention_mask=None, **kwargs):
        encoder_out = self.encoder(input_features).last_hidden_state  # (B, T, d_model)
        logits = self.lm_head(self.dropout(encoder_out))              # (B, T, V)

        loss = None
        if labels is not None:
            # CTC expects (T, B, V) log-probs in float32.
            log_probs = F.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)
            input_lengths = torch.full(
                (logits.shape[0],),
                logits.shape[1],
                dtype=torch.long,
                device=logits.device,
            )
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flat_targets = labels.masked_select(labels_mask)
            # cuDNN's CTC has stricter shape constraints; the native kernel is safer.
            with torch.backends.cudnn.flags(enabled=False):
                loss = F.ctc_loss(
                    log_probs,
                    flat_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction="mean",
                    zero_infinity=True,
                )
        return CausalLMOutput(loss=loss, logits=logits)
