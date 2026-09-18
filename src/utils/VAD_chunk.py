import torch
import torchaudio

import numpy as np
from rVADfast import rVADfast
import logging
from pathlib import Path
import tempfile
import subprocess
import soundfile as sf
import torch
import os
import numpy as np
#from espnet2.bin.asr_inference import Speech2Text

logger = logging.getLogger(__name__)
def vad_to_speech_ts(vad_labels, vad_timestamps, sampling_rate):
    speech_ts = []
    start = None

    for label, t in zip(vad_labels, vad_timestamps):
        if label == 1 and start is None:
            start = int(t * sampling_rate)

        elif label == 0 and start is not None:
            end = int(t * sampling_rate)
            speech_ts.append({"start": start, "end": end})
            start = None

    if start is not None:
        speech_ts.append({
            "start": start,
            "end": int(vad_timestamps[-1] * sampling_rate)
        })

    return speech_ts
def transcribe_audio(
    bash_script,
    wav_path=None,
    audio=None,
    sr=16000,
    work_dir=None,
    min_duration=0.3,  # secondes
):
    """
    Transcrit un audio avec Kaldi (recognizer.sh).

    Entrées possibles :
      - wav_path : chemin vers un fichier wav
      - audio    : chunk audio (torch.Tensor ou np.ndarray)
    """

    bash_script = Path(bash_script)
    if not bash_script.exists():
        raise FileNotFoundError(bash_script)

    tmp_wav_path = None

    # =====================================================
    # 1) Préparation de l'audio
    # =====================================================
    if audio is not None:
        # Tensor -> numpy
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().numpy()

        # mono
        if audio.ndim == 2:
            audio = audio.squeeze()

        if audio.ndim != 1:
            raise ValueError(f"Audio must be mono, got shape {audio.shape}")

        # durée minimale (Kaldi n'aime pas les chunks trop courts)
        if len(audio) < int(min_duration * sr):
            return ""

        # format sûr pour Kaldi / libsndfile
        audio = audio.astype(np.float32)

        tmp_wav = tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False
        )
        tmp_wav_path = Path(tmp_wav.name)

        sf.write(
            tmp_wav_path,
            audio,
            sr,
            format="WAV",
            subtype="PCM_16"
        )

        wav_path = tmp_wav_path

    else:
        # =================================================
        # 2) Cas wav sur disque
        # =================================================
        if wav_path is None:
            raise ValueError("Provide either wav_path or audio")

        wav_path = Path(wav_path)
        if not wav_path.exists():
            raise FileNotFoundError(wav_path)

    # =====================================================
    # 3) Dossier de travail
    # =====================================================
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp())
    else:
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

    ctm_path = work_dir / f"{wav_path.stem}.ctm"

    # =====================================================
    # 4) Appel Kaldi
    # =====================================================
    cmd = [
        str(bash_script),
        str(wav_path),
        str(ctm_path)
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"ASR failed:\n{result.stderr}")

    # =====================================================
    # 5) CTM → texte
    # =====================================================
    words = []
    if ctm_path.exists():
        with open(ctm_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    word = parts[4]
                    if word != "<eps>":
                        words.append(word)

    # =====================================================
    # 6) Nettoyage
    # =====================================================
    if tmp_wav_path is not None and tmp_wav_path.exists():
        tmp_wav_path.unlink()

    return " ".join(words)
def vad_chunk_with_timestamps_v1(
    wav,
    sampling_rate=16000,
    max_chunk_duration=30.0,
    max_pause_duration=0.6,
    min_chunk_duration=2.0      # ← new parameter
):
    logger.info("Using rVAD model")
    vad = rVADfast()
    vad_labels, vad_timestamps = vad(wav, sampling_rate)
    speech_ts = vad_to_speech_ts(vad_labels, vad_timestamps, sampling_rate)

    chunks = []
    chunk_start = None
    chunk_end = None

    for seg in speech_ts:
        seg_start = seg["start"] / sampling_rate
        seg_end   = seg["end"]   / sampling_rate

        if chunk_start is None:
            chunk_start = seg_start
            chunk_end   = seg_end
            continue

        pause = seg_start - chunk_end

        if pause <= max_pause_duration:
            if seg_end - chunk_start <= max_chunk_duration:
                chunk_end = seg_end
            else:
                chunks.append({"start": chunk_start, "end": chunk_end})
                chunk_start = seg_start
                chunk_end   = seg_end
        else:
            chunks.append({"start": chunk_start, "end": chunk_end})
            chunk_start = seg_start
            chunk_end   = seg_end

    if chunk_start is not None:
        chunks.append({"start": chunk_start, "end": chunk_end})

    # ── merge short chunks with their neighbour ──────────────────────────
    merged = []
    for chunk in chunks:
        if merged and (chunk["end"] - chunk["start"]) < min_chunk_duration:
            merged[-1]["end"] = chunk["end"]   # absorb into previous
        else:
            merged.append(chunk)

    # edge case: first chunk is short → absorb into next
    if len(merged) > 1 and (merged[0]["end"] - merged[0]["start"]) < min_chunk_duration:
        merged[1]["start"] = merged[0]["start"]
        merged.pop(0)

    return merged
    
"""def vad_chunk_with_timestamps(
    wav,
    sampling_rate=16000,
    max_chunk_duration=30.0,
    max_pause_duration=0.6
):
    
    wav: torch.Tensor (1D, 16kHz)
    returns: list of dicts with start/end in ORIGINAL time
    
    logger.info("Using Silero VAD model")
    vad_model, utils = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=False)

    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
    speech_ts = get_speech_timestamps(wav,vad_model,sampling_rate=sampling_rate)
    logger.info("Using rVAD model")
    vad = rVADfast()
    vad_labels, vad_timestamps = vad(wav, sampling_rate)
    speech_ts = vad_to_speech_ts(vad_labels, vad_timestamps, sampling_rate)
    chunks = []

    chunk_start = None
    chunk_end = None

    for seg in speech_ts:
        seg_start = seg["start"] / sampling_rate
        seg_end = seg["end"] / sampling_rate

        if chunk_start is None:
            chunk_start = seg_start
            chunk_end = seg_end
            continue

        pause = seg_start - chunk_end

        # ❗ règle 1 : pause courte → on fusionne
        if pause <= max_pause_duration:
            # mais on respecte la durée max
            if seg_end - chunk_start <= max_chunk_duration:
                chunk_end = seg_end
            else:
                chunks.append({
                    "start": chunk_start,
                    "end": chunk_end
                })
                chunk_start = seg_start
                chunk_end = seg_end

        # ❗ règle 2 : pause longue → nouvelle unité
        else:
            chunks.append({
                "start": chunk_start,
                "end": chunk_end
            })
            chunk_start = seg_start
            chunk_end = seg_end

    # dernier chunk
    if chunk_start is not None:
        chunks.append({
            "start": chunk_start,
            "end": chunk_end
        })

    return chunks
"""

def extract_chunk_audio(wav, start, end, sr=16000):
    s = int(start * sr)
    e = int(end * sr)
    return wav[s:e].unsqueeze(0)   # (1, T)

def whisper_transcribe_chunks(
    asr_model,model,
    wav,
    chunks,
    sr=16000,
output_dir="/vol/experiments3/imbenamor/TAPAS-FRAIS/data/chunk_audio"
):
    os.makedirs(output_dir, exist_ok=True)
    results = []

    for i, ch in enumerate(chunks):
        chunk_wav = extract_chunk_audio(
            wav,
            ch["start"],
            ch["end"],
            sr
        )
        # Ensure shape = [1, num_samples]
        #if chunk_wav.dim() == 1:
            #chunk_wav = chunk_wav.unsqueeze(0)

        # 🔹 Save chunk to file
        #chunk_path = os.path.join(output_dir, f"chunk_{i}.wav")
        #torchaudio.save(chunk_path, chunk_wav.cpu(), sr)
        wav_lens = torch.tensor([1.0])

        pred = asr_model.transcribe_batch(
            chunk_wav,
            wav_lens)
        #print(pred)
        if model=="whisper-VAD-chunk" or model=="whisper-large-VAD-chunk":
            hyp = " ".join(pred[0][0]).strip()
        else:
            hyp = " ".join(pred[0]).strip()
        results.append({
            "id": i,
            "start": ch["start"],
            "end": ch["end"],
            "text": hyp.strip()
        })
        #print(f"Durée chunk = {ch['end'] - ch['start']}")

    return results
def wav2vec2spont_transcribe_chunks(
    model,
    wav,
    chunks,
    sr=16000
):
    results = []

    for i, ch in enumerate(chunks):
        chunk_wav = extract_chunk_audio(
            wav,
            ch["start"],
            ch["end"],
            sr
        )

        # remove channel dimension if exists
        chunk_wav = chunk_wav.squeeze()

        # resample if needed
        if sr != 16000:
            chunk_wav = torchaudio.functional.resample(chunk_wav, sr, 16000)

        inputs = processor(
            chunk_wav,
            sampling_rate=16000,
            return_tensors="pt"
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits

        pred_ids = torch.argmax(logits, dim=-1)
        text = processor.batch_decode(pred_ids)[0]

        print(f"Chunk {i}: {text}")
        results.append({
            "id": i,
            "start": ch["start"],
            "end": ch["end"],
            "text": text.strip()
        })
        #print(f"Durée chunk = {ch['end'] - ch['start']}")

    return results
def hmmtdnn_transcribe_chunks(script,wav,chunks,work_dir,sr=16000 ):
    results = []

    for i, ch in enumerate(chunks):
        chunk_wav = extract_chunk_audio(
            wav,
            ch["start"],
            ch["end"],
            sr
        )

        wav_lens = torch.tensor([1.0])

        pred = transcribe_audio(bash_script=script,audio =chunk_wav, work_dir=work_dir)
        hyp = pred


        results.append({
            "id": i,
            "start": ch["start"],
            "end": ch["end"],
            "text": hyp.strip()
        })
        #print(results)

    return results


def espnet_transcribe_chunks(
    model,
    wav,
    chunks,
    sr=16000
):
    results = []

    for i, ch in enumerate(chunks):
        dur = ch["end"] - ch["start"]
        if dur < 0.5:  # skip very short segments
            continue
        chunk_wav = extract_chunk_audio(
            wav,
            ch["start"],
            ch["end"],
            sr
        )
        if isinstance(chunk_wav, np.ndarray):
            chunk_wav = torch.from_numpy(chunk_wav)

        chunk_wav = chunk_wav.float().squeeze().cpu()
        #wav_lens = torch.tensor([1.0])
        res = model(speech=chunk_wav)
        nbests = [text for text, token, token_int, hyp in res]
        pred_transcriptions = nbests[0] if nbests is not None and len(nbests) > 0 else ""


        results.append({
            "id": i,
            "start": ch["start"],
            "end": ch["end"],
            "text": pred_transcriptions
        })

    return results

def ref_text_for_chunk(words, start, end):
    return " ".join(
        w for w, s, e in words
        if e > start and s < end
    )
