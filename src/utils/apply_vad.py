import torch
import soundfile as sf
import librosa
import numpy as np
from rVADfast import rVADfast

import soundfile as sf
import librosa
import numpy as np

def read_audio_16k(path):
    audio, sr = sf.read(path)

    # mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # resample if needed
    if sr != 16000:
        audio = librosa.resample(
            audio.astype("float32"),
            orig_sr=sr,
            target_sr=16000
        )

    return audio.astype("float32"),sr




def apply_vad_to_wav(
    wav_path,
    out_path,
    sampling_rate=16000
):
    vad_model, utils = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=False)

    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
    # read & resample
    wav,sr = read_audio_16k(wav_path)
    wav_torch = torch.from_numpy(wav)
    # detect speech
    speech_timestamps = get_speech_timestamps(
        wav_torch,
        vad_model,
        sampling_rate=sampling_rate
    )

    # keep only speech
    speech_wav = collect_chunks(
        speech_timestamps,
        wav_torch
    )

    # save speech-only audio
    sf.write(out_path, speech_wav, sampling_rate)

    return speech_timestamps
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

def apply_silero_vad_return_audio(
    wav_path,
    sampling_rate=16000,
    return_tensor=False
):
    """
    Apply Silero VAD and return speech-only audio in memory.

    Args:
        wav_path (str): path to input wav file
        sampling_rate (int): expected sample rate (Silero expects 16k)
        return_tensor (bool): if True, return torch.Tensor instead of np.ndarray

    Returns:
        np.ndarray or torch.Tensor: concatenated speech audio
    """
    # Load Silero VAD
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        trust_repo=True
    )

    (get_speech_timestamps,
     save_audio,
     read_audio,
     VADIterator,
     collect_chunks) = utils
    # Load audio (mono, float32)
    wav,sr = read_audio_16k(wav_path)
    audio = torch.from_numpy(wav)
    # Get speech timestamps
    speech_timestamps = get_speech_timestamps(
        audio,
        model,
        sampling_rate=16000
    )


    if len(speech_timestamps) == 0:
        # No speech detected
        return torch.empty(0) if return_tensor else np.array([])

    # Collect speech chunks (tensor)
    speech_chunks = collect_chunks(speech_timestamps, audio)

    if return_tensor:
        return speech_chunks

    return speech_chunks
def apply_rvad_return_audio(
    wav_path,
    sampling_rate=16000,
):
    """
    Apply rVAD and return speech-only audio in memory.

    Returns:
        np.ndarray: concatenated speech waveform
    """

    # --- Load audio ---
    audio, sr = sf.read(wav_path)

    # Mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resample
    if sr != sampling_rate:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sampling_rate)

    audio = audio.astype(np.float32)

    # --- rVAD ---

    vad = rVADfast()
    vad_labels, vad_timestamps = vad(audio, sampling_rate)
    speech_segments = vad_to_speech_ts(vad_labels, vad_timestamps, sampling_rate)
    # --- Concatenate speech ---
    speech_chunks = []
    for seg in speech_segments:
        speech_chunks.append(audio[seg["start"]:seg["end"]])

    if len(speech_chunks) == 0:
        return np.zeros(0, dtype=np.float32), []

    speech_audio = np.concatenate(speech_chunks)

    return speech_audio

