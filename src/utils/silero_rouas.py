import torch
import numpy as np


def silero_to_frame_labels(waveform, sr, frame_duration=0.02):
    """
    Convert Silero speech timestamps to frame-level labels
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        trust_repo=True
    )
    model = model.to(device).eval()
    (get_speech_timestamps,
     save_audio,
     read_audio,
     VADIterator,
     collect_chunks) = utils
    if isinstance(waveform, np.ndarray):
        waveform = torch.from_numpy(waveform).float()
    waveform = waveform.to(device)
    speech_segments = get_speech_timestamps(
        waveform,
        model,
        sampling_rate=sr
    )
    waveform = waveform.cpu()
    frame_samples = int(frame_duration * sr)
    total_frames = int(len(waveform) / frame_samples)

    vad_labels = np.zeros(total_frames, dtype=int)
    vad_timestamps = np.arange(total_frames) * frame_duration

    for seg in speech_segments:
        start_f = int(seg['start'] / frame_samples)
        end_f = int(seg['end'] / frame_samples)
        vad_labels[start_f:end_f] = 1

    return vad_labels, vad_timestamps
