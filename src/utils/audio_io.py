"""Audio loading helpers shared by the alignment front-ends.

Nothing here knows about a particular corpus: every function takes a path and
returns plain arrays / seconds.
"""

import numpy as np
import soundfile as sf

DEFAULT_SAMPLE_RATE = 16000


def load_audio_mono(audio_path, target_sr=DEFAULT_SAMPLE_RATE):
    """Read an audio file as mono float32 at `target_sr`.

    Returns:
        (samples, sample_rate)
    """
    audio, sr = sf.read(audio_path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != target_sr:
        import librosa  # lazy: only needed when the file needs resampling

        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return np.asarray(audio, dtype=np.float32), target_sr


def audio_duration(audio_path):
    """Duration in seconds, read from the file header (no decoding)."""
    info = sf.info(str(audio_path))
    return info.frames / float(info.samplerate)
