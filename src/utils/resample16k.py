import os
import soundfile as sf
import librosa
import numpy as np
import logging

def resample_dir(
    in_dir,
    out_dir,
    target_sr=16000,
    ext=".wav"
):
    os.makedirs(out_dir, exist_ok=True)

    for fname in os.listdir(in_dir):
        if fname.startswith("._"):
            continue
        if not fname.lower().endswith(ext):
            continue

        in_path = os.path.join(in_dir, fname)

        # read header only
        info = sf.info(in_path)

        # already correct → skip
        if info.samplerate == target_sr and info.channels == 1:
            #print(f"= {fname} already {target_sr} Hz → skipped")
            continue

        # load full audio
        audio, sr = sf.read(in_path)

        # mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        # resample
        audio = librosa.resample(
            audio.astype(np.float32),
            orig_sr=sr,
            target_sr=target_sr
        )

        out_path = os.path.join(out_dir, fname)
        sf.write(out_path, audio, target_sr)

        logging.info(f"✔ {fname} resampled {sr} → {target_sr}")

    logging.info("Done.")
