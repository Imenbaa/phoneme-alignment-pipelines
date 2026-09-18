import pandas as pd
import ast
import os
import torch
import torchaudio
import shutil

df1=pd.read_csv("/vol/experiments3/imbenamor/TAPAS-FRAIS/src/utils/pred_rhap.csv")
#this column is specific to Rhapsodie
df1["filename1"]=[i.split("-")[1].split(".")[0]+"-"+i.split("-")[0]+".wav" for i in df1["filename"]]
df1["speaker_id"]=[i.split("-")[0] for i in df1["filename1"]]

corpus_dir = "/vol/experiments3/imbenamor/TAPAS-FRAIS/data/mfa_rhap_w2vctc"
if os.path.exists(corpus_dir):
    shutil.rmtree(corpus_dir)  

os.makedirs(corpus_dir)
target_sr = 16000
wav_path = "/vol/corpora/Rhapsodie/wav16k_corrected"
for idx, row in df1.iterrows():
    audio_path = os.path.join(wav_path, row["filename"])
    tokens = row["predicted_phonemes"]
    audio_path1 = os.path.join(wav_path, row["filename1"])
    utt_id = os.path.splitext(os.path.basename(audio_path1))[0]
    speaker_id = str(row["speaker_id"])   

    # Create speaker folder
    speaker_dir = os.path.join(corpus_dir, speaker_id)
    os.makedirs(speaker_dir, exist_ok=True)
    # Load audio
    waveform, sr = torchaudio.load(audio_path)
    # Resample 
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)

    # Save wav inside speaker folder
    torchaudio.save(
        os.path.join(speaker_dir, f"{utt_id}.wav"),
        waveform,
        target_sr
    )

    # Save lab inside speaker folder
    with open(os.path.join(speaker_dir, f"{utt_id}.lab"), "w", encoding="utf-8") as f:
        f.write(tokens.strip())

#Check phoneme inventory
hyp_inventory = set()
for seq in df1["predicted_phonemes"].dropna():
    tokens = seq.split()
    hyp_inventory.update(tokens)
print(sorted(hyp_inventory))
print("Number of unique phonemes:", len(hyp_inventory))

#Create dictionary for mfa
with open("/vol/experiments3/imbenamor/TAPAS-FRAIS/data/output_w2vrecognition/phoneme_rhap_ctc.txt", "w", encoding="utf-8") as f:
    for ph in hyp_inventory:
        f.write(f"{ph} {ph}\n")

