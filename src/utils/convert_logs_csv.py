import re
import csv
import sys
from pathlib import Path
import os

# Regex to match lines like:
# File: CCM-002710-01_L01.wav | WER=74.874372 | S=80 D=60 I=9
PATTERN = re.compile(
    r"File:\s+(?P<file>\S+)\s+\|\s+"
    r"WER=(?P<wer>[0-9.]+)\s+\|\s+"
    r"S=(?P<s>\d+)\s+D=(?P<d>\d+)\s+I=(?P<i>\d+)"
)

def extract_metrics(log_path, output_csv):
    rows = []

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            match = PATTERN.search(line)
            if match:
                rows.append({
                    "file": match.group("file"),
                    "WER": float(match.group("wer")),
                    "S": int(match.group("s")),
                    "D": int(match.group("d")),
                    "I": int(match.group("i")),
                })

    if not rows:
        print(" No matching lines found in the log.")
        return

    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=["file", "WER", "S", "D", "I"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Extracted {len(rows)} entries into {output_csv}")



if __name__ == "__main__":
    log_path ="/vol/experiments3/imbenamor/TAPAS-FRAIS/logs/silero/"
    csv_folder = "/vol/experiments3/imbenamor/TAPAS-FRAIS/logs/csv_files/typaloc/"

    for f in os.listdir(log_path):
        if f.endswith(".log") and "CEREB" in f or "PARK" in f or "SLA" in f or "CTRL" in f or "CTR" in f:
            extract_metrics(log_path + f, csv_folder+f.split(".")[0]+"silero.csv")
    #extract_metrics("/vol/experiments3/imbenamor/TAPAS-FRAIS/logs/wer_typaloc_SLA_wav2vec.log","/vol/experiments3/imbenamor/TAPAS-FRAIS/logs/csv_files/wer_typaloc_SLA_wav2vec.csv")