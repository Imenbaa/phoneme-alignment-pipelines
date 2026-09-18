import json
from datasets import load_from_disk

preprocessed_path = "results/w2vCTC_VAD_SIL/preprocessed"
lengths_path = "/vol/experiments/cache_imbenamor/lengths_w2vCTC_VAD_SIL.json"

dataset = load_from_disk(preprocessed_path)

lengths = {}
for split in ["train", "validation", "test"]:
    print(f"Computing lengths for {split}...")
    split_lengths = []
    for i in range(0, len(dataset[split]), 1000):
        batch = dataset[split][i:i+1000]["input_values"]
        split_lengths.extend([len(iv) for iv in batch])
    lengths[split] = split_lengths
    print(f"{split}: {len(split_lengths)} examples")

with open(lengths_path, "w") as f:
    json.dump(lengths, f)

print("Done.")