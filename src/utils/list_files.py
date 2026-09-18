import os

def list_files(trans_dir):
    tg_to_wav = {}
    for f in sorted(os.listdir(trans_dir)):
        if f.endswith(".TextGrid") or f.endswith(".textgrid") or f.endswith(".txt") and not f.endswith("pr_analyse.TextGrid") and not f.endswith("_extralinguistiques.txt") and not f.startswith("._"):
            if "Rhapsodie" in trans_dir:
                tg_to_wav[f] = f.split("-")[0] + "-" + f.split("-")[1] + ".wav"
            elif "Monologue" in trans_dir:
                tg_to_wav[f] = f.split(".")[0] + "-image.wav"
            else:
                tg_to_wav[f] = f.split(".")[0] + ".wav"
        if "spont" in trans_dir:
            if f.endswith(".TextGrid") or f.endswith(".textgrid"):
                tg_to_wav[f] = f.split("_")[0]+"_"+f.split("_")[1]+".wav"

    return tg_to_wav