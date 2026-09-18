import argparse
import logging
import sys
import os
from speechbrain.inference.ASR import WhisperASR
from speechbrain.inference.ASR import EncoderASR
from speechbrain.utils.metric_stats import ErrorRateStats
from textgrid import TextGrid
from utils.read_transcription import *
from utils.normalise_text import normalization
from pathlib import Path
from hyperpyyaml import load_hyperpyyaml




parser = argparse.ArgumentParser(description="Calculate wer of rouas models")
parser.add_argument("--pred_trans", type=str,required=True, help="The path to the wav files")
parser.add_argument("--ref_trans", type=str,required=True, help="The reference transcription")
parser.add_argument("--log_file", type=str,required=True, help="The logfile name")

args = parser.parse_args()
log_file = Path("/vol/experiments3/imbenamor/TAPAS-FRAIS/logs/"+ args.log_file + ".log")
wer_hparams = load_hyperpyyaml("""wer_stats: !new:speechbrain.utils.metric_stats.ErrorRateStats""")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode="w"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)
def main(args):
    #-------------------------------- Model inference -------------------------------
    #--------------------------------------------------------------------------------
    WERs = []

    tg_map={}
    for f in os.listdir(args.ref_trans):
        print(f)
        if "Rhapsodie" in args.ref_trans:
            tg_map[f] = f.split("-")[0]+"-"+f.split("-")[1]+".txt" #Rhap-D0005-Pro.TextGrid = Rhap-D0005.TextGrid
        else:
            tg_map[f] = f

        wer_hparams["wer_stats"].clear()
        ref_transcriptions = get_textgrid_transcription_tapas(os.path.join(args.ref_trans, f))
        with open(os.path.join(args.pred_trans, tg_map[f].split(".")[0]+".trn"), "r", encoding="utf-8") as file:
            pred_transcriptions = file.read()
        wer_hparams["wer_stats"].append(ids=list(range(len(ref_transcriptions))),predict=[normalization(pred_transcriptions)[:-1]],target=[normalization(ref_transcriptions)])

        stats = wer_hparams["wer_stats"].summarize()
        logger.info(
                        "File: %s | WER=%f | S=%d D=%d I=%d",
                        f.split(".")[0],
                        stats["WER"],
                        stats["substitutions"],
                        stats["deletions"],
                        stats["insertions"],
                    )
        logger.info("-" * 30)
        WERs.append(stats["WER"])
    
if __name__ == "__main__":
    main(args)