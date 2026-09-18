import argparse
import logging
import os
import torch
from utils.read_transcription import *
from utils.normalise_text import *
from pathlib import Path
from hyperpyyaml import load_hyperpyyaml
from espnet2.bin.asr_inference import Speech2Text
from utils.apply_vad import *
from utils.list_files import list_files
from utils.VAD_chunk import *
from utils.wer_chunk import wer_chunk
from utils.logging_config import setup_logging
from utils.wer_segment import wer_segment

parser = argparse.ArgumentParser(description='Process an audio file with ESPnet2 ASR.')
parser.add_argument('--wav_data', help='Path to the input WAV file or folder')
parser.add_argument("--ref_trans", type=str,required=True, help="The reference transcription")
parser.add_argument('--config', default='/home/rouas/experiments/SpeechRecognition/saved_models/espnet2-conformer-FR/asr_conformer_config.yaml', help='Path to the ASR config file (default: asr_config.yml)')
parser.add_argument('--model', default='/home/rouas/experiments/SpeechRecognition/saved_models/espnet2-conformer-FR/asr_conformer.pth', help='Path to the ASR model file (default: asr.pth)')
#parser.add_argument("--vad", type=str,required=True, help="The logfile name")
parser.add_argument("--log_file", type=str,required=True, help="The logfile name")

args = parser.parse_args()
log_file = Path("/vol/experiments3/imbenamor/TAPAS-FRAIS/logs/espnet/ester/"+ args.log_file + ".log")
wer_hparams = load_hyperpyyaml("""wer_stats: !new:speechbrain.utils.metric_stats.ErrorRateStats""")
setup_logging(log_file)
logger = logging.getLogger(__name__)

def main(args):
    # -------------------------------- ASR models      -------------------------------
    # --------------------------------------------------------------------------------
    speech2text = Speech2Text(args.config, args.model,nbest=1,minlenratio=0.1,beam_size=40, device="cuda",dtype="float32")

    WERs = []
    number_files = 0
    args.ref_trans = unicodedata.normalize("NFD", args.ref_trans)
    print(os.path.exists(args.ref_trans))
    if args.wav_data.endswith("PD") or args.wav_data.endswith("MSA"):

        tg_to_wav = {}
        for w in os.listdir(args.wav_data):
            for t in os.listdir(args.ref_trans):
                if not t.startswith("._"):
                    if w.split("-")[1] in t:
                        tg_to_wav[t] = w
    else:
        tg_to_wav = list_files(args.ref_trans)
    for tg,wav in tg_to_wav.items():
        wav_file = os.path.join(args.wav_data, wav)
        trans_file = os.path.join(args.ref_trans, tg)
        if os.path.exists(wav_file) and os.path.exists(trans_file) and tg !="CCM-004773-01_L01.TextGrid" and wav != "CCM-004773-01_L01.wav":
            number_files += 1
            logging.info(f" File duration: {librosa.get_duration(filename=wav_file)} seconds")
            # load audio+apply silero VAD
            #if args.vad=="rvad":
               # audio_np=apply_rvad_return_audio(wav_file)
            #else:
               # audio_np=apply_silero_vad_return_audio(wav_file)
            # load audio
            audio_np, sr = read_audio_16k(wav_file)
            wav = torch.from_numpy(audio_np)
            # VAD + chunking
            chunks = vad_chunk_with_timestamps(wav)

            results=espnet_transcribe_chunks(speech2text,wav,chunks,sr=16000)
            if "Daoudi" not in args.ref_trans:

                if "Rhapsodie" in args.ref_trans:
                    words = get_textgrid_transcription_rhap_chunk(trans_file)
                else:
                    words = get_textgrid_transcription_chunk(trans_file)

                ref_transcriptions, pred_transcriptions = wer_chunk(results, words)
            else:
                ref_transcriptions = read_preprocess_transcription(trans_file)
                ref_transcriptions = remove_words(ref_transcriptions)
                #ref_transcriptions = clean_french_disfluencies(remove_words(ref_transcriptions))
                pred_transcriptions = " ".join(r["text"] for r in results if r["text"].strip())

            logging.info(normalization(pred_transcriptions))
            logging.info(normalization(ref_transcriptions))
            # -------------------------------- WER per file   -------------------------------
            # --------------------------------------------------------------------------------
            WER = wer_segment(wav_file, ref_transcriptions, pred_transcriptions)
            WERs.append(WER)
            #break
        else:
            logging.warning(f"The file {wav_file} does not exist")
    
if __name__ == "__main__":
    main(args)
