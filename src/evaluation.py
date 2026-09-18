import argparse
import logging
import os
import csv
from transformers import AutoProcessor, AutoModelForCTC
import torchaudio
from utils.save_trans import *
from speechbrain.inference.ASR import WhisperASR
from speechbrain.inference.ASR import EncoderASR
from utils.read_transcription import *
from utils.normalise_text import *
from pathlib import Path
from hyperpyyaml import load_hyperpyyaml
import librosa
import torch
from utils.meta import get_audio_info
from utils.apply_vad import *
from utils.list_files import list_files
from utils.VAD_chunk import *
from utils.wer_chunk import wer_chunk
from utils.logging_config import setup_logging
from utils.wer_segment import wer_segment
from pathlib import Path
models = ["wav2vec2-spont","wav2vec","whisper-VAD-chunk","whisper-large","whisper-large-VAD-chunk","wav2vec2-VAD-chunk","conf_cv","conf_ester"]

parser = argparse.ArgumentParser(description="Evaluate multiple ASR models on french datasets")
parser.add_argument("--model", type=str,choices = models ,required= True, help="The ASR model")
parser.add_argument("--data_name", type=str,required=True, help="The name of dataset")
parser.add_argument("--wav_data", type=str,required=True, help="The path to the wav files")
parser.add_argument("--ref_trans", type=str,required=True, help="The reference transcription")
parser.add_argument("--log_file", type=str,required=True, help="The logfile name")

args = parser.parse_args()

log_file = Path("/vol/experiments3/imbenamor/TAPAS-FRAIS/logs/"+ args.log_file + ".log")
trans = "/vol/experiments3/imbenamor/TAPAS-FRAIS/logs/transcription_files/"+args.data_name+"/"
csv_path = "/vol/experiments3/imbenamor/TAPAS-FRAIS/logs/csv_files/"+args.data_name+"/"
wer_hparams = load_hyperpyyaml("""wer_stats: !new:speechbrain.utils.metric_stats.ErrorRateStats""")
setup_logging(log_file)
logger = logging.getLogger(__name__)
os.makedirs(csv_path, exist_ok=True)
os.makedirs(trans, exist_ok=True)

def main(args):
    #-------------------------------- ASR models      -------------------------------
    #--------------------------------------------------------------------------------
    if args.model == "wav2vec" or args.model == "wav2vec2-VAD-chunk":
        asr_model = EncoderASR.from_hparams(source="/vol/experiments3/imbenamor/TAPAS-FRAIS/models/asr-wav2vec2-commonvoice-fr", savedir="/vol/experiments3/imbenamor/TAPAS-FRAIS/models/asr-wav2vec2-commonvoice-fr", run_opts={"device":"cuda"})
    if args.model == "wav2vec2-spont":
        asr_model = EncoderASR.from_hparams(source="/vol/experiments3/imbenamor/TAPAS-FRAIS/models/asr-wav2vec2-LB7K-spontaneous-fr", savedir="/vol/experiments3/imbenamor/TAPAS-FRAIS/models/asr-wav2vec2-LB7K-spontaneous-fr", run_opts={"device":"cuda"})

    if args.model == "whisper-medium" or args.model == "whisper-VAD-chunk":
        asr_model = WhisperASR.from_hparams(source="/vol/experiments3/imbenamor/TAPAS-FRAIS/models/asr-whisper-medium-commonvoice-fr",savedir="/vol/experiments3/imbenamor/TAPAS-FRAIS/models/asr-whisper-medium-commonvoice-fr", run_opts={"device":"cuda"})
    if args.model == "whisper-large" or args.model == "whisper-large-VAD-chunk":
        asr_model = WhisperASR.from_hparams(source="/vol/experiments3/imbenamor/TAPAS-FRAIS/models/asr-whisper-large-v2-commonvoice-fr",savedir="/vol/experiments3/imbenamor/TAPAS-FRAIS/models/asr-whisper-large-v2-commonvoice-fr", run_opts={"device":"cuda"})

    number_files = 0
    args.ref_trans = unicodedata.normalize("NFD", args.ref_trans)
    if args.wav_data.endswith("PD") or args.wav_data.endswith("MSA"):
        tg_to_wav = {}
        for w in os.listdir(args.wav_data):
            for t in os.listdir(args.ref_trans):
                #if w != '1HC-IAJC_éléments_extralinguisitiques-image.wav' and t !='1HC-IAJC_éléments_extralinguisitiques.txt':
                if not t.startswith("._"):
                    if w.split("-")[1] in t:
                            tg_to_wav[t] = w
    else:
        tg_to_wav = list_files(args.ref_trans)
    #print(tg_to_wav.keys(),tg_to_wav.values())
    csv_results = Path(csv_path+args.model+"-results.csv")
    csv_trans = Path(trans + args.model+"-trans.csv")
    fieldnames1 = ["filename", "duration_sec", "samplerate", "channels", "wer", "S", "D", "I", "N"]
    fieldnames2 = ["filename", "ref_trans", "pred_trans"]
    with open(csv_results, "a", newline="", encoding="utf-8") as f1, \
            open(csv_trans, "a", newline="", encoding="utf-8") as f2:

        writer1 = csv.DictWriter(f1, fieldnames=fieldnames1)
        writer2 = csv.DictWriter(f2, fieldnames=fieldnames2)
        writer1.writeheader()
        writer2.writeheader()
        for tg,w in tg_to_wav.items():
            wav_file = os.path.join(args.wav_data, w)
            trans_file = os.path.join(args.ref_trans, tg)
            if os.path.exists(wav_file) and os.path.exists(trans_file) and tg !="CCM-004773-01_L01.TextGrid" and w !="CCM-004773-01_L01.wav":
                number_files += 1
                logging.info(f" File duration: {librosa.get_duration(filename=wav_file)} seconds")
                info = get_audio_info(wav_file)
                if "Rhapsodie" in args.ref_trans:
                    ref_transcriptions = get_textgrid_transcription_rhap(trans_file)
                elif trans_file.endswith(".txt"):
                    ref_transcriptions = read_preprocess_transcription(trans_file)
                #elif "spont" in args.ref_trans:
                    #ref_transcriptions = extract_words_text(trans_file)
                else:
                    ref_transcriptions = get_textgrid_transcription_tapas(trans_file)

                # load audio
                audio_np, sr = read_audio_16k(wav_file)
                wav = torch.from_numpy(audio_np)
                # VAD + chunking
                chunks = vad_chunk_with_timestamps(wav)
                logging.info("Number of chunks: %d", len(chunks))
                #if args.model == "wa2vec2-spont":
                    #results=wav2vec2spont_transcribe_chunk(model,wav,chunks)
                #if args.model == "whisper-VAD-chunk" or args.model == "wav2vec2-VAD-chunk" or args.model == "whisper-large-VAD-chunk":
                    #results = whisper_transcribe_chunks(asr_model, args.model, wav, chunks)
                if args.model == "whisper-VAD-chunk" or args.model == "wav2vec2-VAD-chunk" or args.model == "whisper-large-VAD-chunk" or args.model == "wav2vec2-spont":
                    results = whisper_transcribe_chunks(asr_model, args.model, wav, chunks)

                    if "Daoudi" not in args.ref_trans:

                        if "Rhapsodie" in args.ref_trans:
                            words = get_textgrid_transcription_rhap_chunk(trans_file)
                        else:
                            words = get_textgrid_transcription_chunk(trans_file)

                        ref_transcriptions,pred_transcriptions=wer_chunk(results,words)
                    else:
                        ref_transcriptions = remove_words(ref_transcriptions)
                        pred_transcriptions = " ".join(r["text"] for r in results if r["text"].strip())
                else:
                    pred_transcriptions = asr_model.transcribe_file(wav_file)
                    #if args.model == "whisper-medium" or args.model == "whisper-large":
                    pred_transcriptions = " ".join(seg.words for seg in pred_transcriptions)

                logger.info("-" * 30)
                logger.info("-" * 30)
                # -------------------------------- WER per file   -------------------------------
                # --------------------------------------------------------------------------------
                WER,S,D,I,N = wer_segment(wav_file,ref_transcriptions,pred_transcriptions)
                writer1.writerow({"filename": w,**info,"wer":WER,"S":S,"D":D,"I":I,"N":N})
                writer2.writerow({"filename":w,"ref_trans":normalization(ref_transcriptions),"pred_trans":normalization(pred_transcriptions)})
                #break
            else:
                logging.warning(f"The file {wav_file} does not exist")

if __name__ == "__main__":
    main(args)