import argparse
import logging
from espnet2.bin.asr_inference import Speech2Text
import gc
import os
import csv
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
models = ["wav2vec","whisper-VAD-chunk","whisper-large","whisper-medium","whisper-large-VAD-chunk","wav2vec2-VAD-chunk"]

parser = argparse.ArgumentParser(description="Evaluate multiple ASR models on french datasets")
#parser.add_argument("--model", type=str,choices = models ,required= True, help="The ASR model")
parser.add_argument("--wav_data", type=str,required=True, help="The path to the wav files")
parser.add_argument("--ref_trans", type=str,required=True, help="The reference transcription")
parser.add_argument("--log_file", type=str,required=True, help="The logfile name")
parser.add_argument("--csv_path", type=str,required=True, help="The csv name")
parser.add_argument("--pred_folder", type=str,default="/vol/experiments3/imbenamor/TAPAS-FRAIS/logs/transcription/rhap/", help="The csv name")

args = parser.parse_args()

log_file = Path("/vol/experiments3/imbenamor/TAPAS-FRAIS/logs/"+ args.log_file + ".log")
wer_hparams = load_hyperpyyaml("""wer_stats: !new:speechbrain.utils.metric_stats.ErrorRateStats""")
setup_logging(log_file)
logger = logging.getLogger(__name__)


def main(args):
    #-------------------------------- ASR models      -------------------------------
    #--------------------------------------------------------------------------------
    w2v = EncoderASR.from_hparams(source="/vol/experiments3/imbenamor/TAPAS-FRAIS/models/asr-wav2vec2-commonvoice-fr", savedir="/vol/experiments3/imbenamor/TAPAS-FRAIS/models/asr-wav2vec2-commonvoice-fr", run_opts={"device":"cuda"})
    whisper_med = WhisperASR.from_hparams(source="/vol/experiments3/imbenamor/TAPAS-FRAIS/models/asr-whisper-medium-commonvoice-fr",savedir="/vol/experiments3/imbenamor/TAPAS-FRAIS/models/asr-whisper-medium-commonvoice-fr", run_opts={"device":"cuda"})
    whisper_large = WhisperASR.from_hparams(source="/vol/experiments3/imbenamor/TAPAS-FRAIS/models/asr-whisper-large-v2-commonvoice-fr",savedir="/vol/experiments3/imbenamor/TAPAS-FRAIS/models/asr-whisper-large-v2-commonvoice-fr", run_opts={"device":"cuda"})

    tg_to_wav = list_files(args.ref_trans)
    print(tg_to_wav.keys(),len(tg_to_wav.values()))
    with open(args.csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["filename", "duration_sec", "samplerate", "channels","trans_w2vec","WER_w2vec","trans_w2vec_vad_chunk","WER_w2vec_vad_chunk","trans_whisper_vad_chunk","WER_whisper_vad_chunk","trans_whisper_large_vad_chunk","WER_whisper_large_vad_chunk","trans_conf_cv_vad_chunk","WER_conf_cv_vad_chunk","trans_conf_ester_vad_chunk","WER_conf_ester_vad_chunk","trans_hmm_tdnn_vad_chunk","WER_hmm_tdnn_vad_chunk"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i,(tg,wav) in enumerate(tg_to_wav.items()):
            if i<35:
                continue
            logging.info(f"=================================={wav}=========================================")
            wav_file = os.path.join(args.wav_data, wav)
            trans_file = os.path.join(args.ref_trans, tg)
            if os.path.exists(wav_file) and os.path.exists(trans_file) and tg !="CCM-004773-01_L01.TextGrid" and wav !="CCM-004773-01_L01.wav":

                info = get_audio_info(wav_file)
                writer.writerow({"filename": wav,**info})
                # load audio
                audio_np, sr = read_audio_16k(wav_file)
                wav = torch.from_numpy(audio_np)
                # VAD + chunking
                chunks = vad_chunk_with_timestamps(wav)
                words = get_textgrid_transcription_rhap_chunk(trans_file)
                #================W2VEC===========
                #pred_transcriptions_w2Vec = w2v.transcribe_file(wav_file)
                #writer.writerow({"trans_w2vec": normalization(pred_transcriptions_w2Vec)})
                #ref_transcriptions = get_textgrid_transcription_rhap(trans_file)
                #ref_transcriptions = remove_words(ref_transcriptions)
                #ref_transcriptions = clean_french_disfluencies_repetition(ref_transcriptions)
                #WER = wer_segment(wav_file, ref_transcriptions, pred_transcriptions_w2Vec)
                #writer.writerow({"WER_w2vec": WER})
               # logging.info("w2v Done!")

                #=========W2VEC vad chunk=========
                results = whisper_transcribe_chunks(w2v, "wav2vec2-VAD-chunk", wav, chunks)
                ref_transcriptions, pred_transcriptions_w2v2_chunk = wer_chunk(results, words)
                ref_transcriptions = clean_french_disfluencies_repetition(ref_transcriptions)
                writer.writerow({"trans_w2vec_vad_chunk": pred_transcriptions_w2v2_chunk})
                WER = wer_segment(wav_file, ref_transcriptions, pred_transcriptions_w2v2_chunk)
                writer.writerow({"WER_w2vec_vad_chunk": WER})
                logging.info("w2vec vad chunk Done!")
                del pred_transcriptions_w2v2_chunk
                gc.collect()
                #=========whisper vad chunk=======
                results = whisper_transcribe_chunks(whisper_med, "whisper-VAD-chunk", wav, chunks)
                ref_transcriptions, pred_transcriptions_whisper_med_chunk = wer_chunk(results, words)
                ref_transcriptions = clean_french_disfluencies_repetition(ref_transcriptions)
                writer.writerow({"trans_whisper_vad_chunk": pred_transcriptions_whisper_med_chunk})
                WER = wer_segment(wav_file, ref_transcriptions, pred_transcriptions_whisper_med_chunk)
                writer.writerow({"WER_whisper_vad_chunk": WER})
                logging.info("whisper med Done!")
                del pred_transcriptions_whisper_med_chunk

                gc.collect()
                #=========whisper large vad chunk=======
                results = whisper_transcribe_chunks(whisper_large, "whisper-large-VAD-chunk", wav, chunks)
                ref_transcriptions, pred_transcriptions_whisper_large_chunk = wer_chunk(results, words)
                ref_transcriptions = clean_french_disfluencies_repetition(ref_transcriptions)
                writer.writerow({"trans_whisper_large_vad_chunk": pred_transcriptions_whisper_large_chunk})
                WER = wer_segment(wav_file, ref_transcriptions, pred_transcriptions_whisper_large_chunk)
                writer.writerow({"WER_whisper_large_vad_chunk": WER})
                logging.info("whisper large Done!")
                del pred_transcriptions_whisper_large_chunk

                gc.collect()
                # =========conf cv=======
                speech2text = Speech2Text("/vol/experiments3/rouas/SpeechRecognition/saved_models/espnet2-commonvoice-conformer-FR/asr_commonvoice_conformer_FR_config.yaml", "/vol/experiments3/rouas/SpeechRecognition/saved_models/espnet2-commonvoice-conformer-FR/asr_commonvoice_conformer_FR.pth", nbest=1, minlenratio=0.1, beam_size=40,
                                          device="cuda", dtype="float32")
                results = espnet_transcribe_chunks(speech2text, wav, chunks, sr=16000)
                ref_transcriptions, pred_transcriptions_conf_cv = wer_chunk(results, words)
                ref_transcriptions = clean_french_disfluencies_repetition(ref_transcriptions)
                writer.writerow({"trans_conf_cv_vad_chunk": pred_transcriptions_conf_cv})
                WER = wer_segment(wav_file, ref_transcriptions, pred_transcriptions_conf_cv)
                writer.writerow({"WER_conf_cv_vad_chunk": WER})
                logging.info("Conf cv Done!")
                del pred_transcriptions_conf_cv
                gc.collect()
                # =========conf ester=======
                speech2text = Speech2Text(
                    "/home/rouas/experiments/SpeechRecognition/saved_models/espnet2-conformer-FR/asr_conformer_config.yaml",
                    "/home/rouas/experiments/SpeechRecognition/saved_models/espnet2-conformer-FR/asr_conformer.pth",
                    nbest=1, minlenratio=0.1, beam_size=40,
                    device="cuda", dtype="float32")
                results = espnet_transcribe_chunks(speech2text, wav, chunks, sr=16000)
                ref_transcriptions, pred_transcriptions_conf_ester = wer_chunk(results, words)
                ref_transcriptions = clean_french_disfluencies_repetition(ref_transcriptions)
                writer.writerow({"trans_conf_ester_vad_chunk": pred_transcriptions_conf_ester})
                WER = wer_segment(wav_file, ref_transcriptions, pred_transcriptions_conf_ester)
                writer.writerow({"WER_conf_ester_vad_chunk": WER})
                logging.info("Conf ester Done!")
                del speech2text
                del pred_transcriptions_conf_ester
                gc.collect()
                # =========hmm-tdnn=======
                results = hmmtdnn_transcribe_chunks("/vol/experiments3/imbenamor/TAPAS-FRAIS/src/asr_FR_kaldi_hmm_tdnn.sh", wav, chunks,"/vol/experiments3/imbenamor/TAPAS-FRAIS/logs/transcription/rhap", sr=16000)
                ref_transcriptions, pred_transcriptions_hmm_tdnn = wer_chunk(results, words)
                ref_transcriptions = clean_french_disfluencies_repetition(ref_transcriptions)
                writer.writerow({"trans_hmm_tdnn_vad_chunk": pred_transcriptions_hmm_tdnn})
                WER = wer_segment(wav_file, ref_transcriptions, pred_transcriptions_hmm_tdnn)
                writer.writerow({"WER_hmm_tdnn_vad_chunk": WER})
                logging.info("hmm_tdnn Done!")
                del pred_transcriptions_hmm_tdnn
                del words
                del results
                gc.collect()
                #=================================
            else:
                logging.warning(f"The file {wav_file} does not exist")

if __name__ == "__main__":
    main(args)