import subprocess
from pathlib import Path
import tempfile
import logging
import argparse
import logging
from pathlib import Path
import tempfile
import subprocess
import soundfile as sf
import torch
import numpy as np
import os
from utils.read_transcription import *
from utils.normalise_text import *
from pathlib import Path
from hyperpyyaml import load_hyperpyyaml
from utils.apply_vad import *
from utils.list_files import list_files
from utils.VAD_chunk import *
from utils.wer_chunk import wer_chunk
from utils.logging_config import setup_logging
from utils.wer_segment import wer_segment
import librosa
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Transcription ASR via Kaldi (recognizer.sh)")
    parser.add_argument("--wav_data",required=True,help="Chemin vers le dossier des WAV")
    parser.add_argument("--ref_trans", type=str, required=True, help="The reference transcription")
    parser.add_argument("--bash",default="/vol/experiments3/imbenamor/TAPAS-FRAIS/src/asr_FR_kaldi_hmm_tdnn.sh",help="Chemin vers le script recognizer.sh")
    parser.add_argument("--work-dir",default="/vol/experiments3/imbenamor/TAPAS-FRAIS/logs/transcriptions/hmm_tdnn/",help="Dossier de travail pour les fichiers CTM (optionnel)")
    parser.add_argument("--log_file", type=str, required=True, help="The logfile name")

    args = parser.parse_args()
    log_file = Path("/vol/experiments3/imbenamor/TAPAS-FRAIS/logs/hmm_tdnn/" + args.log_file + ".log")
    wer_hparams = load_hyperpyyaml("""wer_stats: !new:speechbrain.utils.metric_stats.ErrorRateStats""")
    setup_logging(log_file)
    logger = logging.getLogger(__name__)
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
    for tg, wav in tg_to_wav.items():
        wav_file = os.path.join(args.wav_data, wav)
        trans_file = os.path.join(args.ref_trans, tg)
        if os.path.exists(wav_file) and os.path.exists(trans_file) and tg != "CCM-004773-01_L01.TextGrid" and wav != "CCM-004773-01_L01.wav":
            number_files += 1
            logging.info(f" File duration: {librosa.get_duration(filename=wav_file)} seconds")
            #if "Rhapsodie" in args.ref_trans:
                #ref_transcriptions = get_textgrid_transcription_rhap(trans_file)
            #else:
               #ref_transcriptions = get_textgrid_transcription_tapas(trans_file)
            # load audio
            audio_np, sr = read_audio_16k(wav_file)
            wav = torch.from_numpy(audio_np)
            # VAD + chunking
            chunks = vad_chunk_with_timestamps(wav)
            logging.info("Number of chunks: %d", len(chunks))
            results =hmmtdnn_transcribe_chunks(args.bash,wav,chunks,args.work_dir,sr=16000 )
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
            #pred_transcriptions = transcribe_audio(wav_path=wav_file,bash_script=args.bash,work_dir=args.work_dir)
            #ref_transcriptions = remove_words(ref_transcriptions)
            print(normalization(pred_transcriptions))
            print(normalization(ref_transcriptions))
            WER = wer_segment(wav_file, ref_transcriptions, pred_transcriptions)
            WERs.append(WER)
            #break
    


