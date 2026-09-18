import wave
import numpy as np
import collections
import pandas as pd
from datetime import timedelta
import audiofile
import librosa
import soundfile as sf
import logging
from rVADfast import rVADfast
import argparse
import os
from utils.read_transcription import *
from utils.normalise_text import *
from pathlib import Path
from espnet2.bin.asr_inference import Speech2Text
from utils.apply_vad import *
from utils.list_files import list_files
from utils.VAD_chunk import *
from utils.logging_config import setup_logging
from utils.wer_segment import wer_segment
from utils.silero_rouas import silero_to_frame_labels

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("espnet").setLevel(logging.ERROR)
class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration

def read_wave(path):
    with wave.open(path, 'rb') as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000, 48000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate

def frame_generator(frame_duration_ms, audio, sample_rate):
    n = int(sample_rate * frame_duration_ms / 1000) * 2
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n

def vad_collector(sample_rate, frame_duration_ms, vad, frames, padding_duration_ms=300):
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False

    voiced_frames = []
    segments = []

    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                segments.extend([f for f, s in ring_buffer])
                ring_buffer.clear()
        else:
            segments.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                triggered = False
                yield b''.join([f.bytes for f in segments])
                ring_buffer.clear()
                segments = []

    if segments:
        yield b''.join([f.bytes for f in segments])

def merge_small_segments(df_speech_labels) :
    seuil = 2 # 15seconds minimum
    i = 0
    while i < df_speech_labels.shape[0] - 1 :
        if (df_speech_labels.iloc[i,1] - df_speech_labels.iloc[i,0]) < seuil :
            df_speech_labels.iloc[i,1] = df_speech_labels.iloc[i+1,1]
            df_speech_labels.drop(index=df_speech_labels.iloc[i+1].name, axis = 0, inplace = True)
        else :
            i = i+1
    return df_speech_labels


def inferSegment(waveform, sample_rate, vad_timestamps) :
    MIN_DUR = 0.3
    begin = int(vad_timestamps.start * sample_rate)
    end = int(vad_timestamps.end * sample_rate)

    # safety checks
    if end <= begin:
        return ""

    if (end - begin) < int(MIN_DUR * sample_rate):
        return ""

    segment = waveform[begin:end]

    try:
        results = speech2text(speech=segment)
    except Exception:
        return ""
    nbests = [text for text, token, token_int, hyp in results]
    text = nbests[0] if nbests else ""

    return text + " "

def computeOneFile(args,wav_file) :
    # si l'output est un fichier ou un dossier
    if args.vad =="rvad":
        res_file = args.output+wav_file.split("/")[-1].split(".")[0]+".trn"
    else:
        res_file = args.output + wav_file.split("/")[-1].split(".")[0] + "_silero.trn"

    if os.path.isfile(res_file) == False : # si le fichier n'existe pas déjà
        target_sr = 16000

        # read header only
        info = sf.info(wav_file)
        print(info)
        # already correct → skip
        if info.samplerate == target_sr and info.channels == 1:
            print(f"= Already {target_sr} Hz → skipped")

        waveform_vad, sr = audiofile.read(wav_file)
        logger.info(waveform_vad.shape)
        if waveform_vad.ndim > 1:
            waveform_vad = np.mean(waveform_vad, axis=0)
        logger.info(waveform_vad.shape)
        # resample
        waveform_vad = librosa.resample(waveform_vad.astype(np.float32),orig_sr=sr,target_sr=16000)
        sr = target_sr
        #vad
        if args.vad == "rvad":
            vad = rVADfast()
            vad_labels, vad_timestamps = vad(waveform_vad, sr)
        else:
            vad_labels, vad_timestamps = silero_to_frame_labels(waveform_vad, sr)
        speech_labels =pd.DataFrame([vad_timestamps,vad_labels]).T
        speech_labels.columns = ['time', 'speech']
        speech_labels['id_segment'] = np.append(0,np.cumsum(np.diff(speech_labels['speech']==1)))
        speech_labels= speech_labels.groupby('id_segment').aggregate(start=("time",'first'),
                                                   end=("time",'last'),
                                                   label=('speech','first'))
        speech_labels = speech_labels.loc[speech_labels.label == 1,]
        speech_labels = merge_small_segments(speech_labels)
        results = ""
        for i in range(speech_labels.shape[0]) :
            #print(i+1)
            #print(timedelta(seconds=speech_labels.iloc[i,0]),"-->",timedelta(seconds=speech_labels.iloc[i,1])) #,", duration:",  speech_labels.iloc[i,1] -  speech_labels.iloc[i,0])
            text =  inferSegment(waveform_vad, sr, speech_labels.iloc[i,:])
            results += text
        if len(args.output) > 0 :
            f = open(res_file, "w")
            f.write(results)
            f.close()
    else:
        with open(res_file,"r") as f:
            results=f.read()
    return results

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Process an audio file with ESPnet2 ASR.')
    parser.add_argument('-i', '--input', help='Path to the input WAV file or folder')
    parser.add_argument("--ref_trans", type=str, required=True, help="The reference transcription")
    parser.add_argument('-o', '--output', default="/vol/experiments3/imbenamor/TAPAS-FRAIS/logs/transcriptions/typaloc/", help='Path to the input WAV file or folder')
    parser.add_argument('-c', '--config', default='/home/rouas/experiments/SpeechRecognition/saved_models/espnet2-commonvoice-conformer-FR/asr_commonvoice_conformer_FR_config.yaml', help='Path to the ASR config file (default: asr_config.yml)')
    parser.add_argument('-m', '--model', default='/home/rouas/experiments/SpeechRecognition/saved_models/espnet2-commonvoice-conformer-FR/asr_commonvoice_conformer_FR.pth', help='Path to the ASR model file (default: asr.pth)')
    parser.add_argument("--vad", type=str, required=True, help="The logfile name")
    parser.add_argument("--log_file", type=str, required=True, help="The logfile name")

    args = parser.parse_args()
    log_file = Path("/vol/experiments3/imbenamor/TAPAS-FRAIS/logs/espnet/" + args.log_file + ".log")
    setup_logging(log_file)
    logger = logging.getLogger(__name__)
    # Initialize the Speech2Text instance
    global speech2text
    speech2text = Speech2Text(args.config, args.model,nbest=1,minlenratio=0.1,beam_size=40,device="cuda")

    WERs = []
    number_files = 0
    tg_to_wav = list_files(args.ref_trans)
    for tg, wav in tg_to_wav.items():
        wav_file = os.path.join(args.input, wav)
        trans_file = os.path.join(args.ref_trans, tg)
        logging.info(wav_file)
        if os.path.exists(wav_file) and os.path.exists(trans_file) and tg != "CCM-004773-01_L01.TextGrid" and wav!="CCM-004773-01_L01.wav":
            number_files += 1
            logging.info(f" File duration: {librosa.get_duration(filename=wav_file)} seconds")
            if "Rhapsodie" in args.ref_trans:
                ref_transcriptions = get_textgrid_transcription_rhap(trans_file)
            else:
                ref_transcriptions = get_textgrid_transcription_tapas(trans_file)

            pred_transcriptions = computeOneFile(args,wav_file)

            ref_transcriptions = remove_words(ref_transcriptions)
            print(normalization(pred_transcriptions))
            print(normalization(ref_transcriptions))
            # -------------------------------- WER per file   -------------------------------
            # --------------------------------------------------------------------------------
            WER = wer_segment(wav_file, ref_transcriptions, pred_transcriptions)
            WERs.append(WER)

if __name__ == "__main__":
    main()