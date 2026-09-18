import soundfile as sf
import os
import csv

def get_audio_info(audio_path):
    with sf.SoundFile(audio_path) as audio:
        return {
            "duration_sec": len(audio) / audio.samplerate,
            "samplerate": audio.samplerate,
            "channels": audio.channels
        }


