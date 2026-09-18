
import os
import soundfile as sf
from textgrid import TextGrid
import torch
from speechbrain.inference.ASR import WhisperASR
from apply_vad import *
from hyperpyyaml import load_hyperpyyaml
from normalise_text import normalization
from pathlib import Path
import logging
from read_transcription import *

log_file = Path("/vol/experiments3/imbenamor/TAPAS-FRAIS/logs/whisper_CTR_vad_chunk.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode="w"),
        logging.StreamHandler(),  # remove if you want file-only
    ],
)
logger = logging.getLogger(__name__)
def vad_chunk_with_timestamps(
    wav,
    sampling_rate=16000,
    max_chunk_duration=30.0,
    max_pause_duration=0.6
):
    """
    wav: torch.Tensor (1D, 16kHz)
    returns: list of dicts with start/end in ORIGINAL time
    """
    vad_model, utils = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=False)

    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
    speech_ts = get_speech_timestamps(
        wav,
        vad_model,
        sampling_rate=sampling_rate
    )

    chunks = []

    chunk_start = None
    chunk_end = None

    for seg in speech_ts:
        seg_start = seg["start"] / sampling_rate
        seg_end = seg["end"] / sampling_rate

        if chunk_start is None:
            chunk_start = seg_start
            chunk_end = seg_end
            continue

        pause = seg_start - chunk_end

        # pause courte → on fusionne
        if pause <= max_pause_duration:
            # mais on respecte la durée max
            if seg_end - chunk_start <= max_chunk_duration:
                chunk_end = seg_end
            else:
                chunks.append({
                    "start": chunk_start,
                    "end": chunk_end
                })
                chunk_start = seg_start
                chunk_end = seg_end

        # pause longue → nouvelle unité
        else:
            chunks.append({
                "start": chunk_start,
                "end": chunk_end
            })
            chunk_start = seg_start
            chunk_end = seg_end

    # dernier chunk
    if chunk_start is not None:
        chunks.append({
            "start": chunk_start,
            "end": chunk_end
        })

    return chunks
def load_words_from_textgrid_rhap(tg_path, tier_name=None):
    tg = TextGrid()
    tg.read(tg_path)

    names = tg.getNames()

    # Auto-detect word tier if not explicitly provided
    if tier_name is None:
        for candidate in ["ORT-MAU", "words", "word"]:
            if candidate in names:
                tier_name = candidate
                break

    if tier_name not in names:
        raise ValueError(
            f"No suitable word tier found. Available: {names}"
        )

    tier = tg.tiers[names.index(tier_name)]

    words = []
    for interval in tier.intervals:
        mark = interval.mark.strip()
        if mark:
            words.append(
                (mark, interval.minTime, interval.maxTime)
            )

    return words

def extract_chunk_audio(wav, start, end, sr=16000):
    s = int(start * sr)
    e = int(end * sr)
    return wav[s:e].unsqueeze(0)   # (1, T)

def load_words_from_textgrid(tg_path, tier_name="transcription"):
    tg = TextGrid()
    tg.read(tg_path)

    names = tg.getNames()
    print(names)
    if tier_name not in names:
        raise ValueError(
            f"Tier '{tier_name}' not found. Available: {names}"
        )

    tier_index = names.index(tier_name)
    tier = tg.tiers[tier_index]

    words = []
    for interval in tier.intervals:
        if interval.mark.strip():
            words.append(
                (interval.mark.strip(),
                 interval.minTime,
                 interval.maxTime)
            )
    return words





def whisper_transcribe_chunks(
    asr_model,
    wav,
    chunks,
    sr=16000
):
    results = []

    for i, ch in enumerate(chunks):
        chunk_wav = extract_chunk_audio(
            wav,
            ch["start"],
            ch["end"],
            sr
        )

        wav_lens = torch.tensor([1.0])

        pred = asr_model.transcribe_batch(
            chunk_wav,
            wav_lens)
        hyp = " ".join(pred[0][0]).strip()
        results.append({
            "id": i,
            "start": ch["start"],
            "end": ch["end"],
            "text": hyp.strip()
        })

    return results
def merge_transcriptions(results):
    results = sorted(results, key=lambda x: x["start"])
    return " ".join(r["text"] for r in results if r["text"])
def ref_text_for_chunk(words, start, end):
    return " ".join(
        w for w, s, e in words
        if e > start and s < end
    )

if __name__ == "__main__":
    #wav_path = "/vol/corpora/TAPAS_FRAIS/Data_Partagees_Mons/Data_Mons/Description"
    #wav_path = "/vol/corpora/Rhapsodie/wav16k_corrected"
    wav_path = "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/12-CTRL"
    tg_path = "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/12-CTRL"
    #tg_path = "/vol/corpora/Rhapsodie/TextGrids-fev2013"
    asr_model = WhisperASR.from_hparams(
        source="/vol/experiments3/imbenamor/TAPAS-FRAIS/models/asr-whisper-medium-commonvoice-fr",
        savedir="/vol/experiments3/imbenamor/TAPAS-FRAIS/models/asr-whisper-medium-commonvoice-fr", run_opts={"device":"cuda"})
    wer_hparams = load_hyperpyyaml("""wer_stats: !new:speechbrain.utils.metric_stats.ErrorRateStats""")
    WERs= []
    number_files=0
    tg_to_wav = {}
    for f in sorted(os.listdir(tg_path)):
        if f.endswith(".TextGrid") and not f.endswith("pr_analyse.TextGrid"):
            if "Rhapsodie" in tg_path:
                tg_to_wav[f] = f.split("-")[0] + "-" + f.split("-")[1] + ".wav"
            else:
                tg_to_wav[f] = f.split(".")[0] + ".wav"
    for tg, wav in tg_to_wav.items():
        wav_file = os.path.join(wav_path, wav)
        trans_file = os.path.join(tg_path, tg)
        if os.path.exists(wav_file):
            number_files += 1
            logging.info(f" File duration: {librosa.get_duration(path=wav_file)} seconds")
            # load audio
            audio_np,sr = read_audio_16k(wav_file)
            wav = torch.from_numpy(audio_np)
            # VAD + chunking
            chunks = vad_chunk_with_timestamps(wav,sampling_rate=16000,max_chunk_duration=30.0)
            print("Number of chunks:", len(chunks))
            results = whisper_transcribe_chunks(asr_model,wav,chunks)
            words = get_textgrid_transcription_typaloc(trans_file)
            # Attach references
            for r in results:
                r["ref"] = ref_text_for_chunk(words, r["start"], r["end"])
            for id,r in enumerate(results):
                print(f"[{r['start']:.2f}-{r['end']:.2f}]")
                print("REF:", normalization(r["ref"]))
                print("HYP:", normalization(r["text"]))
                if r["ref"].strip():
                    wer_hparams["wer_stats"].clear()
                    wer_hparams["wer_stats"].append(ids=list(range(len(r["ref"]))),
                                                predict=[normalization(r["text"])],
                                                target=[normalization(r["ref"])])

                    stats = wer_hparams["wer_stats"].summarize()
                    print(f'WER= {stats["WER"]},S = {stats["substitutions"]},D = {stats["deletions"]}, I = {stats["insertions"]}')
                    logger.info(
                        "Chunk: %d | WER=%f | S=%d D=%d I=%d",
                        id,
                        stats["WER"],
                        stats["substitutions"],
                        stats["deletions"],
                        stats["insertions"],
                    )
            logger.info("-" * 30)
            logger.info("-" * 30)
            ref_full = " ".join(
                r["ref"] for r in results if r["ref"].strip()
            )

            hyp_full = " ".join(
                r["text"] for r in results if r["text"].strip()
            )
            wer_hparams["wer_stats"].clear()
            wer_hparams["wer_stats"].append(ids=list(range(len(ref_full))),
                                            predict=[normalization(hyp_full)],
                                            target=[normalization(ref_full)])
            stats = wer_hparams["wer_stats"].summarize()
            logger.info(
                "File: %s | WER=%f | S=%d D=%d I=%d",
                wav_file,
                stats["WER"],
                stats["substitutions"],
                stats["deletions"],
                stats["insertions"],
            )
            logger.info("-" * 30)
            logger.info("-" * 30)
            WERs.append(stats["WER"])
        else:
            logging.warning(f"The file {wav_file} does not exist")

    # Compute final WER
    global_stats = sum(WERs) / len(WERs)

    logger.info("===================== GLOBAL WER ======================")
    #logger.info(f"The number of files is {number_files}")
    logger.info("WER: %.2f%%", global_stats)
    logger.info("=============================================================")