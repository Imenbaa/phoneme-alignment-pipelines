from utils.VAD_chunk import ref_text_for_chunk
from utils.normalise_text import *
from hyperpyyaml import load_hyperpyyaml

import logging

logger = logging.getLogger(__name__)

def wer_chunk(results,words):
    wer_hparams = load_hyperpyyaml("""wer_stats: !new:speechbrain.utils.metric_stats.ErrorRateStats""")

    for r in results:
        r["ref"] = remove_words(ref_text_for_chunk(words, r["start"], r["end"]))
    for id, r in enumerate(results):
        print(f"[{r['start']:.2f}-{r['end']:.2f}]")
        print("REF:", normalization(r["ref"]))
        print("HYP:", normalization(r["text"]))
        if r["ref"].strip():
            wer_hparams["wer_stats"].clear()
            wer_hparams["wer_stats"].append(ids=list(range(len(r["ref"]))),predict=[normalization(r["text"])],target=[normalization(r["ref"])])
            stats = wer_hparams["wer_stats"].summarize()
            print(f'WER= {stats["WER"]},S = {stats["substitutions"]},D = {stats["deletions"]}, I = {stats["insertions"]}')
            logger.info("Chunk: %d | WER=%f | S=%d D=%d I=%d",id,stats["WER"],stats["substitutions"],stats["deletions"],stats["insertions"])
    ref_full = " ".join(
        r["ref"] for r in results if r["ref"].strip()
    )

    hyp_full = " ".join(
        r["text"] for r in results if r["text"].strip()
    )
    return ref_full, hyp_full