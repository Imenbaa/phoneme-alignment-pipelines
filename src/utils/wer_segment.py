import logging
from hyperpyyaml import load_hyperpyyaml
from utils.normalise_text import normalization

logger = logging.getLogger(__name__)

def wer_segment(wav_file,ref_transcriptions,pred_transcriptions):
    wer_hparams = load_hyperpyyaml("""wer_stats: !new:speechbrain.utils.metric_stats.ErrorRateStats""")
    wer_hparams["wer_stats"].clear()
    wer_hparams["wer_stats"].append(ids=list(range(len(ref_transcriptions))),
                                    predict=[normalization(pred_transcriptions)],
                                    target=[normalization(ref_transcriptions)])
    stats = wer_hparams["wer_stats"].summarize()
    logger.info("File: %s | WER=%f | S=%d D=%d I=%d N=%d", wav_file.split("/")[-1], stats["WER"], stats["substitutions"],
                stats["deletions"], stats["insertions"],stats["num_scored_tokens"])
    logger.info("-" * 30)
    return stats["WER"], stats["substitutions"], stats["deletions"], stats["insertions"],stats["num_scored_tokens"]