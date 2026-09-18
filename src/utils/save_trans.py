def save_trn(utt_id, text, filepath, mode="a"):
    """
    utt_id: string
    text: predicted transcription (string)
    filepath: path to .trn file
    mode: "w" to overwrite, "a" to append (default)
    """
    with open(filepath, mode, encoding="utf-8") as f:
        f.write(f"{text} ({utt_id})\n")
