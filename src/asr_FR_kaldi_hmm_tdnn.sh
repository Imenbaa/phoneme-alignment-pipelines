#!/bin/bash
set -e

# === Kaldi environment ===
export KALDI_ROOT=/home/rouas/Sources/git/kaldi
source /home/imbenamor/kaldi_path.sh
if [ "$#" -ne 2 ]; then
    echo "Usage : ./recognizer.sh <wav-audio> output_words.ctm"
    exit 1;
fi

record=$1
output=$2
utt_id=$(basename "$record" .wav | sed 's/[^a-zA-Z0-9_]/_/g')

tmpdir=$(mktemp -d /tmp/kaldi_asr_XXXXXX)
wav_scp="$tmpdir/wav.scp"
utt2spk="$tmpdir/utt2spk"
spk2utt="$tmpdir/spk2utt"

echo "$utt_id sox \"$record\" -r 16000 -c 1 -t wav - |" > "$wav_scp"
echo "$utt_id $utt_id" > "$utt2spk"
echo "$utt_id $utt_id" > "$spk2utt"

echo "TMPDIR = $tmpdir"
ls -l "$tmpdir"

if [ ! -e $record ]; then
    echo "Error $record not found";
    exit 1;
fi

rootdir=/vol/experiments3/rouas/SpeechRecognition/saved_models
model=$rootdir/Kaldi_HMM-TDNN/ester_epac_classic/exp/chain/tdnn1g_sp_comb/final.mdl
online_model=$rootdir/Kaldi_HMM-TDNN/ester_epac_classic/exp/chain/tdnn_online/final.mdl
graph=$rootdir/Kaldi_HMM-TDNN/ester_epac_classic/exp/chain/tree_a_sp/graph/HCLG.fst
conf=$rootdir/Kaldi_HMM-TDNN/ester_epac_classic/exp/chain/tdnn_online/conf/online.conf
word_sym=$rootdir/Kaldi_HMM-TDNN/ester_epac_classic/exp/chain/tree_a_sp/graph/words.txt
phn_table=$rootdir/Kaldi_HMM-TDNN/ester_epac_classic/data/lang_chain/phones.txt

[ ! -f $model ] && echo "$model is missing" && exit 1;
[ ! -f $lat ] && echo "$lat is missing" && exit 1;

#output=words.ctm
shift_factor=0.03 # frame-subsampling of 3 for chain model

/home/rouas/Sources/git/kaldi/src/online2bin/online2-wav-nnet3-latgen-faster \
    --do-endpointing=false \
    --online=false \
    --max-active=7000 \
    --acoustic-scale=1.0 \
    --beam=16.0 \
    --lattice-beam=10.0 \
    --frame-subsampling-factor=3 \
    --config=${conf} \
    --word-symbol-table=${word_sym} \
    ${online_model} \
    ${graph} \
    "ark:cat $utt2spk |" \
    "scp,p:cat $wav_scp |" \
    "ark:| lattice-align-words \
        $rootdir/Kaldi_HMM-TDNN/ester_epac_classic/data/lang_chain/phones/word_boundary.int \
        ${model} ark:- ark:- | \
      lattice-to-ctm-conf \
        --frame-shift=$shift_factor \
        --acoustic-scale=1.0 \
        ark:- - | \
      /home/rouas/Sources/git/kaldi/egs/swbd/s5c/utils/int2sym.pl \
        -f 5 \
        $rootdir/Kaldi_HMM-TDNN/ester_epac_classic/exp/chain/tree_a_sp/graph/words.txt \
      > $output"
