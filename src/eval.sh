#
cd /vol/experiments3/imbenamor/TAPAS-FRAIS/src

#python -m evaluation --model "wav2vec" --wav_data "/vol/corpora/Rhapsodie/wav16k_corrected" --ref_trans "/vol/corpora/Rhapsodie/TextGrids-fev2013" --log_file wer_rhap_corrected

#python -m evaluation --model "wav2vec" --wav_data "/vol/corpora/Rhapsodie/wav16k_corrected" --ref_trans "/vol/corpora/Rhapsodie/TextGrids-fev2013" --log_file wer_rhap_corrected
#python -m eval_rouas_models --pred_trans "/vol/experiments3/imbenamor/TAPAS-FRAIS/data/transcription_tdnn_hmm/e2e_transcriptions_original" --ref_trans "/vol/corpora/Rhapsodie/TextGrids-fev2013" --log_file wer_conformer_rouas

#python -m eval_rouas_models --pred_trans "/vol/experiments3/rouas/TAPAS-FRAIS/transcriptions/HMM-TDNN/Data_Mons/Description" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_Partagees_Mons/Mons TextGrid verifie + 50 ans" --log_file wer_tapas_hmm_tdnn_rouas
#python -m eval_rouas_models --pred_trans "/vol/experiments3/rouas/TAPAS-FRAIS/transcriptions/E2E-conformer-ester+vad/Data_Mons/Description" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_Partagees_Mons/Mons TextGrid verifie + 50 ans" --log_file wer_tapas_conf_rouas
#python -m eval_rouas_models --pred_trans "/vol/experiments3/rouas/TAPAS-FRAIS/transcriptions/E2E-conformer-commonvoice/Data_Mons/Description" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_Partagees_Mons/Mons TextGrid verifie + 50 ans" --log_file wer_tapas_conf_cv_rouas

#python -m evaluation --model "whisper-medium" --wav_data "/vol/corpora/TAPAS_FRAIS/Data_Partagees_Mons/Data_Mons/Description" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_Partagees_Mons/Mons TextGrid verifie + 50 ans" --log_file wer_tapas_vad_whisper_medium
#python -m evaluation --model "whisper-medium" --wav_data "/vol/corpora/TAPAS_FRAIS/Data_Partagees_Mons/Data_Mons/Description" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_Partagees_Mons/Mons TextGrid non verifie" --log_file wer_tapas_nonverif_whisper_medium
#python -m evaluation --model "whisper-medium" --wav_data "/vol/corpora/Rhapsodie/wav16k_corrected" --ref_trans "/vol/corpora/Rhapsodie/TextGrids-fev2013" --log_file wer_rhap_whisper_medium
#python -m evaluation --model "wav2vec" --wav_data "/vol/corpora/TAPAS_FRAIS/Data_Partagees_Mons/Data_Mons/Description" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_Partagees_Mons/Mons TextGrid non verifie" --wav_16k /vol/experiments3/imbenamor/TAPAS-FRAIS/data/tapas16k-nonverif --log_file wer_tapas16k_nonverif_wav2vec
#python -m evaluation --model "wav2vec" --wav_data "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/8-PARK" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/8-PARK" --log_file wer_typaloc_PARK_wav2vec
#python -m evaluation --model "wav2vec" --wav_data "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/8-CEREB" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/8-CEREB" --log_file wer_typaloc_CEREB_wav2vec
#python -m evaluation --model "wav2vec" --wav_data "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/12-SLA" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/12-SLA" --log_file wer_typaloc_SLA_wav2vec
#python -m evaluation --model "wav2vec" --wav_data "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/12-CTRL" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/12-CTRL" --log_file wer_typaloc_CTR_wav2vec


#python -m evaluation --model "whisper-medium" --wav_data "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/8-PARK" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/8-PARK" --log_file wer_typaloc_PARK_whisper
#python -m evaluation --model "whisper-VAD-chunk" --wav_data "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/8-PARK" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/8-PARK" --log_file whisper_PARK_VAD_chunk_rvad

#python -m evaluation --model "whisper-VAD-chunk" --wav_data "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/8-CEREB" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/8-CEREB" --log_file wer_typaloc_CEREB_whisper-VAD-chunk_rvad
#python -m evaluation --model "whisper-medium" --wav_data "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/12-SLA" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/12-SLA" --log_file wer_typaloc_SLA_whisper
#python -m evaluation --model "whisper-VAD-chunk" --wav_data "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/12-SLA" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/12-SLA" --log_file whisper_SLA_vad_chunk_rvad

#python -m evaluation --model "whisper-medium" --wav_data "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/12-CTRL" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/12-CTRL" --log_file wer_typaloc_CTR_whisper
#python -m evaluation --model "whisper-VAD-chunk" --wav_data "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/12-CTRL" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/12-CTRL" --log_file whisper_CTR_vad_chunk_rvad

#python -m evaluation --model "whisper-large-VAD-chunk" --wav_data "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/8-CEREB" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/8-CEREB" --log_file wer_whisper_large-VAD-chunk_CEREB_rvad
#python -m evaluation --model "whisper-large" --wav_data "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/8-PARK" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/8-PARK" --log_file wer_whisper_large_PARK
#python -m evaluation --model "whisper-large-VAD-chunk" --wav_data "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/8-PARK" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/8-PARK" --log_file wer_whisper_large-VAD-chunk_PARK_rvad

#python -m evaluation --model "whisper-large" --wav_data "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/12-SLA" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/12-SLA" --log_file wer_whisper_large_SLA
#python -m evaluation --model "whisper-large-VAD-chunk" --wav_data "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/12-SLA" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/12-SLA" --log_file wer_whisper_large-VAD-chunk_SLA_rvad

#python -m evaluation --model "whisper-large" --wav_data "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/12-CTRL" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/12-CTRL" --log_file wer_whisper_large_CTRL
#python -m evaluation --model "whisper-large-VAD-chunk" --wav_data "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/12-CTRL" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/12-CTRL" --log_file wer_whisper_large-vad-chunk_CTRL_rvad

#python -m evaluation --model "whisper-VAD-chunk" --wav_data "/vol/corpora/TAPAS_FRAIS/Data_Partagees_Mons/Data_Mons/Description" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_Partagees_Mons/Mons TextGrid verifie + 50 ans" --log_file whisper-VAD-chunk_-tapas-rvad

#python -m evaluation --model "whisper-large-VAD-chunk" --wav_data "/vol/corpora/TAPAS_FRAIS/Data_Partagees_Mons/Data_Mons/Description" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_Partagees_Mons/Mons TextGrid verifie + 50 ans" --log_file whisper-VAD-chunk_large-tapas-rvad

#python -m evaluation --model "wav2vec2-VAD-chunk" --wav_data "/vol/corpora/Rhapsodie/wav16k_corrected" --ref_trans "/vol/corpora/Rhapsodie/TextGrids-fev2013" --log_file whisper_large-vad-chunk_rhap_rvad
#python -m evaluation --model "whisper-large-VAD-chunk" --wav_data "/vol/corpora/Rhapsodie/wav16k_corrected" --ref_trans "/vol/corpora/Rhapsodie/TextGrids-fev2013" --log_file whisper_large-vad-chunk_rhap_rvad
#python -m evaluation --model "whisper-VAD-chunk" --wav_data "/vol/corpora/Rhapsodie/wav16k_corrected" --ref_trans "/vol/corpora/Rhapsodie/TextGrids-fev2013" --log_file whisper_vad-chunk_rhap_rvad

#python -m evaluation --model "wav2vec" --wav_data "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/donnnees_spont_ParisTypaloc-TapasFrais (copie)" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/donnnees_spont_ParisTypaloc-TapasFrais (copie)" --log_file typaloc_all_w2v
#python -m evaluation --model "wav2vec" --wav_data "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/donnnees_spont_ParisTypaloc-TapasFrais (copie)" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/donnnees_spont_ParisTypaloc-TapasFrais (copie)" --log_file typaloc_all_w2v


#python -m evaluation --model "wav2vec2-VAD-chunk" --data_name tapas --wav_data "/vol/corpora/TAPAS_FRAIS/Data_Partagees_Mons/Data_Mons/Description" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_Partagees_Mons/Mons TextGrid verifie + 50 ans" --log_file w2vec2-VAD-chunk-tapas-rvad.log
#python -m evaluation --model "wav2vec2-VAD-chunk" --data_name typaloc-cereb --wav_data "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/8-CEREB" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/8-CEREB" --log_file w2vec2-VAD-chunk-cereb-rvad.log
#python -m evaluation --model "wav2vec2-VAD-chunk" --data_name typaloc-park --wav_data "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/8-PARK" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/8-PARK" --log_file w2vec2-VAD-chunk-park-rvad.log
#python -m evaluation --model "wav2vec2-VAD-chunk" --data_name typaloc-sla --wav_data "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/12-SLA" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/12-SLA" --log_file w2vec2-VAD-chunk-sla-rvad.log
#python -m evaluation --model "wav2vec2-VAD-chunk" --data_name typaloc-ctrl --wav_data "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/12-CTRL" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/12-CTRL" --log_file w2vec2-VAD-chunk-ctrl-rvad.log

python -m evaluation --model "wav2vec2-VAD-chunk" --wav_data "/vol/corpora/Daoudi/Data/Monologue/HC" --ref_trans "/vol/corpora/Daoudi/Data/Transcription_monlogue_HC" --log_file HC_spon_w2V-VAD-chunk.log 
python -m evaluation --model "wav2vec2-VAD-chunk" --wav_data "/vol/corpora/Daoudi/Data/Monologue/MSA" --ref_trans "/vol/corpora/Daoudi/Data/Transcriptions/TXT Chloé" --log_file MSA_spon_w2V-VAD-chunk 
python -m evaluation --model "wav2vec2-VAD-chunk" --wav_data "/vol/corpora/Daoudi/Data/Monologue/PD" --ref_trans "/vol/corpora/Daoudi/Data/Transcriptions/TXT Robin" --log_file PD_spon_w2V-VAD-chunk 

#python -m evaluation --model "whisper-VAD-chunk" --wav_data "/vol/corpora/Daoudi/Data/Monologue/HC" --ref_trans "/vol/corpora/Daoudi/Data/Transcription_monlogue_HC" --log_file HC_spon_whisper-VAD-chunk --csv_path HC_spon_meta.csv
#python -m evaluation --model "whisper-VAD-chunk" --wav_data "/vol/corpora/Daoudi/Data/Monologue/MSA" --ref_trans "/vol/corpora/Daoudi/Data/Transcriptions/TXT Chloé" --log_file MSA_spon_whisper-VAD-chunk --csv_path MSA_spon_meta.csv
#python -m evaluation --model "whisper-VAD-chunk" --wav_data "/vol/corpora/Daoudi/Data/Monologue/PD" --ref_trans "/vol/corpora/Daoudi/Data/Transcriptions/TXT Robin" --log_file PD_spon_whisper-VAD-chunk --csv_path PD_spon_meta.csv

python -m evaluation --model "whisper-large-VAD-chunk" --wav_data "/vol/corpora/Daoudi/Data/Monologue/HC" --ref_trans "/vol/corpora/Daoudi/Data/Transcription_monlogue_HC" --log_file HC_spon_whisper-large-VAD-chunk 
python -m evaluation --model "whisper-large-VAD-chunk" --wav_data "/vol/corpora/Daoudi/Data/Monologue/MSA" --ref_trans "/vol/corpora/Daoudi/Data/Transcriptions/TXT Chloé" --log_file MSA_spon_whisper-large-VAD-chunk 
python -m evaluation --model "whisper-large-VAD-chunk" --wav_data "/vol/corpora/Daoudi/Data/Monologue/PD" --ref_trans "/vol/corpora/Daoudi/Data/Transcriptions/TXT Robin" --log_file PD_spon_whisper-large-VAD-chunk 


#python -m evaluation --model "wav2vec2-spont" --data_name rhap --wav_data "/vol/corpora/Rhapsodie/wav16k_corrected" --ref_trans "/vol/corpora/Rhapsodie/TextGrids-fev2013" --log_file wa2vec2_spont_rhap
#python -m evaluation --model "wav2vec2-spont" --data_name tapas --wav_data "/vol/corpora/TAPAS_FRAIS/Data_Partagees_Mons/Data_Mons/Description" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_Partagees_Mons/Mons TextGrid verifie + 50 ans" --log_file wa2vec2_spont_tapas
#python -m evaluation --model "wav2vec2-spont" --data_name typaloc-cereb --wav_data "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/8-CEREB" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/8-CEREB" --log_file wa2vec2_spont_cereb
#python -m evaluation --model "wav2vec2-spont" --data_name typaloc-park --wav_data "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/8-PARK" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/8-PARK" --log_file wa2vec2_spont_park
#python -m evaluation --model "wav2vec2-spont" --data_name typaloc-sla --wav_data "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/12-SLA" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/12-SLA" --log_file wa2vec2_spont_sla
#python -m evaluation --model "wav2vec2-spont" --data_name typaloc-ctrl --wav_data "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/12-CTRL" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/12-CTRL" --log_file wa2vec2_spont_ctrl
