#python -m eval_hmm_tdnn --wav_data "/vol/corpora/TAPAS_FRAIS/Data_Partagees_Mons/Data_Mons/Description" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_Partagees_Mons/Mons TextGrid verifie + 50 ans" --log_file hmm_tdnn_tapas
#python -m eval_hmm_tdnn --wav_data "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/12-SLA" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/12-SLA" --log_file hmm_tdnn_SLA
#python -m eval_hmm_tdnn --wav_data "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/8-CEREB" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/8-CEREB" --log_file hmm_tdnn_CEREB
#python -m eval_hmm_tdnn --wav_data "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/8-PARK" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/8-PARK" --log_file hmm_tdnn_PARK
#python -m eval_hmm_tdnn --wav_data "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/12-CTRL" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/12-CTRL" --log_file hmm_tdnn_CTRL

#python -m eval_hmm_tdnn --wav_data "/vol/corpora/Rhapsodie/wav16k_corrected" --ref_trans "/vol/corpora/Rhapsodie/TextGrids-fev2013" --log_file hmm_tdnn_rhap

#python -m eval_hmm_tdnn --wav_data "/vol/corpora/TAPAS_FRAIS/Data_Partagees_Mons/Data_Mons/Description" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_Partagees_Mons/Mons TextGrid verifie + 50 ans" --log_file hmm_tdnn_tapas
#python -m eval_hmm_tdnn --wav_data "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/12-SLA" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/12-SLA" --log_file hmm_tdnn_SLA
#python -m eval_hmm_tdnn --wav_data "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/8-CEREB" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/8-CEREB" --log_file hmm_tdnn_CEREB
#python -m eval_hmm_tdnn --wav_data "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/8-PARK" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/8-PARK" --log_file hmm_tdnn_PARK
#python -m eval_hmm_tdnn --wav_data "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/12-CTRL" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/12-CTRL" --log_file hmm_tdnn_CTRL


python -m eval_hmm_tdnn --wav_data "/vol/corpora/Daoudi/Data/Monologue/HC" --ref_trans "/vol/corpora/Daoudi/Data/Transcription_monlogue_HC" --log_file HC_spon_hmmtdnn-VAD-chunk #--csv_path HC_spon_meta.csv
python -m eval_hmm_tdnn --wav_data "/vol/corpora/Daoudi/Data/Monologue/MSA" --ref_trans "/vol/corpora/Daoudi/Data/Transcriptions/TXT Chloé" --log_file MSA_spon_hmmtdnn-VAD-chunk #--csv_path MSA_spon_meta.csv
python -m eval_hmm_tdnn --wav_data "/vol/corpora/Daoudi/Data/Monologue/PD" --ref_trans "/vol/corpora/Daoudi/Data/Transcriptions/TXT Robin" --log_file PD_spon_hmmtdnn-VAD-chunk