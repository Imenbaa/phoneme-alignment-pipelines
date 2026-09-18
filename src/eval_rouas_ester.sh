#python -m espnet_vad_rouas_ester --i "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/8-CEREB" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/8-CEREB" -o "/vol/experiments3/imbenamor/TAPAS-FRAIS/logs/transcriptions/ester/typaloc/" --vad silero --log_file CEREB_conf_silero
#python -m espnet_vad_rouas_ester --i "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/8-CEREB" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/8-CEREB" -o "/vol/experiments3/imbenamor/TAPAS-FRAIS/logs/transcriptions/ester/typaloc/" --vad rvad --log_file CEREB_conf_rvad

#python -m espnet_vad_rouas_ester --i "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/12-CTRL" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/12-CTRL" -o "/vol/experiments3/imbenamor/TAPAS-FRAIS/logs/transcriptions/ester/typaloc/" --vad silero --log_file CTRL_conf_silero
#python -m espnet_vad_rouas_ester --i "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/12-CTRL" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/12-CTRL" -o "/vol/experiments3/imbenamor/TAPAS-FRAIS/logs/transcriptions/ester/typaloc/" --vad rvad --log_file CTRL_conf_rvad

#python -m espnet_vad_rouas_ester --i "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/8-PARK" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/8-PARK" -o "/vol/experiments3/imbenamor/TAPAS-FRAIS/logs/transcriptions/ester/typaloc/" --vad silero --log_file PARK_conf_silero
#python -m espnet_vad_rouas_ester --i "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/8-PARK" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/8-PARK" -o "/vol/experiments3/imbenamor/TAPAS-FRAIS/logs/transcriptions/ester/typaloc/" --vad rvad --log_file PARK_conf_rvad

#python -m espnet_vad_rouas_ester --i "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/12-SLA" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/12-SLA" -o "/vol/experiments3/imbenamor/TAPAS-FRAIS/logs/transcriptions/ester/typaloc/" --vad rvad --log_file SLA_conf_rvad
#python -m espnet_vad_rouas_ester --i "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/12-SLA" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/12-SLA" -o "/vol/experiments3/imbenamor/TAPAS-FRAIS/logs/transcriptions/ester/typaloc/" --vad silero --log_file SLA_conf_silero

#python -m espnet_vad_rouas_ester --i "/vol/corpora/TAPAS_FRAIS/Data_Partagees_Mons/Data_Mons/Description" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_Partagees_Mons/Mons TextGrid verifie + 50 ans" -o "/vol/experiments3/imbenamor/TAPAS-FRAIS/logs/transcriptions/ester/tapas/" --vad rvad --log_file tapas_conf_rvad
#python -m espnet_vad_rouas_ester --i "/vol/corpora/TAPAS_FRAIS/Data_Partagees_Mons/Data_Mons/Description" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_Partagees_Mons/Mons TextGrid verifie + 50 ans" -o "/vol/experiments3/imbenamor/TAPAS-FRAIS/logs/transcriptions/ester/tapas/" --vad silero --log_file tapas_conf_silero

#python -m espnet_vad_rouas_ester --i "/vol/corpora/Rhapsodie/wav16k_corrected" --ref_trans "/vol/corpora/Rhapsodie/TextGrids-fev2013" -o "/vol/experiments3/imbenamor/TAPAS-FRAIS/logs/transcriptions/ester/rhap/" --vad rvad --log_file rhap_conf_rvad
#python -m espnet_vad_rouas_ester --i "/vol/corpora/Rhapsodie/wav16k_corrected" --ref_trans "/vol/corpora/Rhapsodie/TextGrids-fev2013" -o "/vol/experiments3/imbenamor/TAPAS-FRAIS/logs/transcriptions/ester/rhap/" --vad silero --log_file rhap_conf_silero

#python -m espnet_vad_rouas_ester --wav_data "/vol/corpora/Rhapsodie/wav16k_corrected" --ref_trans "/vol/corpora/Rhapsodie/TextGrids-fev2013" --log_file conf_vad_chunk_ester_rhap_rvad
#python -m espnet_vad_rouas_ester --wav_data "/vol/corpora/TAPAS_FRAIS/Data_Partagees_Mons/Data_Mons/Description" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_Partagees_Mons/Mons TextGrid verifie + 50 ans"  --log_file conf_vad_chunk_ester_tapas_rvad
#python -m espnet_vad_rouas_ester --wav_data "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/8-PARK" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/8-PARK"  --log_file conf_vad_chunk_ester_park_rvad
#python -m espnet_vad_rouas_ester --wav_data "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/12-SLA" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/12-SLA"  --log_file conf_vad_chunk_ester_sla_rvad
#python -m espnet_vad_rouas_ester --wav_data "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/12-CTRL" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/12-CTRL"  --log_file conf_vad_chunk_ester_ctrl_rvad
#python -m espnet_vad_rouas_ester --wav_data "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/8-CEREB" --ref_trans "/vol/corpora/TAPAS_FRAIS/Data_partagees_ParisTypaloc-TapasFrais/8-CEREB"  --log_file conf_vad_chunk_ester_cereb_rvad



python -m espnet_vad_rouas_ester --wav_data "/vol/corpora/Daoudi/Data/Monologue/HC" --ref_trans "/vol/corpora/Daoudi/Data/Transcription_monlogue_HC" --log_file HC_spon_conf_ester-VAD-chunk #--csv_path HC_spon_meta.csv
python -m espnet_vad_rouas_ester --wav_data "/vol/corpora/Daoudi/Data/Monologue/MSA" --ref_trans "/vol/corpora/Daoudi/Data/Transcriptions/TXT Chloé" --log_file MSA_spon_conf_ester-VAD-chunk #--csv_path MSA_spon_meta.csv
python -m espnet_vad_rouas_ester --wav_data "/vol/corpora/Daoudi/Data/Monologue/PD" --ref_trans "/vol/corpora/Daoudi/Data/Transcriptions/TXT Robin" --log_file PD_spon_conf_ester-VAD-chunk

#python -m eval_hmm_tdnn --wav_data "/vol/corpora/Rhapsodie/wav16k_corrected" --ref_trans "/vol/corpora/Rhapsodie/TextGrids-fev2013" --log_file hmm_tdnn_rhap


