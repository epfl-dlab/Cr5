bash count_words.sh

python training_concepts.py --voc_size 200000 --min_lang 2 --upper_threshold 1000 --lower_threshold 50 --lang_codes en_it
python training_concepts.py --voc_size 200000 --min_lang 2 --upper_threshold 1000 --lower_threshold 50 --lang_codes da_en_it_vi
python training_concepts.py --voc_size 200000 --min_lang 2 --upper_threshold 1000 --lower_threshold 50 --lang_codes da_vi
python training_concepts_no_intersection.py --voc_size 200000 --min_lang 2 --upper_threshold 1000 --lower_threshold 50 --lang_codes da_vi_en --intersection_to_exclude da_vi
python training_concepts.py --voc_size 200000 --min_lang 2 --upper_threshold 700 --lower_threshold 100 --lang_codes bg_ca_cs_da_de_el_en_es_et_fi_fr_hr_hu_id_it_mk_nl_no_pl_pt_ro_ru_sk_sl_sv_tr_uk_vi
python training_concepts.py --voc_size 200000 --min_lang 2 --upper_threshold 700 --lower_threshold 100 --lang_codes da_en_it_vi

python validation_concepts.py --training_concepts_file_name=da_en_it_vi_200000_lt_100_ut_700_ml_2 --voc_size=200000 --val_set_size=2000 --get_full_intersection=False
python validation_concepts.py --training_concepts_file_name=bg_ca_cs_da_de_el_en_es_et_fi_fr_hr_hu_id_it_mk_nl_no_pl_pt_ro_ru_sk_sl_sv_tr_uk_vi_200000_lt_100_ut_700_ml_2 --voc_size=200000 --val_set_size=2000 --get_full_intersection=False --lang_codes_to_include=da_en_vi_it_el_ru
python target_concepts_no_intersection.py --validation_concepts_file_for_exclusion bg_ca_cs_da_de_el_en_es_et_fi_fr_hr_hu_id_it_mk_nl_no_pl_pt_ro_ru_sk_sl_sv_tr_uk_vi_200000_lt_100_ut_700_ml_2_size_2000_inter_false_lang_codes_da_en_vi_it_el_ru --search_space_size 200000 --suffix no_int_50_1000 --lower_threshold 50 --upper_threshold 1000 --voc_size 200000

python generate_concept_id_concept_name_mapping.py
