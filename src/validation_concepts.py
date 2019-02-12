# python validation_concepts.py --training_concepts_file_name=bg_ca_cs_da_de_el_en_es_et_fi_fr_hr_hu_id_it_mk_nl_no_pl_pt_ro_ru_sk_sl_sv_tr_uk_vi_200000_lt_100_ut_700_ml_2 --voc_size=200000 --val_set_size=2000 --get_full_intersection=False --lang_codes_to_include=da_en_vi_it_gr_ru

import pickle
import random
import itertools
import utils
import math
import argparse
import os

parser = argparse.ArgumentParser(description='Evaluation concepts file generation')
parser.add_argument("--training_concepts_file_name", type=str, default=None, help="Required. The training concepts file you want to sample from.")
parser.add_argument("--voc_size", type=int, default=None, help="Required. The vocabulary size considered when doing the unique word counting.")
parser.add_argument("--lang_codes_to_include", type=str, default=None, help="The training concepts file you want to sample from. Ex. da_it")
parser.add_argument("--val_set_size", type=int, default=2000, help="The desired size of the validation concepts set. Default value is 1000")
parser.add_argument("--get_full_intersection", type=str, default='False', help="Whether on top of the pairwise intersection, a validation set in the full intersection of the languages should be include. Possible values: True or False. Default value is True.")
parser.add_argument("--max_val_set_size_full_inter_ratio", type=float, default=0.3, help="Defines the maximum ratio between the validation set taken out and the training concepts available (it refers to the full intersection). Default value is 0.3.")
params = parser.parse_args()

assert(params.training_concepts_file_name is not None)
assert(params.voc_size is not None)
assert(os.path.isfile(utils.get_training_concepts_file_path(params.training_concepts_file_name)))
assert(params.get_full_intersection == 'True' or params.get_full_intersection == 'False')

def is_superset(set_identifier, min_intersection):
	min_intersection = set(min_intersection)
	langs_in_set = set(set_identifier.split('_'))
	return langs_in_set.issuperset(min_intersection)

training_concepts_file_name = params.training_concepts_file_name # 'da_de_en_es_vi_65000_lt_100_ut_600_ml_2'
voc_considered = params.voc_size # 65000
val_set_size = params.val_set_size # 2000
get_full_intersection = True if params.get_full_intersection == 'True' else False
max_val_set_size_full_inter_ratio = params.max_val_set_size_full_inter_ratio # 0.3

dump_file_name = '{}_size_{}_inter_{}'.format(training_concepts_file_name, val_set_size, "true" if get_full_intersection else "false")

# Reading data

training_concepts_dump = pickle.load(open(utils.get_training_concepts_file_path(training_concepts_file_name), 'rb'))
lang_codes = training_concepts_dump['lang_codes']

if params.lang_codes_to_include is not None:
	lang_codes_to_include = set(params.lang_codes_to_include.split('_'))

	# Check that all the languages that we try to include are contained in the training_concepts_dump
	assert(len(lang_codes_to_include - set(lang_codes)) == 0)
else:
	lang_codes_to_include = set(lang_codes)
	params.lang_codes_to_include = "_".join(lang_codes)
dump_file_name = '{}_lang_codes_{}'.format(dump_file_name, params.lang_codes_to_include)
overlapping_concepts = training_concepts_dump['overlapping_concepts']

params = {'voc_considered' : voc_considered, 'get_full_intersection' : get_full_intersection, 'val_set_size' : val_set_size, \
		  'max_val_set_size_full_inter_ratio' : max_val_set_size_full_inter_ratio, 'training_concepts_file_name' : training_concepts_file_name, 'lang_codes_to_include' : lang_codes_to_include}

validation_dump_file = {}
validation_dump_file['lang_codes'] = lang_codes
validation_dump_file['params'] = params

# Generating pairwise validation sets
random.seed(123)

pairs = sorted(list(itertools.combinations(lang_codes, 2)))
for pair in pairs:
	pair_identifier = '_'.join(sorted(pair))

	# Check if both languages of the pair are in the set of languages to be included
	lang_1, lang_2 = pair
	if (lang_1 not in lang_codes_to_include) or (lang_2 not in lang_codes_to_include):
		validation_dump_file[pair_identifier] = set()
		continue
	
	concepts_in_intersection = set() 
	for set_identifier in overlapping_concepts.keys():
		if is_superset(set_identifier, pair):
			concepts_in_intersection = concepts_in_intersection.union(overlapping_concepts[set_identifier])
	
	random_validation_set = random.sample(sorted(set(concepts_in_intersection)), min(len(concepts_in_intersection), val_set_size))
	validation_dump_file[pair_identifier] = set(random_validation_set)

 # Generating full intersection validaiton set (if requested)

if get_full_intersection:
	concepts_in_full_intersection = set() 
	for set_identifier in overlapping_concepts.keys():
		if is_superset(set_identifier, lang_codes):
			concepts_in_full_intersection = concepts_in_full_intersection.union(overlapping_concepts[set_identifier])

	# Get max set size
	set_size = min(len(concepts_in_full_intersection), val_set_size)
	if max_val_set_size_full_inter_ratio is not None:
		set_size = min(set_size, math.ceil(len(concepts_in_full_intersection)*max_val_set_size_full_inter_ratio))
		
	# Get full intersection from pairwise validation sets
	current_full_intersection = None
	for pair in pairs:
		pair_identifier = '_'.join(sorted(pair))
		pairwise_val_set = validation_dump_file[pair_identifier]
		
		if current_full_intersection == None:
			current_full_intersection = pairwise_val_set
		else:
			current_full_intersection = current_full_intersection & pairwise_val_set

	current_full_intersection_size = len(current_full_intersection)
	print("Pairwise validation_set contains %d elements from full intersection" % current_full_intersection_size)

	# Add concepts if current validation set is smaller than desired
	if len(current_full_intersection) < set_size:
		set_diff = concepts_in_full_intersection - current_full_intersection
		to_add = set_size - current_full_intersection_size
		
		random_validation_set = random.sample(sorted(set(set_diff)), to_add)
		current_full_intersection = current_full_intersection.union(random_validation_set)
		
	set_identifier = '_'.join(sorted(lang_codes))
	validation_dump_file[set_identifier] = current_full_intersection

# Dump generated file
pickle.dump(validation_dump_file, open(utils.get_validation_concepts_file_path(dump_file_name), 'wb'))
