# python target_concepts.py --training_concepts_file_for_sampling 28_langs_200000_lt_100_ut_700_ml_2 --validation_concepts_file_for_exclusion 28_langs_inter_false_da_en_vi_it_el_ru --search_space_size 200000 --suffix 100_700
import random
import utils
import argparse
import os
import pickle

parser = argparse.ArgumentParser(description='Evaluation concepts file generation')
parser.add_argument("--training_concepts_file_for_sampling", type=str, default=None, help="Required. The training concepts file from which we will sample the target concepts set for each language.")
parser.add_argument("--validation_concepts_file_for_exclusion", type=str, default=None, help="Required. The validation concepts should not be included in the search space (they are added later) and all the languages present in the validation set must have a target space defined in order to perform evaluation.")
parser.add_argument("--suffix", type=str, default=200000, help="Required. The suffix added to the language code in order to specify the concrete target space")
parser.add_argument("--search_space_size", type=int, default=200000, help="The desired size of search space. Default value is 200000")
params = parser.parse_args()
print(params.suffix)

assert(params.training_concepts_file_for_sampling is not None)
assert(params.validation_concepts_file_for_exclusion is not None)
assert(params.suffix is not None)

training_concepts_file_for_sampling = params.training_concepts_file_for_sampling
validation_concepts_file_for_exclusion = params.validation_concepts_file_for_exclusion
search_space_size = params.search_space_size

random.seed(123)

training_concepts = pickle.load(open(utils.get_training_concepts_file_path(training_concepts_file_for_sampling), 'rb'))
validation_concepts_per_lang = utils.get_validation_concepts_per_lang(validation_concepts_file_for_exclusion)

validation_set_dump_obj = pickle.load(open(utils.get_validation_concepts_file_path(validation_concepts_file_for_exclusion), 'rb'))
validation_lang_codes = validation_set_dump_obj['params']['lang_codes_to_include']

search_space_size = 200000

for lang_code in validation_lang_codes:
	dump_file = utils.get_target_concepts_file_path(lang_code, params.suffix)

	if os.path.isfile(dump_file):
		print("Target concepts file for language '%s' already exists." % lang_code)
		continue

	concept_pool = training_concepts['to_keep_per_lang'][lang_code]
	validation_concepts = validation_concepts_per_lang[lang_code]

	concept_pool_outside_validation = concept_pool - validation_concepts
	num_potential_concepts = len(concept_pool_outside_validation)
	
	keys_num = min(search_space_size, num_potential_concepts)
	print("Language %s, keys %d" % (lang_code, keys_num))
	keys = random.sample(sorted(concept_pool_outside_validation), keys_num)
	pickle.dump(set(keys), open(dump_file, 'wb'))