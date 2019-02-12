# python target_concepts_no_intersection.py --validation_concepts_file_for_exclusion 28_langs_inter_false_da_en_vi_it_el_ru --search_space_size 200000 --suffix no_int_100_1000 --lower_threshold 100 --upper_threshold 1000 --voc_size 2000

import random
import utils
import argparse
import os
import pickle

parser = argparse.ArgumentParser(description='Evaluation target concepts file generation')
parser.add_argument("--validation_concepts_file_for_exclusion", type=str, default=None, help="Required. The validation concepts should not be included in the search space (they are added later) and all the languages present in the validation set must have a target space defined in order to perform evaluation.")
parser.add_argument("--suffix", type=str, default=None, help="Required. The suffix added to the language code in order to specify the concrete target space")
parser.add_argument("--search_space_size", type=int, default=200000, help="The desired size of search space. Default value is 200000")
parser.add_argument("--voc_size", type=int, default=200000, help="Vocabulary size considered in the counting. Default value is 200000")
parser.add_argument("--upper_threshold", type=int, default=1000, help="The upper threshold of unique words in a document. Default value is 1000")
parser.add_argument("--lower_threshold", type=int, default=100, help="The lower threshold of unique words in a document. Default value is 100")
params = parser.parse_args()

assert(params.validation_concepts_file_for_exclusion is not None)
assert(params.suffix is not None)

validation_concepts_file_for_exclusion = params.validation_concepts_file_for_exclusion
search_space_size = params.search_space_size
upper_threshold = params.upper_threshold
lower_threshold = params.lower_threshold
words_considered = params.voc_size

random.seed(123)

validation_concepts_per_lang = utils.get_validation_concepts_per_lang(validation_concepts_file_for_exclusion)

validation_set_dump_obj = pickle.load(open(utils.get_validation_concepts_file_path(validation_concepts_file_for_exclusion), 'rb'))
if 'lang_codes_to_include' in validation_set_dump_obj['params']:
	validation_lang_codes = validation_set_dump_obj['params']['lang_codes_to_include']
else:
	validation_lang_codes = validation_set_dump_obj['lang_codes']

search_space_size = 200000

def is_in_range(num_words, upper_threshold, lower_threshold):
	if num_words > upper_threshold:
		return 1

	if num_words < lower_threshold:
		return -1

	return 0

range_check_status = {}

#for lang_code in validation_lang_codes:
for lang_code in ['el', 'ru', 'en', 'vi', 'da', 'it']:
	dump_file = utils.get_target_concepts_file_path(lang_code, params.suffix)

	if os.path.isfile(dump_file):
		print("Target concepts file for language '%s' already exists." % lang_code)
		continue

	range_check_status[lang_code] = {'in_range': 0, 'more': 0, 'less': 0}

	lang_id = '%s_%d' % (lang_code, words_considered)
	dump_file_name = "unique_words_per_concept_%s.p" % lang_id
	dump_file_name = utils.get_word_counts_file_path(dump_file_name)

	word_counts = pickle.load(open(dump_file_name, 'rb'))
	concepts = word_counts['concepts']

	concept_pool = set()

	for concept_id in concepts.keys():
		status = is_in_range(concepts[concept_id], upper_threshold, lower_threshold)
		if status == 0:
			range_check_status[lang_code]['in_range'] += 1
			concept_pool.add(concept_id)
		elif status == 1:
			range_check_status[lang_code]['more'] += 1
		else:
			range_check_status[lang_code]['less'] += 1

	validation_concepts = validation_concepts_per_lang[lang_code]

	concept_pool_outside_validation = concept_pool - validation_concepts
	num_potential_concepts = len(concept_pool_outside_validation)
	
	keys_num = min(search_space_size, num_potential_concepts)
	print("Language %s, keys %d" % (lang_code, keys_num))
	keys = random.sample(sorted(concept_pool_outside_validation), keys_num)
	pickle.dump(set(keys), open(dump_file, 'wb'))

print(range_check_status)
