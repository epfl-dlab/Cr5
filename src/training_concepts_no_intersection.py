import pickle
import os
import utils
import sys
import argparse


parser = argparse.ArgumentParser(description='Training concepts file generation')
parser.add_argument("--lang_codes", type=str, default=None, help="Required. String that represents the languages you want to embed. Ex. da_vi_en")
parser.add_argument("--intersection_to_exclude", type=str, default=None, help="Required. String that represents the languages you want to embed. Ex. da_vi")
parser.add_argument("--voc_size", type=int, default=65000, help="Required. The vocabulary size considered when doing the counting")
parser.add_argument("--min_lang", type=int, default=2, help="The minimum number of languages a concept need occur in, to be considered. Default value is 2")
parser.add_argument("--upper_threshold", type=int, default=600, help="The maximum number of unique words an article can have in order to be considered. Default value is 600")
parser.add_argument("--lower_threshold", type=int, default=100, help="The minimum number of unique words an article can have in order to be considered. Default value is 100")
params = parser.parse_args()

assert(params.lang_codes is not None)

lang_codes = params.lang_codes.split('_')
lang_codes = sorted(lang_codes)

intersection_to_exclude = params.intersection_to_exclude.split('_')
intersection_to_exclude = sorted(intersection_to_exclude)

words_considered = params.voc_size
min_lang = params.min_lang
upper_threshold = params.upper_threshold
lower_threshold = params.lower_threshold

modeling_through = set(lang_codes) - set(intersection_to_exclude)

assert(len(modeling_through) != 0)

def is_in_range(num_words, upper_threshold, lower_threshold):
	if num_words > upper_threshold:
		return 1

	if num_words < lower_threshold:
		return -1

	return 0

def filter_docs(docs, upper_threshold, lower_threshold):
	invalid = set()
	for index, words_lang in docs.items():
		if words_lang[0] > upper_threshold or words_lang[1] > upper_threshold or words_lang[2] > upper_threshold:
			invalid.add(index)

		if words_lang[0] < lower_threshold or words_lang[1] < lower_threshold or words_lang[2] < lower_threshold:
			invalid.add(index)

	return invalid

training_concepts_id = '{}_{}_through_{}_lt_{}_ut_{}_ml_{}'.format('_'.join(lang_codes), words_considered, '_'.join(modeling_through), lower_threshold, upper_threshold, min_lang)
dump_file = utils.get_training_concepts_file_path(training_concepts_id)

if os.path.isfile(dump_file):
	print("Training concepts file already exists.")
	sys.exit()

if min_lang > len(lang_codes):
	print("Too big intersection, not enough languages.")
	sys.exit()
	
all_concepts = {}
concepts = {}
potential_concepts = set()

for lang_code in lang_codes:
	lang_id = '%s_%d' % (lang_code, words_considered)
	dump_file_name = "unique_words_per_concept_%s.p" % lang_id
	dump_file_name = utils.get_word_counts_file_path(dump_file_name)
	result = pickle.load(open(dump_file_name, 'rb'))
	
	all_concepts[lang_code] = result['all_concepts']
	concepts[lang_code] = result['concepts']
	potential_concepts = potential_concepts.union(concepts[lang_code])

to_keep_per_lang = {}
print("Number of potential concepts: ", len(potential_concepts))
print("Upper treshold: ", upper_threshold)
print("Lower treshold: ", lower_threshold)
print("Presence in minimum number of languages for a concept to be valid: ", min_lang)

overlap = {}
overlapping_concepts = {}
range_check_fail = {}
intersection_to_exclude = set(intersection_to_exclude)

for lang_code in lang_codes:
	to_keep_per_lang[lang_code] = set()
	range_check_fail[lang_code] = {'less' : 0, 'more' : 0}

for concept_id in potential_concepts:
	langs_with_doc_in_range = []

	for lang_code in lang_codes:
		if concept_id in concepts[lang_code]:
			status = is_in_range(concepts[lang_code][concept_id], upper_threshold, lower_threshold)
			if status == 0:
				langs_with_doc_in_range.append(lang_code)
			elif status == 1:
				range_check_fail[lang_code]['more'] += 1 
			else:
				range_check_fail[lang_code]['less'] += 1 

	if len(langs_with_doc_in_range) >= min_lang:
		if len(intersection_to_exclude - set(langs_with_doc_in_range)) == 0:
			continue

		identifier = "_".join(langs_with_doc_in_range)

		if identifier not in overlap:
			overlap[identifier] = 1
			overlapping_concepts[identifier] = set([concept_id])
		else:
			overlap[identifier] += 1
			overlapping_concepts[identifier].add(concept_id)
		
		for lang_code in langs_with_doc_in_range:
			to_keep_per_lang[lang_code].add(concept_id)

dump_object = {'to_keep_per_lang' : to_keep_per_lang, 'overlapping_concepts' : overlapping_concepts, 'overlap' : overlap, 'range_check_fail' : range_check_fail, 'lang_codes' : lang_codes}
pickle.dump(dump_object, open(dump_file, 'wb'))

print(overlap)