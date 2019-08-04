import os
import numpy as np
import scipy.sparse as sparse
import gzip
import codecs
import pickle
import matplotlib.pyplot as plt
import random
try:
	import evaluator
except ModuleNotFoundError:
	print("Evaluator could not be included")
	print("Execution continues (assuming that error occurred in training)")


# HOME_DIR_1 = '/dlabdata1/west/crosslingwiki/' # Wikipedia data location

# HOME_DIR_2 = '/scratch/josifosk/' # Home directory for the intermediate data files and result dumps
HOME_DIR_2 = './..' # Home directory for the intermediate data files and result dumps
HOME_DIR_2 = os.path.join(HOME_DIR_2, 'data')
HOME_DIR_1 = HOME_DIR_2

WORD_INDICES = os.path.join(HOME_DIR_1, 'matlab_input/word_indices')
SPARSE_MATRICES = os.path.join(HOME_DIR_1, 'matlab_input/sparse_matrices')
CONCEPT_INDICES = os.path.join(HOME_DIR_1, 'matlab_input/concept_indices')
EMBEDDINGS_DUMPS = os.path.join(HOME_DIR_2, 'embeddings_dumps')
TRAINING_CONCEPTS_PATH = os.path.join(HOME_DIR_2, 'training_concepts')
TRAINING_DATASETS_PATH = os.path.join(HOME_DIR_2, 'training_datasets')
VALIDATION_DATASETS_PATH = os.path.join(HOME_DIR_2, 'validation_datasets')
VALIDATION_CONCEPTS_PATH = os.path.join(HOME_DIR_2, 'validation_concepts')
RESULTS_DUMPS_PATH = os.path.join(HOME_DIR_2, 'results_dumps')
TARGET_CONCEPTS_DUMPS_PATH = os.path.join(HOME_DIR_2, 'target_concepts')
WORD_COUNTS_PATH = os.path.join(HOME_DIR_2, 'word_counts')
VOCABULARRY_MAPPING_PATH = os.path.join(HOME_DIR_2, 'voc_mappings')
LOGS_PATH = os.path.join(HOME_DIR_2, 'logs')

#####
# File system navigation
#####

def evaluate_experiment(experiment_id, lang_codes_to_evaluate, vocabulary_size, target_concepts_suffix, axes, _dir = 'avg', dimensions = 300, validation_set_file_name = None):
	for i, test_set_flag in enumerate([False, True]):
		matched_runs = filter_runs_on_field(experiment_id, 'dimensions', dimensions, verbose = False)
		experiment_run_ids = [run['id'] for run in matched_runs]

		assert(len(experiment_run_ids) != 0)

		mr_eval_obj = evaluator.Multirun_evaluator(experiment_run_ids, lang_codes_to_evaluate, vocabulary_size = 200000, test_set_flag = test_set_flag, target_concepts_suffix = target_concepts_suffix, validation_set_file_name = validation_set_file_name)

		ax = axes[i]

		if test_set_flag:
			title = 'Test performance'
		else:
			title = 'Validation performance'

		ax.set_xlabel('Regularization parameter')
		ax.set_ylabel('Harmonic mean of ranks')
		mr_eval_obj.plot_performance(lang_codes_to_evaluate, _dir, title, ax)

def plot_processing_time(experiment_id, dimensions = 300, eigs_tol = 1):
	matched_runs = filter_runs_on_field(experiment_id, 'dimensions', dimensions, verbose = False)
	experiment_run_ids = [run['id'] for run in matched_runs]

	assert(len(experiment_run_ids) != 0)
	fig, ax = plt.subplots(1, 1, figsize=(7,5), sharey=True)
	fig.suptitle("Experiment {}".format(experiment_id))

	ax.set_xlabel('Regularization parameter')
	ax.set_ylabel('Time in minutes')

	plot_execution_time(experiment_run_ids, eigs_tol, ax)

		
def evaluate_experiment_diff_lang_pairs(lang_pairs_to_evaluate, experiment_id, vocabulary_size, target_concepts_suffix, _dir = 'avg', dimensions = 300, validation_set_file_name = None):
	for lang_pair in lang_pairs_to_evaluate:
		r = 1
		c = 2
		fig, axes = plt.subplots(1, 2, figsize=(20,7), sharey=True)
		fig.suptitle("Language pair {}, Experiment {}, Target concepts {} - {}".format('_'.join(lang_pair), experiment_id, target_concepts_suffix, _dir))
		
		evaluate_experiment(experiment_id, lang_pair, vocabulary_size, target_concepts_suffix, axes, _dir = 'avg', dimensions = 300, validation_set_file_name = validation_set_file_name)
		
def evaluate_experiment_diff_target_concepts(target_concepts_suffixes, experiment_id, lang_codes_to_evaluate, vocabulary_size, _dir = 'avg', dimensions = 300, validation_set_file_name = None):
	for target_concepts_suffix in target_concepts_suffixes:
		r = 1
		c = 2
		fig, axes = plt.subplots(1, 2, figsize=(20,7), sharey=True)
		fig.suptitle("Language pair {}, Experiment {}, Target concepts {} - {}".format('_'.join(lang_codes_to_evaluate), experiment_id, target_concepts_suffix, _dir))
		
		evaluate_experiment(experiment_id, lang_codes_to_evaluate, vocabulary_size, target_concepts_suffix, axes, _dir = 'avg', dimensions = 300, validation_set_file_name = validation_set_file_name)

def get_params_learned_exp_id(experiment_id, emb_dim, dico_eval, src_lang, tgt_lang):
    params_dict = {'cuda': False,
     'experiment_id' : experiment_id,
     'results_file_name' : '',
     'dico_eval': dico_eval,
     'emb_dim': emb_dim,
     'exp_id': '',
     'exp_name': 'debug',
     'exp_path': '/dlabdata1/josifosk/MUSE/dumped/debug/2en2y27clm',
     'normalize_embeddings': '',
     'max_vocab': 200000,
     'src_lang': src_lang,
     'tgt_lang': tgt_lang,
     'verbose': 2}

    params = lambda: None
    for key in params_dict.keys():
        setattr(params, key, params_dict[key])
    
    return params

def get_params_learned_exp_run(experiment_run, emb_dim, dico_eval, src_lang, tgt_lang):
    params_dict = {'cuda': False,
     'results_file_name' : experiment_run,
     'dico_eval': dico_eval,
     'emb_dim': emb_dim,
     'exp_id': '',
     'exp_name': 'debug',
     'exp_path': '/dlabdata1/josifosk/MUSE/dumped/debug/2en2y27clm',
     'max_vocab': 200000,
     'normalize_embeddings': '',
     'src_lang': src_lang,
     'tgt_lang': tgt_lang,
     'verbose': 2}

    params = lambda: None
    for key in params_dict.keys():
        setattr(params, key, params_dict[key])
        
    return params

def get_params_baseline(src_lang, tgt_lang, dico_eval):
    params_dict = {'cuda': False,
     'dico_eval': dico_eval,
     'emb_dim': 300,
     'exp_id': '',
     'exp_name': 'debug',
     'exp_path': '/dlabdata1/josifosk/MUSE/dumped/debug/2en2y27clm',
     'max_vocab': 200000,
     'normalize_embeddings': '',
     'src_emb': 'data/wiki.multi.%s.vec' % src_lang,
     'src_lang': src_lang,
     'tgt_emb': 'data/wiki.multi.%s.vec' % tgt_lang,
     'tgt_lang': tgt_lang,
     'verbose': 2}

    params = lambda: None
    for key in params_dict.keys():
        setattr(params, key, params_dict[key])
        
    return params

def get_full_logs_path(experiment_run_identifier):
	experiments_log_file_name = "{}_full".format(experiment_run_identifier)
	return get_logs_path(experiments_log_file_name)
	
def get_runs_logs_path(experiment_identifier):
	experiment_parameter_logs_file_name = "{}_runs".format(experiment_identifier)
	return get_logs_path(experiment_parameter_logs_file_name)

def get_logs_path(file_name):
	return os.path.join(LOGS_PATH, file_name)

def get_vocabulary_mapping_file_path(file_name):
	return os.path.join(VOCABULARRY_MAPPING_PATH, file_name)

def get_word_counts_file_path(file_name):
	return os.path.join(WORD_COUNTS_PATH, file_name)

def get_target_concepts_file_path(lang_code, suffix):
	'''Returns the absolute path of the validation_concepts file that corresponds to the passed file name'''
	if suffix is not None:
		print("Suffix used for target documents: ", suffix)
		return os.path.join(TARGET_CONCEPTS_DUMPS_PATH, '{}_{}_keys'.format(lang_code, suffix))

	return os.path.join(TARGET_CONCEPTS_DUMPS_PATH, '{}_keys'.format(lang_code))
	

def get_results_dump_file_path(file_name):
	'''Returns the absolute path of the validation_concepts file that corresponds to the passed file name'''
	return os.path.join(RESULTS_DUMPS_PATH, file_name)

def get_validation_concepts_file_path(file_name):
	'''Returns the absolute path of the validation_concepts file that corresponds to the passed file name'''
	return os.path.join(VALIDATION_CONCEPTS_PATH, file_name)

def get_validation_datasets_file_path(file_name):
	'''Returns the absolute path of the validation_dataset file that corresponds to the passed file name'''
	return os.path.join(VALIDATION_DATASETS_PATH, file_name)

def get_training_concepts_file_path(file_name):
	'''Returns the absolute path of the training_concepts file that corresponds to the passed file name'''
	return os.path.join(TRAINING_CONCEPTS_PATH, file_name)

def get_training_dataset_file_path(file_name):
	'''Returns the absolute path of the training_dataset file that corresponds to the passed file name'''
	return os.path.join(TRAINING_DATASETS_PATH, file_name)

def get_word_index_file_path(lang_code):
	'''Returns the absolute of the word index file that corresponds to the passed langage code'''
	file_name = 'word_index_{}.tsv'.format(lang_code)
	return os.path.join(WORD_INDICES, file_name)

def get_concept_index_file_path(lang_code):
	'''Returns the absolute path of the concept index file that corresponds to the passed langage code'''
	file_name = 'concept_index_{}.tsv'.format(lang_code)
	return os.path.join(CONCEPT_INDICES, file_name)

def get_sparse_matrix_file_path(lang_code):
	'''Returns the absolute path of the word index file that corresponds to the passed langage code'''
	file_name = 'sparse_matrix_{}.txt.gz'.format(lang_code)
	return os.path.join(SPARSE_MATRICES, file_name)

def get_embeddings_file_path(lang_code):
	'''Returns the absolute path to the embeddings corresponding to the passed langauge code'''
	file_name = 'wiki.multi.{}.vec'.format(lang_code)
	return os.path.join(EMBEDDINGS_DUMPS, file_name)

def get_vocabulary_size(lang_code):
	'''Returns the maximum id for the passed language code'''

	word_index_file = get_word_index_file_path(lang_code)
	
	with codecs.open(word_index_file, "r", "utf-8") as f:
		c = 0
		
		for line in f:
			c = c + 1
			
		parts = line.split()
		if int(parts[0]) != c:
			raise Exception("Last index -> {}; doesn't match the number of lines -> {}.".format(parts[0], c))
		return c
	
def get_num_concepts_from_sparse(lang_code):
	'''Returns the number of concepts represented inside a sparse_matrix file
	Used for testing purposes'''
	with gzip.open(get_sparse_matrix_file_path(lang_code), 'r') as f:
		concepts = set()
		for line in codecs.getreader("utf-8")(f):
			parts = line.split()
			concepts.add(parts[0])
		return len(concepts)

def get_extended_validation_set_file_name(validation_set_file_name, test_set_flag, target_concepts_suffix):
	if target_concepts_suffix != None:
		return "{}_tsf_{}_suffix_{}".format(validation_set_file_name, "true" if test_set_flag else "false", target_concepts_suffix)

	return "{}_{}".format(validation_set_file_name, "true" if test_set_flag else "false")

#####
# Training and logging helpers
#####

def get_validation_concepts_per_lang(file_name):
	'''Returns the concepts used for validation (according to the validation file passed as argument) for each language in the validation file.
	Used in training to exclude the concepts used for validation'''

	validation_set_dump_obj = pickle.load(open(get_validation_concepts_file_path(file_name), 'rb'))
	
	# get all validation combinations, remove lang_codes since it is not one
	lang_codes = validation_set_dump_obj['lang_codes']
	validation_combinations = set(list(validation_set_dump_obj.keys()))
	validation_combinations.remove('lang_codes')
	if 'params' in validation_combinations:
		validation_combinations.remove('params')

	# loop through languages, get concepts where lang_code is one of the languages in the combination
	validation_set_per_langs = {}
	for lang_code in lang_codes:
		validation_set = set()
		for comb_code in validation_combinations:
			comb_lang_set = set(comb_code.split('_'))
			if lang_code in comb_lang_set:
				comb_val_set = validation_set_dump_obj[comb_code]
				validation_set = validation_set.union(comb_val_set)
				
		validation_set_per_langs[lang_code] = validation_set
		
	return validation_set_per_langs

def get_test_concepts_from_validation_set(validation_set_name):
	print("Running get_test_concepts_from_validation_set")
	validation_set_dump_obj = pickle.load(open(get_validation_concepts_file_path(validation_set_name), 'rb')) 
	all_pairs = set(validation_set_dump_obj.keys()) - set(['params', 'lang_codes'])
	
	stratified_validation_sets = {}
	
	for pair_id in all_pairs:
		if len(validation_set_dump_obj[pair_id]) == 0:
			stratified_validation_sets[pair_id] = set()
		else:
			random.seed(123)
			
			full_val_set = sorted(list(validation_set_dump_obj[pair_id]))
			
			test_set_size = len(full_val_set) / 2
			if len(full_val_set) % 2 != 0:
				test_set_size += 1
			
			test_set_size = int(test_set_size)
			test_val_set = random.sample(full_val_set, test_set_size)
			
			stratified_validation_sets[pair_id] = test_val_set
			# print("Test set for lang_code: {}, size: {}, full: {}".format(pair_id, len(test_val_set), len(full_val_set)))
			
	return stratified_validation_sets

def get_val_concepts_from_validation_set(validation_set_name):
	print("Running get_val_concepts_from_validation_set")

	validation_set_dump_obj = pickle.load(open(get_validation_concepts_file_path(validation_set_name), 'rb')) 
	all_pairs = set(validation_set_dump_obj.keys()) - set(['params', 'lang_codes'])
	
	test_concept_sets = get_test_concepts_from_validation_set(validation_set_name)
	val_concept_sets = {}
	
	for pair_id in all_pairs:
		val_concept_sets[pair_id] = set(validation_set_dump_obj[pair_id]) - set(test_concept_sets[pair_id])
	
	return val_concept_sets

def get_filtered_validation_concepts_per_lang(file_name, test_set_flag):
	'''Returns the concepts used for validation (according to the validation file passed as argument) for each language in the validation file.
	Used in training to exclude the concepts used for validation'''

	validation_set_dump_obj = pickle.load(open(get_validation_concepts_file_path(file_name), 'rb'))

	# get all validation combinations, remove lang_codes since it is not one
	lang_codes = validation_set_dump_obj['lang_codes']
	validation_combinations = set(validation_set_dump_obj.keys()) - set(['params', 'lang_codes'])
	
	if test_set_flag:
		print("Getting test concepts")
		filtered_concepts_set = get_test_concepts_from_validation_set(file_name)
	else:
		print("Getting validation concepts")
		filtered_concepts_set = get_val_concepts_from_validation_set(file_name)

	# loop through languages, get concepts where lang_code is one of the languages in the combination
	validation_set_per_langs = {}
	for lang_code in lang_codes:
		validation_set = set()
		for comb_code in validation_combinations:
			comb_lang_set = set(comb_code.split('_'))
			if lang_code in comb_lang_set:
				comb_val_set = filtered_concepts_set[comb_code]
				validation_set = validation_set.union(comb_val_set)

		validation_set_per_langs[lang_code] = validation_set

	return validation_set_per_langs

def is_close_enough(x, y):
	'''Comparator between two floats'''
	diff = x-y
	return abs(x-y) <= 0.015

def read_parameters_from_string(line):
	'''Generates a parameters dictionary from a logged experiment run string'''
	line = line.strip()
	params = {}
	parts = line.split(' ')
	params['id'] = parts[0]
	for i in range(1, len(parts)):
		if parts[i] == 'lambda':
			params['lambda'] = float(parts[i+1])
			i += 1
		
		if parts[i] == 'cg_tol' or parts[i] == 'eigs_tol': 
			params['%s_1' % parts[i]] = float(parts[i+1])
			params['%s_2' % parts[i]] = float(parts[i+2])
			i += 2
		
		if parts[i] == 'dims':
			params['dimensions'] = parts[i+1]
			i += 1
			
	return params

def get_run_identifier(experiment_identifier, dims, _lambda, cg_tol_1, eigs_tol_1, cg_tol_2 = None, eigs_tol_2 = None, return_line = False):
	'''Returns the run identifier for an experiment with given parameters'''

	logs_full_path = get_runs_logs_path(experiment_identifier)

	if not os.path.isfile(logs_full_path):
		print('The runs log file for the given experiment identifier does not exist.')
		return None

	params = {}
	
	params['dimensions'] = dims
	params['lambda'] = _lambda
	params['cg_tol_1'] = cg_tol_1
	params['eigs_tol_1'] = eigs_tol_1

	if cg_tol_2 is None:
		params['cg_tol_2'] = cg_tol_1
	else:
		params['cg_tol_2'] = cg_tol_2

	if eigs_tol_2 is None:
		params['eigs_tol_2'] = eigs_tol_1
	else:
		params['eigs_tol_2'] = eigs_tol_2
	
	d_line = []
	
	with open(logs_full_path, 'r') as f:
		for line in f:
			line = line.strip()
			curr_params = read_parameters_from_string(line)
			all_close = True
			for param in params.keys():
				if not is_close_enough(float(params[param]), float(curr_params[param])):
					all_close = False
					break
			if all_close:
				d_line.append(line.strip())
				
	if len(d_line) == 0:
		print('There is no experiment run with the chosen set of parameters')
		return None

	# print("Experiment found as: ", d_line[0])

	assert(len(d_line) == 1)

	if return_line:
		return (read_parameters_from_string(d_line[0])['id'], d_line)

	return read_parameters_from_string(d_line[0])['id']

def get_exp_num_for_id(experiment_identifier):
	'''Returns the first vacant experiment run number for the experiment with the identifier passed as argument'''

	all_log_files = [i for i in os.listdir(get_logs_path('')) if not i.endswith('runs')]
	max_id = -1
	for log_file in all_log_files:
		exp_id, exp_num, _ = log_file.rsplit('_', 2)
		if experiment_identifier == exp_id:
			max_id = max(max_id, int(exp_num))
	max_id += 1
	return max_id

def generate_params_dict(training_concepts_file_name, validation_set_file_name, case_folding_flag, _lambda, dimensions, cg_tol_1, eigs_tol_1, cg_tol_2, eigs_tol_2, vocabulary_size):
	'''Generates a parameters dictionary corresponding to the arguments passed'''

	params = {}

	params['training_concepts_file_name'] = training_concepts_file_name
	params['validation_set_file_name'] = validation_set_file_name
	params['case_folding_flag'] = case_folding_flag

	params['dimensions'] = dimensions

	params['lambda'] = _lambda

	params['vocabulary_size'] = vocabulary_size

	params['cg_tol_1'] = cg_tol_1
	params['eigs_tol_1'] = eigs_tol_1

	if cg_tol_2 is None:
		params['cg_tol_2'] = cg_tol_1
	else:
		params['cg_tol_2'] = cg_tol_2

	if eigs_tol_2 is None:
		params['eigs_tol_2'] = eigs_tol_1
	else:
		params['eigs_tol_2'] = eigs_tol_2

	return params

def params_to_string(params):
	'''Returns a string representation of the experiment run parameters'''
	_lambda = '{0:.2f}'.format(params['lambda'])
	dims = params['dimensions']

	# CG tolerance
	cg_tol_1 = '{0:.2f}'.format(params['cg_tol_1'])

	if 'cg_tol_2' in params:
		cg_tol_2 = '{0:.2f}'.format(params['cg_tol_2'])
	else:
		cg_tol_2 = cg_tol_2

	# Eigs tolerance
	eigs_tol_1 = '{0:.2f}'.format(params['eigs_tol_1'])

	if 'eigs_tol_2' in params:
		eigs_tol_2 = '{0:.2f}'.format(params['eigs_tol_2'])
	else:
		eigs_tol_2 = eigs_tol_1

	line = 'lambda {} cg_tol {} {} eigs_tol {} {} dims {}'.format(_lambda, cg_tol_1, cg_tol_2, eigs_tol_1, eigs_tol_2, dims)

	return line

#####
# Vocabulary index retrieval
#####

def get_word_id_2_word_mapping(lang_code):
	'''Returns word_id_2_word mapping from the original index for the language passed as argument'''

	word_index_file = get_word_index_file_path(lang_code)

	with codecs.open(word_index_file, "r", "utf-8") as f:
		word_id_2_word = {}
		for line in f:
			line = ' '.join(line.split())
			parts = line.strip().split(' ')
			try:
				word_id_2_word[int(parts[0])] = parts[2]
			except IndexError:
				word_id_2_word[int(parts[0])] = 'Missing word'
				print("Missing word in {0} vocabulary! Line: {1}".format(lang_code, line))
		
	return word_id_2_word

def get_vocabulary_mapping(lang_code, case_folding_flag):
	'''Returns id_2_word mapping for vocabulary in a given language if not case_folded or 
	init_id_2_lowercase_id, word_2_lowercase_id (where words are case folded), lowercase_id_2_word if case folded'''

	dump_name = '{}_{}'.format(lang_code, 'casefolded' if case_folding_flag else 'non_casefolded')
	dump_path = get_vocabulary_mapping_file_path(dump_name)

	if os.path.isfile(dump_path):
		return pickle.load(open(dump_path, 'rb'))

	word_id_2_word = get_word_id_2_word_mapping(lang_code)
	
	if not case_folding_flag:
		pickle.dump(word_id_2_word, open(dump_path, 'wb'))
		return word_id_2_word
	
	ordered_items = sorted(list(word_id_2_word.items()), key = lambda x: x[0])
	lowercase_word_2_lowercase_id = {}
	word_id_2_lowercase_id = {}

	c = 0
	for word_id, word in ordered_items:
		lowercase_word = word.lower()

		if lowercase_word not in lowercase_word_2_lowercase_id:
			lowercase_word_2_lowercase_id[lowercase_word] = c
			c += 1

		lowercase_id = lowercase_word_2_lowercase_id[lowercase_word]
		word_id_2_lowercase_id[word_id] = lowercase_id
	
	lowercase_id_2_lowercase_word = { _id : word for word, _id  in lowercase_word_2_lowercase_id.items()}

	pickle.dump([word_id_2_lowercase_id, lowercase_word_2_lowercase_id, lowercase_id_2_lowercase_word], open(dump_path, 'wb'))
	
	return word_id_2_lowercase_id, lowercase_word_2_lowercase_id, lowercase_id_2_lowercase_word

#####
# Baseline embeddings
#####

def read_baseline_embeddings(lang_code):
	'''Returns word_2_index dictionary and a list of the embeddings corresponding to the index'''
	
	embeddings_file_path = get_embeddings_file_path(lang_code)
	with open(embeddings_file_path, 'r') as f:
		word_2_index = {}
		embs = []
		next(f)
		c = 0
		for line in f:
			parts = line.split(sep=' ')
			word = parts[0]
			emb = np.array(parts[1:], dtype=np.float64)
			embs.append(emb)
			word_2_index[word] = c
			c += 1

	return word_2_index, embs

def get_unsup_embeddings(lang_codes):
	'''Returns index_2_word_id_mapping and the embeddings corresponding to the word_ids in the index, for every language passed as argument.
	The words are lowercased (hence word_ids correspond to a new case_folded word index). 
	The word_embeddings used originate from the baseline embeddings.'''

	index_2_word_id_per_lang = {}
	embs_per_lang = {}

	for lang_code in lang_codes:
		word_2_index, embs = read_baseline_embeddings(lang_code)

		word_id_2_lowercase_id, lowercase_word_2_lowercase_id, lowercase_id_2_lowercase_word = \
			get_vocabulary_mapping(lang_code, case_folding_flag=True)

		word_index_pairs = sorted(list(word_2_index.items()), key = lambda x: x[1])
		ordered_index_2_lowercase_word_id = {}
		
		ordered_embs = []
		c = 0
		for word, emb_idx in word_index_pairs:
			if word in lowercase_word_2_lowercase_id:
				ordered_index_2_lowercase_word_id[c] = lowercase_word_2_lowercase_id[word]
				ordered_embs.append(embs[emb_idx])
				c += 1

		index_2_word_id_per_lang[lang_code] = ordered_index_2_lowercase_word_id
		embs_per_lang[lang_code] = np.array(ordered_embs)
		
	return index_2_word_id_per_lang, embs_per_lang

def get_unsup_embeddings_non_casefolded(lang_codes):
	'''Returns index_2_word_id_mapping and the embeddings corresponding to the word_ids in the index, for every language passed as argument.
	The words are not lowercased (hence the word_ids correspond to the word_index in the dataset). 
	The word_embeddings used originate from the baseline embeddings.'''

	index_2_word_id_per_lang = {}
	embs_per_lang = {}

	for lang_code in lang_codes:
		word_2_index, embs = helpers2.read_baseline_embeddings(lang_code)
			
		word_id_2_word = helpers2.get_vocabulary_mapping(lang_code, case_folded=False)
		word_2_word_id = {v:k for k,v in word_id_2_word.items()}
		
		ind = 0
		index_2_word_id = {}
		ordered_embs = []
		
		for word, emb_idx in word_2_index.items():
			if word in word_2_word_id:
				index_2_word_id[ind] = word_2_word_id[word]
				ordered_embs.append(embs[emb_idx])
				ind += 1

		index_2_word_id_per_lang[lang_code] = index_2_word_id
		embs_per_lang[lang_code] = np.array(ordered_embs)
		print("Matched %d words in %s" % (len(ordered_embs), lang_code))
		
	return index_2_word_id_per_lang, embs_per_lang    

#####
# Sparse matrix manipulation
#####

def get_nnz(M, axis):
	'''Returns the number of nonzero elements on a given axis of a scipy sparse matrix.'''

	assert (axis == 'row' or axis == 'col')
	if axis == 'row':
		return np.bincount(M.tocoo().row)
		
	return np.bincount(M.tocoo().col)
	
def update_index_after_removal(index, idx_to_drop, idx_to_keep = None):
	'''Returns an updated index (words or concepts) by dropping indices passed as parameter to the function.'''

	assert (idx_to_keep is not None or idx_to_drop is not None)
	size = len(index)

	if idx_to_keep is None:
		idx_to_keep = get_idx_to_keep(size, idx_to_drop)
	else:
		idx_to_keep = np.sort(idx_to_keep)
	
	ordered_elements = np.array([index[i] for i in range(size)])
	elements_to_keep = ordered_elements[idx_to_keep]
	
	return {i:elements_to_keep[i] for i in range(len(elements_to_keep))}

def get_idx_to_keep(size, idx_to_drop):
	'''Returns a boolean mask for numpy array that keeps relevant elements'''

	return ~np.in1d(range(size), idx_to_drop)

def remove_rows(M, idx_to_drop):
	'''Efficient in-place removal of rows indexed in idx_to_drop. Passed matrix must be in csr format.'''

	idx_to_drop = sorted(idx_to_drop, reverse=True)
	for idx in idx_to_drop:
		remove_row_csr(M, idx)

def remove_row_csr(M, i):
	'''Efficient removal of row i in-place. Passed matrix must be in csr format.'''

	if not isinstance(M, sparse.csr_matrix):
		raise ValueError("works only for CSR format -- use .tocsr() first")

	n = M.indptr[i+1] - M.indptr[i]
	if n > 0:
		M.data[M.indptr[i]:-n] = M.data[M.indptr[i+1]:]
		M.data = M.data[:-n]
		M.indices[M.indptr[i]:-n] = M.indices[M.indptr[i+1]:]
		M.indices = M.indices[:-n]
	M.indptr[i:-1] = M.indptr[i+1:]
	M.indptr[i:] -= n
	M.indptr = M.indptr[:-1]
	M._shape = (M._shape[0]-1, M._shape[1])

def remove_columns(M, idx_to_drop):
	'''Efficient removal of columns with indices passed in idx_to_drop. Returned matrix is in csr format.'''

	idx_to_drop = np.unique(idx_to_drop)
	C = M.tocoo()
	keep = ~np.in1d(C.col, idx_to_drop)
	C.data, C.row, C.col = C.data[keep], C.row[keep], C.col[keep]
	C.col -= idx_to_drop.searchsorted(C.col)    # decrement column indices
	C._shape = (C.shape[0], C.shape[1] - len(idx_to_drop))
	return C.tocsr()

#####
# Validation helpers
#####
def get_concept_id_concept_name_mapping():
	file_path = os.path.join(HOME_DIR_2, 'concept_id_concept_name_mapping.p')

	if not os.path.isfile(file_path):
		raise Exception('The concept_id_concept_name_mapping file does not exist, please generate it.')

	with open(file_path, 'rb') as f:
		return pickle.load(f) 

def get_different_param_vals(experiment_identifier, key = 'eigs_tol_1'):
	'''Returns a list of all of the different parameters for a paramater field in the '''

	logs_full_path = get_runs_logs_path(experiment_identifier)

	if not os.path.isfile(logs_full_path):
		print('The runs log file for the given experiment identifier does not exist.')
		return None
	
	different_vals = []
	with open(logs_full_path, 'r') as f:
		for line in f:
			line = line.strip()
			curr_params = read_parameters_from_string(line)
			inside = False
			for tol in different_vals:
				if is_close_enough(float(tol), float(curr_params[key])):
					inside = True
					break
			if not inside:
				different_vals.append(curr_params[key])
				
	return different_vals

def filter_runs_on_field(experiment_identifier, field_name, field_value, verbose = True):
	'''Given an experiment identifier, a field in the parameter dictionary and a value for that field, filters the runs from that experiment
	and returns the parameters that pass the filter'''

	logs_full_path = get_runs_logs_path(experiment_identifier)

	if not os.path.isfile(logs_full_path):
		print('The runs log file for the given experiment identifier does not exist.')
		return None
	
	params = {}

	params[field_name] = field_value
		
	matches = []
	
	with open(logs_full_path, 'r') as f:
		for line in f:
			line = line.strip()
			curr_params = read_parameters_from_string(line)
			all_close = True
			for param in params.keys():
				if not is_close_enough(float(params[param]), float(curr_params[param])):
					all_close = False
					break
			if all_close:
				matches.append(curr_params)
				if verbose:
					print("MATCH -> ", line)
			else:
				if verbose:
					print("skip -> ", line)

	if len(matches) == 0:
		print('There is no experiment run with the chosen set of parameters')
		return None
					  
	return matches

def filter_ranking_df_by_direction(df, _dir):
	'''Filters all other lang1_lang2 retrieval directions from df, keeping only the retrival in direction passed as argument _dir.'''

	return df.query('dir == "{}"'.format(_dir))

def print_different_param_vals(experiment_id):
	print("Eigenvalue decomposition tolerance")
	diff_eigs = get_different_param_vals(experiment_id, 'eigs_tol_1')
	print(diff_eigs)
	
	print("CG tolerance")
	diff_cg = get_different_param_vals(experiment_id, 'cg_tol_1')
	print(diff_cg)
	
	print("Dimensions")
	diff_dims = get_different_param_vals(experiment_id, 'dimensions')
	print(diff_dims)

	print("Regularisation parameter")
	diff_lambda = get_different_param_vals(experiment_id, 'lambda')
	print(diff_lambda)
	
	return diff_eigs, diff_cg, diff_dims, diff_lambda

def get_results_obj(experiment_id):
	return pickle.load(open(get_results_dump_file_path(experiment_id), 'rb'))

def get_execution_time(experiment_id):
	res_obj = get_results_obj(experiment_id)
	return res_obj['training_outcome']['time_consumed']['decompose_M2_eigsh'] + res_obj['training_outcome']['time_consumed']['decompose_M_eigsh']

def plot_execution_time(experiment_run_ids, eigs_tol, ax = None):
	plot_data = {}
	for exp_id in experiment_run_ids:
		res_obj = get_results_obj(exp_id)

		exe_time = get_execution_time(exp_id)
		_lambda = res_obj['parameters']['lambda']

		cg_tol_1 = res_obj['parameters']['cg_tol_1']
		eigs_tol_1 = res_obj['parameters']['eigs_tol_1']

		assert(eigs_tol_1 == eigs_tol)

		if cg_tol_1 not in plot_data:
			plot_data[cg_tol_1] = []

		plot_data[cg_tol_1].append((_lambda, exe_time))
		
	if ax == None:
		fig = plt.figure(figsize=(7,7))
		ax = plt.subplot(111)

	for key in sorted(plot_data.keys()):
		plot_data[key] = sorted(plot_data[key])
		x = np.array(plot_data[key])[:, 0]
		y = np.array(plot_data[key])[:, 1]/60
		ax.plot(x, y, label = 'cg_tol = %s' % key)
		ax.scatter(x,y)

	ax.legend()
	ax.set_title('Eigs tol %s' % eigs_tol)

def get_target_concepts_per_lang(lang_codes, suffix):
	'''Returns a dictionary of target concepts for the languages in the list passed as argument, which define the retrieval space in validation.'''

	target_concepts_per_lang = {}

	for lang_code in lang_codes:
		target_concepts_per_lang[lang_code] = set(pickle.load(open(get_target_concepts_file_path(lang_code, suffix), 'rb')))

	return target_concepts_per_lang
