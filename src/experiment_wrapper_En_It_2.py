
from experiment_run import Experiment
import argparse
import utils

parser = argparse.ArgumentParser(description='Experiment wrapper')
parser.add_argument("--voc_size", type=int, default=None, help="Size of considered vocabulary")
parser.add_argument("--exp_num", type=int, default=None, help="Experiment number")
read_params = parser.parse_args()

experiment_number = read_params.exp_num
assert(read_params.voc_size is not None)
voc_size = read_params.voc_size

# lang_codes = ['da', 'vi', 'en', 'it']
# lang_codes = ['da', 'vi']
lang_codes = ['en', 'it']
lang_codes = sorted(lang_codes)
lang_identifier = '_'.join(lang_codes)

# short documents used in training
# training_concepts_file_name = 'da_de_en_es_vi_65000_lt_100_ut_600_ml_2'
# training_concepts_file_name = 'da_vi_65000_lt_100_ut_600_ml_2'
training_concepts_file_name = 'en_it_200000_lt_100_ut_1000_ml_2'

# short documents used in validation
validation_set_file_name = 'da_en_it_vi_200000_lt_100_ut_700_ml_2_size_2000_inter_false'

# casefolded
case_folding_flag = True
experiment_identifier = '{}_200000_short_casefolded'.format(lang_identifier)

if experiment_number is None:
	experiment_number = utils.get_exp_num_for_id(experiment_identifier)

lambdas = [2]

eigs_tolerances_1 = [1]

cg_tolerances_1 = [2]

dimensions = [300]

for eigs_tol_1 in eigs_tolerances_1:
	for cg_tol_1 in cg_tolerances_1:
		for _lambda in lambdas:
			for dims in dimensions:
				# Test whether the experiment has been done before
				search_res = utils.get_run_identifier(experiment_identifier, dims, _lambda, cg_tol_1, eigs_tol_1, return_line = True)
				if search_res is not None:
					print("SKIPPED! Experiment found as ->", search_res[1][0])
					continue

				cg_tol_2 = cg_tol_1
				eigs_tol_2 = eigs_tol_1

				params = utils.generate_params_dict(training_concepts_file_name, validation_set_file_name, case_folding_flag, _lambda, dims, cg_tol_1, eigs_tol_1, cg_tol_2, eigs_tol_2, voc_size)
				exp_obj = Experiment(experiment_identifier, experiment_number, params)
				exp_obj.run_experiment()
				experiment_number = utils.get_exp_num_for_id(experiment_identifier)