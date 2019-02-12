from logger import Logger
from experiment_run import Experiment
import argparse
import utils

parser = argparse.ArgumentParser(description='Experiment wrapper')
parser.add_argument("--exp_num", type=int, default=None, help="Experiment number")
read_params = parser.parse_args()

experiment_number = read_params.exp_num

lang_codes = ['da', 'en', 'vi']
lang_codes = sorted(lang_codes)
lang_identifier = '_'.join(lang_codes)


training_concepts_file_name = 'da_en_vi_65000_through_en_lt_100_ut_600_ml_2'

validation_set_file_name = 'da_de_en_es_vi_65000_lt_100_ut_600_ml_2_size_1000_inter_true'

# casefolded
case_folding_flag = True
experiment_identifier = '{}_short_short_casefolded'.format(lang_identifier)

if experiment_number is None:
	experiment_number = utils.get_exp_num_for_id(experiment_identifier)

lambdas = [6, 4, 2, 1.33]

eigs_tolerances_1 = [0.3, 1]
cg_tolerances_1 = [2, 3]
dimensions = [100, 200, 300]

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

				params = utils.generate_params_dict(training_concepts_file_name, validation_set_file_name, case_folding_flag, _lambda, dims, cg_tol_1, eigs_tol_1, cg_tol_2, eigs_tol_2)
				exp_obj = Experiment(experiment_identifier, experiment_number, params)
				exp_obj.run_experiment()
				experiment_number += 1