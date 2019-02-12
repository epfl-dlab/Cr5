import utils
import pickle
import os

from logger import Logger
from data_class import Data
from operations_wrapper import Operations
from scipy.sparse.linalg import eigsh, ArpackError
from timeit import default_timer

import sys


class Experiment:
	def __init__(self, experiment_identifier, experiment_number, params):
		self.experiment_identifier = experiment_identifier
		self.experiment_number = experiment_number
		self.params = params
		self.experiment_run_identifier = "{}_{}".format(experiment_identifier, experiment_number)
		self.logger = Logger(self)

		self.results_dump_path = utils.get_results_dump_file_path(self.experiment_run_identifier)

		if os.path.isfile(self.results_dump_path):
			print("Results dump %s already exists" % self.results_dump_path)
			sys.exit()

		self.logger.redirect_output()

	def run_experiment(self):
		'''Compleates an actual run of the full pipeline, with the parameters corresponding to the arguments passed in the constructor'''

		params = self.params
		print(params)

		cg_max_iter = params['cg_max_iter'] # Default value 500
		eigs_max_iter = params['eigs_max_iter'] # Default value 250

		training_concepts_file_name = params['training_concepts_file_name']
		validation_set_file_name = params['validation_set_file_name']
		case_folding_flag = params['case_folding_flag']

		_lambda = params['lambda']

		cg_tol_1 = 10**(-1 * params['cg_tol_1'])
		eigs_tol_1 = 10**(-1 * params['eigs_tol_1'])

		# Same for now
		cg_tol_2 = 10**(-1 * params['cg_tol_2'])
		eigs_tol_2 = 10**(-1 * params['eigs_tol_2'])

		dims = params['dimensions']
		vocabulary_size = params['vocabulary_size']

		data_obj = Data()
		data_obj.load_training(training_concepts_file_name, validation_set_file_name, case_folding_flag, vocabulary_size)

		operations_obj = Operations(data_obj, _lambda, cg_max_iter, cg_tol_1)

		start = default_timer()

		try:
			vals, vecs = operations_obj.decompose_M_eigsh(dims, eigs_max_iter, eigs_tol_1)
		except ArpackError as e:
			try:
				print("ERROR occured!")
				print(e)
				vals, vecs = operations_obj.decompose_M_eigsh(dims, eigs_max_iter, eigs_tol_1, True)
			except ArpackError as e:
				print("FAIL! Can't complete the decomposition!")
				return

		end = default_timer()
		time_elapsed = end - start
		print("Finished decomposition one: ", time_elapsed)

		training_outcome = {}

		training_outcome['e_vals'] = vals
		training_outcome['e_vecs'] = vecs

		results_obj = {}
		results_obj['training_outcome'] = training_outcome
		results_obj['parameters'] = params
		results_obj['data'] = data_obj.final_dataset_dump_name

		with open(self.results_dump_path, 'wb') as f:
			pickle.dump(results_obj, f, protocol = 4)

		start = default_timer()

		try:
			vals_m2, vecs_m2 = operations_obj.decompose_M2_eigsh(dims, eigs_max_iter, eigs_tol_1)
			print(vals_m2) # Visual sanity check
		except ArpackError as e:
			try:
				print("ERROR occured!")
				print(e)
				vals_m2, vecs_m2 = operations_obj.decompose_M2_eigsh(dims, eigs_max_iter, eigs_tol_1, True)
			except ArpackError as e:
				print("FAIL! Can't complete the decomposition!")
				return

		end = default_timer()
		time_elapsed = end - start
		print("Finished decomposition two: ", time_elapsed)

		training_outcome['M2_e_vals'] = vals_m2
		training_outcome['M2_e_vecs'] = vecs_m2

		# training_outcome['cg_residuals'] = operations_obj.cg_residuals
		training_outcome['num_iter'] = operations_obj.num_iter
		# training_outcome['cg_residuals2'] = operations_obj.cg_residuals2
		training_outcome['num_iter2'] = operations_obj.num_iter2
		training_outcome['time_consumed'] = operations_obj.time_consumed

		results_obj['training_outcome'] = training_outcome

		with open(self.results_dump_path, 'wb') as f:
			pickle.dump(results_obj, f, protocol = 4)

		self.logger.revert_standard_output()
		self.logger.log_run()