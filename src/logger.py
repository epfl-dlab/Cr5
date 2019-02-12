import os
import sys
import utils

class Logger:
	def __init__(self, experiment_object):
		experiments_parameter_logs_file_name = utils.get_runs_logs_path(experiment_object.experiment_identifier)
		experiment_log_file_name = utils.get_full_logs_path(experiment_object.experiment_run_identifier)


		self.experiments_parameter_logs_file_name = experiments_parameter_logs_file_name
		self.experiment_log_file_name = experiment_log_file_name
		self.experiment_run_identifier = experiment_object.experiment_run_identifier
		self.params_string = utils.params_to_string(experiment_object.params)
		self.old_output = None
		print(self.params_string)

		if os.path.isfile(experiment_log_file_name):
			print("Logs file %s already exists" % experiment_log_file_name)
			sys.exit()

	def log_run(self):
		'''Logs the run in the experiment specific file, which keeps track of all the runs (using different training parameters) for the given experiment identifier.'''
		experiments_parameter_logs_file_name = self.experiments_parameter_logs_file_name
		with open(self.experiments_parameter_logs_file_name, 'a+') as f:
			line = '{} {}'.format(self.experiment_run_identifier, self.params_string)
			f.write('%s\n' % line)

	def redirect_output(self):
		'''Redirects output to the experiment run specific file.'''
		self.old_output = sys.stdout
		sys.stdout = open(self.experiment_log_file_name, 'w')

	def revert_standard_output(self):
		'''Reverts the output to the initial one (before the redirection) if it has been redirected.'''
		if self.old_output is not None:
			sys.stdout = self.old_output