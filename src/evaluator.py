from data_class import Data
import copy
import numpy as np
import scipy.sparse
import scipy
import pickle
import pandas as pd
import itertools
import utils
import sys
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import normalize

import faiss
FAISS_AVAILABLE = True
if not hasattr(faiss, 'StandardGpuResources'):
	sys.stderr.write("Impossible to import Faiss-GPU. "
					 "Switching to FAISS-CPU, "
					 "this will be slower.\n\n")
	
def get_nn_avg_dist(emb, query, knn):
	"""
	Compute the average distance of the `knn` nearest neighbors
	for a given set of embeddings and queries.
	Use Faiss if available.
	"""
	if FAISS_AVAILABLE:
		emb = emb.cpu().numpy()
		query = query.cpu().numpy()
		if hasattr(faiss, 'StandardGpuResources'):
			# gpu mode
			res = faiss.StandardGpuResources()
			config = faiss.GpuIndexFlatConfig()
			config.device = 0
			index = faiss.GpuIndexFlatIP(res, emb.shape[1], config)
		else:
			# cpu mode
			index = faiss.IndexFlatIP(emb.shape[1])
		index.add(emb)
		distances, _ = index.search(query, knn)
		return distances.mean(1)
	else:
		bs = 1024
		all_distances = []
		emb = emb.transpose(0, 1).contiguous()
		for i in range(0, query.shape[0], bs):
			distances = query[i:i + bs].mm(emb)
			best_distances, _ = distances.topk(knn, dim=1, largest=True, sorted=True)
			all_distances.append(best_distances.mean(1).cpu())
		all_distances = torch.cat(all_distances)
		return all_distances.numpy()


class Evaluator:
	'''Class that encapsulates the evaluation framework for a single experiment run'''

	def __init__(self, results_obj_name, languages_to_evaluate, vocabulary_size, test_set_flag, target_concepts_suffix, validation_set_file_name = None):
		'''Initialized by the name of a results_obj, list of langages that you would like to evaluate. By default the validation object is retrieved from the results object, but another validation object can be passed optionally (asuming that it contains validation concepts set for each pair in languages to evaluate)'''
		data_obj = Data()

		if validation_set_file_name is None:
			data_obj.load_validation(results_obj_dump_name=results_obj_name, vocabulary_size = vocabulary_size, test_set_flag = test_set_flag, target_concepts_suffix = target_concepts_suffix)
		else:
			print('Validation_set_file_name:', validation_set_file_name)
			data_obj.load_validation(results_obj_dump_name=results_obj_name, vocabulary_size = vocabulary_size, validation_set_file_name = validation_set_file_name, test_set_flag = test_set_flag, target_concepts_suffix = target_concepts_suffix)

		self.results_obj_name = results_obj_name
		self.embedding_types = {'baseline', 'learned', 'ev_learned'}
		# self.data_obj = data_obj
		self.embeddings = data_obj.embeddings
		self.training_languages = data_obj.training_languages
		self.e_vals = data_obj.e_vals
		self.training_params = data_obj.results_dump_obj['parameters']
		self.validation_set_file_name = data_obj.validation_set_file_name

		# langugages to consider when doing evaluation
		self.lang_codes_to_evaluate = sorted(languages_to_evaluate)

		# languages for which there are embeddings available in the results object
		self.validation_lang_codes = sorted(data_obj.lang_codes)
		
		# checks whether all of the languages that we want to evaluate actually have trained embeddings
		assert(len(languages_to_evaluate) == len(set(languages_to_evaluate) & set(self.validation_lang_codes)))

		# raw data, index_2_word, index_2_concept, concept_2_index
		self.Z_per_lang = data_obj.Z_per_lang
		self.val_index_2_concept_id_per_lang = data_obj.val_index_2_concept_id_per_lang
		self.val_index_2_word_id_per_lang = data_obj.val_index_2_word_id_per_lang

		concept_id_2_index_per_lang = {}

		for lang_code in self.lang_codes_to_evaluate:
			concept_id_2_index_per_lang[lang_code] = {concept_id : idx for idx, concept_id in \
													  self.val_index_2_concept_id_per_lang[lang_code].items()}

		self.concept_id_2_index_per_lang = concept_id_2_index_per_lang

		self.emb_index_2_word_id_per_lang = data_obj.emb_index_2_word_id_per_lang
		self.emb_case_folding_flag = data_obj.emb_case_folding_flag

		# generating a word_id_2_idf dictionary per language
		idf_per_lang = data_obj.idf_per_lang
		word_id_2_idf_per_lang = {}

		for lang_code in self.lang_codes_to_evaluate:
			idf = idf_per_lang[lang_code]
			emb_index_2_word_id = self.emb_index_2_word_id_per_lang[lang_code]

			word_id_2_idf = {}
			for index, word_id in emb_index_2_word_id.items():
				word_id_2_idf[word_id] = idf[index]

			word_id_2_idf_per_lang[lang_code] = word_id_2_idf

		self.word_id_2_idf_per_lang = word_id_2_idf_per_lang
		self.validation_set = data_obj.validation_set

		self.doc_embs_per_lang = {}
		self.ranking_dfs = {}
		self.concept_id_2_concept_name = utils.get_concept_id_concept_name_mapping()


	def get_embeddings(self, use_eigenvalues = False):
		'''Assings the trained embeddings as a property to the evaluator object'''
		if use_eigenvalues:
			print("Reading embeddings (eigenvalues used)")
		else:
			print("Reading embeddings")

		embeddings = self.embeddings
		embeddings = copy.deepcopy(embeddings)

		if use_eigenvalues:
			e_vals = self.e_vals
			embeddings = np.dot(embeddings, np.diag(np.sqrt(e_vals)))
			self.ev_embs_per_lang = Data.get_embeddings_per_lang(self.training_languages, self.emb_index_2_word_id_per_lang, embeddings)
		else:
			self.embs_per_lang = Data.get_embeddings_per_lang(self.training_languages, self.emb_index_2_word_id_per_lang, embeddings)

	def get_baseline_embeddings(self):
		'''Assings the baseline embeddings as a property to the evaluator object'''

		print("Reading baseline embeddings")
		if not self.emb_case_folding_flag:
			print("WARNING! Initial validation dataset is not lowercased!")
		
		index_2_word_id_per_lang, embeddings = utils.get_unsup_embeddings(self.lang_codes_to_evaluate)
		self.bsln_emb_index_2_word_id_per_lang = index_2_word_id_per_lang
		self.bsln_embs_per_lang = embeddings

	def get_document_embeddings(self, emb_type):
		'''Generates embeddings for the documents in the validation set using the desired embeddings. Options are `baseline` embeddings, `learned` (trained embeddings from the results_obj) or 
		`ev_learned` (trained embeddings from the results_obj, which in addition to the eigenvectors, make use of the learned eigenvalues as well))'''

		assert(emb_type in self.embedding_types)

		print("Generating doc embeddings for", emb_type)

		# retrieving desired embeddings
		if emb_type == 'baseline':
			self.get_baseline_embeddings()
			embs_per_lang = self.bsln_embs_per_lang
			emb_index_2_word_id_per_lang = self.bsln_emb_index_2_word_id_per_lang
		elif emb_type == 'learned':
			self.get_embeddings(False)
			embs_per_lang = self.embs_per_lang
			emb_index_2_word_id_per_lang = self.emb_index_2_word_id_per_lang
		else:
			self.get_embeddings(True)
			embs_per_lang = self.ev_embs_per_lang
			emb_index_2_word_id_per_lang = self.emb_index_2_word_id_per_lang

		val_index_2_word_id_per_lang = self.val_index_2_word_id_per_lang
		lang_codes_to_evaluate = self.lang_codes_to_evaluate
		word_id_2_idf_per_lang = self.word_id_2_idf_per_lang
		Z_per_lang = self.Z_per_lang
		doc_embs_per_lang = {}

		# generating embeddings for each language in lang_codes_to_evaluate
		for lang_code in lang_codes_to_evaluate:
			val_index_2_word_id = val_index_2_word_id_per_lang[lang_code]
			emb_index_2_word_id = emb_index_2_word_id_per_lang[lang_code]
			
			# find the intersection between words that are used in the validation documents and ones for which embeddings are available for
			word_ids_in_both_vocabularies = set(list(val_index_2_word_id.values())) & set(list(emb_index_2_word_id.values()))
			print("%d words in the intersection between training and evaluation in %s" % (len(word_ids_in_both_vocabularies), lang_code))
			
			word_id_2_validation_index = {v:k for k,v in val_index_2_word_id.items()}
			word_id_2_emb_index = {v:k for k,v in emb_index_2_word_id.items()}
			word_id_2_idf = word_id_2_idf_per_lang[lang_code]
			
			Z_columns_order = []
			embeddings_row_order = []
			idf = []
			
			# order the embeddings in the embeddings matrix, tfidf diagonal matrix and the validation documents representaion matrix in the same way 
			word_ids_in_both_vocabularies = sorted(word_ids_in_both_vocabularies)
			
			for word_id in word_ids_in_both_vocabularies:
				Z_columns_order.append(word_id_2_validation_index[word_id])
				embeddings_row_order.append(word_id_2_emb_index[word_id])
				idf.append(word_id_2_idf[word_id])
			
			# generate embeddings
			Z = Z_per_lang[lang_code][:,Z_columns_order]
			
			# transformer = TfidfTransformer(norm = 'l2', smooth_idf=False)
			# idfed_Z = transformer.fit_transform(Z)
			
			idf = scipy.sparse.diags(idf, format='csc')
			idfed_Z = Z.dot(idf)
			# normalized_idfed_Z = normalize(idfed_Z, norm='l2', axis=1)
			normalized_idfed_Z = idfed_Z
			
			ordered_embs = embs_per_lang[lang_code][embeddings_row_order]
			
			doc_embs = normalized_idfed_Z.dot(ordered_embs)
			doc_embs = normalize(doc_embs, norm='l2', axis=1)
			
			doc_embs_per_lang[lang_code] = doc_embs

		self.doc_embs_per_lang[emb_type] = doc_embs_per_lang

	def perform_ranking(self, doc_embs_per_lang, lang_pair, metric = 'cosine', k = 10):
		'''The method that performs the evaluation. The passed arguments are the document embeddings per lang, and a list (or any iterable) of a language pair codes. The retrieval task is performed for all the concepts in the validation set, with the rank of the target document 
		(based on the cosine similarity between document embeddings), as well as the concept names of best ranked k concepts (passed as an optional) being recorded. NOT IMPLEMENTED: The metric name is passed as an argument, so that different similarity measures can be implemented in the future'''

		concept_id_2_concept_name = self.concept_id_2_concept_name
		concept_id_2_index_per_lang = self.concept_id_2_index_per_lang
		val_index_2_concept_id_per_lang = self.val_index_2_concept_id_per_lang

		dir_1 = sorted(lang_pair)
		pair_id = '_'.join(lang_pair)
		val_concepts = sorted(self.validation_set[pair_id])
		val_concepts_names = [concept_id_2_concept_name[concept_id] for concept_id in val_concepts]

		dir_2 = dir_1[::-1]

		# Get the indices of the matching documents in both of the languages for all of the concepts in the validation set
		val_concepts_idx_in_lang = {}

		for lang_code in lang_pair:
			val_concepts_idx_in_lang[lang_code] = []
			
		for val_concept in sorted(val_concepts):
			for lang_code in lang_pair:
				val_concepts_idx_in_lang[lang_code].append(concept_id_2_index_per_lang[lang_code][val_concept])
				
		# Get embeddigs for validation concepts
		val_concept_embs_per_lang = {}
		for lang_code in lang_pair:
			val_concept_embs_per_lang[lang_code] = doc_embs_per_lang[lang_code][val_concepts_idx_in_lang[lang_code]]

		# generate ranking dataframes
		dfs = []
		for _dir in [dir_1, dir_2]:
			src_lang, targ_lang = _dir

			if metric.startswith('csls_knn_'):
				print("Ranking with", metric)
				queries = torch.from_numpy(val_concept_embs_per_lang[src_lang]).float()
				keys = torch.from_numpy(doc_embs_per_lang[targ_lang]).float()

				knn = metric[len('csls_knn_'):]
				assert knn.isdigit()
				knn = int(knn)
				# average distances to k nearest neighbors
				knn = metric[len('csls_knn_'):]
				assert knn.isdigit()
				knn = int(knn)
				average_dist_keys = torch.from_numpy(get_nn_avg_dist(queries, keys, knn))
				average_dist_queries = torch.from_numpy(get_nn_avg_dist(keys, queries, knn))
				# scores
				scores = keys.mm(queries.transpose(0, 1)).transpose(0, 1)
				scores.mul_(2)
				scores.sub_(average_dist_queries[:, None].float() + average_dist_keys[None, :].float())
				scores = scores.cpu()
				distance_matrix = scores.numpy() * -1
			else:
				print("Ranking with", metric)
				# find the similarity/distance between the validation document embeddings in the source language, and all of the target documents in the target language
				distance_matrix = scipy.spatial.distance.cdist(val_concept_embs_per_lang[src_lang], doc_embs_per_lang[targ_lang], metric = metric)

			rankings = np.argsort(distance_matrix)

			# find the ranking of the documents in the target language that correspond to the concept described in each of the queries
			target_rank = [np.where(rankings[i] == val_concepts_idx_in_lang[targ_lang][i])[0][0] for i in range(len(rankings))]
			target_rank = np.array(target_rank) + 1

			# get the top k ranked concepts in the target language for each query
			top_k_concept_names = []
			for i in range(len(rankings)):
				temp_top_k = rankings[i][:k]
				temp_top_k_concept_names = [concept_id_2_concept_name[val_index_2_concept_id_per_lang[targ_lang][concept_id]] for concept_id in temp_top_k]
				top_k_concept_names.append(temp_top_k_concept_names)

			# construct the rankings dataframe
			dir_id = np.full_like(target_rank, '_'.join(_dir), dtype=np.object)
			df = pd.DataFrame.from_dict({'concept_id' : val_concepts, 'concept_name' : val_concepts_names, 'dir' : dir_id, 'target_rank' : target_rank, 'top_%d' % k : top_k_concept_names})

			dfs.append(df)

		# concatenate the two dataframes to get the concept ranking in both directions easily accesible
		full_df = pd.concat(dfs)
		full_df = full_df.sort_values(['concept_id', 'dir'])
		full_df = full_df.set_index(['concept_id', 'dir'], drop=True)

		return full_df


	def get_ranked_concepts(self, lang_pair, emb_type = 'learned', metric = 'cosine'):
		'''Given a language pair list (or any interable), embeddings type (baseline, learned or ev_learned explanation on get_embeddings function docstring) and a similarity metric
		returns a dataframe that summarizes the result from the cross-language retrieval task'''
		assert(emb_type in self.embedding_types)
		
		lang_pair = sorted(lang_pair)
		pair_id = '_'.join(lang_pair)

		# Check whether the evaluation for the given language pair, embeddings and metric has been already performed
		ranking_dfs = self.ranking_dfs

		if emb_type not in ranking_dfs:
			self.ranking_dfs[emb_type] = {}
			ranking_dfs = self.ranking_dfs

		if metric not in ranking_dfs[emb_type]:
			self.ranking_dfs[emb_type][metric] = {}
			ranking_dfs = self.ranking_dfs

		# return the dataframe if it has been already generated, otherwise generate it
		ranking_dfs = ranking_dfs[emb_type][metric]
		if pair_id in ranking_dfs:
			return ranking_dfs[pair_id]

		print('Getting ranked concepts for ', lang_pair)

		if emb_type not in self.doc_embs_per_lang:
			self.get_document_embeddings(emb_type = emb_type)

		doc_embs_per_lang = self.doc_embs_per_lang[emb_type]
		ranking_df = self.perform_ranking(doc_embs_per_lang, lang_pair, metric = metric)

		self.ranking_dfs[emb_type][metric][pair_id] = ranking_df

		return ranking_df

	def get_ranked_concepts_in_one_dir(self, pair_id, emb_type = 'learned', metric = 'cosine'):
		'''Get ranked concepts in only one direction instead of both. Ex. pair_id is da_vi if we want to have only the retireval from Danish to Vietnamese'''

		assert(emb_type in self.embedding_types)
		
		lang_pair = pair_id.split('_')
		ranking_df = self.get_ranked_concepts(lang_pair, emb_type, metric)

		return utils.filter_ranking_df_by_direction(ranking_df, pair_id)


	def get_MRR(self, pair_id, emb_type = 'learned', metric = 'cosine', inverted = False):
		'''Returns the mean reciprocal rank induced from the cross-language retrieval task for the given combination of parameters.
		Returns the harmonic mean of ranks (1 / MRR) instead of MRR if inverted flag is set to True'''

		lang_pairs = pair_id.split('_')
		ranking_df = self.get_ranked_concepts(lang_pairs, emb_type, metric)

		ranking_df = ranking_df.query('dir == "{}"'.format(pair_id))
		mrr = np.mean(1 / ranking_df['target_rank'])

		if inverted:
			return 1 / mrr

		return mrr

	def get_average_MRR(self, lang_pair, emb_type = 'learned', metric = 'cosine', inverted = False):
		'''Returns the average of the mean reciprocal rank in both direction of the language pair, 
		induced from the cross-language retrieval task for the given combination of parameters.
		Returns the average harmonic mean of ranks (1 / MRR) instead of MRR if inverted flag is set to True'''

		dir_1 = sorted(lang_pair)
		dir_1_id = '_'.join(dir_1)

		dir_2 = dir_1[::-1]
		dir_2_id = '_'.join(dir_2)
		
		pair_ids = [dir_1_id, dir_2_id]
		
		result = 0
		for pair_id in pair_ids:
			result += self.get_MRR(pair_id, emb_type, metric, inverted)
		
		return result/2

	def plot_ranking_count_for_pair_id(self, pair_id, color, ax = None, emb_type = 'learned', metric = 'cosine', k = 70):
		'''Visualize the relationship between the rank, and the query frequency with the given rank'''
		if ax == None:
			fig = plt.figure(figsize=(7,7))
			ax = plt.subplot(111)
			
		rankings_df = self.get_ranked_concepts_in_one_dir(pair_id, emb_type = emb_type, metric = metric)
		rank_count_pairs = sorted(rankings_df.target_rank.value_counts().items(), key = lambda x: x[0])
		rank_count_pairs = rank_count_pairs[:k]
		
		counts = [i[1] for i in rank_count_pairs]
		rankings = [i[0] for i in rank_count_pairs]
		ax.plot(rankings, counts, label = '{}, HMR: {:.2f}'.format(pair_id, self.get_MRR(pair_id, emb_type, inverted = True)), color = color)

	def plot_ranking_count(self, lang_pair, emb_type):
		colors = itertools.cycle(['navy', 'navy', 'orange', 'orange'])
		dir_1 = sorted(lang_pair)
		dir_1_id = '_'.join(dir_1)

		dir_2 = dir_1[::-1]
		dir_2_id = '_'.join(dir_2)
		
		pair_ids = [dir_1_id, dir_2_id]
		
		r = 1
		c = 2
		fig, axes = plt.subplots(1,2, figsize=(20,7), sharey=True)
		fig.suptitle("{} - {}".format(lang_pair[0].capitalize(), lang_pair[1].capitalize()))
		titles = [emb_type, 'baseline']
		
		for i, ax in enumerate(axes):
			ax.set_ylabel(str('Query rank frequency'))
			ax.set_xlabel(str('Query rank'))
			ax.set_xscale('log')
			ax.set_yscale('log')
			ax.set_title('{} embedding (Average MRR: {:.2f}, Average HMR {:.2f})'.format(\
				   titles[i].capitalize(), self.get_average_MRR(lang_pair, titles[i]), self.get_average_MRR(lang_pair, titles[i], inverted = True)))

		for pair_id in pair_ids:
			self.plot_ranking_count_for_pair_id(pair_id, next(colors), axes[0])
			self.plot_ranking_count_for_pair_id(pair_id, next(colors), axes[1], emb_type = 'baseline')

		axes[0].legend()
		axes[1].legend()
	
	def plot_ranking_count_cdf(self, lang_pair, emb_type, x_logscale = False, path_to_dump = None, metric_learned = 'cosine', metric_baseline = 'cosine', ll_x=0.05, hl_x=2.1):
		colors = itertools.cycle(['navy', 'navy', 'orange', 'orange'])
		dir_1 = sorted(lang_pair)
		dir_1_id = '_'.join(dir_1)

		dir_2 = dir_1[::-1]
		dir_2_id = '_'.join(dir_2)
		
		pair_ids = [dir_1_id, dir_2_id]
		
		r = 1
		c = 2
		fig, axes = plt.subplots(1,2, figsize=(20,7), sharey=True)
		fig.suptitle("{} - {} CDF plot".format(lang_pair[0].capitalize(), lang_pair[1].capitalize()))
		titles = [emb_type, 'baseline']
		metrics = [metric_learned, metric_baseline]
		for i, ax in enumerate(axes):
			ax.set_ylabel(str('Recall'))
			ax.set_xlabel(str('Query rank'))
			if x_logscale:
				ax.set_xscale('log')
				ax.set_xlim(10**-ll_x, 10**hl_x)
			# else:
				# ax.set_xlim(-2, 100)

			ax.set_title('{} embedding (MRR: {:.2f}, HMR {:.2f})'.format(\
				   titles[i].capitalize(), self.get_average_MRR(lang_pair, titles[i], metric = metrics[i]), self.get_average_MRR(lang_pair, titles[i], inverted = True, metric = metrics[i])))
			
		for pair_id in pair_ids:
			self.plot_ranking_count_cdf_for_pair_id(pair_id, next(colors), axes[0], emb_type = emb_type, metric = metric_learned)
			self.plot_ranking_count_cdf_for_pair_id(pair_id, next(colors), axes[1], emb_type = 'baseline', metric = metric_baseline)

		axes[0].legend()
		axes[1].legend()

		if path_to_dump is not None:
			plt.savefig(path_to_dump, bbox_inches='tight')
		
	def plot_ranking_count_cdf_for_pair_id(self, pair_id, color, ax = None, emb_type = 'learned', metric = 'cosine', x_logscale = False, k = 70):
		'''Visualize the relationship between the rank, and the query frequency with the given rank as the percentage of queries that have rank lower or equal than the given number'''
		
		if ax == None:
			fig = plt.figure(figsize=(7,7))
			ax = plt.subplot(111)
			ax.set_ylabel(str('Recall'))
			ax.set_xlabel(str('query ranking'))
			if x_logscale:
				ax.set_xscale('log')
			
		rankings_df = self.get_ranked_concepts_in_one_dir(pair_id, emb_type = emb_type, metric = metric)
		rank_count_pairs = sorted(rankings_df.target_rank.value_counts().items(), key = lambda x: x[0])
		total_num_of_docs = rankings_df.shape[0]
		
		counts = [i[1] for i in rank_count_pairs]
		recall = np.cumsum(counts) / total_num_of_docs * 100
		rankings = [i[0] for i in rank_count_pairs]
		ax.plot(rankings, recall, label = '{}, HMR: {:.2f}'.format(pair_id, self.get_MRR(pair_id, emb_type, inverted = True, metric = metric), color = color))

	def get_precision_per_lang_pair(self, lang_pair, emb_type = 'learned', metric = 'cosine'):
		'''Returns the summary of the cross language retrieval task in terms of precision i.e how many valdidation concepts have their matching
		document ranked first, how many have it in the first 5 target documents, and how many have it in the first 10 target documents'''

		ranking_df = self.get_ranked_concepts(lang_pair, emb_type, metric)

		dir_1 = sorted(lang_pair)
		dir_1_id = '_'.join(dir_1)

		dir_2 = dir_1[::-1]
		dir_2_id = '_'.join(dir_2)

		dir_ids = [dir_1_id, dir_2_id]

		pair, p_1, p_5, p_10 = [], [], [], []

		for _id in dir_ids:
			dir_df = utils.filter_ranking_df_by_direction(ranking_df, _id)
			pair.append(_id)

			p_score = np.sum(dir_df['target_rank'] <= 1)
			p_1.append(float(p_score) * 100 / dir_df['target_rank'].shape[0])

			p_score = np.sum(dir_df['target_rank'] <= 5)
			p_5.append(float(p_score) * 100 / dir_df['target_rank'].shape[0])

			p_score = np.sum(dir_df['target_rank'] <= 10)
			p_10.append(float(p_score) * 100 / dir_df['target_rank'].shape[0])

		df = pd.DataFrame.from_dict({'pair' : pair, 'P@1' : p_1, 'P@5' : p_5, 'P@10' : p_10})
		df = df[['pair', 'P@1', 'P@5', 'P@10']]
		df = df.set_index('pair', drop=True)

		return df.round(2)

	def get_overall_precision(self, emb_type = 'learned', metric = 'cosine'):
		'''Returns a dataframe that summarizes precision for every possible pair in languages to evaluate.'''
		print("Embeddings type: ", emb_type)
		dfs = []

		all_pairs = list(itertools.combinations(self.lang_codes_to_evaluate, 2))
		for lang_pair in all_pairs:
			df = self.get_precision_per_lang_pair(lang_pair, emb_type, metric)
			dfs.append(df)

		return pd.concat(dfs)

	def print_important_numbers(self):
		results_dump_obj = pickle.load(open(utils.get_results_dump_file_path(self.results_obj_name), 'rb'))

		training_dataset_name = results_dump_obj['data']
		training_dataset_dump_obj = pickle.load(open(utils.get_training_dataset_file_path(training_dataset_name), 'rb'))

		training_Z_per_lang = training_dataset_dump_obj['Z_per_lang']
		
		print("*** Training phase ***")
		target_docs_num = dict([(lang_code, temp_Z.shape[0]) for lang_code, temp_Z in training_Z_per_lang.items()])
		print("# Training documents:", target_docs_num)

		validation_docs_voc_size = dict([(lang_code, temp_Z.shape[1]) for lang_code, temp_Z in training_Z_per_lang.items()])
		print("# Training sets vocabulary size:", validation_docs_voc_size)

		print("")
		
		print("*** Evaluation phase ***")
		n_queries = dict([(key, len(set(val_set))) for key, val_set in self.validation_set.items() if len(val_set) != 0])
		print("# Queries:", n_queries)

		target_docs_num = dict([(lang_code, temp_Z.shape[0]) for lang_code, temp_Z in self.Z_per_lang.items()])
		print("# Target documents:", target_docs_num)

		validation_docs_voc_size = dict([(lang_code, temp_Z.shape[1]) for lang_code, temp_Z in self.Z_per_lang.items()])
		print("# Evaluation datasets vocabulary size:", validation_docs_voc_size)

class Multirun_evaluator:
	'''Class that encapsulates the evaluation framework for all experiment runs (different parameter combinations) 
	for a given experiment identifier (same dataset)'''

	def __init__(self, experiment_run_ids, lang_codes_to_evaluate, vocabulary_size, test_set_flag, target_concepts_suffix, validation_set_file_name = None):
		'''Initialized by an iterable of the desired experiment run ids, and an iterable of languages that need to be evaluated'''

		self.experiment_run_ids = experiment_run_ids
		self.lang_codes_to_evaluate = lang_codes_to_evaluate
		self.performance = {}

		evaluators = []
		for experiment_run_id in experiment_run_ids:
			if validation_set_file_name == None:
				eval_obj = Evaluator(experiment_run_id, lang_codes_to_evaluate, vocabulary_size = vocabulary_size, test_set_flag = test_set_flag, target_concepts_suffix = target_concepts_suffix)
			else:
				eval_obj = Evaluator(experiment_run_id, lang_codes_to_evaluate, vocabulary_size = vocabulary_size, validation_set_file_name = validation_set_file_name, test_set_flag = test_set_flag, target_concepts_suffix = target_concepts_suffix)
			evaluators.append(eval_obj)

		self.evaluators = evaluators

	def get_performance_for_lang_pair(self, lang_pair, emb_type = 'learned', metric = 'cosine'):
		'''Returns a list of dictionaries that summarize the retrieval task for the passed language pair and parameters'''

		evaluators = self.evaluators
		performance = self.performance

		if emb_type not in performance:
			self.performance[emb_type] = {}
			performance = self.performance

		if metric not in performance[emb_type]:
			self.performance[emb_type][metric] = {}
			performance = self.performance

		# return the performance obj if it has been already generated, otherwise generate it
		pair_id = '_'.join(sorted(lang_pair))
		performance = performance[emb_type][metric]
		if pair_id in performance:
			return performance[pair_id]

		curr_perf = []
		for eval_obj in evaluators:
			dir_1 = sorted(lang_pair)
			dir_1_id = '_'.join(dir_1)
			mrr_dir_1 = eval_obj.get_MRR(dir_1_id, emb_type = emb_type, metric = metric)

			dir_2 = dir_1[::-1]
			dir_2_id = '_'.join(dir_2)
			mrr_dir_2 = eval_obj.get_MRR(dir_2_id)

			curr_perf.append({'dir_1' : dir_1_id, 'dir_2' : dir_2_id, 'mrr_dir_1' : mrr_dir_1,\
								'mrr_dir_2' : mrr_dir_2, 'params' : eval_obj.training_params})

		self.performance[emb_type][metric][pair_id] = curr_perf
		
		return curr_perf

	def plot_performance(self, lang_pair, _dir, title, ax = None, emb_type = 'learned', metric = 'cosine'):
		'''Plots the harmonic mean of ranks for the retrieval in _dir direction on the y axis, 
		regularization paramter on x axis, and the cg tolrances as different lines on the given figure. 
		Assumes that all the parameters except the cg_tolerance and the regularization paremter are the same in the self object (achived through filtering).'''

		# Sanitiy check
		dir_1 = sorted(lang_pair)
		dir_1_id = '_'.join(dir_1)

		dir_2 = dir_1[::-1]
		dir_2_id = '_'.join(dir_2)

		dir_ids = [dir_1_id, dir_2_id]

		assert(_dir == 'avg' or _dir in dir_ids)

		performance = self.get_performance_for_lang_pair(lang_pair, emb_type, metric)
		plot_data = self.generate_plot_data_from_perf_obj(performance, _dir)

		if ax == None:
			fig = plt.figure(figsize=(7,7))
			ax = plt.subplot(111)
			
		for key in plot_data.keys():
			x = np.array(plot_data[key])[:, 0]
			y = np.array(plot_data[key])[:, 1]
			ax.plot(x, y, label = key.split(',')[0])
			ax.scatter(x,y)

		ax.legend()
		ax.set_title("{}, direction {}, emb_type {}".format(title, _dir, emb_type))

	def generate_plot_data_from_perf_obj(self, performance, _dir):
		'''Parses a performance object from the multirun evaluator, and returns a dictionary 
		that can be easily used to plot the data'''

		data = {}

		for perf_obj in performance:
			# Read parameters
			_lambda = perf_obj['params']['lambda']
			cg_tol_1 = perf_obj['params']['cg_tol_1']
			eigs_tol_1 = perf_obj['params']['eigs_tol_1']

			# Get the performance score according to the _dir parameter (langA_lang_B, langB_langA, or avg)
			if _dir == 'avg':
				score = (1.0/perf_obj['mrr_dir_1'] + 1.0/perf_obj['mrr_dir_2'])/2
			else:
				if perf_obj['dir_1'] == _dir:
					score = 1.0/perf_obj['mrr_dir_1']   
				elif perf_obj['dir_2'] == _dir:
					score = 1.0/perf_obj['mrr_dir_2']
				else:
					raise Exception('Retrieval direction not available')

			label = 'cg_tol = {0:.2f}, eigs_tol = {1:.2f}'.format(cg_tol_1, eigs_tol_1)
			if label not in data:
				data[label] = []
			data[label].append([_lambda, score])

		for key in data.keys():
			data[key] = sorted(data[key], key = lambda x: x[0])
			
		return data

	# def plot_performance(self, title, emb_type = 'learned', metric = 'cosine'):
	# 	plot_data = utils.generate_plot_data_from_perf_obj(performance, _dir)

	# 	if ax == None:
	# 		fig = plt.figure(figsize=(7,7))
	# 		ax = plt.subplot(111)
			
	# 	for key in data.keys():
	# 		x = np.array(data[key])[:, 0]
	# 		y = np.array(data[key])[:, 1]
	# 		ax.plot(x, y, label = key.split(',')[0])
	# 		ax.scatter(x,y)
	# 		ax.legend()
		
	# 	ax.set_title(title)





