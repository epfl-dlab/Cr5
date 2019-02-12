import unittest
import numpy as np
import gzip
import codecs
import itertools
import scipy.sparse as sps
from old_data_class import Data
from operations_wrapper import Operations
import pickle
import numpy as np
import scipy.sparse as sparse
from ctypes import POINTER,c_void_p,c_int,c_char,c_double,byref,cdll
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize

class TestClass(unittest.TestCase):
	'''Unittesting module for the functions implemented in the Operations class'''

	# Functions for generating random "faithfull" data (data that has the same structure as in our setting)
	def generate_v(self, size, seed = 123):
		"""Generate a random vector v"""

		np.random.seed(seed)

		elems = size//4

		normal = np.random.randn(elems) 
		zeros = np.zeros(elems)
		uniform = np.random.rand(elems)
		uniform2 = np.random.rand(size - 3*elems)*100

		v = np.hstack([normal, zeros, uniform, uniform2])
		sign_flip = np.random.choice(range(size), size=size//10, replace=False)
		v[sign_flip] = - v[sign_flip]

		np.random.shuffle(v)
		return v.reshape(size,1)

	def generate_Y(self, num_documents, num_concepts, seed = 12345):
		"""Generate a random matrix according to the structure that Y has in our setting (sparse matrix having only one element 1 in a row)"""

		np.random.seed(seed)

		concepts = list(range(num_concepts))
		concepts = concepts + list(np.random.randint(0, num_concepts, num_documents-num_concepts))
		np.random.shuffle(concepts)

		rows = range(len(concepts))
		data = np.ones_like(concepts)

		Y = sps.coo_matrix((data, (rows, concepts)), dtype=np.int32, shape=(num_documents, num_concepts))
		Y = Y.asformat('csr')
		return Y

	def generate_elements_for_lang(self, num_docs, voc_size, density):
		"""Generate random documents and representing them in a bag of words format"""

		elements = max(int(voc_size * num_docs * density), voc_size, num_docs)

		columns = np.array(range(voc_size))
		rem_columns = np.array(list(np.random.randint(0, voc_size, elements-voc_size)))
		columns = np.hstack((columns, rem_columns))
		np.random.shuffle(columns)

		rows = np.array(range(num_docs))
		rem_rows = np.array(list(np.random.randint(0, num_docs, elements-num_docs)))
		rows = np.hstack((rows, rem_rows))
		np.random.shuffle(rows)

		data = np.random.randint(1, 30, elements)

		return rows, columns, data

	def generate_Z(self, num_docs_per_lang, voc_size_per_lang, density, seed = 1358):
		""""Generate a random matrix according to the structure that Z has in our setting (blockdiagonal)"""

		np.random.seed(seed)
		document_offset = 0
		vocabulary_offset = 0

		rows, columns, data = np.array([]), np.array([]), np.array([])
		for num_docs, voc_size in zip(num_docs_per_lang, voc_size_per_lang):
			temp_rows, temp_cols, temp_data = self.generate_elements_for_lang(num_docs, voc_size, density)
			temp_rows = temp_rows + document_offset
			temp_cols = temp_cols + vocabulary_offset

			document_offset = document_offset + num_docs
			vocabulary_offset = vocabulary_offset + voc_size

			rows = np.hstack((rows, temp_rows))
			columns = np.hstack((columns, temp_cols))
			data = np.hstack((data, temp_data))

		Z = sps.coo_matrix((data, (rows, columns)), dtype=np.float64, shape=(document_offset, vocabulary_offset))
		Z_T_COO = sps.coo_matrix.transpose(Z)
		Z = Z.asformat('csr')
		Z_T = Z_T_COO.asformat('csr')

		return (Z, Z_T)

	def generate_data_obj(self, num_concepts, num_docs_per_lang, voc_size_per_lang, density=0.25, seed_z = 1358, seed_y = 12345):
		"""Generate a random object from the Data class"""

		num_documents = np.sum(num_docs_per_lang)
		Z, Z_T = self.generate_Z(num_docs_per_lang, voc_size_per_lang, density, seed_z)
		Y = self.generate_Y(num_documents, num_concepts, seed_y)

		data_obj = Data()
		data_obj.assign_Y(Y)
		data_obj.assign_Z(Z)
		data_obj.Z_T = Z_T

		return data_obj

	# Testing functions
	def tes_multiply_by_Y_left(self):
		"""Tests the functionality of multiplying a vector with Y from the left"""

		num_documents_per_language_list = [[8], [8], [100], [123]] # controls Y.shape[0]
		num_concepts_list = [5, 8, 35, 62] # controls Y.shape[1]

		seeds_y = [12345, 1111, 222222] # generates different Y's
		seeds_v = [123, 1111, 22222, 333333] # generates different v's
		for i in range(len(num_concepts_list)):
			for seed_y in seeds_y:
				num_docs_per_lang = voc_size_per_lang = num_documents_per_language_list[i]
				num_concepts = num_concepts_list[i]

				data_obj = self.generate_data_obj(num_concepts, num_docs_per_lang, voc_size_per_lang, seed_y = seed_y)
				operations_obj = Operations(data_obj)
				Y_dense = data_obj.Y.todense()

				for seed_v in seeds_v:
					v = self.generate_v(num_concepts, seed_v)

					dense_multiply = Y_dense @ v

					custom_multiply = operations_obj.multiply_by_Y_left(v)
					self.assertTrue(np.allclose(custom_multiply, dense_multiply))

	def tes_multiply_by_Y_T_left(self):
		"""Tests the functionality of multiplying a vector with Y transpose from the left"""

		num_documents_per_language_list = [[8], [8], [100], [123]] # controls Y.shape[0]
		num_concepts_list = [5, 8, 35, 62] # controls Y.shape[1]

		seeds_y = [12345, 1111, 222222] # generates different Y's
		seeds_v = [1233, 111111, 22, 22222] # generates different v's
		for i in range(len(num_concepts_list)):
			num_docs_per_lang = voc_size_per_lang = num_documents_per_language_list[i]
			num_concepts = num_concepts_list[i]
			total_num_of_docs = np.sum(num_docs_per_lang)
			for seed_y in seeds_y:
				data_obj = self.generate_data_obj(num_concepts, num_docs_per_lang, voc_size_per_lang, seed_y = seed_y)
				operations_obj = Operations(data_obj)
				Y_dense = data_obj.Y.todense()
				Y_dense_T = Y_dense.T

				for seed_v in seeds_v:
					v = self.generate_v(total_num_of_docs, seed_v)

					dense_multiply = Y_dense_T @ v

					custom_multiply = operations_obj.multiply_by_Y_T_left(v)
					self.assertTrue(np.allclose(custom_multiply, dense_multiply))

	def tes_multiply_by_const_left(self):
		"""Tests the functionality of multiplying a vector with the constant factor from the left"""

		num_documents_per_language_list = [[8], [8], [100], [123]] # controls the value of n in the constant factor
		seeds_v = [123, 1111, 22222, 333333] # generates different v's
		seed_y = 123
		for i in range(len(num_documents_per_language_list)):
			num_docs_per_lang = voc_size_per_lang = num_documents_per_language_list[i]
			num_concepts = num_docs_per_lang[0]

			data_obj = self.generate_data_obj(num_concepts, num_docs_per_lang, voc_size_per_lang, seed_y = seed_y)
			operations_obj = Operations(data_obj)

			n = data_obj.Z.shape[0]
			const_dense = np.eye(n) - 1/n

			for seed_v in seeds_v:
				v = self.generate_v(num_concepts, seed_v)
				dense_multiply = const_dense @ v
				custom_multiply = operations_obj.multiply_by_const_left(v)
				cython_multiply = operations_obj.multiply_by_const_left_cython(v)
				self.assertTrue(np.allclose(custom_multiply, dense_multiply))
				self.assertTrue(np.allclose(dense_multiply, cython_multiply.reshape(v.shape[0], 1)))
	# ####

	# def SpMV_viaMKL( self, A, x ):
	# 	mkl = cdll.LoadLibrary("libmkl_rt.so")

	# 	SpMV = mkl.mkl_cspblas_dcsrgemv
	# 	# Dissecting the "cspblas_dcsrgemv" name:
	# 	# "c" - for "c-blas" like interface (as opposed to fortran)
	# 	#    Also means expects sparse arrays to use 0-based indexing, which python does
	# 	# "sp"  for sparse
	# 	# "d"   for double-precision
	# 	# "csr" for compressed row format
	# 	# "ge"  for "general", e.g., the matrix has no special structure such as symmetry
	# 	# "mv"  for "matrix-vector" multiply

	# 	if not sparse.isspmatrix_csr(A):
	# 		raise Exception("Matrix must be in csr format")
	        
	# 	(m,n) = A.shape

	# 	# The data of the matrix
	# 	data    = A.data.ctypes.data_as(POINTER(c_double))
	# 	indptr  = A.indptr.ctypes.data_as(POINTER(c_int))
	# 	indices = A.indices.ctypes.data_as(POINTER(c_int))

	# 	# Allocate output, using same conventions as input
	# 	nVectors = 1
	# 	if x.ndim is 1:
	# 		y = np.empty(m,dtype=np.double,order='F')
	# 		if x.size != n:
	# 			raise Exception("x must have n entries. x.size is %d, n is %d" % (x.size,n))
	# 	elif x.shape[1] is 1:
	# 		y = np.empty((m,1),dtype=np.double,order='F')
	# 		if x.shape[0] != n:
	# 			raise Exception("x must have n entries. x.size is %d, n is %d" % (x.size,n))
	# 	else:
	# 		nVectors = x.shape[1]
	# 		y = np.empty((m,nVectors),dtype=np.double,order='F')
	# 		if x.shape[0] != n:
	# 			raise Exception("x must have n entries. x.size is %d, n is %d" % (x.size,n))

	# 	# Check input
	# 	if x.dtype.type is not np.double:
	# 		x = x.astype(np.double,copy=True)
	        
	# 	# Put it in column-major order, otherwise for nVectors > 1 this FAILS completely
	# 	if x.flags['F_CONTIGUOUS'] is not True:
	# 		x = x.copy(order='F')

	# 	if nVectors == 1:
	# 		np_x = x.ctypes.data_as(POINTER(c_double))
	# 		np_y = y.ctypes.data_as(POINTER(c_double))
	# 		# now call MKL. This returns the answer in np_y, which links to y
	# 		SpMV(byref(c_char(b"N")), byref(c_int(m)), data , indptr, indices, np_x, np_y) 
	# 	else:
	# 		for columns in xrange(nVectors):
	# 			xx = x[:,columns]
	# 			yy = y[:,columns]
	# 			np_x = xx.ctypes.data_as(POINTER(c_double))
	# 			np_y = yy.ctypes.data_as(POINTER(c_double))
	# 			SpMV(byref(c_char("N")), byref(c_int(m)),data,indptr, indices, np_x, np_y) 

	# 	return y

	# ####

	# def tes_simple(self):
	# 	total_vocabulary = 30000
	# 	v = self.generate_v(total_vocabulary, 123)
	# 	with open('ms_csr.p', 'rb') as f:
	# 		ms_csr = pickle.load(f)
	# 	res1 = self.SpMV_viaMKL(ms_csr, v)
	# 	res2 = ms_csr.dot(v)
	# 	self.assertTrue(np.allclose(res1, res2))


	def tes_multiply_by_Z_left(self):
		"""Tests the functionality of multiplying a vector with Z from the left"""

		num_documents_per_language_list = [[13], [4], [12, 22, 10], [132, 123, 123], [123, 120, 130]] # controls Z.shape[0]
		voc_size_per_lang_list = [[6], [10], [23, 30, 4], [40, 70,50], [123, 20, 120]] # controls Z.shape[1]
		num_concepts_list = np.ones_like(voc_size_per_lang_list) # dummy used to create the data object

		seeds_z = [12345, 1111, 222222] # generates different Z's
		seeds_v = [123, 1111, 22222, 333333] # generates different v's

		for i in range(len(num_documents_per_language_list)):
			num_docs_per_lang = num_documents_per_language_list[i]
			voc_size_per_lang = voc_size_per_lang_list[i]
			num_concepts = num_concepts_list[i]
			total_vocabulary = np.sum(voc_size_per_lang)

			for seed_z in seeds_z:
				data_obj = self.generate_data_obj(num_concepts, num_docs_per_lang, voc_size_per_lang, seed_z = seed_z)
				operations_obj = Operations(data_obj)
				Z_dense = data_obj.Z.todense()

				for seed_v in seeds_v:
					v = self.generate_v(total_vocabulary, seed_v)
					v = v.reshape(v.shape[0])

					dense_multiply = Z_dense @ v
					custom_multiply = operations_obj.multiply_by_Z_left(v)
					mkl_multiply = operations_obj.multiply_by_Z_viaMKL(v)

					self.assertTrue(np.allclose(custom_multiply, dense_multiply))
					self.assertTrue(np.allclose(mkl_multiply, custom_multiply))

	def tes_multiply_by_Z_T(self):
		"""Tests the functionality of multiplying a vector with Z transpose from the left"""

		num_documents_per_language_list = [[13], [4], [12, 22, 10], [132, 123, 123], [123, 120, 130]] # controls Z.shape[0]
		voc_size_per_lang_list = [[6], [10], [23, 30, 4], [40, 70,50], [123, 20, 120]] # controls Z.shape[1]
		num_concepts_list = np.ones_like(voc_size_per_lang_list) # dummy  used to create the data object

		seeds_z = [12345, 1111, 222222] # generates different Z's
		seeds_v = [123, 1111, 22222, 333333] # generates different v's

		for i in range(len(num_documents_per_language_list)):
			num_docs_per_lang = num_documents_per_language_list[i]
			voc_size_per_lang = voc_size_per_lang_list[i]
			num_concepts = num_concepts_list[i]
			total_num_of_docs = np.sum(num_docs_per_lang)

			for seed_z in seeds_z:
				data_obj = self.generate_data_obj(num_concepts, num_docs_per_lang, voc_size_per_lang, seed_z = seed_z)
				operations_obj = Operations(data_obj)
				Z_dense = data_obj.Z.todense()
				Z_dense_T = data_obj.Z_T.todense()

				for seed_v in seeds_v:
					v = self.generate_v(total_num_of_docs, seed_v)
					dense_multiply = Z_dense_T @ v
					custom_multiply = operations_obj.multiply_by_Z_T_left(v)
					mkl_multiply = operations_obj.multiply_by_Z_T_viaMKL(v)
					self.assertTrue(np.allclose(mkl_multiply, custom_multiply))
					self.assertTrue(np.allclose(custom_multiply, dense_multiply))

	def get_dense_B(self, operations_obj):
		"""Helper for getting a dense version of the matrix B"""

		data_obj = operations_obj.data_obj
		lambda_ = operations_obj.lambda_
		Z_dense = data_obj.Z.todense()

		n = data_obj.Z.shape[0]
		eye_sub  = np.eye(n) - 1/n
		B = Z_dense.T @ eye_sub @ Z_dense
		B = lambda_ * np.eye(data_obj.Z.shape[1], data_obj.Z.shape[1]) + B

		return B

	def tes_multiply_by_B_left(self):
		"""Tests the functionality of multiplying a vector with A from the left. 
		The tested routine is used in the CG method."""

		num_documents_per_language_list = [[13], [4], [12, 22, 10], [132, 123, 123], [123, 120, 130]] # controls Z.shape[0] and the value of n
		voc_size_per_lang_list = [[6], [10], [23, 30, 4], [40, 70,50], [123, 20, 120]] # controls Z.shape[1]
		num_concepts_list = np.ones_like(voc_size_per_lang_list) # dummy used to create the data object

		lambda_values = [1, 0.5, 0.33, 1.66, 2]

		seeds_z = [12345, 1111, 222222] # generates different Z's
		seeds_v = [123, 1111, 22222, 333333] # generates different v's
		for i in range(len(num_documents_per_language_list)):
			num_docs_per_lang = num_documents_per_language_list[i]
			voc_size_per_lang = voc_size_per_lang_list[i]
			num_concepts = num_concepts_list[i]
			total_vocabulary = np.sum(voc_size_per_lang)

			for seed_z in seeds_z:
				data_obj = self.generate_data_obj(num_concepts, num_docs_per_lang, voc_size_per_lang, seed_z = seed_z)
				Z_dense = data_obj.Z.todense()

				for lambda_ in lambda_values:
					operations_obj = Operations(data_obj, lambda_)

					for seed_v in seeds_v:
						v = self.generate_v(total_vocabulary, seed_v)

						dense_B = self.get_dense_B(operations_obj)
						dense_multiply = dense_B @ v

						v = v.reshape(v.shape[0])
						custom_multiply = Operations.multiply_by_B_left(v, operations_obj)
						self.assertTrue(np.allclose(custom_multiply, dense_multiply.reshape(v.shape[0])))

	def is_pos_def(self, x):
		"""Helper for testing the positive definiteness of a matrix"""

		return np.all(np.linalg.eigvals(x) > 0)

	def test_multiply_by_inverse_cg(self):
		"""Tests the functionality of multiplying a vector by A inverse from the left. 
		The product is generated using CG method."""

		num_documents_per_language_list = [[13], [4], [12, 22, 10], [132, 123, 123], [123, 120, 130]] # controls Z.shape[0] and the value of n
		voc_size_per_lang_list = [[6], [10], [23, 30, 4], [40, 70,50], [123, 20, 120]] # controls Z.shape[1]
		num_concepts_list = np.ones_like(voc_size_per_lang_list) # dummy used to create the data object

		lambda_values = [1, 0.5, 0.33, 1.66, 2]

		seeds_z = [12345, 1111, 222222] # generates different Z's
		seeds_v = [123, 1111, 22222, 333333] # generates different v's
		for i in range(len(num_documents_per_language_list)):
			num_docs_per_lang = num_documents_per_language_list[i]
			voc_size_per_lang = voc_size_per_lang_list[i]
			num_concepts = num_concepts_list[i]
			total_vocabulary = np.sum(voc_size_per_lang) # dimension of vector v

			for seed_z in seeds_z:
				data_obj = self.generate_data_obj(num_concepts, num_docs_per_lang, voc_size_per_lang, seed_z = seed_z)
				#Z_dense = data_obj.Z.todense()

				for lambda_ in lambda_values:
					operations_obj = Operations(data_obj, lambda_)

					for seed_v in seeds_v:
						v = self.generate_v(total_vocabulary, seed_v)

						dense_B = self.get_dense_B(operations_obj)
						#print(self.is_pos_def(dense_A))
						dense_multiply = np.linalg.inv(dense_B) @ v

						return_obj = operations_obj.multiply_by_inverse_cg(v)
						custom_multiply = return_obj[0].reshape(total_vocabulary, 1)

						#if not np.allclose(custom_multiply, dense_multiply):
							#print(dense_multiply)
							# print(Z_dense.shape)
							# print(i)
							# print(seed_z)
							# print(lambda_)
							# print(seed_v)
						self.assertTrue(np.allclose(custom_multiply, dense_multiply))

	def get_dense_M(self, operations_obj):
		"""Helper for getting a dense version of the matrix M"""

		data_obj = operations_obj.data_obj
		lambda_ = operations_obj.lambda_

		n = data_obj.Z.shape[0]
		eye_sub  = np.eye(n) - 1/n
		Z_dense = data_obj.Z.todense()
		Y_dense = data_obj.Y.todense()
		B = self.get_dense_B(operations_obj)

		M = Y_dense.T @ eye_sub @ Z_dense @ np.linalg.inv(B) @ Z_dense.T @ eye_sub @ Y_dense
		return M

	def tf_idf_sparse(self, Z, normalization = 'l1'):
		cols, occurences = np.unique(Z.nonzero()[1], return_counts=True)
		occurences = occurences[np.argsort(cols)]
		num_docs = np.ones_like(occurences)*Z.shape[0]
		idf = np.log(num_docs/occurences) + 1
		idf = sps.diags(idf, format='csc')
		final = Z.dot(idf)
		if normalization != None:
			final = normalize(final, norm=normalization, axis=1)
		return final


	def test_tf_idf(self):
		num_documents_per_language_list = [[13], [4], [12, 22, 10], [132, 123, 123], [123, 120, 130]] # controls Z.shape[0]
		voc_size_per_lang_list = [[6], [10], [23, 30, 4], [40, 70,50], [123, 20, 120]] # controls Z.shape[1]
		num_concepts_list = np.ones_like(voc_size_per_lang_list) # dummy  used to create the data object

		seeds_z = [12345, 1111, 222222] # generates different Z's
		seeds_v = [123, 1111, 22222, 333333] # generates different v's

		normalization = ['l1', 'l2', None]

		for norm in normalization:
			for i in range(len(num_documents_per_language_list)):
				num_docs_per_lang = num_documents_per_language_list[i]
				voc_size_per_lang = voc_size_per_lang_list[i]
				num_concepts = num_concepts_list[i]
				total_num_of_docs = np.sum(num_docs_per_lang)

				for seed_z in seeds_z:
					data_obj = self.generate_data_obj(num_concepts, num_docs_per_lang, voc_size_per_lang, seed_z = seed_z)

					tf_idf_sparse_res = self.tf_idf_sparse(data_obj.Z, norm)

					data_obj.tf_idf_scikit(norm)

					self.assertTrue(sparse.isspmatrix_csr(data_obj.Z))
					self.assertTrue((data_obj.Z != tf_idf_sparse_res).nnz == 0)

			print("Done with %s" % norm)

		for norm in normalization:
			data_obj = Data()
			data_obj.load_obj('da_it_en_65000.p')

			tf_idf_sparse_res = self.tf_idf_sparse(data_obj.Z, norm)

			data_obj.tf_idf_scikit(norm)

			self.assertTrue(sparse.isspmatrix_csr(data_obj.Z))
			self.assertTrue((data_obj.Z != tf_idf_sparse_res).nnz == 0)



	def tes_multiply_by_M(self):
		"""Tests the functionality of multiplying a vector by M from the left. 
		The tested routine is used by the iterative method for finding the SVD/Eigendecomposition decomposition of M."""

		num_documents_per_language_list = [[13], [4], [12, 22, 10], [132, 123, 123], [123, 120, 130]] # controls Z.shape[0], Y.shape[0] and n
		voc_size_per_lang_list = [[6], [10], [23, 30, 4], [40, 70,50], [123, 20, 120]] # controls Z.shape[1]
		num_concepts_list = [3, 4, 15, 100, 130] # controls Y.shape[1]

		lambda_values = [1, 0.5, 0.33, 1.66, 2]

		seeds_z = [12345, 1111, 222222] # generates different Z's
		seeds_y = [1, 123123123, 50000] # generates different Y's
		seeds_v = [123, 1111, 22222, 333333] # generates different v's

		for i in range(len(num_documents_per_language_list)):
			num_docs_per_lang = num_documents_per_language_list[i]
			voc_size_per_lang = voc_size_per_lang_list[i]
			num_concepts = num_concepts_list[i] # dimensions of vector v

			for seed_z in seeds_z:
				for seed_y in seeds_y:
					data_obj = self.generate_data_obj(num_concepts, num_docs_per_lang, voc_size_per_lang, seed_y = seed_y, seed_z = seed_z)

					for lambda_ in lambda_values:
						operations_obj = Operations(data_obj, lambda_)

						for seed_v in seeds_v:
							v = self.generate_v(num_concepts, seed_v)
							v = v.reshape(v.shape[0])

							dense_M = self.get_dense_M(operations_obj)
							dense_multiply = dense_M @ v
							custom_multiply = Operations.multiply_by_M_left(v, operations_obj)

							self.assertTrue(np.allclose(custom_multiply, dense_multiply))

	def get_dense_M2(self, operations_obj):
		"""Helper for getting a dense version of the matrix M2"""

		data_obj = operations_obj.data_obj
		lambda_ = operations_obj.lambda_

		n = data_obj.Z.shape[0]
		eye_sub  = np.eye(n) - 1/n
		Z_dense = data_obj.Z.todense()
		Y_dense = data_obj.Y.todense()
		B_inv = np.linalg.inv(self.get_dense_B(operations_obj))
		vecs = operations_obj.V

		M2 = B_inv @ Z_dense.T @ eye_sub @ Y_dense @ vecs @ vecs.T @ Y_dense.T @ eye_sub @ Z_dense @ B_inv
		return M2

	def tes_decompose_M_eigsh(self):
		"""Tests the whole framework and the resulting SVD/Eigendecomposition of M."""
		
		num_documents_per_language_list = [[13], [4], [12, 22, 10], [132, 123, 123], [123, 120, 130]] # controls Z.shape[0], Y.shape[0] and n
		voc_size_per_lang_list = [[6], [10], [23, 30, 4], [40, 70,50], [123, 20, 120]] # controls Z.shape[1]
		num_concepts_list = [3, 4, 15, 100, 130] # controls Y.shape[1]

		first_k_list = [1, 2, 6, 50, 30]

		lambda_values = [1, 0.33, 1.66]

		seeds_z = [12345, 1111, 222222] # generates different Z's
		seeds_y = [1, 123123123, 50000] # generates different Y's
		seeds_v = [123, 1111, 22222, 333333] # generates different v's for the multiplication test

		for i in range(len(num_documents_per_language_list)):
			print("Starting %d" % i)
			num_docs_per_lang = num_documents_per_language_list[i]
			voc_size_per_lang = voc_size_per_lang_list[i]
			num_concepts = num_concepts_list[i] # dimensions of vector v
			total_vocabulary = np.sum(voc_size_per_lang)

			k = first_k_list[i]

			for seed_z in seeds_z:
				for seed_y in seeds_y:
					data_obj = self.generate_data_obj(num_concepts, num_docs_per_lang, voc_size_per_lang, seed_y = seed_y, seed_z = seed_z)

					for lambda_ in lambda_values:
						operations_obj = Operations(data_obj, lambda_)
						dense_M = self.get_dense_M(operations_obj)

						u, s, vh = np.linalg.svd(dense_M, full_matrices=False)

						e_vals = s[:k]
						e_vals = np.flip(e_vals, axis=0)
						e_vecs = np.flip(vh[:k].T, axis=1)

						vals, vecs = operations_obj.decompose_M_eigsh(k)
						#print(e_vals)

						self.assertTrue(np.allclose(vals, e_vals))
						self.assertTrue(np.allclose(np.abs(e_vecs), np.abs(vecs)))

						## Test multiply_by_V

						for seed_v in seeds_v:
							v = self.generate_v(k, seed_v)

							dense_multiply = vecs @ v

							custom_multiply = operations_obj.multiply_by_V_left(v)
							self.assertTrue(np.allclose(custom_multiply, dense_multiply))

						## Test multiply_by_V_T

						for seed_v in seeds_v:
							v = self.generate_v(num_concepts, seed_v)

							dense_multiply = vecs.T @ v

							custom_multiply = operations_obj.multiply_by_V_T_left(v)
							self.assertTrue(np.allclose(custom_multiply, dense_multiply))

						## Test multiply_by_M2

						dense_M2 = self.get_dense_M2(operations_obj)

						for seed_v in seeds_v:
							v = self.generate_v(total_vocabulary, seed_v)

							dense_multiply = dense_M2 @ v

							custom_multiply = Operations.multiply_by_M2_left(v, operations_obj)

							self.assertTrue(np.allclose(custom_multiply, dense_multiply))

						## Test M2 eigendecomposition

						u, s, vh = np.linalg.svd(dense_M2, full_matrices=False)

						e_vals = s[:k]
						e_vals = np.flip(e_vals, axis=0)
						e_vecs = np.flip(vh[:k].T, axis=1)

						vals, vecs = operations_obj.decompose_M2_eigsh(k)
						#print(e_vals)

						self.assertTrue(np.allclose(vals, e_vals))
						self.assertTrue(np.allclose(np.abs(e_vecs), np.abs(vecs)))


if __name__ == '__main__':
	unittest.main()	