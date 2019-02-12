import numpy as np
import scipy
import inspect
import scipy.sparse as sparse

from multiply_cython import multiply_by_const_left_c
from scipy.sparse.linalg import LinearOperator
from timeit import default_timer
from ctypes import POINTER,c_void_p,c_int,c_char,c_double,byref,cdll
import sys

class Operations:
	"""A wrapper class that implements all of the operations done on the data matrices"""

	# Function that use the routine for transposing a matrix should be changed if the format of the sparse matrix changes.

	def __init__(self, data_obj, lambda_ = 1, cg_max_iter = None, cg_tolerance = None):
		Operations.cg_residuals = []
		Operations.cg_residuals2 = []
		Operations.num_iter = 0
		Operations.num_iter2 = 0

		mkl = cdll.LoadLibrary("libmkl_rt.so")
		self.SpMV = mkl.mkl_cspblas_dcsrgemv
		self.time_consumed = {}
		self.data_obj = data_obj
		self.n = data_obj.Z.shape[0]
		self.shapes = {'Z': data_obj.Z.shape, 'Y': data_obj.Y.shape, 'M': (data_obj.Y.shape[1], data_obj.Y.shape[1]), \
			'B': (data_obj.Z.shape[1], data_obj.Z.shape[1]), 'M2': (data_obj.Z.shape[1], data_obj.Z.shape[1])}

		# Checks whether the data is from the type assumed in the fortran native library (double) and
		# whether the data in memory is fortran contiguous

		if data_obj.Z.data.dtype.type is not np.double:
			print("WARNING! Z datatype is not double (np.float64)!")

		if data_obj.Z.data.flags['F_CONTIGUOUS'] is not True:
			print("WARNING! Z.data is not F_CONTIGUOUS")

		if data_obj.Z.indptr.flags['F_CONTIGUOUS'] is not True:
			print("WARNING! Z.indptr is not F_CONTIGUOUS")

		if data_obj.Z.indices.flags['F_CONTIGUOUS'] is not True:
			print("WARNING! Z.indices is not F_CONTIGUOUS")

		if data_obj.Z_T.data.dtype.type is not np.double:
			print("WARNING! Z datatype is not double (np.float64)!")

		if data_obj.Z_T.data.flags['F_CONTIGUOUS'] is not True:
			print("WARNING! Z.data is not F_CONTIGUOUS")
		
		if data_obj.Z_T.indptr.flags['F_CONTIGUOUS'] is not True:
			print("WARNING! Z.indptr is not F_CONTIGUOUS")

		if data_obj.Z_T.indices.flags['F_CONTIGUOUS'] is not True:
			print("WARNING! Z.indices is not F_CONTIGUOUS")

		self.Z_data = data_obj.Z.data.ctypes.data_as(POINTER(c_double))
		self.Z_indptr = data_obj.Z.indptr.ctypes.data_as(POINTER(c_int))
		self.Z_indices = data_obj.Z.indices.ctypes.data_as(POINTER(c_int))

		self.Z_T_data = data_obj.Z_T.data.ctypes.data_as(POINTER(c_double))
		self.Z_T_indptr = data_obj.Z_T.indptr.ctypes.data_as(POINTER(c_int))
		self.Z_T_indices = data_obj.Z_T.indices.ctypes.data_as(POINTER(c_int))

		
		self.M_LO = None
		self.B_LO = None
		self.M2_LO = None
		self.lambda_ = lambda_

		self.cg_max_iter = cg_max_iter
		self.cg_tolerance = cg_tolerance

	def multiply_by_Z_viaMKL( self, x ):
		'''Multiplies the vector passed as argument by the matrix Z'''
		code = 'multiply_by_Z_viaMKL'
		start = default_timer()

		# Dissecting the "cspblas_dcsrgemv" name:
		# "c" - for "c-blas" like interface (as opposed to fortran)
		#    Also means expects sparse arrays to use 0-based indexing, which python does
		# "sp"  for sparse
		# "d"   for double-precision
		# "csr" for compressed row format
		# "ge"  for "general", e.g., the matrix has no special structure such as symmetry
		# "mv"  for "matrix-vector" multiply

		A = self.data_obj.Z

		if not sparse.isspmatrix_csr(A):
			raise Exception("Matrix must be in csr format")
	        
		(m,n) = A.shape

		# # The data of the matrix
		# data    = A.data.ctypes.data_as(POINTER(c_double))
		# indptr  = A.indptr.ctypes.data_as(POINTER(c_int))
		# indices = A.indices.ctypes.data_as(POINTER(c_int))

		# Allocate output, using same conventions as input
		nVectors = 1
		if x.ndim is 1:
			y = np.empty(m,dtype=np.double,order='F')
			if x.size != n:
				raise Exception("x must have n entries. x.size is %d, n is %d" % (x.size,n))
		elif x.shape[1] is 1:
			y = np.empty((m,1),dtype=np.double,order='F')
			if x.shape[0] != n:
				raise Exception("x must have n entries. x.size is %d, n is %d" % (x.size,n))
		else:
			nVectors = x.shape[1]
			y = np.empty((m,nVectors),dtype=np.double,order='F')
			if x.shape[0] != n:
				raise Exception("x must have n entries. x.size is %d, n is %d" % (x.size,n))

		# Check input
		if x.dtype.type is not np.double:
			x = x.astype(np.double,copy=True)
	        
		# Put it in column-major order, otherwise for nVectors > 1 this FAILS completely
		if x.flags['F_CONTIGUOUS'] is not True:
			x = x.copy(order='F')

		if nVectors == 1:
			np_x = x.ctypes.data_as(POINTER(c_double))
			np_y = y.ctypes.data_as(POINTER(c_double))
			# now call MKL. This returns the answer in np_y, which links to y
			self.SpMV(byref(c_char(b"N")), byref(c_int(m)), self.Z_data , self.Z_indptr, self.Z_indices, np_x, np_y) 
		else:
			for columns in xrange(nVectors):
				xx = x[:,columns]
				yy = y[:,columns]
				np_x = xx.ctypes.data_as(POINTER(c_double))
				np_y = yy.ctypes.data_as(POINTER(c_double))
				self.SpMV(byref(c_char("N")), byref(c_int(m)),data,indptr, indices, np_x, np_y) 

		end = default_timer()
		time_elapsed = end - start
		self.update_time(code, time_elapsed)

		return y

	def multiply_by_Z_T_viaMKL( self, x ):
		'''Multiplies the vector passed as argument by the matrix Z_T'''

		code = 'multiply_by_Z_T_viaMKL'
		start = default_timer()

		A = self.data_obj.Z_T

		if not sparse.isspmatrix_csr(A):
			raise Exception("Matrix must be in csr format")
	        
		(m,n) = A.shape

		# # The data of the matrix
		# data    = A.data.ctypes.data_as(POINTER(c_double))
		# indptr  = A.indptr.ctypes.data_as(POINTER(c_int))
		# indices = A.indices.ctypes.data_as(POINTER(c_int))

		# Allocate output, using same conventions as input
		nVectors = 1
		if x.ndim is 1:
			y = np.empty(m,dtype=np.double,order='F')
			if x.size != n:
				raise Exception("x must have n entries. x.size is %d, n is %d" % (x.size,n))
		elif x.shape[1] is 1:
			y = np.empty((m,1),dtype=np.double,order='F')
			if x.shape[0] != n:
				raise Exception("x must have n entries. x.size is %d, n is %d" % (x.size,n))
		else:
			nVectors = x.shape[1]
			y = np.empty((m,nVectors),dtype=np.double,order='F')
			if x.shape[0] != n:
				raise Exception("x must have n entries. x.size is %d, n is %d" % (x.size,n))

		# Check input
		if x.dtype.type is not np.double:
			x = x.astype(np.double,copy=True)
	        
		# Put it in column-major order, otherwise for nVectors > 1 this FAILS completely
		if x.flags['F_CONTIGUOUS'] is not True:
			x = x.copy(order='F')

		if nVectors == 1:
			np_x = x.ctypes.data_as(POINTER(c_double))
			np_y = y.ctypes.data_as(POINTER(c_double))
			# now call MKL. This returns the answer in np_y, which links to y
			self.SpMV(byref(c_char(b"N")), byref(c_int(m)), self.Z_T_data , self.Z_T_indptr, self.Z_T_indices, np_x, np_y) 
		else:
			for columns in xrange(nVectors):
				xx = x[:,columns]
				yy = y[:,columns]
				np_x = xx.ctypes.data_as(POINTER(c_double))
				np_y = yy.ctypes.data_as(POINTER(c_double))
				self.SpMV(byref(c_char("N")), byref(c_int(m)),data,indptr, indices, np_x, np_y) 

		end = default_timer()
		time_elapsed = end - start
		self.update_time(code, time_elapsed)

		return y

	def reset_time(self):
		'''Reset the time consumed per function dictionary'''

		self.time_consumed = {}

	def set_lambda(self, lambda_):
		'''Change the regularization parameter for the Operations instance'''

		self.labmda_ = lambda_

	def update_time(self, operation, time):
		'''Utility function used by the other routines to update the time in the time consumed per function dictionary'''

		if operation in self.time_consumed:
			self.time_consumed[operation] += time
			return

		self.time_consumed[operation] = time

	def multiply_by_const_left(self, v):
		'''Multiplies the vector passed as argument by the constant factor in the equation'''

		code = 'multiply_by_const_left'
		start = default_timer()

		n = self.n
		non_diag_value = -1/n
		diag_value = 1 - 1/n
		init = 0.0
		for i in range(len(v)):
			init = init + non_diag_value * v[i]

		result = np.array([init - val * non_diag_value + val * diag_value for val in v])
		end = default_timer()
		time_elapsed = end - start
		self.update_time(code, time_elapsed)

		return result

	def multiply_by_const_left_cython(self, v):
		'''Multiplies the vector passed as argument by the constant factor in the equation'''

		code = 'multiply_by_const_left_cython'
		start = default_timer()

		if v.flags['C_CONTIGUOUS'] is not True:
			v = v.copy(order='C')
			print("Not C_CONTIGUOUS")

		if v.ndim is 2:
			v = v.reshape(v.shape[0])

		multiply_by_const_left_c(v, self.n)

		end = default_timer()
		time_elapsed = end - start
		self.update_time(code, time_elapsed)

		return v

	def multiply_by_B_left(v, operations_obj):
		'''Multiplies the vector passed as argument by the matrix B, constructed according to the data in the passed operations object.
		It is used in the linear operator for the conjugate gradient method.'''

		code = 'multiply_by_B_left'
		start = default_timer()

		#print(v.shape)
		inter = operations_obj.multiply_by_Z_viaMKL(v)	 #operations_obj.multiply_by_Z_left(v)
		#print(inter.shape)
		inter = operations_obj.multiply_by_const_left_cython(inter)
		#print(inter.shape)
		inter = operations_obj.multiply_by_Z_T_viaMKL(inter)
		#print(inter.shape)

		result = inter + v * operations_obj.lambda_

		end = default_timer()
		time_elapsed = end - start
		operations_obj.update_time(code, time_elapsed)

		return result

	def report_cg_error(xk):
		'''The function is called on every conjugate gradient iteration (in the first decomposition) to report on the residual. 
		Used in the conjugate gradient convergence analysis'''

		if Operations.num_iter % 1000 ==0:
			print(Operations.num_iter)
			sys.stdout.flush()

		Operations.num_iter += 1
		# Reporting the residual
		# frame = inspect.currentframe().f_back
		# residual = frame.f_locals['resid']
		# Operations.cg_residuals.append(residual)

	def report_cg_error2(xk):
		'''The function is called on every conjugate gradient iteration (in the second decomposition) to report on the residual. 
		Used in the conjugate gradient convergence analysis'''

		if Operations.num_iter2 % 1000 ==0:
			print(Operations.num_iter2)
			sys.stdout.flush()
			
		Operations.num_iter2 += 1
		# Reporting the residual
		# frame = inspect.currentframe().f_back
		# residual = frame.f_locals['resid']
		# Operations.cg_residuals2.append(residual)

	def multiply_by_inverse_cg(self, v, maxiter = 250, callback = None, tol = 0): # maxiter = None, callback = None, tol = 1e-8):
		'''Returns the result from the A^-1 * v multiplication, implemented using the conjugate gradient method.
		NB: the tolerance parameter represents relative tolerance i.e ||b - A * xk|| / ||b|| < tol'''

		code = 'multiply_by_inverse_cg'
		start = default_timer()

		if self.B_LO == None:
			self.B_LO = LinearOperator(self.shapes['B'], matvec=lambda v: Operations.multiply_by_B_left(v, self))

		LO = self.B_LO

		if self.cg_max_iter is None:
			if maxiter is None:
				if callback is not None:
					result = scipy.sparse.linalg.cg(LO, v, tol = tol, callback = callback)
				else:
					result = scipy.sparse.linalg.cg(LO, v, tol = tol)
			else:
				if callback is not None:
					result = scipy.sparse.linalg.cg(LO, v, maxiter = maxiter, tol = tol, callback = callback)
				else:
					result = scipy.sparse.linalg.cg(LO, v, maxiter = maxiter, tol = tol)
		else:
			maxiter = self.cg_max_iter
			tol = self.cg_tolerance

			if callback is not None:
				result = scipy.sparse.linalg.cg(LO, v, maxiter = maxiter, tol = tol, callback = callback)
			else:
				result = scipy.sparse.linalg.cg(LO, v, maxiter = maxiter, tol = tol)


		end = default_timer()
		time_elapsed = end - start
		self.update_time(code, time_elapsed)

		return result

	def multiply_by_Z_left(self, v):
		'''Multiplies the passed vector by the matrix Z from the left.'''

		code = 'multiply_by_Z_left'
		start = default_timer()

		Z = self.data_obj.Z
		result = Z.dot(v)

		end = default_timer()
		time_elapsed = end - start
		self.update_time(code, time_elapsed)

		return result

	def multiply_by_Z_T_left(self, v):
		'''Multiplies the passed vector by the matrix Z transposed, from the left.'''

		code = 'multiply_by_Z_T_left'
		start = default_timer()

		Z = self.data_obj.Z
		Z_T = scipy.sparse.csr_matrix.transpose(Z)
		result = Z_T.dot(v)

		end = default_timer()
		time_elapsed = end - start
		self.update_time(code, time_elapsed)

		return result

	def multiply_by_Y_left(self, v):
		'''Multiplies the passed vector by the matrix Y from the left.'''

		code = 'multiply_by_Y_left'
		start = default_timer()

		Y = self.data_obj.Y
		result = v[Y.indices]

		end = default_timer()
		time_elapsed = end - start
		self.update_time(code, time_elapsed)

		return result

	def multiply_by_Y_T_left(self, v):
		'''Multiplies the passed vector by the matrix Y transposed, from the left.'''

		code = 'multiply_by_Y_T_left'
		start = default_timer()

		Y = self.data_obj.Y
		Y_T = scipy.sparse.csr_matrix.transpose(Y)
		result = Y_T.dot(v)

		end = default_timer()
		time_elapsed = end - start
		self.update_time(code, time_elapsed)

		return result

	def multiply_by_M_left(v, operations_obj):
		'''Multiplies the vector passed as argument by the matrix M, constructed according to the data in the passed operations object.
		It is used in the linear operator for the eigendecomposition.'''

		inter = operations_obj.multiply_by_Y_left(v) # Y @ v
		inter = operations_obj.multiply_by_const_left_cython(inter) # const @ Y @ v
		inter = operations_obj.multiply_by_Z_T_viaMKL(inter) # Z_T @ const @ Y @ v
		# Operations.cg_residuals.append(-20)
		inter = operations_obj.multiply_by_inverse_cg(inter, callback = Operations.report_cg_error)[0].reshape(operations_obj.shapes['Z'][1], 1) # (B)^-1 @ Z_T @ const @ Y @ v
		inter = operations_obj.multiply_by_Z_viaMKL(inter) # Z @ (B)^-1 @ Z_T @ const @ Y @ v
		inter = operations_obj.multiply_by_const_left_cython(inter) # const @ Z @ (B)^-1 @ Z_T @ const @ Y @ v
		inter = operations_obj.multiply_by_Y_T_left(inter) # Y_T @ const @ Z @ (B)^-1 @ Z_T @ const @ Y @ v
		return inter

	def decompose_M_eigsh(self, k, maxiter = 200, tol = 0, increase_ncv = False): #maxiter = 800, tol=0): # maxiter = None, tol = 1e-8):
		'''Returns the `k` largest eigenvalues and eigenvectors from the eigendecomposition of M.
		Implemented using the eigsh routine.'''

		print("First decomposition started.")
		code = 'decompose_M_eigsh'
		start = default_timer()

		if self.M_LO == None:
			self.M_LO = LinearOperator(self.shapes['M'], matvec=lambda v: Operations.multiply_by_M_left(v, self))

		LO = self.M_LO
		if maxiter is None:
			vals, vecs = scipy.sparse.linalg.eigsh(LO, k=k, tol=tol)
		else:
			if increase_ncv:
				print("Default NCV.")
				print("Eigs tolerance: ", tol)
				vals, vecs = scipy.sparse.linalg.eigsh(LO, k=k, maxiter = maxiter, tol = tol, ncv = 5*k)
			else:
				print("Default NCV.")
				print("Eigs tolerance: ", tol)
				vals, vecs = scipy.sparse.linalg.eigsh(LO, k=k, maxiter = maxiter, tol=tol)

		end = default_timer()
		time_elapsed = end - start
		self.update_time(code, time_elapsed)
		self.V = vecs
		self.S = vals

		return vals, vecs


	def multiply_by_V_left(self, v):
		'''Multiplies the passed vector by the matrix of eigenvectors V from the left.'''

		code = 'multiply_by_v_left'
		start = default_timer()

		vecs = self.V
		result = vecs @ v

		end = default_timer()
		time_elapsed = end - start
		self.update_time(code, time_elapsed)

		return result

	def multiply_by_V_T_left(self, v):
		'''Multiplies the passed vector by the matrix of eigenvectors V transposed, from the left.'''

		code = 'multiply_by_v_T_left'
		start = default_timer()

		vecs_T = self.V.T
		result = vecs_T @ v

		end = default_timer()
		time_elapsed = end - start
		self.update_time(code, time_elapsed)

		return result

	def multiply_by_M2_left(v, operations_obj):
		'''Multiplies the vector passed as argument by the matrix M2, constructed according to the data in the passed operations object.
		It is used in the linear operator for the eigendecomposition.'''

		# Operations.cg_residuals2.append(-10)
		inter = operations_obj.multiply_by_inverse_cg(v, callback = Operations.report_cg_error2)[0].reshape(operations_obj.shapes['Z'][1], 1) # (B)^-1 @ v
		inter = operations_obj.multiply_by_Z_viaMKL(inter) # Z @ (B)^-1 @ v
		inter = operations_obj.multiply_by_const_left_cython(inter) # const @ Z @ (B)^-1 @ v
		inter = operations_obj.multiply_by_Y_T_left(inter) # Y_T @ const @ Z @ (B)^-1 @ v
		inter = operations_obj.multiply_by_V_T_left(inter) # V_T @ Y_T @ const @ Z @ (B)^-1 @ v
		inter = operations_obj.multiply_by_V_left(inter) # V @ V_T @ Y_T @ const @ Z @ (B)^-1 @ v
		inter = operations_obj.multiply_by_Y_left(inter) # Y @ V @ V_T @ Y_T @ const @ Z @ (B)^-1 @ v
		inter = operations_obj.multiply_by_const_left_cython(inter) # const @ Y @ ?? @ ?? @ Y_T @ const @ Z @ (B)^-1 @ v
		inter = operations_obj.multiply_by_Z_T_viaMKL(inter) # Z_T @ const @ Y @ ?? @ ?? @ Y_T @ const @ Z @ (B)^-1 @ v
		# Operations.cg_residuals2.append(-10)
		inter = operations_obj.multiply_by_inverse_cg(inter, callback = Operations.report_cg_error2)[0].reshape(operations_obj.shapes['Z'][1], 1) # (B)^-1 Z_T @ const @ Y @ ?? @ ?? @ Y_T @ const @ Z @ (B)^-1 @ v
		return inter

	def decompose_M2_eigsh(self, k, maxiter = 200, tol = 0, increase_ncv = False): #maxiter = 800, tol = 0): # maxiter = None, tol = 1e-8):
		'''Returns the `k` largest eigenvalues and eigenvectors from the eigendecomposition of M2.
		Implemented using the eigsh routine.'''

		print("Second decomposition started.")
		code = 'decompose_M2_eigsh'
		start = default_timer()

		if self.M2_LO == None:
			self.M2_LO = LinearOperator(self.shapes['M2'], matvec=lambda v: Operations.multiply_by_M2_left(v, self))

		LO = self.M2_LO
		if maxiter is None:
			vals, vecs = scipy.sparse.linalg.eigsh(LO, k=k, tol = tol)
		else:
			if increase_ncv:
				print("NCV increased.")
				print("Eigs tolerance: ", tol)
				vals, vecs = scipy.sparse.linalg.eigsh(LO, k=k, maxiter = maxiter, tol = tol, ncv = 5*k)
			else:
				print("Default NCV.")
				print("Eigs tolerance: ", tol)
				vals, vecs = scipy.sparse.linalg.eigsh(LO, k=k, maxiter = maxiter, tol = tol)

		end = default_timer()
		time_elapsed = end - start
		self.update_time(code, time_elapsed)
		self.U = vecs
		self.S2 = vals

		return vals, vecs