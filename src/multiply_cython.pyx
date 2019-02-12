import numpy as np
cimport numpy as np

def multiply_by_const_left_c(np.ndarray[np.float64_t, ndim=1] v, int n):
	cdef np.float64_t init = 0.0
	cdef int size = len(v)
	cdef double non_diag_value = - 1.0/n
	cdef double diag_value = 1 - 1.0/n

	for i in range(size):
		init = init + non_diag_value * v[i]

	for i in range(size):
		v[i] = init - v[i] * non_diag_value + v[i] * diag_value