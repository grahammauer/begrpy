# Matrix Transpose
def transpose(matrix):

	"""
	Transpose the elements of a 2-D list across the primary diagonal

	Parameters
	----------
	matrix : list
		(n x m) 2-D list

	Returns
	-------
	transpose : list
		(m x n) 2-D list
	"""

	shape = [len(matrix[0]), len(matrix)]


	transpose = [[0 for i in range(shape[0])] for j in range(shape[1])]

	for i in range(shape[0]):

		for j in range(shape[1]):

			transpose[j][i] = matrix[i][j]

	return transpose

# Matrix Multiplication

# Vector Dot Product
def vectorDotProduct(vector1, vector2):
	"""
	Take the dot product of two 1-D lists of length n. Add
	the product of each component of the vectors. 

	Parameters
	----------
	vector1 : list
		n length 1-D list

	vector2 : list
		n length 1-D list

	Returns
	-------
	Sum the product of each vector components
	"""

	return matrixMultiply(vector1, matrixTranspose(vector2))
