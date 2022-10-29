# Matrix Transpose
def matrixTranspose(matrix):

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
	
	# What is the shape of the matrix being transposed?
	shape = [len(matrix[0]), len(matrix)]

	# Create a the transposed matrix. Flip the dimensions and then run through each original element and assign new positions.
	transpose = [[matrix[i][j] for i in range(shape[0])] for j in range(shape[1])]

	return transpose

# Matrix Multiplication
def matrixMultiply(matrix1, matrix2):
    """
    Multiplies two matrices: matrix1 and matrix2.
    
    Parameters:
    -----------
        matrix1 : list
            (n x m) 2-D list
            
        matrix2 : list
            (m x k) 2-D list
            
    Returns:
    --------
        output : list
            (n x k) 2-D list
    
    """
    
    # Transposing matrix2 for elt-by-elt multiplication
    matrix2 = matrixTranspose(matrix2)
    
    # Multiplying matrices
    output = [[sum([i*j
                    for i, j in zip(matrix1[r],matrix2[c])])
               for c in range(len(matrix2))]
              for r in range(len(matrix1))]
    
    return output

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
