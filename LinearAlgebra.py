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
	transpose = [[matrix[i][j] for i in range(shape[1])] for j in range(shape[0])]

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

# Row Echelon Form
def forwardSolve(matrix):
    	'''
    	Row reduces a dense matrix to an upper triangular matrix in O(n^3) time. 
    
    	Assumptions
    	-----------
    	- Matrix is one-to-one
    	- Linearly independent rows
    	- No '0' rows or columns
    	- No '0's in pivot positions
    
    	Parameters
   	 ----------
   	 matrix : list 
    	    (m x n) size 2-D list following assumptions above 
    
    	Returns
    	-------
    	Row reduced upper triangular form of matrix
    	'''
	# Identify the dimensions of the matrix
	rows = len(matrix)
	columns = len(matrix[0])

	# Check for 0 rows and 0 columns later
    
    	# Assumes that the matrix is 1-1, if matrix is onto then take the min(columns, rows).
	for count in range(columns - 1):

		# For each valid row under the pivot
		for r in range(rows - count - 1):

			# Calculate the pivot modifier
			m = matrix[r+count+1][count] / matrix[count][count]

			for c in range(columns):

				matrix[r+count+1][c] = matrix[r+count+1][c] - m * matrix[count][c]

	return matrix

# Row Echelon form to Reduced Row Echelon Form
def backwardSolve(matrix):
	'''
	Take an upper triangular matrix and reduce to row echelon form. 

	Assumptions
	-----------
	- Matrix is in row reduced form
	- Matrix of size (n, n+1)
	- No '0' rows or columns

	Parameters
	----------
	matrix : list
		(m x n) size 2-D list following assumptions above

	Returns
	-------
	Row echelon form of the matrix

	NOTE:
	-----
	- Weird '-0.0' output in the lower triangular portion
	'''

	# Identify the dimensions of the matrix
	rows = len(matrix)
	columns = len(matrix[0])

	# Make the diagonal 1

	for r in range(rows):

		m = matrix[r][r]

		for c in range(columns):

			matrix[r][c] = matrix[r][c] / m

	# Because n, n+1 matrix, omit the first column, also omit last column
	for c in range(columns - 2):

		# Start c at the second to last column
		c = -1 * (c + 2) 

		# Iterates through each row above the pivot position
		for r in range(rows + 1 + c):

			# Make sure the last column changes. Everything else will be 0
			matrix[r][-1] = matrix[r][-1] - matrix[c + 1][-1] * matrix[r][c]

	# Make other entries besides diagonal and right column 0
	for c in range(columns - 2):

		for r in range(c + 1):

			matrix[r][c + 1] = 0.0

	return matrix

# Gauss Jordan Elimination
def GaussJordanElim(matrix):
	'''
	Take a matrix and reduce to row echelon form.

	Assumptions
	-----------
	- Matrix of size (n, n+1)
	- No '0' rows or columns

	Parameters
	----------
	matrix : list
		(m x n) size 2-D list following assumptions above

	Returns
	-------
	Reduced row echelon form of the matrix
	'''

	gaussJordan = backwardSolve(forwardSolve(matrix))

	return gaussJordan

# Matrix inversion
def matrixInversion(matrix):
	'''
	Take a matrix and find the inverse

	Assumptions
	-----------
	- Matrix of size (n, n)
	- No '0' rows or columns

	Parameters
	----------
	matrix : list
		(n x n) size 2-D list following assumptions above

	Returns
	-------
	Inversion of the matrix
	'''

	# Identify the dimensions of the matrix
	rows = len(matrix)
	columns = len(matrix[0])

	# Create an augmented matrix
	matrixAugmented = [[0 for i in range(2 * columns)] for j in range(rows)]

	for c in range(columns):

		for r in range(rows):

			matrixAugmented[r][c] = matrix[r][c]

	for c in range(columns):

		matrixAugmented[c][c + columns] = 1

	matrix = forwardSolve(matrixAugmented)

	inverseMatrix = [[matrixAugmented[r][c + columns] for c in range(columns)] for r in range(rows)]

	return inverseMatrix
