import Discrete_LDPC_decoding.GF2MatrixMul_c as GF2MatrixMul_c
import numpy as np
import scipy.sparse as sp

__author__ = "Maximilian Stark"
__copyright__ = "05.07.2016, Institute of Communications, University of Technology Hamburg"
__credits__ = ["Maximilian Stark"]
__version__ = "1.0"
__email__ = "maximilian.stark@tuhh.de"
__status__ = "Production"
__name__ = "LDPC Encoder"
__doc__ = "This class implements an LDPC encoder."


class LDPCEncoder:
    """This class implements an LDPC encoder. The constructor takes the path to a saved Parity Check Matrix as input. The
    file should be in the alist Format. Similar to the LDPCencoder from the Matlab communication toolbox the  last
    N−K columns in the parity check matrix must be an invertible matrix in GF(2).
    This is because the encoding is done only based on parity check matrix, by evaluating a_k' = inv(H_k)*H_l*a_L.
    Input X must be a numeric or logical column vector with length equal K. The length of the encoded data output
    vector, Y, is N. It is a solution to the parity-check equation, with the first K bits equal to the input, X."""

    def __init__(self, filename, alist_file = True):
        #if alist_file:
        #    self.H = self.load_check_mat(filename)
        #else:
        #    self.H = np.load(filename)

        # if sp.issparse(self.H):
        #    self.H = (self.H).toarray()
        # self.H_sparse = sp.csr_matrix(self.H)
        # self.setParityCheckMatrix(self.H)

        self.H_sparse = self.load_check_mat(filename)
        self.setParityCheckMatrix(self.H_sparse)




    def alistToNumpy(self,lines):
        """Converts a parity-check matrix in AList format to a 0/1 numpy array. The argument is a
       list-of-lists corresponding to the lines of the AList format, already parsed to integers
        if read from a text file.
        The AList format is introduced on http://www.inference.phy.cam.ac.uk/mackay/codes/alist.html.
        This method supports a "reduced" AList format where lines 3 and 4 (containing column and row
        weights, respectively) and the row-based information (last part of the Alist file) are omitted.
        Example:
             >>> alistToNumpy([[3,2], [2, 2], [1,1,2], [2,2], [1], [2], [1,2], [1,2,3,4]])
            array([[1, 0, 1],
                  [0, 1, 1]])
        """

        nCols, nRows = lines[0]
        if len(lines[2]) == nCols and len(lines[3]) == nRows:
            startIndex = 4
        else:
            startIndex = 2
        matrix = np.zeros((nRows, nCols), dtype=np.int)
        for col, nonzeros in enumerate(lines[startIndex:startIndex + nCols]):
            for rowIndex in nonzeros:
                if rowIndex != 0:
                    matrix[rowIndex - 1, col] = 1

        return matrix

    def load_sparse_csr(self,filename):
        loader = np.load(filename)
        return sp.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                              shape=loader['shape'])

    def load_check_mat(self, filename):
        if filename.endswith('.npy') or filename.endswith('.npz'):
            if filename.endswith('.npy'):
                H = np.load(filename)
                H = sp.csr_matrix(H)
            else:
                H = self.load_sparse_csr(filename)

        else:
            arrays = [np.array(list(map(int, line.split()))) for line in open(filename)]
            H = self.alistToNumpy(arrays)
            H = sp.csr_matrix(H)

        return H

    def encode(self, X):

        EncodingMethod = self.EncodingMethod.copy()

        if self.RowOrder[0] >= 0:
            # only if the last (N-K) columns of H are not triangular or if they are lower/upper triangular along the
            # antidiagonal
            # Todo
            pass
        else:
            pass

        # compute matrix product between first K_columns of H and information bits.
        result = self.GF2MatrixMul(X, np.zeros(self.NumParityBits, dtype=int), self.NumInfoBits, self.MatrixA_RowIndices,
                                   self.MatrixA_RowStartLoc, self.MatrixA_ColumnSum, 1)

        # need to perform another substitution if last (N-K) columns are not triangular
        if EncodingMethod == 0:
            # forward substitution for lower triangular matrix obtained from factorization in GF(2)
            result = self.GF2MatrixMul(result, result, self.NumParityBits, self.MatrixL_RowIndices,
                                       self.MatrixL_RowStartLoc, self.MatrixL_ColumnSum, 1)
            # now we need to perform backward substitution since B will be upper triangular
            EncodingMethod = -1

        if self.RowOrder[0] >= 0:
            #first version loop
            #for counter in range(self.NumParityBits):
            #    result[counter] = result[self.RowOrder[counter]]
            # second option
            result = result[self.RowOrder]

        # Solve for the Parity Check Bits.
        # Common step for all shapes.
        parity_check_bits = self.GF2MatrixMul(result, result, self.NumParityBits, self.MatrixB_RowIndices,
                                              self.MatrixB_RowStartLoc, self.MatrixB_ColumnSum, EncodingMethod)

        codeword = np.append(X, parity_check_bits)
        return codeword

    def encode_c(self, X):
        EncodingMethod = self.EncodingMethod

        if self.RowOrder[0] >= 0:
            # only if the last (N-K) coloums of H are not triangula or if they are lower/upper triangular along the
            # antidiagonal
            # Todo
            pass
        else:
            pass

        # compute matrix product between first K_columns of H and information bits.
        result = GF2MatrixMul_c.GF2MatrixMul_c(X.astype(np.int32).copy(), np.zeros(self.NumParityBits, dtype=np.int32),
                                               self.NumInfoBits, self.MatrixA_RowIndices,self.MatrixA_RowStartLoc,
                                               self.MatrixA_ColumnSum, 1)

        # need to perform another substitution if last (N-K) columns are not triangular
        if EncodingMethod == 0:
            # forward substitution for lower triangular matrix obtained from factorization in GF(2)
            result = GF2MatrixMul_c.GF2MatrixMul_c(result, result, self.NumParityBits, self.MatrixL_RowIndices,
                                       self.MatrixL_RowStartLoc, self.MatrixL_ColumnSum, 1)
            # now we need to perform backward substitution since B will be upper triangular
            EncodingMethod = -1

        if self.RowOrder[0] >= 0:
            #first version loop
            #for counter in range(self.NumParityBits):
            #    result[counter] = result[self.RowOrder[counter]]
            # second option
            result = result[self.RowOrder]

        # Solve for the Parity Check Bits.
        # Common step for all shapes.
        parity_check_bits = GF2MatrixMul_c.GF2MatrixMul_c(result, result, self.NumParityBits, self.MatrixB_RowIndices,
                                              self.MatrixB_RowStartLoc, self.MatrixB_ColumnSum, EncodingMethod)

        codeword = np.append(X, parity_check_bits)
        return codeword

    def GF2MatrixMul(self, source, dest, srclen ,RowIndices, RowLoc, ColumnSum, direction):
        """ example: 
            source: InformationBits
            dest: MatrixProductbuffer (return value)
            srclen: NumInfoBits
            RowIndices: A_RowIndices (of matrix A which is the H(:,1:K)
            RowLoc: A_RowStartLoc
            CoulumnSum
            direction: 1 or -1 forward backward substitution
            """

        if direction == 1:
            columnindex = 0 # Start from the first column for forward substitution
        else:
            columnindex = srclen - 1 # Start from the last column for backward substitution

        
        for col_counter in range(srclen):
            if not source[columnindex] == 0:
                for row_counter in range(ColumnSum[columnindex]):
                    rowindex = RowIndices[RowLoc[columnindex] + row_counter]
                    dest[rowindex] = 1 - dest[rowindex]

            columnindex += direction


        return dest
            

    def setParityCheckMatrix(self,H):
        params = self.getLDPCEncoderParamters(H)
        self.storedParityCheckMatrix = H

    def getLDPCEncoderParamters(self,H):
        self.N = H.shape[1]
        self.K = self.N -H.shape[0]

        #extract last (N-K) columns of parity check matrix
        last_Part = H[:,self.K:]

        # check if last_Part is triangular
        shape = self.isfulldiagtriangular(last_Part)

        if shape == 1:
            algo = 'Forward Substitution'
            rowOrder = np.array([-1])                   # Don't need to reverse the order
        elif shape == -1:
            algo = 'Backward Substitution'
            rowOrder = np.array([-1])                   # Don't need to reverse the order
        else:
            # Reverse the order of rows in last_Part, but keep lastPart, since if PB is not triangular
            # we need to factorize it in GF(2)
            Reversed_last = last_Part[::-1,:].copy()
            rev_shape = self.isfulldiagtriangular(Reversed_last)
            if rev_shape == 1:
                algo = 'Forward Substitution'
                rowOrder = np.arange((self.N-self.K))[::-1]
                last_Part =  Reversed_last
            elif rev_shape == -1:
                algo = 'Backward Substitution'
                rowOrder = np.arange((self.N-self.K))[::-1]
                last_Part = Reversed_last
            else:
                algo = 'Matrix Inverse'

        # now we preallocate variable for the encode function
        self.MatrixL_RowIndices = np.int32(0)
        self.MatrixL_ColumnSum = np.int32(0)
        self.MatrixL_RowStartLoc = np.int32(0)

        if algo == 'Forward Substitution':
            self.EncodingMethod = np.int8(1)
            #P = np.tril(last_Part, -1) # remove diagonal
            P = sp.tril(last_Part, -1) # remove diagonal
        elif algo == 'Backward Substitution':
            self.EncodingMethod = np.int8(1)
            #P = np.triu(last_Part, 1) # remove diagonal
            P = sp.triu(last_Part, 1) # remove diagonal
        else:
            # algo is 'Matrix Inverse' so we need to work a bit. So we factorize in GF(2) first.
            PL, last_Part, rowOrder, invertible = self.gf2factorize(last_Part.toarray())

            if not invertible:
                print('Not invertible Matrix')
            self.EncodingMethod = np.int8(0)
            #self.MatrixL_RowIndices, self.MatrixL_RowStartLoc, self.MatrixL_ColumnSum = \
            #    self.ConvertMatrixFormat(np.tril(PL, -1))

            self.MatrixL_RowIndices, self.MatrixL_RowStartLoc, self.MatrixL_ColumnSum = \
                self.ConvertMatrixFormat(sp.tril(PL, -1))

            last_Part = last_Part[rowOrder, :]
            #P = np.triu(last_Part, 1)
            P = sp.triu(last_Part, 1)

        # Update all internal data structures for the encoding
        self.RowOrder = np.int32(rowOrder)

        self.MatrixA_RowIndices, self.MatrixA_RowStartLoc, self.MatrixA_ColumnSum = self.ConvertMatrixFormat(H[:, :self.K])
        self.MatrixB_RowIndices, self.MatrixB_RowStartLoc, self.MatrixB_ColumnSum = self.ConvertMatrixFormat(P)

        # Update all external properties.
        self.NumInfoBits = self.K
        self.NumParityBits = self.N - self.K
        self.BlockLength = self.N
        self.EncodingAlgorithm = algo

    def ConvertMatrixFormat(self, X):
        """Create an alternative representation of zero-one matrix"""

        # j, i = np.nonzero(np.transpose(X.toarray()))
        # RowIndices = np.int32(i)
        # ColumnSum = np.int32((X.toarray()).sum(0))
        # # For each row find the corresponding row indices start in RowIndicies.
        # CumulativeSum = np.cumsum(np.double(ColumnSum))
        # RowStartLoc = np.int32(np.append([0], CumulativeSum[:-1]))

        RowIndices = ((X.tocsc()).indices).astype(np.int32)
        RowStartLoc = np.int32((X.tocsc()).indptr[:-1])
        ColumnSum = np.int32((X.tocsc().sum(0)).A[0,:])

        return RowIndices, RowStartLoc, ColumnSum

    def gf2factorize(self,X):
        """This function factorizes a square matrix in GF(2) using Gaussian elimination.
        X= A * B using modulo 2 arithmetic.
        X may be sparse.
        A and B will be sparse

        A is always lower triangular. If X is invertible in GF(2), then B(chosen_pivot,:) is upper triangular and
        invertible.
        """

        n = X.shape[0]
        if not n == X.shape[1]:
            print("error non square matrix")

        Y1 = np.eye(n,n,0,bool)
        Y2 = np.zeros([n,n]).astype(bool)
        Y2[np.nonzero(X)] = 1
        chosen_pivots = np.zeros(n).astype(int)
        invertible = True

        for col in range(n):
            candidate_rows = Y2[:, col].copy()

            candidate_rows[chosen_pivots[:col]] = 0 # never use a chosen pivot
            candidate_rows = np.nonzero(candidate_rows)[0]

            if candidate_rows.size ==0:
                invertible = False # not invertible
                break
            else:
                pivot = candidate_rows[0] # chose first candidate as pivot
                chosen_pivots[col] = pivot # record pivot
                # find all nonzero elements in pivot row and xor with corresponding other candidate rows
                columnind = np.nonzero(Y2[pivot, :])

                # subtraction step. but is NOT in GF2


                Y2[candidate_rows[1:, np.newaxis], columnind] = \
                    np.logical_not(Y2[candidate_rows[1:, np.newaxis], columnind])


                Y1[candidate_rows[1:], pivot] = 1

        A = sp.csr_matrix(Y1)
        B = sp.csr_matrix(Y2)
        #A = Y1
        #B = Y2

        #if not invertible return empty pivot
        if not invertible:
            chosen_pivots = np.zeros(n).astype(int)

        return A, B, chosen_pivots, invertible

    def isfulldiagtriangular(self,X):
        """X must be a square logical matrix.
        shape = 1 if X is ower triangular and has a full diagonal
        shape = -1 if X is upper triangular and has a full diagonal
        shape = 0"""

        N = X.shape[0]
        NumNonZeros = (X != 0).sum()
        #if not np.all(np.diagonal(X)):
        if not np.all(X.diagonal()):
            shape = 0
        else:
            #NumNonzerosInLowerPart = (np.tril(X) != 0).sum()
            NumNonzerosInLowerPart = (sp.tril(X) != 0).sum()
            if NumNonzerosInLowerPart == NumNonZeros:
                shape = 1 # X is lower triangular
            elif  NumNonzerosInLowerPart == N:
                shape = -1
            else:
                shape = 0
        return shape


