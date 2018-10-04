import numpy as np
cimport numpy as np

DTYPE = np.int
#for linux in VM change to np.int_32_t
ctypedef np.int_t DTYPE_t
cimport cython
@cython.boundscheck(False) # turn of bounds-checking for entire function
def GF2MatrixMul_c( np.ndarray[DTYPE_t, ndim=1] source,
                    np.ndarray[DTYPE_t, ndim=1] dest,
                    int srclen,
                    np.ndarray[DTYPE_t, ndim = 1] RowIndices,
                    np.ndarray[DTYPE_t, ndim = 1] RowLoc,
                    np.ndarray[DTYPE_t, ndim = 1] ColumnSum, int direction):
    """ example:
        source: InformationBits
        dest: MatrixProductbuffer (return value)
        srclen: NumInfoBits
        RowIndices: A_RowIndices (of matrix A which is the H(:,1:K)
        RowLoc: A_RowStartLoc
        CoulumnSum
        direction: 1 or -1 forward backward substitution
        """

    cdef unsigned int columnindex, col_counter, row_counter, rowindex;

    if direction == 1:
        columnindex = 0  # Start from the first column for forward substitution
    else:
        columnindex = srclen - 1  # Start from the last column for backward substitution

    for col_counter in  range(srclen):
        if not source[columnindex] == 0:
            for row_counter in range(ColumnSum[columnindex]):
                rowindex = RowIndices[RowLoc[columnindex] + row_counter]
                dest[rowindex] = 1 - dest[rowindex]
        columnindex += direction
    return dest
