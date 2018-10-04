import numpy as np
import scipy as sci

from Discrete_LDPC_decoding.LDPC_encoder import LDPCEncoder

__author__ = "Maximilian Stark"
__copyright__ = "10.08.2016, Institute of Communications, University of Technology Hamburg"
__credits__ = ["Maximilian Stark"]
__version__ = "1.0"
__email__ = "maximilian.stark@tuhh.de"
__status__ = "Production"
__name__ = "LDPC encoded BPSK Transmitter"
__doc__ = """This class generates a random sequence of BPSK symbols. These symbols are encoded using a given LDPC code
            parity check matrix."""

class LDPC_BPSK_Transmitter:

    def __init__(self, filename_H_,  msg_at_time=1):
        self.filename_H = filename_H_
        self.H_sparse = self.load_check_mat(self.filename_H)

        self.encoder = LDPCEncoder(self.filename_H)

        # analyze the H matrix and set all decoder variables
        self.set_code_parameters()

        self.data_len = (self.R_c * self.codeword_len).astype(int)

        self.last_transmitted_bits = []
        self.msg_at_time = msg_at_time

    def set_code_parameters(self):
        self.degree_checknode_nr = ((self.H_sparse).sum(1)).astype(np.int).A[:, 0]  # which check node has which degree?
        self.degree_varnode_nr = ((self.H_sparse).sum(0)).astype(np.int).A[0,
                                 :]  # which variable node has which degree?

        self.N_v = self.H_sparse.shape[1]  # How many variable nodes are present?
        self.N_c = self.H_sparse.shape[0]  # How many checknodes are present?

        self.d_c_max = self.degree_checknode_nr.max()
        self.d_v_max = self.degree_varnode_nr.max()

        self.codeword_len = self.H_sparse.shape[1]
        row_sum = self.H_sparse.sum(0).A[0, :]
        col_sum = self.H_sparse.sum(1).A[:, 0]
        d_v_dist_val = np.unique(row_sum)
        d_v_dist = np.zeros(int(d_v_dist_val.max()))

        for d_v in np.sort(d_v_dist_val).astype(np.int):
            d_v_dist[d_v - 1] = (row_sum == d_v).sum()
        d_v_dist = d_v_dist / d_v_dist.sum()

        d_c_dist_val = np.unique(col_sum)
        d_c_dist = np.zeros(int(d_c_dist_val.max()))

        for d_c in np.sort(d_c_dist_val).astype(np.int):
            d_c_dist[d_c - 1] = (col_sum == d_c).sum()

        d_c_dist = d_c_dist / d_c_dist.sum()
        nom = np.dot(d_v_dist, np.arange(d_v_dist_val.max()) + 1)
        den = np.dot(d_c_dist, np.arange(d_c_dist_val.max()) + 1)

        self.R_c = 1 - nom / den

    def alistToNumpy(self, lines):
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

    def load_sparse_csr(self, filename):
        loader = np.load(filename)
        return sci.sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                              shape=loader['shape'])

    def load_check_mat(self, filename):
        if filename.endswith('.npy') or filename.endswith('.npz'):
            if filename.endswith('.npy'):
                H = np.load(filename)
                H_sparse = sci.sparse.csr_matrix(H)
            else:
                H_sparse = self.load_sparse_csr(filename)
        else:
            arrays = [np.array(list(map(int, line.split()))) for line in open(filename)]
            H = self.alistToNumpy(arrays)
            H_sparse = sci.sparse.csr_matrix(H)
        return H_sparse

    def transmit(self):

        uncoded_msgs = np.random.randint(0,2, (self.data_len, self.msg_at_time))

        #uncoded_msgs = np.zeros( (self.data_len, self.msg_at_time) )
        encoded_msgs = np.zeros((self.codeword_len, self.msg_at_time))


        for i in range(self.msg_at_time):

            encoded_msgs[:, i]=self.encoder.encode_c(uncoded_msgs[:, i])

        self.last_transmitted_bits = uncoded_msgs

        data = self.BPSK_mapping(encoded_msgs)

        return data

    def BPSK_mapping(self, X):

        data = np.ones((self.codeword_len, self.msg_at_time))
        data[X == 1] = -1

        return data

class LDPC_QAM_Transmitter(LDPC_BPSK_Transmitter):
    def __init__(self, filename_H_, encoding_table, sqrt_M, msg_at_time=1):

        LDPC_BPSK_Transmitter.__init__(self,filename_H_, msg_at_time)

        self.encoding_table = encoding_table
        self.sqrt_M = sqrt_M
        self.num_bits = int(np.log2(sqrt_M) * 2)

        self.amplitude_values = np.zeros(self.sqrt_M)
        self.determine_amplitudes_for_encoding_values()

    def transmit(self):
        uncoded_msgs = np.random.randint(0, 2, (self.data_len, self.msg_at_time))

        encoded_msgs = np.zeros((self.codeword_len, self.msg_at_time))
        data = np.zeros((int(self.codeword_len/self.num_bits), self.msg_at_time),dtype=complex)

        for i in range(self.msg_at_time):
            encoded_msgs[:, i] = self.encoder.encode_c(uncoded_msgs[:, i])
            data[:, i] = self.QAM_mapping(encoded_msgs[:,i])[:,0]

        self.last_transmitted_bits = uncoded_msgs

        return data

    def QAM_mapping(self, X):

        data_real=np.reshape(self.amplitude_values[((np.reshape(X.T,(-1, self.num_bits )) [:,:int(self.num_bits/2)] *
                                2**np.arange( self.num_bits/2 )[::-1]).sum(1)).astype(np.int)],(-1,1))

        data_imag=np.reshape(self.amplitude_values[((np.reshape(X.T,(-1, self.num_bits )) [:,int(self.num_bits/2):] *
                                2**np.arange( self.num_bits/2 )[::-1]).sum(1)).astype(np.int)],(-1,1))

        data = (data_real + 1j*data_imag) * self.d_min/2
        return data

    def determine_amplitudes_for_encoding_values(self):
        natural_values = ((self.encoding_table * 2**np.arange( self.num_bits/2 )[::-1]).sum(1)).astype(np.int)

        self.amplitude_values[natural_values] = np.arange(-self.sqrt_M+1,self.sqrt_M,2)
        self.d_min = np.sqrt(6/ (self.sqrt_M**2 -1) )

class LDPC_MPSK_Transmitter(LDPC_BPSK_Transmitter):
    def __init__(self, filename_H_, encoding_table, M, msg_at_time=1):
        LDPC_BPSK_Transmitter.__init__(self, filename_H_, msg_at_time)

        self.encoding_table = encoding_table
        self.M = M
        self.num_bits = int(np.log2(M))

        self.phase_values = np.zeros(self.M, dtype=complex)
        self.determine_phase_for_encoding_values()

    def transmit(self):
        uncoded_msgs = np.random.randint(0, 2, (self.data_len, self.msg_at_time))
        #uncoded_msgs = np.zeros((self.data_len, self.msg_at_time))

        encoded_msgs = np.zeros((self.codeword_len, self.msg_at_time))
        data = np.zeros((int(self.codeword_len / self.num_bits), self.msg_at_time), dtype=complex)

        for i in range(self.msg_at_time):
            encoded_msgs[:, i] = self.encoder.encode_c(uncoded_msgs[:, i])
            data[:, i] = self.MPSK_mapping(encoded_msgs[:, i])[:, 0]

        self.last_transmitted_bits = uncoded_msgs

        return data

    def MPSK_mapping(self, X):
        data = np.reshape(
            self.phase_values[((np.reshape(X.T, (-1, self.num_bits))[:, :int(self.num_bits)] *
                                    2 ** np.arange(self.num_bits)[::-1]).sum(1)).astype(np.int)], (-1, 1))
        #print(np.angle(data,deg=True))
        return data

    def determine_phase_for_encoding_values(self):
        natural_values = ((self.encoding_table * 2 ** np.arange(self.num_bits)[::-1]).sum(1)).astype(np.int)

        angle_ind = np.arange(self.M)
        angles = 2 * np.pi/self.M * angle_ind
        self.phase_values[natural_values] = 1 * np.exp(1j * angles)
