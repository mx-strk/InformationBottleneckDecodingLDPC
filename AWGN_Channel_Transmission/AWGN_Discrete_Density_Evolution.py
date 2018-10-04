import pickle

import scipy as sci

from AWGN_Channel_Transmission.AWGN_Quantizer_BPSK import AWGN_Channel_Quantizer as AWGN_Channel_Quantizer_BPSK
from AWGN_Channel_Transmission.AWGN_Quantizer_Mary import AwgnChannelQuantizer as AWGN_Channel_Quantizer_QAM
from AWGN_Channel_Transmission.AWGN_Quantizer_Mary import AwgnChannelQuantizer_MPSK as AWGN_Channel_Quantizer_MPSK
from Discrete_LDPC_decoding.Discrete_Density_Evolution import Discrete_Density_Evolution_class as discrete_DE
from Discrete_LDPC_decoding.Discrete_Density_Evolution_irreg import \
    Discrete_Density_Evolution_class_irregular as discrete_DE_irregular
from Discrete_LDPC_decoding.Information_Matching import *

__author__ = "Maximilian Stark"
__copyright__ = "05.07.2016, Institute of Communications, University of Technology Hamburg"
__credits__ = ["Maximilian Stark"]
__version__ = "1.0"
__email__ = "maximilian.stark@tuhh.de"
__status__ = "Production"
__name__ = "Discrete Density Evolution for a given noise level of an AWGN channel"
__doc__ = """ This module contains classes which combine AWGN channel quantizers and discrete Density Evolution.
              They can be used to generate and save lookup tables for a certain design-Eb/N0. Different classes are
              found by inheritance of the base class, i.e. AWGN_Discrete_Density_Evolution_class to support different
              modulation schemes and irregular LDPC codes.
              """

class AWGN_Discrete_Density_Evolution_class:
    """ Generates a discrete LDPC decoder for a AWGN channel and a regular LDPC code for a certain design-Eb/N0.

    The assumed modulation is BPSK which is considered in the quantizer design.
    Attributes:
        sigma_n2: noise variance corresponding to the desired design-Eb/N0 of the decoder
        AD_max_abs: limits of the quantizer
        cardinality_Y_channel: number of steps used for the fine quantization of the input distribution of the quantizer
        cardinality_T_channel: cardinality of the compression variable representing the quantizer output

        cardinality_T_decoder_ops: cardinality of the compression variables inside the decoder

        d_c: check node degree
        d_v: variable node degree

        imax: maximum number of iterations
        nror: number of runs of the Information Bottleneck algorithm

        Trellis_checknodevector_a:  vectorized version of the trellis which holds the resulting outputs for a certain
                                    input and iteration at a check node
        Trellis_varnodevector_a:  vectorized version of the trellis which holds the resulting outputs for a certain
                                    input and iteration at a variable node
    """
    def __init__(self, sigma_n2_, AD_max_abs_,cardinality_Y_channel_, cardinality_T_channel_,
                 cardinality_T_decoder_ops_,d_v_, d_c_, i_max_, nror_):
        """Inits the AWGN_Discrete_Density_Evolution_class with the following arguments

        Args:
            sigma_n2_: noise variance corresponding to the desired design-Eb/N0 of the decoder
            AD_max_abs_: limits of the quantizer
            cardinality_Y_channel_: number of steps used for the fine quantization of the input distribution of the quantizer
            cardinality_T_channel_: cardinality of the compression variable representing the quantizer output

            cardinality_T_decoder_ops_: cardinality of the compression variables inside the decoder

            d_c_: check node degree
            d_v_: variable node degree

            i_max_: maximum number of iterations
            nror_: number of runs of the Information Bottleneck algorithm
        """
        # copy input arguments to class attributes
        self.sigma_n2 = sigma_n2_
        self.AD_max_abs = AD_max_abs_

        self.cardinality_Y_channel = cardinality_Y_channel_
        self.cardinality_T_channel = cardinality_T_channel_
        self.cardinality_T_decoder_ops = cardinality_T_decoder_ops_

        self.d_v = d_v_
        self.d_c = d_c_

        R_c = 1 - self.d_v / self.d_c
        if R_c > 0:
            self.EbN0 = -10 * np.log10(self.sigma_n2 * 2 * R_c)

        self.imax = i_max_
        self.nror = nror_

        self.build_quantizer()


        self.Trellis_checknodevector_a = 0
        self.Trellis_varnodevector_a = 0

    def set_code_parameters(self):
        """Analysis of the given parity check matrix.
        Determines node-degree distribution, edge-degree distribution and code rate
        """
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

        self.lambda_vec = convert_node_to_edge_degree(d_v_dist)
        self.rho_vec = convert_node_to_edge_degree(d_c_dist)

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
        """Performs loading of a sparse parity check matrix which is stored in a *.npy file."""
        loader = np.load(filename)
        return sci.sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                                     shape=loader['shape'])

    def load_check_mat(self, filename):
        """Performs loading of a predefined parity check matrix."""
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

    def build_quantizer(self):
        """Generates instance of a quantizer for BPSK and an AWGN channel for the given characteristics."""
        quanti = AWGN_Channel_Quantizer_BPSK(self.sigma_n2,self.AD_max_abs,self.cardinality_T_channel,self.cardinality_Y_channel)
        self.p_x_and_t_input = quanti.p_x_and_t

    def run_discrete_density_evolution(self):
        """Performs the discrete density evolution using the input distributions obtained from the quantizer.
        The resulting trellis diagram is stored in a vector that can be used for the real decoder later.
        """
        DDE_inst = discrete_DE(self.p_x_and_t_input, self.cardinality_T_decoder_ops,
                               self.d_v, self.d_c, self.imax, self.nror)

        DDE_inst.run_discrete_Density_Evolution()

        self.Trellis_checknodevector_a = DDE_inst.Trellis_checknodevector_a
        self.Trellis_varnodevector_a = DDE_inst.Trellis_varnodevector_a

        self.DDE_inst_data = DDE_inst.__dict__

    def save_config(self,text=''):
        """Saves the instance."""
        #timestr = time.strftime("%Y%m%d-%H%M%S")
        timestr =''

        output = open('decoder_config_EbN0_gen_' + str(self.EbN0) + '_' + str(
            self.cardinality_T_decoder_ops) + timestr + text + '.pkl', 'wb')

        # Pickle dictionary using protocol -1.
        pickle.dump(self.__dict__, output, protocol=-1)

class AWGN_Discrete_Density_Evolution_class_irregular(AWGN_Discrete_Density_Evolution_class):
    """Inherited from base class AWGN_Discrete_Density_Evolution_class.

    Generalization for irregular codes. Thus a new discrete density evolution schemes is used.

    Attributes:
        filename_H: filename of the parity check matrix of the considered code
        H_sparse: corresponding parity check matrix for the considered code

        matching_vector_varnode: holds the deterministic mapping found when performing message alginment for a
                                 variable noce
        matching_vector_checknode: holds the deterministic mapping found when performing message alginment for a
                                   check node

        match: boolean indicating if alignment should be used or not
    """
    def __init__(self, sigma_n2_, AD_max_abs_,cardinality_Y_channel_, cardinality_T_channel_,
                 cardinality_T_decoder_ops_,filename_H_, i_max_, nror_,match=True):
        """Inits AWGN_Discrete_Density_Evolution_class_irregular class.

        Args:
            filename_H_: filename of parity check matrix
            match: boolean indicating if alignment should be used or not
        """
        self.filename_H = filename_H_
        self.H_sparse = self.load_check_mat(self.filename_H)
        self.set_code_parameters()
        AWGN_Discrete_Density_Evolution_class.__init__(self, sigma_n2_,
                                                       AD_max_abs_, cardinality_Y_channel_, cardinality_T_channel_,
                                                       cardinality_T_decoder_ops_, self.d_v_max, self.d_c_max ,
                                                       i_max_, nror_)

        self.EbN0 = -10 * np.log10(self.sigma_n2 * 2 * self.R_c)
        self.match = match

    def run_discrete_density_evolution(self):
        """Runs discrete density evolution for irregular codes

        Returns also two matching vectors describing the deterministic transformation obtained by message alingment
        """
        DDE_inst = discrete_DE_irregular(self.p_x_and_t_input, self.cardinality_T_decoder_ops,
                               self.lambda_vec, self.rho_vec, self.imax, self.nror, match=self.match)

        DDE_inst.run_discrete_Density_Evolution()

        self.Trellis_checknodevector_a = DDE_inst.Trellis_checknodevector_a
        self.Trellis_varnodevector_a = DDE_inst.Trellis_varnodevector_a
        self.matching_vector_checknode = DDE_inst.matching_vector_checknode
        self.matching_vector_varnode = DDE_inst.matching_vector_varnode


        self.DDE_inst_data = DDE_inst.__dict__

class AWGN_Discrete_Density_Evolution_class_irregular_QAM(AWGN_Discrete_Density_Evolution_class_irregular):
    """Inherited from base class AWGN_Discrete_Density_Evolution_class_irregular.

    Adapted version for an irregular LDPC code, an AWGN channel and QAM modulation.
    Thus, the quantizer is replaced.
    """

    def __init__(self, sigma_n2_,
                 AD_max_abs_,
                 cardinality_Y_channel_,
                 cardinality_T_channel_,
                 cardinality_T_decoder_ops_,
                 filename_H_,
                 encoding_table,
                 sqrt_M,
                 i_max_,
                 nror_,
                 match=True):
        """Inits AWGN_Discrete_Density_Evolution_class_irregular_QAM class"""

        self.encoding_table = encoding_table
        self.sqrt_M = sqrt_M
        self.num_bits = int(np.log2(sqrt_M) * 2)

        AWGN_Discrete_Density_Evolution_class_irregular.__init__(self,sigma_n2_,
                                                                 AD_max_abs_,
                                                                 cardinality_Y_channel_,
                                                                 cardinality_T_channel_,
                                                                 cardinality_T_decoder_ops_,
                                                                 filename_H_,
                                                                 i_max_,
                                                                 nror_,
                                                                 match)



        self.EbN0 = -10 * np.log10(self.sigma_n2 * self.R_c * self.num_bits)

    def build_quantizer(self):
        """Generates a quantizer of the AWGN channel output where the used modulation scheme is QAM."""

        quanti = AWGN_Channel_Quantizer_QAM(self.sigma_n2,
                                                 self.AD_max_abs,
                                                 self.cardinality_T_channel,
                                                 self.cardinality_Y_channel,
                                                 self.encoding_table,
                                                 sqrt_M=self.sqrt_M )
        self.p_x_and_t_input = quanti.p_b_and_u_matched

class AWGN_Discrete_Density_Evolution_class_QAM(AWGN_Discrete_Density_Evolution_class):
    """Inherited from base class AWGN_Discrete_Density_Evolution_class.

        Adapted version for an regular LDPC code, an AWGN channel and QAM modulation.
        Thus, the quantizer is replaced.
    """

    def __init__(self, sigma_n2_, AD_max_abs_,cardinality_Y_channel_, cardinality_T_channel_,
                 cardinality_T_decoder_ops_,filename_H_,
                 encoding_table,sqrt_M,i_max_, nror_,match=True):

        self.match = match
        self.filename_H = filename_H_
        self.H_sparse = self.load_check_mat(self.filename_H)
        self.set_code_parameters()
        self.encoding_table = encoding_table
        self.sqrt_M = sqrt_M
        self.num_bits = int(np.log2(sqrt_M) * 2)

        AWGN_Discrete_Density_Evolution_class.__init__(self, sigma_n2_,
                                                       AD_max_abs_, cardinality_Y_channel_, cardinality_T_channel_,
                                                       cardinality_T_decoder_ops_, self.d_v_max, self.d_c_max,
                                                       i_max_, nror_)



        self.EbN0 = -10 * np.log10(self.sigma_n2 * self.R_c * self.num_bits)
        pass


    def build_quantizer(self):
        """Generates a quantizer of the AWGN channel output where the used modulation scheme is QAM."""

        quanti = AWGN_Channel_Quantizer_QAM(self.sigma_n2,
                                                 self.AD_max_abs,
                                                 self.cardinality_T_channel,
                                                 self.cardinality_Y_channel,
                                                 self.encoding_table,
                                                 sqrt_M=self.sqrt_M )
        if self.match:
            self.p_x_and_t_input = quanti.p_b_and_u_matched
        else:
            self.p_x_and_t_input = quanti.p_b_and_u_matched_no_match

class AWGN_Discrete_Density_Evolution_class_irregular_MPSK(AWGN_Discrete_Density_Evolution_class_irregular):
    """Inherited from base class AWGN_Discrete_Density_Evolution_class_irregular.

       Adapted version for an irregular LDPC code, an AWGN channel and MPSK modulation.
       Thus, the quantizer is replaced.
    """

    def __init__(self, sigma_n2_,
                 AD_max_abs_,
                 cardinality_Y_channel_,
                 cardinality_T_channel_,
                 cardinality_T_decoder_ops_,
                 filename_H_,
                 encoding_table,
                 M,
                 i_max_,
                 nror_,
                 match=True):

        self.encoding_table = encoding_table
        self.M = M
        self.num_bits = int(np.log2(M))

        self.filename_H = filename_H_
        self.H_sparse = self.load_check_mat(self.filename_H)
        self.set_code_parameters()

        AWGN_Discrete_Density_Evolution_class.__init__(self, sigma_n2_,
                                                       AD_max_abs_, cardinality_Y_channel_, cardinality_T_channel_,
                                                       cardinality_T_decoder_ops_, self.d_v_max, self.d_c_max,
                                                       i_max_, nror_)
        self.match = match
        self.EbN0 = -10 * np.log10(self.sigma_n2 * self.R_c * self.num_bits)

    def build_quantizer(self):
        """Generates a quantizer of the AWGN channel output where the used modulation scheme is 8 PSK."""

        quanti = AWGN_Channel_Quantizer_MPSK(self.sigma_n2,
                                            self.AD_max_abs,
                                            self.cardinality_T_channel,
                                            self.cardinality_Y_channel,
                                            self.encoding_table,
                                            M=self.M)

        self.p_x_and_t_input = quanti.p_b_and_u_matched