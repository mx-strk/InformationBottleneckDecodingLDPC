import matplotlib.pyplot as plt
import numpy as np
from information_bottleneck.information_bottleneck_algorithms.lin_sym_sIB_class import lin_sym_sIB
from information_bottleneck.tools.inf_theory_tools import mutual_information as mutual_inf

__author__ = "Maximilian Stark"
__copyright__ = "05.07.2016, Institute of Communications, University of Technology Hamburg"
__version__ = "1.0"
__email__ = "maximilian.stark@tuhh.de"
__status__ = "Production"
__name__ = "Discrete Density Evolution"
__doc__ = "This code implements the discrete density evolution algorithm presented in " \
          "J. Lewandowsky and G. Bauch, 'Trellis based node operations for LDPC decoders " \
          "from the Information Bottleneck method,' (ICSPCS), 2015 , pp. 1-10."


class Discrete_Density_Evolution_class:
    """This class enforces symmetric Trellis diagram by using the lin_sym_sIB class. See the documentation for further
    information about this Information Bottleneck algorithm
    Attributes:
    input parameter
        p_x_and_t_input             initial pdf p(x,t)
        cardinality_T_decoder_ops   message alphabet cardinality of the passed decoder messages
        d_v                         variable node degree
        d_c                         check node degree
        i_max                       number of iterations
    IB related parameters
        cardinality_T               number of clusters
        beta                        set to inf, due to deterministic mapping
        eps
        nror                        Number Of Runs, namely number of runs with different initial clusterings
    discrete DE parameters

    """
    PROBABILITY_MIN_JOINT_PDF = 1e-15
    PROBABILITY_MAX_JOINT_PDF = 0.5-1e-15

    def __init__(self, p_x_and_t_input_, cardinality_T_decoder_ops_, d_v_, d_c_, i_max_, nror_):
        # initialize parameters
        self.p_x_and_t_channel_init = p_x_and_t_input_
        #determine cardinalities
        self.cardinality_T_channel = self.p_x_and_t_channel_init.shape[0]
        self.cardinality_T_decoder_ops = cardinality_T_decoder_ops_

        # code related parameters
        self.d_v = d_v_
        self.d_c = d_c_

        # discrete DE paramters
        self.i_max = i_max_

        # IB parameters
        self.nror = nror_

        # generate look up tables (LUT) for all input combinations. Basically this is a linear remapping from a 2D space
        # onto a 1D vector. Due to the possible mismatch cardinality_T_channel != cardinality_T_decoder_ops, a specific
        # loop up table is generated for this case, since the number of possible input combination of a partial node
        # operation could vary.


        ###### ------ CHECK NODES Preallocation START ------ ######
        # As explained earlier due to the possible mismatch of cardinality_T_channel and cardinality_T_decoder_ops the
        # first iteration has to be treated separately. In this first iteration the first partial node operation at the
        # check node we have incoming messages from the channel taking cardinality_T_channel different values. The next
        # partial node operation gets an input vector containing the result of the first partial node operation being
        # cardinality cardinality_T_decoder_ops and a second input value from a variable node with cardinality_T_channel.
        # In the next iteration all incoming and outgoing messages can take only cardinality_T_decoder_ops values.

        # Preallocate LUTs

        # Case iter = 0
        self.all_two_input_combinations_first_partial_op_check_first_iter = np.hstack((
        np.kron(np.arange(self.cardinality_T_channel)[:, np.newaxis], np.ones([self.cardinality_T_channel, 1])),
        np.tile(np.arange(self.cardinality_T_channel)[:, np.newaxis], (self.cardinality_T_channel, 1)) ))

        # Case iter = 0, channel and cardinality_T_decoder_ops input
        self.all_two_input_combinations_other_partial_ops_check_first_iter = np.hstack((
        np.kron(np.arange(self.cardinality_T_decoder_ops)[:, np.newaxis], np.ones([self.cardinality_T_channel, 1])),
        np.tile(np.arange(self.cardinality_T_channel)[:, np.newaxis], (self.cardinality_T_decoder_ops, 1))))

        # Case iter >0, only cardinality_T_decoder_ops
        self.all_two_input_combinations_other_partial_ops_check = np.hstack((
        np.kron(np.arange(self.cardinality_T_decoder_ops)[:, np.newaxis], np.ones([self.cardinality_T_decoder_ops, 1])),
        np.tile(np.arange(self.cardinality_T_decoder_ops)[:, np.newaxis], (self.cardinality_T_decoder_ops, 1))))

        self.p_t_l_c_given_vec_y_l_c_collection = np.empty((self.i_max, self.d_c-2), dtype=object)
        self.p_x_l_c_given_t_l_c_collection = np.empty((self.i_max, self.d_c-2), dtype=object)
        self.sorted_vec_y_l_c_collection = np.empty((self.i_max, self.d_c-2), dtype=object)
        self.p_t_l_c_collection = np.empty((self.i_max, self.d_c-2), dtype=object)


        self.Trellis_checknodevector_a = \
            np.zeros(self.cardinality_T_channel**2 \
                     + (self.d_c-3)*self.cardinality_T_decoder_ops*self.cardinality_T_channel \
                     + (self.i_max-1)*self.cardinality_T_decoder_ops**2*(self.d_c-2))

        ###### ------ CHECK NODES Preallocation End ------ ######

        ###### ------ VARIABLE NODES Preallocation Start ------ ######

        # First message at the variable nodes is the quantized channel output. Consequently, this message can take
        # cardinality_T_channel values. The other incomming message are from check nodes and therefore are limited to
        # cardinality_T_decoder_ops values.

        self.all_two_input_combinations_first_partial_ops_var = np.hstack((
            np.kron(np.arange(self.cardinality_T_channel)[:, np.newaxis], np.ones([self.cardinality_T_decoder_ops, 1])),
            np.tile(np.arange(self.cardinality_T_decoder_ops)[:, np.newaxis], (self.cardinality_T_channel, 1))))


        self.all_two_input_combinations_other_partial_ops_var = np.hstack((
            np.kron(np.arange(self.cardinality_T_decoder_ops)[:, np.newaxis], np.ones([self.cardinality_T_decoder_ops, 1])),
            np.tile(np.arange(self.cardinality_T_decoder_ops)[:, np.newaxis], (self.cardinality_T_decoder_ops, 1))))

        self.p_t_l_v_given_vec_y_l_v_collection = np.empty((self.i_max, self.d_v), dtype=object)
        self.p_x_l_v_given_t_l_v_collection = np.empty((self.i_max, self.d_v), dtype=object)
        self.sorted_vec_y_l_v_collection = np.empty((self.i_max, self.d_v), dtype=object)
        self.p_t_l_v_collection = np.empty((self.i_max, self.d_v), dtype=object)


        self.Trellis_varnodevector_a = \
            np.zeros( (self.i_max) * (self.cardinality_T_channel * self.cardinality_T_decoder_ops
                                      + (self.d_v - 1) * self.cardinality_T_decoder_ops ** 2))

        ###### ------ VARIABLE NODES Preallocation END ------ ######

        # preallocate vectors for mutual information results
        self.MI_T_dvm1_v_X_dvm1_v = np.zeros(self.i_max)
        self.MI_Y_dvm1_v_X_dvm1_v = np.zeros(self.i_max)
        self.mutual_inf_gain_matrix = np.zeros((self.i_max,self.d_v) )

    def check_node_density_evolution(self, iteration, p_x_lminus1_c_and_t_lminus1_c, p_b_lplus1_c_and_y_lplus1_c):
        print("check_node_density_evolution"+str(iteration))
        for w in range(self.d_c - 2):
            p_x_l_c_and_vec_y_l_c_lin = self.checknode_in_joint_pdf_y_lin(p_x_lminus1_c_and_t_lminus1_c,
                                                                          p_b_lplus1_c_and_y_lplus1_c)

            # run Information Bottleneck algorithm on p_x_l_c_and_vec_y_l_c_lin
            IB_instance = lin_sym_sIB(self.numerical_quard(p_x_l_c_and_vec_y_l_c_lin),
                                      self.cardinality_T_decoder_ops,
                                      self.nror)

            IB_instance.run_IB_algo()

            # get results and feed back
            p_t_l_c_given_vec_y_l_c, p_x_lminus1_c_given_t_lminus1_c, p_t_lm1_c = IB_instance.get_results()
            p_x_lminus1_c_and_t_lminus1_c = p_x_lminus1_c_given_t_lminus1_c * p_t_lm1_c[:,np.newaxis]


            IB_instance.display_MIs(short=True)

            # save message mappings for the check nodes explicitly in  cell array
            self.p_t_l_c_given_vec_y_l_c_collection[iteration, w] = p_t_l_c_given_vec_y_l_c
            self.p_x_l_c_given_t_l_c_collection[iteration, w] = p_x_lminus1_c_given_t_lminus1_c
            self.p_t_l_c_collection[iteration, w] = p_t_lm1_c

            if iteration == 0 and w == 0:
                self.sorted_vec_y_l_c_collection[iteration, w] = \
                    self.all_two_input_combinations_first_partial_op_check_first_iter
            if iteration == 0 and w > 0:
                self.sorted_vec_y_l_c_collection[iteration, w] = \
                    self.all_two_input_combinations_other_partial_ops_check_first_iter
            if iteration > 0:

                self.sorted_vec_y_l_c_collection[iteration, w] = self.all_two_input_combinations_other_partial_ops_check

        return p_x_lminus1_c_and_t_lminus1_c

    def variable_node_density_evolution(self, iteration, p_x_lminus1_c_and_t_lminus1_c):
        print("variable_node_density_evolution number"+str(iteration))

        # The first message is from the channel and the second message is a message from the other nodes per definition
        p_b_lplus1_v_and_y_lplus1_v = self.p_x_and_t_channel_init # channel
        p_x_lminus1_v_and_t_lminus1_v = p_x_lminus1_c_and_t_lminus1_c # output of DE for the check nodes

        p_x_l_v_and_vec_y_l_v_lin = self.varnode_in_joint_pdf_y_lin(p_b_lplus1_v_and_y_lplus1_v,
                                                                    p_x_lminus1_v_and_t_lminus1_v)


        # run Information Bottleneck algorithm on p_x_l_v_and_vec_y_l_v_lin
        IB_instance = lin_sym_sIB(self.numerical_quard(p_x_l_v_and_vec_y_l_v_lin),
                                  self.cardinality_T_decoder_ops,
                                  self.nror)

        IB_instance.run_IB_algo()

        # get results and feed back
        p_t_l_v_given_vec_y_l_v, p_x_lminus1_v_given_t_lminus1_v, p_t_lm1_v = IB_instance.get_results()

        p_x_lminus1_v_and_t_lminus1_v = p_x_lminus1_v_given_t_lminus1_v * p_t_lm1_v[:, np.newaxis]

        IB_instance.display_MIs(short=True)
        self.mutual_inf_gain_matrix[iteration,0] = IB_instance.get_mutual_inf()[0]

        # save message mappings for the variable nodes explicitly in  cell array
        self.p_t_l_v_given_vec_y_l_v_collection[iteration, 0] = p_t_l_v_given_vec_y_l_v
        self.p_x_l_v_given_t_l_v_collection[iteration, 0] = p_x_lminus1_v_given_t_lminus1_v
        self.p_t_l_v_collection[iteration, 0] = p_t_lm1_v


        self.sorted_vec_y_l_v_collection[iteration, 0] = self.all_two_input_combinations_first_partial_ops_var

        # the rest of the message pdfs are all from check nodes
        p_b_lplus1_v_and_y_lplus1_v = p_x_lminus1_c_and_t_lminus1_c

        for w in range(1, self.d_v - 1):
            p_x_l_v_and_vec_y_l_v_lin = self.varnode_in_joint_pdf_y_lin(p_x_lminus1_v_and_t_lminus1_v,
                                                                        p_b_lplus1_v_and_y_lplus1_v)

            # run Information Bottleneck algorithm on p_x_l_v_and_vec_y_l_v_lin
            IB_instance = lin_sym_sIB(self.numerical_quard(p_x_l_v_and_vec_y_l_v_lin),
                                      self.cardinality_T_decoder_ops,
                                      self.nror)

            IB_instance.run_IB_algo()

            # get results and feed back
            p_t_l_v_given_vec_y_l_v, p_x_lminus1_v_given_t_lminus1_v, p_t_lm1_v = IB_instance.get_results()

            p_x_lminus1_v_and_t_lminus1_v = p_x_lminus1_v_given_t_lminus1_v * p_t_lm1_v[:, np.newaxis]

            IB_instance.display_MIs(short=True)
            self.mutual_inf_gain_matrix[iteration,w] = IB_instance.get_mutual_inf()[0] - self.mutual_inf_gain_matrix[iteration,:].sum()

            # save message mappings for the variable nodes explicitly in  cell array
            self.p_t_l_v_given_vec_y_l_v_collection[iteration, w] = p_t_l_v_given_vec_y_l_v
            self.p_x_l_v_given_t_l_v_collection[iteration, w] = p_x_lminus1_v_given_t_lminus1_v
            self.p_t_l_v_collection[iteration, w] = p_t_lm1_v

            self.sorted_vec_y_l_v_collection[iteration, w] = self.all_two_input_combinations_other_partial_ops_var

        de_varnode_out = p_x_lminus1_v_and_t_lminus1_v / p_x_lminus1_v_and_t_lminus1_v.sum()

        # and run the IB algorithm for the last time to create the message mappings of the decision mapping for the
        # variable nodes
        p_x_l_v_and_vec_y_l_v_lin = self.varnode_in_joint_pdf_y_lin(p_x_lminus1_v_and_t_lminus1_v,
                                                                  p_b_lplus1_v_and_y_lplus1_v)


        IB_instance = lin_sym_sIB(self.numerical_quard(p_x_l_v_and_vec_y_l_v_lin),
                                  self.cardinality_T_decoder_ops,
                                  self.nror)

        IB_instance.run_IB_algo()

        # get results and save
        p_t_l_v_given_vec_y_l_v, p_x_lminus1_v_given_t_lminus1_v, p_t_lm1_v = IB_instance.get_results()

        IB_instance.display_MIs(short=True)
        self.mutual_inf_gain_matrix[iteration,-1] = IB_instance.get_mutual_inf()[0] - \
                                                    self.mutual_inf_gain_matrix[iteration,:].sum()
        # and save the resulting message mapping
        self.p_t_l_v_given_vec_y_l_v_collection[iteration, -1] = p_t_l_v_given_vec_y_l_v
        self.p_x_l_v_given_t_l_v_collection[iteration, -1] = p_x_lminus1_v_given_t_lminus1_v
        self.p_t_l_v_collection[iteration, -1] = p_t_lm1_v

        # finally store I(T_(d_v - 1} ^ v, X{d_v - 1} ^ v)
        self.MI_T_dvm1_v_X_dvm1_v[iteration], self.MI_Y_dvm1_v_X_dvm1_v[iteration] = IB_instance.get_mutual_inf()

        self.sorted_vec_y_l_v_collection[iteration, -1] = self.all_two_input_combinations_other_partial_ops_var

        return de_varnode_out

    def run_discrete_Density_Evolution(self):
        """This function combines all subroutines to implement discrete density evolution"""

        #  in the first iteration all incoming messages are directly from the channel
        p_x_lminus1_c_and_t_lminus1_c = self.p_x_and_t_channel_init / self.p_x_and_t_channel_init.sum()
        p_b_lplus1_c_and_y_lplus1_c = self.p_x_and_t_channel_init / self.p_x_and_t_channel_init.sum()

        # create theIB_calc object(init values are arbitrary, in density evolution loop, the member function init()
        # is called to set up the object properly, but we do not have a standard constructor in this class )

        self.ext_mi_varnode_in_iter = np.empty(self.i_max+1)
        self.ext_mi_checknode_in_iter = np.empty(self.i_max)

        self.ext_mi_varnode_in_iter[0] = mutual_inf(self.p_x_and_t_channel_init)

        for i in range(self.i_max):

            # check node density evolution
            de_checknode_out = self.check_node_density_evolution(i, p_x_lminus1_c_and_t_lminus1_c, p_b_lplus1_c_and_y_lplus1_c)
            self.ext_mi_checknode_in_iter[i] = mutual_inf(de_checknode_out)

            # variable node density evolution
            de_varnode_out = self.variable_node_density_evolution(i, de_checknode_out)
            self.ext_mi_varnode_in_iter[i+1] = mutual_inf(de_varnode_out)

            # set feed back variables to the varnode density evolution
            # normalize for stability
            p_b_lplus1_c_and_y_lplus1_c = de_varnode_out
            p_x_lminus1_c_and_t_lminus1_c = de_varnode_out


        # calculate the Trellis vector vec_a for the checknodes for all partial steps in the first decoder iteration
        # first partial node operation:
        # @param iteration: denotes the iteration
        # @param l: indicates teh partial node operation index

        l=0
        self.Trellis_checknodevector_a[:(l+1)*self.cardinality_T_channel**2] = \
            np.argmax(self.p_t_l_c_given_vec_y_l_c_collection[0, l], axis=1).astype(int)

        # and now for the other partial node operations of the first iteration, where l>0
        for l in range(1, self.d_c-2):
            # the offset marks the jump in address for this iteration
            offset = 1*self.cardinality_T_channel**2+(l-1)*self.cardinality_T_channel*self.cardinality_T_decoder_ops

            # length denotes the number new samples, which are filled in Trellis_Trellis_checknodevector_a
            length = self.cardinality_T_channel * self.cardinality_T_decoder_ops

            self.Trellis_checknodevector_a[offset:offset+length] = \
                np.argmax(self.p_t_l_c_given_vec_y_l_c_collection[0, l], axis=1).astype(int)


        # calc for iterations > 0
        initial_offset = 1*self.cardinality_T_channel**2+(self.d_c-3)*self.cardinality_T_channel*self.cardinality_T_decoder_ops
        for iteration in range(1, self.i_max):
            iter_offset = (iteration-1) * (self.d_c-2) * self.cardinality_T_decoder_ops**2
            for l in range(self.d_c - 2):
                offset = initial_offset + iter_offset + l*self.cardinality_T_decoder_ops**2
                length = self.cardinality_T_decoder_ops**2

                self.Trellis_checknodevector_a[offset:offset + length] = \
                    np.argmax(self.p_t_l_c_given_vec_y_l_c_collection[iteration, l], axis=1).astype(int)


        # Trellis vector for variable nodes
        for iteration in range(self.i_max):
            l = 0
            iter_offset = iteration * (self.cardinality_T_channel * self.cardinality_T_decoder_ops +
                                      (self.d_v - 1) * self.cardinality_T_decoder_ops ** 2)

            length = self.p_t_l_v_given_vec_y_l_v_collection[iteration, l].shape[0]

            self.Trellis_varnodevector_a[iter_offset:iter_offset + length] = \
                np.argmax(self.p_t_l_v_given_vec_y_l_v_collection[iteration, l], axis=1).astype(int)

            offset_after_first_op = iter_offset+length

            for l in range(1, self.d_v):
                offset = offset_after_first_op + (l - 1) * (self.cardinality_T_decoder_ops ** 2)
                length = self.cardinality_T_decoder_ops ** 2
                self.Trellis_varnodevector_a[offset:offset+length] = \
                    np.argmax(self.p_t_l_v_given_vec_y_l_v_collection[iteration, l], axis=1).astype(int)

    def checknode_in_joint_pdf_y_lin(self, p_t_lminus1_c_and_x_lminus1_c, p_y_lplus1_c_and_b_lplus1_c ):
        """ This function calculates the joint pdf p(vec_y_l ^ c, x_l ^ c) that is needed for the density evolution. The
        result is  a matrix. Each row corresponds to a linearized vector vec_y_l ^ c

        Please note that according to the
        paper the vector vec_y_l ^ c = [t_{l - 1} ^ c, y_{l + 1} ^ c] and that the order is extremely important, because
        a linear coordinate is calculated from this vector by the rule
        vec_y_lin = | \mathcal {Y} ^ c | t_ {l - 1} ^ c + y_  {l + 1} ^ c
        Args:
            :param p_t_lminus1_c_and_x_lminus1_c:   the density p(t_lm1_c,x_lm1_c)
            :param p_y_lplus1_c_and_b_lplus1_c:   the density p(y_lp1_c,b_lp1_c)
        Return
            :return p_x_c_and_y_vec_c:
        """

        # determine the cardinality of the first input. Referring to the paper this is the message entering from the top
        cardinality_Y_i_first_in = p_t_lminus1_c_and_x_lminus1_c.shape[0]
        cardinality_Y_i_second_in = p_y_lplus1_c_and_b_lplus1_c.shape[0]
        cardinality_vec_y_l = cardinality_Y_i_first_in*cardinality_Y_i_second_in

        p_x_c_and_y_vec_c = np.zeros((cardinality_vec_y_l, 2))
        p_x_c_and_y_vec_c2 = np.zeros((cardinality_vec_y_l, 2))

        # In the following the sum is evaluated. There exist two case where x^c can be 0 or 1. Furthermore the \oplus
        # sum of b_0 and b_1 results in each of the previously mentioned cases, on two different ways. These are denoted
        # as part 1 and part 2. The sum of these parts equals the result for either x^c=0 or 1.

        y_lp1_vec = np.kron(np.arange(cardinality_Y_i_second_in)[:, np.newaxis], np.ones([cardinality_Y_i_first_in, 1])).astype(int)
        t_lm1_vec = np.tile(np.arange(cardinality_Y_i_first_in)[:, np.newaxis], (cardinality_Y_i_second_in,1))

        # case x ^ c = 0, where x ^ c = b_0 ^ c \oplus b_1 ^ c
        # case b_0 ^ c = 0 b_1 ^ c = 0 ,case b_0 ^ c = 1 b_1 ^ c = 1
        part10 = p_y_lplus1_c_and_b_lplus1_c[y_lp1_vec, :] * p_t_lminus1_c_and_x_lminus1_c[t_lm1_vec, :]

        p_x_c_and_y_vec_c[cardinality_Y_i_second_in * t_lm1_vec + y_lp1_vec, 0] = part10.sum(2) #+ part20

        # case x ^ c = 1, where x ^ c = b_0 ^ c \oplus b_1 ^ c
        # case b_0 ^ c = 0 b_1 ^ c = 1 , case b_0 ^ c = 1  b_1 ^ c = 0
        part11 = p_y_lplus1_c_and_b_lplus1_c[y_lp1_vec, :] * p_t_lminus1_c_and_x_lminus1_c[t_lm1_vec, ::-1]

        p_x_c_and_y_vec_c[cardinality_Y_i_second_in * t_lm1_vec + y_lp1_vec, 1] = part11.sum(2)

        return p_x_c_and_y_vec_c

    def varnode_in_joint_pdf_y_lin(self, p_t_lminus1_v_and_x_lminus1_v, p_y_lplus1_v_and_b_lplus1_v):
        """ This function calculates the joint pdf p(vec_y_l ^ v, x_l ^ v) that is needed for the density evolution. The
        result is  a matrix. Each row corresponds to a linearized vector vec_y_l ^ v

        Please note that according to the
        paper the vector vec_y_l ^ c = [t_{l - 1} ^ v, y_{l + 1} ^ v] and that the order is extremely important, because
        a linear coordinate is calculated from this vector by the rule
        vec_y_lin = | \mathcal {Y} ^ v | t_ {l - 1} ^ v + y_  {l + 1} ^ v
        Args:
            :param p_t_lminus1_v_and_x_lminus1_v:   the density p(t_lm1_v,x_lm1_v)
            :param p_y_lplus1_v_and_b_lplus1_v:   the density p(y_lp1_v,b_lp1_v)
        Return
            :return p_x_v_and_y_vec_v:
        """

        # determine the cardinality of the first input. Referring to the paper this is the message entering from the top
        cardinality_Y_i_first_in = p_t_lminus1_v_and_x_lminus1_v.shape[0]
        cardinality_Y_i_second_in = p_y_lplus1_v_and_b_lplus1_v.shape[0]
        cardinality_vec_y_l = cardinality_Y_i_first_in * cardinality_Y_i_second_in

        p_x_v_and_y_vec_v = np.zeros((cardinality_vec_y_l, 2))

        # In the following the sum is evaluated. There exist two case where x^c can be 0 or 1. Furthermore the \oplus
        # sum of b_0 and b_1 results in each of the previously mentioned cases, on two different ways. These are denoted
        # as part 1 and part 2. The sum of these parts equals the result for either x^c=0 or 1.

        y_lp1_vec = np.kron(np.arange(cardinality_Y_i_second_in)[:, np.newaxis],
                            np.ones([cardinality_Y_i_first_in, 1])).astype(int)
        t_lm1_vec = np.tile(np.arange(cardinality_Y_i_first_in)[:, np.newaxis], (cardinality_Y_i_second_in, 1))

        # case x ^ c = 0, where x ^ c = b_0 ^ c \oplus b_1 ^ c
        # case b_0 ^ c = 0 b_1 ^ c = 0 ,case b_0 ^ c = 1 b_1 ^ c = 1
        part10 = 2 * p_y_lplus1_v_and_b_lplus1_v[y_lp1_vec, 0] * p_t_lminus1_v_and_x_lminus1_v[t_lm1_vec, 0]

        p_x_v_and_y_vec_v[cardinality_Y_i_second_in * t_lm1_vec + y_lp1_vec, 0] = part10

        # case x ^ c = 1, where x ^ c = b_0 ^ c \oplus b_1 ^ c
        # case b_0 ^ c = 0 b_1 ^ c = 1 , case b_0 ^ c = 1  b_1 ^ c = 0
        part11 = 2 * p_y_lplus1_v_and_b_lplus1_v[y_lp1_vec, 1] * p_t_lminus1_v_and_x_lminus1_v[t_lm1_vec, 1]

        p_x_v_and_y_vec_v[cardinality_Y_i_second_in * t_lm1_vec + y_lp1_vec, 1] = part11

        return p_x_v_and_y_vec_v

    def numerical_quard(self, pdf):
        """Function to avoid numerical instabilities."""
        limited_pdf = pdf
        limited_pdf[limited_pdf <= self.PROBABILITY_MIN_JOINT_PDF] = self.PROBABILITY_MIN_JOINT_PDF
        limited_pdf[limited_pdf >= self.PROBABILITY_MAX_JOINT_PDF] = self.PROBABILITY_MAX_JOINT_PDF
        limited_pdf = limited_pdf / limited_pdf.sum()
        return limited_pdf

    def visualize_mi_evolution(self):
        plt.plot(self.MI_T_dvm1_v_X_dvm1_v)
        plt.show()

    

