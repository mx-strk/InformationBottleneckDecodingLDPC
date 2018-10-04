from information_bottleneck.information_bottleneck_algorithms.lin_sym_sIB_class import lin_sym_sIB
from information_bottleneck.tools.inf_theory_tools import mutual_information

from Discrete_LDPC_decoding.Discrete_Density_Evolution import Discrete_Density_Evolution_class
from Discrete_LDPC_decoding.Information_Matching import *

__author__ = "Maximilian Stark"
__copyright__ = "05.07.2016, Institute of Communications, University of Technology Hamburg"
__version__ = "1.0"
__email__ = "maximilian.stark@tuhh.de"
__status__ = "Production"
__name__ = "Discrete Density Evolution for irregular Codes"
__doc__ = """This module holds a class which implements an adapted version of discrete density evolution which works
            for irregular LDPC codes."""

class Discrete_Density_Evolution_class_irregular(Discrete_Density_Evolution_class):
    """This class enforces symmetric Trellis diagram by using the lin_sym_sIB class. See the documentation for further
    information about this Information Bottleneck algorithm
    Args:
    input parameter
        p_x_and_t_input             initial pdf p(x,t)
        cardinality_T_decoder_ops   message alphabet cardinality of the passed decoder messages
        lambda_vec                  variable node degree distribution from an edge perspective
        rho_vec                     check node degree distribution from an edge perspective
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

    def __init__(self, p_x_and_t_input_, cardinality_T_decoder_ops_, lambda_vec_, rho_vec_, i_max_, nror_, match = True,theta = 0):
        self.lambda_vec = lambda_vec_
        self.rho_vec = rho_vec_
        self.match = match
        self.theta = theta
        self.d_c_max = self.rho_vec.shape[0]
        self.d_v_max = self.lambda_vec.shape[0]

        Discrete_Density_Evolution_class.__init__(self,p_x_and_t_input_,cardinality_T_decoder_ops_,
                                                  self.d_v_max,self.d_c_max, i_max_, nror_)

        # the matching collection contains the results of new information matching step
        self.matching_c_collection = np.zeros((self.i_max, self.d_c_max,cardinality_T_decoder_ops_), dtype=np.int)
        self.matching_v_collection = np.zeros((self.i_max, self.d_v_max,cardinality_T_decoder_ops_), dtype=np.int)

        self.matching_vector_checknode = np.array([])
        self.matching_vector_varnode = np.array([])
        self.cost_vector = np.empty(self.i_max)
        self.cost_vector_no = np.empty(self.i_max)
        self.ratio = np.empty(self.i_max)


    def check_node_density_evolution(self, iteration, p_x_lminus1_c_and_t_lminus1_c, p_b_lplus1_c_and_y_lplus1_c):
        print("check_node_density_evolution"+str(iteration))
        for w in range(self.d_c_max - 2):
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

        #######################################################################################
        p_x_lminus1_c_and_t_lminus1_c_weighted = np.zeros_like(p_x_lminus1_c_and_t_lminus1_c)
        p_x_lminus1_c_and_t_lminus1_c_weighted_no_match = np.zeros_like(p_x_lminus1_c_and_t_lminus1_c)

        max_abs = np.zeros(self.d_c_max)
        for rho in range(self.d_c_max):
            if self.rho_vec[rho] > 0:
                max_abs[rho] = (np.abs(np.log(self.p_x_l_c_given_t_l_c_collection[iteration, rho - 2][:, 0] /
                                                   self.p_x_l_c_given_t_l_c_collection[iteration, rho - 2][:,
                                                   1])) * 1 / 16).sum()


        lowest_degree_index =  np.argmax(max_abs) #- 1 #np.argmax(self.rho_vec)   # degree 7 corresponds to d_c_max-4
        p_x_given_t_lowest_degree = self.p_x_l_c_given_t_l_c_collection[iteration, lowest_degree_index-2]
        p_x_and_t_lowest_degree = p_x_given_t_lowest_degree * self.p_t_l_c_collection[iteration, lowest_degree_index-2][:,np.newaxis]

        theta = self.theta
        identity_transform = np.arange(self.cardinality_T_decoder_ops)

        for rho in range(self.d_c_max):
            if self.rho_vec[rho] > 0:
                cur_p_x_given_t = self.p_x_l_c_given_t_l_c_collection[iteration, rho-2]
                cur_pt =  self.p_t_l_c_collection[iteration, rho-2]
                cur_p_x_and_t = cur_p_x_given_t*cur_pt[:,np.newaxis]
                if rho != lowest_degree_index:
                    p_x_given_z_stars, p_x_and_z_stars, p_star_z, z_stars, _ = information_matching_v2(
                         self.cardinality_T_decoder_ops,
                         cur_p_x_and_t,
                         p_x_and_t_lowest_degree, theta)

                    self.matching_c_collection[iteration,rho,:] = z_stars
                else:
                    p_x_and_z_stars = cur_p_x_and_t
                    self.matching_c_collection[iteration, rho, :] = identity_transform
                    # not really needed, but better for generality

                p_x_lminus1_c_and_t_lminus1_c_weighted += self.rho_vec[rho] * p_x_and_z_stars
                p_x_lminus1_c_and_t_lminus1_c_weighted_no_match += self.rho_vec[rho] * cur_p_x_and_t

            else:
                pass

        print(mutual_information(p_x_lminus1_c_and_t_lminus1_c_weighted))
        print(mutual_information(p_x_lminus1_c_and_t_lminus1_c_weighted_no_match))

        ##########################################################################
        if self.match:
            return p_x_lminus1_c_and_t_lminus1_c_weighted
        else:
            return p_x_lminus1_c_and_t_lminus1_c_weighted_no_match

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

        for w in range(1, self.d_v_max - 1):
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

        ##################################################################

        p_x_lminus1_v_and_t_lminus1_v_weighted = np.zeros_like(p_x_lminus1_v_and_t_lminus1_v)
        p_x_lminus1_v_and_t_lminus1_v_weighted_no_match = np.zeros_like(p_x_lminus1_v_and_t_lminus1_v)

        max_abs = np.zeros(self.d_v_max)
        for lambda_i in range(1,self.d_v_max): #start with 1 because varnode degree 1 will only forward channel message
            if self.lambda_vec[lambda_i] > 0:
                max_abs[lambda_i] = (np.abs(np.log(self.p_x_l_v_given_t_l_v_collection[iteration, lambda_i-1][:, 0] /
                                self.p_x_l_v_given_t_l_v_collection[iteration, lambda_i-1][:, 1])) * 1 / self.cardinality_T_decoder_ops).sum()

                p_x_given_t_highest_test = self.p_x_l_v_given_t_l_v_collection[iteration, lambda_i - 1]
                p_t_test = self.p_t_l_v_collection[iteration, lambda_i - 1][:,np.newaxis]

                #max_abs[lambda_i] = mutual_information(p_x_given_t_highest_test*p_t_test)

        matching_degree = np.argmax(max_abs)-1
        p_x_given_t_highest_degree =  self.p_x_l_v_given_t_l_v_collection[iteration, matching_degree]
        p_x_and_t_highest_degree = p_x_given_t_highest_degree * self.p_t_l_v_collection[iteration, matching_degree][:,np.newaxis]

        theta = self.theta
        identity_transform = np.arange(self.cardinality_T_decoder_ops)


        #################### Cascade ################### NEW #########################
        p_x_and_t_desired = p_x_and_t_highest_degree.copy()
        nom =  self.lambda_vec[matching_degree+1] * p_x_and_t_highest_degree
        den = self.lambda_vec[matching_degree+1]

        p_x_given_z_stars_mat = np.empty(self.d_v_max, dtype=object)
        p_star_z_mat = np.empty(self.d_v_max, dtype=object)

        for lambda_i in range(1,self.d_v_max): #start with 1 because varnode degree 1 will only forward channel message
            if self.lambda_vec[lambda_i] > 0:
                cur_p_x_given_t = self.p_x_l_v_given_t_l_v_collection[iteration, lambda_i-1]
                cur_pt =  self.p_t_l_v_collection[iteration, lambda_i-1]
                cur_p_x_and_t = cur_p_x_given_t*cur_pt[:,np.newaxis]
                if lambda_i != matching_degree+1:
                    p_x_given_z_stars, p_x_and_z_stars, p_star_z, z_stars, _ = information_matching_v2(self.cardinality_T_decoder_ops,
                                                                                                 cur_p_x_and_t,
                                                                                                 p_x_and_t_desired)
                    p_x_given_z_stars_mat[lambda_i] = p_x_given_z_stars
                    p_star_z_mat[lambda_i] = p_star_z

                    self.matching_v_collection[iteration, lambda_i, :] = z_stars

                    nom += self.lambda_vec[lambda_i] * p_x_and_z_stars
                    den += self.lambda_vec[lambda_i]
                    p_x_and_t_desired = nom / den
                else:
                     p_x_and_z_stars = cur_p_x_and_t
                     self.matching_v_collection[iteration, lambda_i, :] = identity_transform



                p_x_lminus1_v_and_t_lminus1_v_weighted += self.lambda_vec[lambda_i] * p_x_and_z_stars
                p_x_lminus1_v_and_t_lminus1_v_weighted_no_match += self.lambda_vec[lambda_i] * cur_p_x_and_t

        # match again against result
        p_x_given_z1_stars, p_x_and_z1_stars, p_star_z1, z1_stars, _ = information_matching_v2(self.cardinality_T_decoder_ops,
                                                                                                p_x_and_t_highest_degree,
                                                                                                p_x_lminus1_v_and_t_lminus1_v_weighted)

        self.matching_v_collection[iteration, matching_degree, :] = z1_stars

        p_x_given_z_stars_mat[matching_degree+1] = p_x_given_z1_stars
        p_star_z_mat[matching_degree+1] = p_star_z1

        # remove old entry from highest match, rememeber matching degree is already -1
        #
        p_x_lminus1_v_and_t_lminus1_v_weighted -= self.lambda_vec[matching_degree+1] * p_x_and_t_highest_degree
        p_x_lminus1_v_and_t_lminus1_v_weighted +=  self.lambda_vec[matching_degree+1] * p_x_and_z1_stars

        p_x_lminus1_v_given_t_lminus1_v_weighted = p_x_lminus1_v_and_t_lminus1_v_weighted/ \
                                                   p_x_lminus1_v_and_t_lminus1_v_weighted.sum(1)[:,np.newaxis]
        # calculate costs

        local_cost = np.zeros(self.d_v_max)
        global_costs = 0
        for lambda_i in range(1,self.d_v_max):
            if self.lambda_vec[lambda_i] > 0:
                local_cost[lambda_i] = np.dot( p_star_z_mat[lambda_i],
                                               kl_divergence(p_x_given_z_stars_mat[lambda_i], p_x_lminus1_v_given_t_lminus1_v_weighted))

                global_costs += self.lambda_vec[lambda_i]  * local_cost[lambda_i]

        p_x_lminus1_v_given_t_lminus1_v_weighted_no_match  = p_x_lminus1_v_and_t_lminus1_v_weighted_no_match / \
                                                           p_x_lminus1_v_and_t_lminus1_v_weighted_no_match.sum(1)[:,np.newaxis]

        local_cost_no = np.zeros(self.d_v_max)
        global_costs_no = 0
        for lambda_i in range(1,self.d_v_max):
            if self.lambda_vec[lambda_i] > 0:
                local_cost_no[lambda_i] = np.dot(p_star_z_mat[lambda_i],
                                              kl_divergence(p_x_given_z_stars_mat[lambda_i],
                                                            p_x_lminus1_v_given_t_lminus1_v_weighted_no_match))

                global_costs_no += self.lambda_vec[lambda_i] * local_cost_no[lambda_i]

        print("global=",'{:.2e}'.format( global_costs))
        print("local=", local_cost)
        self.cost_vector[iteration] = global_costs
        self.cost_vector_no[iteration] = global_costs_no
        self.ratio[iteration] = global_costs / global_costs_no
        print(mutual_information(p_x_lminus1_v_and_t_lminus1_v_weighted))
        ##################################################################

        de_varnode_out_no_match = p_x_lminus1_v_and_t_lminus1_v_weighted_no_match / p_x_lminus1_v_and_t_lminus1_v_weighted_no_match.sum()
        de_varnode_out = p_x_lminus1_v_and_t_lminus1_v_weighted / p_x_lminus1_v_and_t_lminus1_v_weighted.sum()

        print(mutual_information(de_varnode_out_no_match))

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

        self.MI_T_dvm1_v_X_dvm1_v[iteration] = mutual_information(de_varnode_out)
        self.MI_Y_dvm1_v_X_dvm1_v[iteration] = mutual_information(de_varnode_out_no_match)

        self.sorted_vec_y_l_v_collection[iteration, -1] = self.all_two_input_combinations_other_partial_ops_var
        if self.match:
            return de_varnode_out
        else:
            return de_varnode_out_no_match

    def run_discrete_Density_Evolution(self):
        """This function combines all subroutines to implement discrete density evolution"""

        #  in the first iteration all incoming messages are directly from the channel
        p_x_lminus1_c_and_t_lminus1_c = self.p_x_and_t_channel_init / self.p_x_and_t_channel_init.sum()
        p_b_lplus1_c_and_y_lplus1_c = self.p_x_and_t_channel_init / self.p_x_and_t_channel_init.sum()

        # create theIB_calc object(init values are arbitrary, in density evolution loop, the member function init()
        # is called to set up the object properly, but we do not have a standard constructor in this class )

        for i in range(self.i_max):

            # check node density evolution
            de_checknode_out = self.check_node_density_evolution(i, p_x_lminus1_c_and_t_lminus1_c, p_b_lplus1_c_and_y_lplus1_c)

            # variable node density evolution
            de_varnode_out = self.variable_node_density_evolution(i, de_checknode_out)

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
        for l in range(1, self.d_c_max-2):
            # the offset marks the jump in address for this iteration
            offset = 1*self.cardinality_T_channel**2+(l-1)*self.cardinality_T_channel*self.cardinality_T_decoder_ops

            # length denotes the number new samples, which are filled in Trellis_Trellis_checknodevector_a
            length = self.cardinality_T_channel * self.cardinality_T_decoder_ops

            self.Trellis_checknodevector_a[offset:offset+length] = \
                np.argmax(self.p_t_l_c_given_vec_y_l_c_collection[0, l], axis=1).astype(int)


        # calc for iterations > 0
        initial_offset = 1*self.cardinality_T_channel**2+(self.d_c_max-3)*self.cardinality_T_channel*self.cardinality_T_decoder_ops
        for iteration in range(1, self.i_max):
            iter_offset = (iteration-1) * (self.d_c_max-2) * self.cardinality_T_decoder_ops**2
            for l in range(self.d_c_max - 2):
                offset = initial_offset + iter_offset + l*self.cardinality_T_decoder_ops**2
                length = self.cardinality_T_decoder_ops**2

                self.Trellis_checknodevector_a[offset:offset + length] = \
                    np.argmax(self.p_t_l_c_given_vec_y_l_c_collection[iteration, l], axis=1).astype(int)


        # Trellis vector for variable nodes
        for iteration in range(self.i_max):
            l = 0
            iter_offset = iteration * (self.cardinality_T_channel * self.cardinality_T_decoder_ops +
                                      (self.d_v_max - 1) * self.cardinality_T_decoder_ops ** 2)

            length = self.p_t_l_v_given_vec_y_l_v_collection[iteration, l].shape[0]

            self.Trellis_varnodevector_a[iter_offset:iter_offset + length] = \
                np.argmax(self.p_t_l_v_given_vec_y_l_v_collection[iteration, l], axis=1).astype(int)

            offset_after_first_op = iter_offset+length

            for l in range(1, self.d_v_max):
                offset = offset_after_first_op + (l - 1) * (self.cardinality_T_decoder_ops ** 2)
                length = self.cardinality_T_decoder_ops ** 2
                self.Trellis_varnodevector_a[offset:offset+length] = \
                    np.argmax(self.p_t_l_v_given_vec_y_l_v_collection[iteration, l], axis=1).astype(int)

        # Generate matching vectors
        self.matching_vector_checknode = np.reshape(self.matching_c_collection,self.matching_c_collection.size)
        self.matching_vector_varnode = np.reshape(self.matching_v_collection, self.matching_v_collection.size)

    

