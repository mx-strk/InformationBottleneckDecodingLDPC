import os

import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import scipy as sci
from mako.template import Template
from pyopencl.reduction import get_sum_kernel

from Discrete_LDPC_decoding.discrete_LDPC_decoder import Discrete_LDPC_Decoder_class

__author__ = "Maximilian Stark"
__copyright__ = "05.07.2016, Institute of Communications, University of Technology Hamburg"
__credits__ = ["Maximilian Stark", "Jan Lewandowsky"]
__version__ = "1.0"
__email__ = "maximilian.stark@tuhh.de"
__status__ = "Production"
__name__ = "Discrete LPDC Decoder"
__doc__ = """This class uses the results of Discrete Density Evolution to set up LDPC decoder that purely works on
            lookups of integers."""

class Discrete_LDPC_Decoder_class_irregular(Discrete_LDPC_Decoder_class):
    """This class uses the results of Discrete Density Evolution to set up LDPC decoder that purely works on
       lookups of integers.
    Args:
    input parameter
        H                           the parity check matrix of the Low-Density-Parity Check Code
        i_max                       the number of iteration, that should be performed by the decoder
        cardinality_Y_channel       the resolution of the continuous channel (typically a large number)
        cardinality_T_channel       number of clusters of the channel quantizer
        cardinality_T_decoder_ops   number of clusters used by the decoder, typically similar to cardinality_T_channel
    """

    def __init__(self, filename, imax_, cardinality_T_channel_,
                 cardinality_T_decoder_ops_,
                 Trellis_checknode_vector_a_,
                 Trellis_varnode_vector_a_,
                 matching_vector_checknode_,
                 matching_vector_varnode_,
                 msg_at_time_,
                 match='true'):

        # initialize parameters
        self.H_sparse = self.load_check_mat(filename)

        self.imax = imax_

        # Quantizer parameters
        self.cardinality_T_channel = cardinality_T_channel_

        # Discrete DE related
        self.cardinality_T_decoder_ops = cardinality_T_decoder_ops_
        self.Trellis_checknode_vector_a = Trellis_checknode_vector_a_.astype(int)
        self.Trellis_varnode_vector_a = Trellis_varnode_vector_a_.astype(int)

        # analyze the H matrix and set all decoder variables

        self.set_code_parameters()
        self.data_len = (self.R_c * self.codeword_len).astype(int)


        self.msg_at_time = msg_at_time_
        self.map_node_connections()

        self.matching_vector_checknode = matching_vector_checknode_
        self.matching_vector_varnode = matching_vector_varnode_
        self.match = match

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
                #H = self.H_sparse.toarray()
        else:
            arrays = [np.array(list(map(int, line.split()))) for line in open(filename)]
            H = self.alistToNumpy(arrays)
            H_sparse = sci.sparse.csr_matrix(H)
        return H_sparse

    def map_node_connections(self):
        """ The overall idea of this function is to store the connections between var- and check nodes in a new structure
        namely two vectors. This vectors are called inboxes, because the entries should be seen as memory for incoming
        messages. Therefore it is important to track which check node output rights in which var node input and vince
        versa. """

        self.inbox_memory_start_checknodes = np.append([0], np.cumsum(self.degree_checknode_nr[:-1]) ).astype(np.int)
        self.inbox_memory_start_varnodes = np.append([0], np.cumsum(self.degree_varnode_nr[:-1]) ).astype(np.int)

        # At first it is determined which check node delivers to which var node
        # This information is encoded in the non-zero columns of each row
        # non-zero return the indices in the desired way.

        self.customers_checknode_nr = self.H_sparse.indices

        # Now it is determined which var node delivers to which check node
        # This information is encoded in the non-zero rows of each column
        # non-zero return the indices in the desired way.

        self.customers_varnode_nr = (self.H_sparse.tocsc() ).indices

        # now we now the connections but, since one node has multiple inputs the node number is node enough.
        # An additional offset vector needs to be defined. If one node was already connected, then the memory box is
        # filled. Performing cumsum on the rows only allows to generate this offset vector at check nodes destinations.

        H_copy = self.H_sparse.tocsc().copy()
        for i in range(self.N_v):
            H_copy.data[H_copy.indptr[i] : H_copy.indptr[i+1] ] = \
                np.arange(H_copy.indptr[i+1]-H_copy.indptr[i])

        self.offset_at_dest_checknodes = H_copy.tocsr().data


        self.target_memory_cells_checknodes = (self.inbox_memory_start_varnodes[self.customers_checknode_nr] + \
                                              self.offset_at_dest_checknodes).astype(np.int)

        H_copy = self.H_sparse.copy()
        for i in range(self.N_c):
            H_copy.data[H_copy.indptr[i] : H_copy.indptr[i+1] ] = \
                np.arange(H_copy.indptr[i+1]-H_copy.indptr[i])

        self.offset_at_dest_varnodes = H_copy.tocsc().data

        self.target_memory_cells_varnodes = (self.inbox_memory_start_checknodes[self.customers_varnode_nr] + \
                                            self.offset_at_dest_varnodes).astype(np.int)


        self.inbox_memory_checknodes = np.zeros((self.degree_checknode_nr.sum().astype(np.int),self.msg_at_time)).astype(int)
        self.inbox_memory_varnodes = np.zeros((self.degree_varnode_nr.sum().astype(np.int),self.msg_at_time)).astype(int)
        self.memory_channel_values = np.zeros(self.N_v)

    def init_OpenCL_decoding(self,msg_at_time_, context_=False):

        if not context_ :
            self.context = cl.create_some_context()
        else:
            self.context = context_

        print(self.context.get_info(cl.context_info.DEVICES))
        path = os.path.split(os.path.abspath(__file__))

        kernelsource = open(os.path.join(path[0], "kernels_template_irreg.cl")).read()
        tpl = Template(kernelsource)
        rendered_tp = tpl.render(cn_degree=self.d_c_max, vn_degree=self.d_v_max,match=self.match,
                                 msg_at_time=msg_at_time_)

        self.program = cl.Program(self.context, str(rendered_tp)).build()

        self.queue = cl.CommandQueue(self.context)

        self.inbox_memory_start_varnodes_buffer = cl_array.to_device(self.queue,
                                                                self.inbox_memory_start_varnodes.astype(np.int32))

        self.inbox_memory_start_checknodes_buffer = cl_array.to_device(self.queue,
                                                                  self.inbox_memory_start_checknodes.astype(np.int32))

        self.degree_varnode_nr_buffer = cl_array.to_device(self.queue, self.degree_varnode_nr.astype(np.int32))

        self.degree_checknode_nr_buffer = cl_array.to_device(self.queue, self.degree_checknode_nr.astype(np.int32))

        self.target_memorycells_varnodes_buffer = cl_array.to_device(self.queue,
                                                                self.target_memory_cells_varnodes.astype(np.int32))
        self.target_memorycells_checknodes_buffer = cl_array.to_device(self.queue,
                                                                  self.target_memory_cells_checknodes.astype(np.int32))

        self.Trellis_checknode_vector_a_buffer = cl_array.to_device(self.queue, self.Trellis_checknode_vector_a.astype(np.int32))

        self.Trellis_varnode_vector_a_buffer = cl_array.to_device(self.queue, self.Trellis_varnode_vector_a.astype(np.int32))

        self.matching_vector_varnode_buffer = cl_array.to_device(self.queue, self.matching_vector_varnode.astype(np.int32))

        self.matching_vector_checknode_buffer = cl_array.to_device(self.queue, self.matching_vector_checknode.astype(np.int32))

        self.checknode_inbox_buffer = cl_array.empty(self.queue, self.inbox_memory_checknodes.shape, dtype=np.int32)

        self.varnode_inbox_buffer = cl_array.empty(self.queue, self.inbox_memory_varnodes.shape, dtype=np.int32)

        self.syndrom_buffer = cl_array.empty(self.queue,
            (self.degree_checknode_nr.shape[0], self.inbox_memory_varnodes.shape[-1]), dtype=np.int32)

        self.krnl = get_sum_kernel(self.context, None,
                                   dtype_in=self.varnode_inbox_buffer.dtype)  # varnode_output_buffer.dtype )


        # define programs
        self.send_prog = self.program.send_channel_values_to_checknode_inbox

        self.first_iter_prog = self.program.checknode_update_iter0
        self.first_iter_prog.set_scalar_arg_dtypes([None, None, None, None, None, np.int32, np.int32, None, None])

        self.varnode_update_prog = self.program.varnode_update
        self.varnode_update_prog.set_scalar_arg_dtypes([None, None, None, None, None, None, np.int32,
                                                   np.int32, np.int32, None, None])

        self.checknode_update_prog = self.program.checknode_update
        self.checknode_update_prog.set_scalar_arg_dtypes([None, None, None, None, None, np.int32,
                                                   np.int32, np.int32, None, None])

        self.calc_syndrom_prog = self.program.calc_syndrome
        self.calc_syndrom_prog.set_scalar_arg_dtypes([None, None, None, np.int32, None])

        self.varoutput_prog = self.program.calc_varnode_output
        self.varoutput_prog.set_scalar_arg_dtypes([None, None, None, None,np.int32,np.int32,np.int32, None, None ])

    def decode_OpenCL(self, received_blocks,buffer_in=False,return_buffer=False):
        # Set up OpenCL
        if buffer_in:
            channel_values_buffer = received_blocks
        else:
            channel_values_buffer = cl_array.to_device(self.queue,received_blocks.astype(np.int32))

        varnode_output_buffer = cl_array.empty(self.queue, received_blocks.shape, dtype=np.int32)

        self.send_prog(self.queue, received_blocks.shape, None,
                  channel_values_buffer.data,
                  self.inbox_memory_start_varnodes_buffer.data,
                  self.degree_varnode_nr_buffer.data,
                  self.target_memorycells_varnodes_buffer.data,
                  self.checknode_inbox_buffer.data)
        #self.queue.finish()

        self.first_iter_prog(self.queue, (self.degree_checknode_nr.shape[0], received_blocks[:,np.newaxis].shape[-1]), None,
                        self.checknode_inbox_buffer.data,
                        self.inbox_memory_start_checknodes_buffer.data,
                        self.degree_checknode_nr_buffer.data,
                        self.target_memorycells_checknodes_buffer.data,
                        self.varnode_inbox_buffer.data,
                        self.cardinality_T_channel,
                        self.cardinality_T_decoder_ops,
                        self.Trellis_checknode_vector_a_buffer.data,
                        self.matching_vector_checknode_buffer.data)

        syndrome_zero = False
        i_num = 1


        while (i_num<self.imax) and (not syndrome_zero):

            local_size = None

            self.varnode_update_prog(self.queue, received_blocks.shape , None,
                                channel_values_buffer.data,
                                self.varnode_inbox_buffer.data,
                                self.inbox_memory_start_varnodes_buffer.data,
                                self.degree_varnode_nr_buffer.data,
                                self.target_memorycells_varnodes_buffer.data,
                                self.checknode_inbox_buffer.data,
                                self.cardinality_T_channel,
                                self.cardinality_T_decoder_ops,
                                i_num-1,
                                self.Trellis_varnode_vector_a_buffer.data,
                                self.matching_vector_varnode_buffer.data
                                )
            #self.queue.finish()

            self.checknode_update_prog(self.queue, (self.degree_checknode_nr.shape[0], received_blocks[:,np.newaxis].shape[-1]), None,
                                   self.checknode_inbox_buffer.data,
                                   self.inbox_memory_start_checknodes_buffer.data,
                                   self.degree_checknode_nr_buffer.data,
                                   self.target_memorycells_checknodes_buffer.data,
                                   self.varnode_inbox_buffer.data,
                                   self.cardinality_T_channel,
                                   self.cardinality_T_decoder_ops,
                                   i_num-1,
                                   self.Trellis_checknode_vector_a_buffer.data,
                                   self.matching_vector_checknode_buffer.data)

            #self.queue.finish()

            self.calc_syndrom_prog(self.queue, (self.degree_checknode_nr.shape[0], received_blocks[:,np.newaxis].shape[-1]), None,
                                      self.checknode_inbox_buffer.data,
                                      self.inbox_memory_start_checknodes_buffer.data,
                                      self.degree_checknode_nr_buffer.data,
                                      self.cardinality_T_decoder_ops,
                                      self.syndrom_buffer.data)

            #self.queue.finish()

            if cl_array.sum(self.syndrom_buffer).get() == 0:
                syndrome_zero =True


            i_num += 1


        self.varoutput_prog(self.queue, received_blocks.shape , None,
                            channel_values_buffer.data,
                            self.varnode_inbox_buffer.data,
                            self.inbox_memory_start_varnodes_buffer.data,
                            self.degree_varnode_nr_buffer.data,
                            self.cardinality_T_channel,
                            self.cardinality_T_decoder_ops,
                            i_num - 1,
                            self.Trellis_varnode_vector_a_buffer.data,
                            varnode_output_buffer.data)
        self.queue.finish()
        if return_buffer:
            return varnode_output_buffer
        else:
            output_values = varnode_output_buffer.get()
            return output_values

    def return_errors_all_zero(self, varnode_output_buffer):

        # only consider first systematic bits which are R_c * N_var
        errors = self.krnl(
            (varnode_output_buffer[:self.data_len].__lt__(int(self.cardinality_T_decoder_ops / 2)).astype(np.int32))).get()

        return errors

    def discrete_cn_operation(self,vec_y_c,iter_):
        #self.d_c = self.degree_checknode_nr[0]

        if (iter_ == 0):
            d_c_cur = vec_y_c.shape[1]+1

            t_0_c = self.Trellis_checknode_vector_a[vec_y_c[:, 0]*
                                                    self.cardinality_T_channel + vec_y_c[:, 1]]

            t_l_m_1_c= t_0_c

            for l in range(d_c_cur - 3):
                t_l_c = \
                    self.Trellis_checknode_vector_a[t_l_m_1_c * self.cardinality_T_decoder_ops +
                                                    vec_y_c[:, l + 2] +
                                                    self.cardinality_T_channel ** 2 +
                                                    l * self.cardinality_T_decoder_ops * self.cardinality_T_channel]
                t_l_m_1_c= t_l_c

            # Todo transform step
            # transform_offset = iter_ * self.cardinality_T_decoder_ops * self.d_c_max +\
            #                    (d_c_cur-1) * self.cardinality_T_decoder_ops
            # t_l_transformed = self.matching_vector_checknode[transform_offset + t_l_m_1_c]
            # t_l_m_1_c = t_l_transformed

        else:
            d_c_cur = vec_y_c.shape[1]+1


            offset_iteration_0 = 1 * (self.d_c_max - 3) * self.cardinality_T_channel * self.cardinality_T_decoder_ops + \
                                 1 * self.cardinality_T_channel ** 2

            add_offset_iteration_iter = (iter_ - 1) * (self.d_c_max - 2) * self.cardinality_T_decoder_ops ** 2

            t_0_c = self.Trellis_checknode_vector_a[vec_y_c[:, 0]*
                                                                           self.cardinality_T_decoder_ops +
                                                                           vec_y_c[:, 1] +
                                                                           offset_iteration_0 +
                                                                           add_offset_iteration_iter]

            t_l_m_1_c= t_0_c

            for l in range(d_c_cur - 3):
                t_l_c = self.Trellis_checknode_vector_a[t_l_m_1_c* self.cardinality_T_decoder_ops +
                                                        vec_y_c[:, l + 2] +
                                                        (l+1) * self.cardinality_T_decoder_ops ** 2 +
                                                        offset_iteration_0 + add_offset_iteration_iter]
                t_l_m_1_c = t_l_c

            # Todo transform step
            # transform_offset = iter_ * self.cardinality_T_decoder_ops * self.d_c_max + \
            #                    (d_c_cur - 1) * self.cardinality_T_decoder_ops
            # t_l_transformed = self.matching_vector_checknode[transform_offset + t_l_m_1_c]
            # t_l_m_1_c = t_l_transformed

        node_output_msg = t_l_m_1_c
        return node_output_msg

    def discrete_vn_operation(self, vec_y_v, iter_):

        offset_iteration_iter = (1 * self.cardinality_T_channel * self.cardinality_T_decoder_ops + (
            self.d_v_max - 1) * self.cardinality_T_decoder_ops ** 2) * (iter_)

        t_0_v = self.Trellis_varnode_vector_a[vec_y_v[:, 0]*self.cardinality_T_decoder_ops +
                                                                     vec_y_v[:, 1] + offset_iteration_iter]

        t_l_m_1_v= t_0_v

        for l in range(vec_y_v.shape[1]- 2):
            t_l_v = self.Trellis_varnode_vector_a[t_l_m_1_v *self.cardinality_T_decoder_ops +
                                                 vec_y_v[:, l + 2] +
                                                 l * self.cardinality_T_decoder_ops ** 2 +
                                                 offset_iteration_iter +
                                                 1 * self.cardinality_T_channel * self.cardinality_T_decoder_ops]
            t_l_m_1_v = t_l_v

        # Todo transform step
        # if vec_y_v.shape[1] < self.d_v_max+1:
        #     d_v_cur = vec_y_v.shape[1]
        #     transform_offset = iter_ * self.cardinality_T_decoder_ops * self.d_v_max +\
        #                            (d_v_cur-1) * self.cardinality_T_decoder_ops
        #     t_l_transformed = self.matching_vector_varnode[transform_offset + t_l_m_1_v]
        #     t_l_m_1_v = t_l_transformed


        node_output_msg = t_l_m_1_v
        return node_output_msg

    def decode_on_host(self,channel_values_):

        self.memory_channel_values = channel_values_
        d_v_degrees = np.unique(self.degree_varnode_nr)
        d_c_degrees = np.unique(self.degree_checknode_nr)
        for d_v in d_v_degrees:
            var_node_inds = self.degree_varnode_nr == d_v
            start_idx_var = self.inbox_memory_start_varnodes[var_node_inds]
            ind_mat_var = start_idx_var[:, np.newaxis] + np.arange(d_v)
            channel_val_mat = np.kron(self.memory_channel_values[var_node_inds, np.newaxis],
                                      np.ones((d_v, 1))).astype(int)

            self.inbox_memory_checknodes[:, 0][self.target_memory_cells_varnodes[ind_mat_var]] = \
                channel_val_mat

        # checknode first iteration
        iter = 0
        for d_c in d_c_degrees:
            check_node_inds = self.degree_checknode_nr == d_c
            start_idx_check = self.inbox_memory_start_checknodes[check_node_inds]
            ind_mat_check = start_idx_check[:, np.newaxis] + np.arange(d_c)

            all_messages = self.inbox_memory_checknodes[ind_mat_check]
            m = np.kron(np.arange(d_c)[:, np.newaxis],
                        np.ones(d_c))  # '*ones(1,self.degree_checknode_nr(1));
            reduced = all_messages[:, m.transpose()[(1 - np.eye(d_c)).astype(bool)].astype(int)]
            reduced = np.reshape(reduced, (-1, d_c - 1))

            customers_check = np.reshape(self.target_memory_cells_checknodes[ind_mat_check], (-1, 1))[:, 0]
            self.inbox_memory_varnodes[customers_check, 0] = self.discrete_cn_operation(reduced, iter)


        for iter in range(self.imax-1):

            for d_v in d_v_degrees:
                var_node_inds = self.degree_varnode_nr == d_v
                start_idx_var = self.inbox_memory_start_varnodes[var_node_inds]
                ind_mat_var = start_idx_var[:, np.newaxis] + np.arange(d_v)
                channel_val_mat = np.kron(self.memory_channel_values[var_node_inds, np.newaxis],
                                          np.ones((d_v, 1))).astype(int)

                all_messages = self.inbox_memory_varnodes[ind_mat_var]

                m = np.kron(np.arange(d_v)[:, np.newaxis], np.ones(d_v))

                reduced = all_messages[:, m.transpose()[(1 - np.eye(d_v)).astype(bool)].astype(int)]
                reduced = np.reshape(reduced, (-1, d_v - 1))

                customers_var = np.reshape(self.target_memory_cells_varnodes[ind_mat_var], (-1, 1))

                self.inbox_memory_checknodes[:, 0][customers_var] = self.discrete_vn_operation(
                    np.hstack((channel_val_mat, reduced)), iter)

            for d_c in d_c_degrees:
                check_node_inds = self.degree_checknode_nr == d_c
                start_idx_check = self.inbox_memory_start_checknodes[check_node_inds]
                ind_mat_check = start_idx_check[:, np.newaxis] + np.arange(d_c)

                all_messages = self.inbox_memory_checknodes[ind_mat_check]
                m = np.kron(np.arange(d_c)[:, np.newaxis],
                              np.ones(d_c))  # '*ones(1,self.degree_checknode_nr(1));
                reduced = all_messages[:, m.transpose()[(1 - np.eye(d_c)).astype(bool)].astype(int)]
                reduced = np.reshape(reduced, (-1, d_c - 1))

                customers_check = np.reshape(self.target_memory_cells_checknodes[ind_mat_check], (-1, 1))[:, 0]
                self.inbox_memory_varnodes[customers_check, 0] = self.discrete_cn_operation(reduced, iter+1)


        output_vector = np.zeros(self.N_v)

        for d_v in d_v_degrees:
            var_node_inds = self.degree_varnode_nr == d_v
            start_idx_var = self.inbox_memory_start_varnodes[var_node_inds]
            ind_mat_var = start_idx_var[:, np.newaxis] + np.arange(d_v)

            all_messages = self.inbox_memory_varnodes[ind_mat_var]
            output_vector[var_node_inds] = self.discrete_vn_operation(
                 np.hstack((self.memory_channel_values[var_node_inds, np.newaxis], all_messages[:, :, 0])), self.imax - 1)
        return output_vector

