import os

import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
from mako.template import Template
from pyopencl.reduction import get_sum_kernel

__author__ = "Maximilian Stark"
__copyright__ = "05.07.2016, Institute of Communications, University of Technology Hamburg"
__credits__ = ["Maximilian Stark", "Jan Lewandowsky"]
__version__ = "1.0"
__email__ = "maximilian.stark@tuhh.de"
__status__ = "Production"
__name__ = "Discrete LPDC Decoder"
__doc__ = """This class uses the results of Discrete Density Evolution to set up LDPC decoder that purely works on
            look-ups of integers."""

class Discrete_LDPC_Decoder_class:
    """This class uses the results of Discrete Density Evolution to set up LDPC decoder that purely works on
       lookups of integers.
    Attributes:
        H                           the parity check matrix of the Low-Density-Parity Check Code
        i_max                       the number of iteration, that should be performed by the decoder
        cardinality_Y_channel       the resolution of the continuous channel (typically a large number)
        cardinality_T_channel       number of clusters of the channel quantizer
        cardinality_T_decoder_ops   number of clusters used by the decoder, typically similar to cardinality_T_channel
    """

    def __init__(self, filename, imax_, cardinality_T_channel_,
                 cardinality_T_decoder_ops_, Trellis_checknode_vector_a_, Trellis_varnode_vector_a_,msg_at_time_):
        # initialize parameters
        self.H = self.load_check_mat(filename)
        self.imax = imax_

        # Quantizer parameters
        self.cardinality_T_channel = cardinality_T_channel_

        # Discrete DE related
        self.cardinality_T_decoder_ops = cardinality_T_decoder_ops_
        self.Trellis_checknode_vector_a = Trellis_checknode_vector_a_.astype(int)
        self.Trellis_varnode_vector_a = Trellis_varnode_vector_a_.astype(int)

        # analyze the H matrix and set all decoder variables
        self.degree_checknode_nr = (self.H).sum(1) # which check node has which degree?
        self.degree_varnode_nr = (self.H).sum(0) # which variable node has which degree?
        self.N_v = self.H.shape[1]  # How many variable nodes are present?
        self.N_c = self.H.shape[0] # How many checknodes are present?

        self.msg_at_time = msg_at_time_
        self.map_node_connections()

    def update_trellis_vectors(self,Trellis_checknode_vector_a_, Trellis_varnode_vector_a_):
        self.Trellis_checknode_vector_a = Trellis_checknode_vector_a_.astype(int)
        self.Trellis_varnode_vector_a = Trellis_varnode_vector_a_.astype(int)

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

    def load_check_mat(self, filename):
        arrays = [np.array(list(map(int, line.split()))) for line in open(filename)]
        H = self.alistToNumpy(arrays)
        return H

    def map_node_connections(self):
        """ The overall idea of this function is to store the connections between var- and check nodes in a new structure
        namely two vectors. This vectors are called inboxes, because the entries should be seen as memory for incoming
        messages. Therefore it is important to track which check node output rights in which var node input and vince
        versa. """

        self.inbox_memory_start_checknodes = np.append([0], np.cumsum(self.degree_checknode_nr[:-1]) )
        self.inbox_memory_start_varnodes = np.append([0], np.cumsum(self.degree_varnode_nr[:-1]) )

        # At first it is determined which check node delivers to which var node
        # This information is encoded in the non-zero columns of each row
        # non-zero return the indices in the desired way.

        self.customers_checknode_nr = np.nonzero(self.H)[1]

        # Now it is determined which var node delivers to which check node
        # This information is encoded in the non-zero rows of each column
        # non-zero return the indices in the desired way.

        self.customers_varnode_nr = np.nonzero(self.H.transpose())[1]

        # now we now the connections but, since one node has multiple inputs the node number is node enough.
        # An additional offset vector needs to be defined. If one node was already connected, then the memory box is
        # filled. Performing cumsum on the rows only allows to generate this offset vector at check nodes destinations.
        self.offset_at_dest_checknodes = np.cumsum(self.H, 0)
        self.offset_at_dest_checknodes = self.offset_at_dest_checknodes[np.nonzero(self.H)] - 1


        self.target_memory_cells_checknodes = self.inbox_memory_start_varnodes[self.customers_checknode_nr] + \
                                              self.offset_at_dest_checknodes


        self.offset_at_dest_varnodes = np.cumsum(self.H, 1)
        self.offset_at_dest_varnodes = self.offset_at_dest_varnodes.transpose()[np.nonzero(self.H.transpose())] - 1


        self.target_memory_cells_varnodes = self.inbox_memory_start_checknodes[self.customers_varnode_nr] + \
                                            self.offset_at_dest_varnodes


        self.inbox_memory_checknodes = np.zeros((self.degree_checknode_nr.sum(),self.msg_at_time)).astype(int)
        self.inbox_memory_varnodes = np.zeros((self.degree_varnode_nr.sum(),self.msg_at_time)).astype(int)
        self.memory_channel_values = np.zeros(self.N_v)

    def init_OpenCL_decoding(self,msg_at_time_, context_=False):
        if not context_:
            self.context = cl.create_some_context()
        else:
            self.context = context_

        path = os.path.split(os.path.abspath(__file__))

        kernelsource = open(os.path.join(path[0], "kernels_template.cl")).read()
        tpl = Template(kernelsource)
        rendered_tp = tpl.render(cn_degree=self.degree_checknode_nr[0], vn_degree=self.degree_varnode_nr[0],
                                 msg_at_time=msg_at_time_)

        self.program = cl.Program(self.context, str(rendered_tp)).build()

        self.queue = cl.CommandQueue(self.context)
        mem_pool = cl.tools.MemoryPool(cl.tools.ImmediateAllocator(self.queue, cl.mem_flags.READ_ONLY))
        mem_pool2 = cl.tools.MemoryPool(cl.tools.ImmediateAllocator(self.queue))

        #mem_pool = None
        self.inbox_memory_start_varnodes_buffer = cl_array.to_device(self.queue,
                                                                self.inbox_memory_start_varnodes.astype(np.int32),allocator=mem_pool)

        self.inbox_memory_start_checknodes_buffer = cl_array.to_device(self.queue,
                                                                  self.inbox_memory_start_checknodes.astype(np.int32),allocator=mem_pool)

        self.degree_varnode_nr_buffer = cl_array.to_device(self.queue, self.degree_varnode_nr.astype(np.int32),allocator=mem_pool)

        self.degree_checknode_nr_buffer = cl_array.to_device(self.queue, self.degree_checknode_nr.astype(np.int32),allocator=mem_pool)

        self.target_memorycells_varnodes_buffer = cl_array.to_device(self.queue,
                                                                self.target_memory_cells_varnodes.astype(np.int32),allocator=mem_pool)
        self.target_memorycells_checknodes_buffer = cl_array.to_device(self.queue,
                                                                  self.target_memory_cells_checknodes.astype(np.int32),allocator=mem_pool)

        self.Trellis_checknode_vector_a_buffer = cl_array.to_device(self.queue, self.Trellis_checknode_vector_a.astype(np.int32),allocator=mem_pool)

        self.Trellis_varnode_vector_a_buffer = cl_array.to_device(self.queue, self.Trellis_varnode_vector_a.astype(np.int32),allocator=mem_pool)

        self.checknode_inbox_buffer = cl_array.empty(self.queue, self.inbox_memory_checknodes.shape, dtype=np.int32,allocator=mem_pool2)

        self.varnode_inbox_buffer = cl_array.empty(self.queue, self.inbox_memory_varnodes.shape, dtype=np.int32,allocator=mem_pool2)

        self.syndrom_buffer = cl_array.empty(self.queue,
            (self.degree_checknode_nr.shape[0], self.inbox_memory_varnodes.shape[-1]), dtype=np.int32,allocator=mem_pool2)

        self.krnl = get_sum_kernel(self.context, None,
                                   dtype_in=self.varnode_inbox_buffer.dtype)  # varnode_output_buffer.dtype )


        # define programs
        self.send_prog = self.program.send_channel_values_to_checknode_inbox

        self.first_iter_prog = self.program.checknode_update_iter0
        self.first_iter_prog.set_scalar_arg_dtypes([None, None, None, None, None, np.int32, np.int32, None])

        self.varnode_update_prog = self.program.varnode_update
        self.varnode_update_prog.set_scalar_arg_dtypes([None, None, None, None, None, None, np.int32,
                                                   np.int32, np.int32, None])

        self.checknode_update_prog = self.program.checknode_update
        self.checknode_update_prog.set_scalar_arg_dtypes([None, None, None, None, None, np.int32,
                                                   np.int32, np.int32, None])

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
                        self.Trellis_checknode_vector_a_buffer.data)

        syndrome_zero = False
        i_num = 1


        while (i_num<self.imax) and (not syndrome_zero):

            local_size = None #(1000, 1)

            self.varnode_update_prog(self.queue, received_blocks.shape , local_size,
                                channel_values_buffer.data,
                                self.varnode_inbox_buffer.data,
                                self.inbox_memory_start_varnodes_buffer.data,
                                self.degree_varnode_nr_buffer.data,
                                self.target_memorycells_varnodes_buffer.data,
                                self.checknode_inbox_buffer.data,
                                self.cardinality_T_channel,
                                self.cardinality_T_decoder_ops,
                                i_num-1,
                                self.Trellis_varnode_vector_a_buffer.data
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
                                   self.Trellis_checknode_vector_a_buffer.data)

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
            pass
            output_values = varnode_output_buffer.get()
            return output_values

    def return_errors_all_zero(self, varnode_output_buffer):

        errors = self.krnl((varnode_output_buffer.__lt__(int(self.cardinality_T_decoder_ops / 2)).astype(np.int32))).get()
        return errors

    def discrete_cn_operation(self,vec_y_c,iter_):
        self.d_c = self.degree_checknode_nr[0]

        if (iter_ == 0):
            t_0_c = self.Trellis_checknode_vector_a[vec_y_c[:, 0]*self.cardinality_T_channel + vec_y_c[:, 1]]

            t_l_m_1_c = t_0_c

            for l in range(self.d_c - 3):
                t_l_c = self.Trellis_checknode_vector_a[t_l_m_1_c * self.cardinality_T_decoder_ops +
                                                        vec_y_c[:, l + 2] + self.cardinality_T_channel ** 2
                                                        + l * self.cardinality_T_decoder_ops * self.cardinality_T_channel]
                t_l_m_1_c = t_l_c

        else:
            offset_iteration_0 = 1 * (self.d_c - 3) * self.cardinality_T_channel * self.cardinality_T_decoder_ops + \
                                 1 * self.cardinality_T_channel ** 2
            add_offset_iteration_iter = (iter_ - 1) * (self.d_c - 2) * self.cardinality_T_decoder_ops ** 2

            t_0_c = self.Trellis_checknode_vector_a[vec_y_c[:, 0]*self.cardinality_T_decoder_ops +
                                                   vec_y_c[:, 1] + offset_iteration_0 + add_offset_iteration_iter]

            t_l_m_1_c = t_0_c

            for l in range(self.d_c - 3):
                t_l_c = self.Trellis_checknode_vector_a[t_l_m_1_c * self.cardinality_T_decoder_ops +
                                                        vec_y_c[:, l + 2] +
                                                        (l+1) * self.cardinality_T_decoder_ops ** 2 +
                                                        offset_iteration_0 + add_offset_iteration_iter]
                t_l_m_1_c = t_l_c


        node_output_msg = t_l_m_1_c
        return node_output_msg

    def discrete_vn_operation(self, vec_y_v, iter_):
        self.d_v = self.degree_varnode_nr[0]

        offset_iteration_iter = (1 * self.cardinality_T_channel * self.cardinality_T_decoder_ops + (
            self.d_v - 1) * self.cardinality_T_decoder_ops ** 2) * (iter_)

        t_0_v = self.Trellis_varnode_vector_a[vec_y_v[:, 0]*self.cardinality_T_decoder_ops +
                                              vec_y_v[:, 1] + offset_iteration_iter]

        t_l_m_1_v = t_0_v
        for l in range(vec_y_v.shape[1]- 2):
            t_l_v = self.Trellis_varnode_vector_a[t_l_m_1_v * self.cardinality_T_decoder_ops + vec_y_v[:, l + 2] +
                                                 l * self.cardinality_T_decoder_ops ** 2 +
                                                 offset_iteration_iter +
                                                 1 * self.cardinality_T_channel * self.cardinality_T_decoder_ops]
            t_l_m_1_v = t_l_v

        node_output_msg = t_l_m_1_v
        return node_output_msg

    def decode_on_host(self,channel_values_):

        self.memory_channel_values = channel_values_

        channel_val_mat = np.kron(self.memory_channel_values[:,np.newaxis], np.ones( (self.degree_varnode_nr[0],1) )).astype(int)

        start_idx_var = self.inbox_memory_start_varnodes
        ind_mat_var = start_idx_var[:,np.newaxis] + np.arange(self.degree_varnode_nr[0])


        self.inbox_memory_checknodes[:,0][self.target_memory_cells_varnodes[ind_mat_var]] = channel_val_mat

        start_idx_check = self.inbox_memory_start_checknodes
        index_mat_check = start_idx_check[:,np.newaxis] + np.arange(self.degree_checknode_nr[0])


        customers_check = np.reshape(self.target_memory_cells_checknodes[index_mat_check], (-1,1))[:,0]
        customers_var = np.reshape(self.target_memory_cells_varnodes[ind_mat_var],(-1,1))

        for iter in range(self.imax):
            all_messages = self.inbox_memory_checknodes[index_mat_check]

            m = np.kron(np.arange(self.degree_checknode_nr[0])[:,np.newaxis],np.ones(self.degree_checknode_nr[0])) #'*ones(1,self.degree_checknode_nr(1));
            reduced = all_messages[:, m.transpose()[ (1 - np.eye(self.degree_checknode_nr[0])).astype(bool) ].astype(int)]
            reduced = np.reshape(reduced,(-1,self.degree_checknode_nr[0]-1))


            self.inbox_memory_varnodes[customers_check, 0] = self.discrete_cn_operation(reduced, iter)


            all_messages = self.inbox_memory_varnodes[ind_mat_var]

            m = np.kron(np.arange(self.degree_varnode_nr[0])[:, np.newaxis], np.ones(self.degree_varnode_nr[0]))

            reduced = all_messages[:, m.transpose()[(1 - np.eye(self.degree_varnode_nr[0])).astype(bool)].astype(int)]
            reduced = np.reshape(reduced, (-1, self.degree_varnode_nr[0] - 1))

            self.inbox_memory_checknodes[:,0][customers_var] = self.discrete_vn_operation(np.hstack((channel_val_mat, reduced)),iter)

        all_messages = self.inbox_memory_varnodes[ind_mat_var]

        output_vector = self.discrete_vn_operation(np.hstack((self.memory_channel_values[:,np.newaxis], all_messages[:,:,0])), self.imax-1)

        return output_vector

