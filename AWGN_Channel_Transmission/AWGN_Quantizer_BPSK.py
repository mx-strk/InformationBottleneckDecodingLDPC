import numpy as np
from information_bottleneck.information_bottleneck_algorithms.symmetric_sIB import symmetric_sIB
from scipy.stats import norm

try:
    import pyopencl as cl
    import pyopencl.array as cl_array
    from pyopencl.clrandom import rand as clrand
except ImportError:
    Warning("PyOpenCl not installed")
import os
from mako.template import Template

__author__ = "Maximilian Stark"
__copyright__ = "09.08.2016, Institute of Communications, University of Technology Hamburg"
__credits__ = ["Maximilian Stark"]
__version__ = "1.0"
__email__ = "maximilian.stark@tuhh.de"
__status__ = "Production"
__name__ = "AWGN Channel Quantizer"
__doc__ = """This modules contains a class generating a quantizer for an AWGN channel output for BPSK modulation"""


class AWGN_Channel_Quantizer:
    """Implementation of an information optimum quantizer unit assuming BPSK transmission.

    The quantizer is generated using the symmetric, sequential information bottleneck algorithm.
    This class supports OpenCL for faster quantization and even direct quantization and sample generation on the GPU
    (cf. quantize direct).
    Although it is theoretical correct to quantize directly, it is preferable to create a more realistic
    communication chain including an encoder and modulator in your system instead of using this direct quantization approach.

    Attributes:
        sigma_n2: noise variance corresponding to the desired design-Eb/N0 of the decoder
        AD_max_abs: limits of the quantizer
        cardinality_Y: number of steps used for the fine quantization of the input distribution of the quantizer
        cardinality_T: cardinality of the compression variable representing the quantizer output

        limits: borders of the quantizer regions
        y_vec: fine quantization of the input domain
        delta: spacing between two values in the quantized input domain (cf. y_vec)

        x_vec: position of the means of the involved Gaussians

    """
    def __init__(self, sigma_n2_, AD_max_abs_, cardinality_T_, cardinality_Y_, dont_calc = False):
        """Inits the quantizer class."""
        self.nror = 5
        self.limits = np.zeros(cardinality_T_)

        self.sigma_n2 = sigma_n2_
        self.cardinality_T = cardinality_T_
        self.cardinality_Y = cardinality_Y_
        self.AD_max_abs = AD_max_abs_

        self.y_vec = np.linspace(-self.AD_max_abs, +self.AD_max_abs, self.cardinality_Y)
        self.x_vec = np.array([-1, 1])
        self.delta = self.y_vec[1] - self.y_vec[0]
        if not dont_calc:
            self.calc_quanti()

    def calc_quanti(self):
        """Determines the information optimum quantizer for the given input distribution"""

        # calculate p_xy based on sigma_n2 and AD_max_abs;
        # init as normal with mean + 1
        p_y_given_x_equals_zero = norm.pdf(self.y_vec, loc=1, scale=np.sqrt(self.sigma_n2)) * self.delta

        # truncate t account for distortion introduced by the quantizer limits then
        p_y_given_x_equals_zero[-1] += self.gaussian_over_prob(self.AD_max_abs, 1)
        p_y_given_x_equals_zero[0] += self.gaussian_under_prob(-self.AD_max_abs, 1)

        # flip distribution, which realizes mean -1 or a transmitted bit = 1
        p_y_given_x_equals_one = p_y_given_x_equals_zero[::-1]

        self.p_xy = 0.5 * np.hstack((p_y_given_x_equals_zero[:,np.newaxis], p_y_given_x_equals_one[:,np.newaxis]))

        self.p_xy = self.p_xy / self.p_xy.sum() #normalize for munerical stability

        # run the symmetric sequential Information Bottleneck algorithm
        IB_class = symmetric_sIB(self.p_xy, self.cardinality_T, self.nror)
        IB_class.run_IB_algo()

        # store the results
        [self.p_t_given_y, self.p_x_given_t, self.p_t] = IB_class.get_results()

        # calculate
        # p(t | X = 0)=p(X=0 | t)
        # p(t) / p(X=0)
        self.p_x_given_t = self.p_x_given_t / self.p_x_given_t.sum(1)[:,np.newaxis]
        self.p_x_and_t = self.p_x_given_t * self.p_t[:,np.newaxis]
        p_t_given_x_equals_zero = self.p_x_and_t[:, 0] / 0.5

        self.cdf_t_given_x_equals_zero = np.append([0], np.cumsum(p_t_given_x_equals_zero))

        self.output_LLRs = np.log(self.p_x_and_t[:, 0] / self.p_x_and_t[:, 1])
        self.calc_limits()

    @classmethod
    def from_generated(cls, cdf_t_given_x_equals_zero_):
        cdf_t_given_x_equals_zero = cdf_t_given_x_equals_zero_
        return cls(cdf_t_given_x_equals_zero,)

    def gaussian_over_prob(self, x, mu):
        """Compensates the ignored probability mass caused by fixing the region to +- AD_abs_max."""

        prob = norm.sf((x-mu+self.delta/2)/np.sqrt(self.sigma_n2))
        return prob

    def gaussian_under_prob(self, x, mu):
        """Compensates the ignored probability mass caused by fixing the region to +- AD_abs_max."""

        prob = 1-self.gaussian_over_prob(x-self.delta,mu)
        return prob

    def calc_limits(self):
        """Calculates the limits of the quantizer borders"""

        for i in range(self.cardinality_T):
            cur_vec = (self.p_t_given_y[:, i] == 1).nonzero()
            self.limits[i] = self.y_vec[cur_vec[0].min()]

        self.limits[int(self.cardinality_T/2)] = 0
        #self.limits[-1]=self.AD_max_abs

    def quantize_direct(self, input_bits):
        """Direct quantization without the need of a channel in between since the inversion method is used.
        The clusters are directly sampled.
        """
        # create uniform samples
        rand_u = np.random.rand(input_bits.shape[0],input_bits.shape[1])

        # create samples ~ p(t | X = 0) using inversion method
        if input_bits.shape[1] > 1:
            output_integers = ((np.repeat(rand_u[:,:,np.newaxis], self.cardinality_T+1, axis=2)-self.cdf_t_given_x_equals_zero) > 0).sum(2)-1
            output_integers[input_bits.astype(bool)] = self.cardinality_T - 1 - output_integers[input_bits.astype(bool)]
        else:
            output_integers = ((rand_u - self.cdf_t_given_x_equals_zero) > 0).sum(1) - 1
            # "mirror" a sample, when the input bit is 1, otherwise do nothing.
            output_integers[input_bits.astype(bool)[:, 0]] = self.cardinality_T - 1 - output_integers[
            input_bits.astype(bool)[:, 0]]

        return output_integers

    def quantize_on_host(self,x):
        """Quantizes the received samples on the local machine"""
        if x.shape[1] > 1:
            cluster = ((np.repeat(x[:,:,np.newaxis], self.cardinality_T, axis=2)-self.limits) > 0).sum(2)-1
            cluster[cluster == -1] = 0
        else:
            cluster = np.sum((x - self.limits) > 0, 1) -1
            cluster[cluster==-1] = 0

        return cluster

    def init_OpenCL_quanti(self, N_var,msg_at_time,return_buffer_only=False):
        """Inits the OpenCL context and transfers all static data to the device"""

        self.context = cl.create_some_context()

        print(self.context.get_info(cl.context_info.DEVICES))
        path = os.path.split(os.path.abspath(__file__))
        kernelsource = open(os.path.join(path[0], 'kernels_quanti_template.cl')).read()

        tpl = Template(kernelsource)
        rendered_tp = tpl.render(Nvar=N_var)

        self.program = cl.Program(self.context, str(rendered_tp)).build()

        self.return_buffer_only = return_buffer_only

        # Set up OpenCL
        self.queue = cl.CommandQueue(self.context)
        self.quantize = self.program.quantize
        self.quantize.set_scalar_arg_dtypes([np.int32, None, None, None])
        self.quantize_LLR = self.program.quantize_LLR
        self.quantize_LLR.set_scalar_arg_dtypes([np.int32, None, None, None,None])
        self.limit_buff = cl_array.to_device(self.queue, self.cdf_t_given_x_equals_zero.astype(np.float64))
        self.cluster_buff = cl_array.empty(self.queue, (N_var, msg_at_time), dtype=np.int32)
        self.LLR_buff = cl_array.empty(self.queue, (N_var, msg_at_time), dtype=np.float64)
        self.LLR_values_buff = cl_array.to_device(self.queue, self.output_LLRs.astype(np.float64))

    def quantize_OpenCL(self, x):
        """Quantizes the received distorted samples on the graphic card"""

        # Create OpenCL buffers

        x_buff = cl_array.to_device(self.queue,x.astype(np.float64) )
        limit_buff = cl_array.to_device(self.queue, self.limits.astype(np.float64))
        cluster_buff = cl_array.empty_like(x_buff.astype(np.int32))

        self.quantize(self.queue, x.shape, None, self.cardinality_T, x_buff.data, limit_buff.data, cluster_buff.data)
        self.queue.finish()

        if self.return_buffer_only:
            return cluster_buff
        else:
            clusters = cluster_buff.get()
            return clusters

    def quantize_direct_OpenCL(self,N_var,msg_at_time):
        """Direct quantization without the need of a channel in between since the inversion method is used.
        The clusters are directly sampled. In this scenario the all-zeros codeword is considered such that no data
        needs to be transferred to the graphic card.

        """

        #rand_u_buff = clrand(self.queue, (N_var,msg_at_time), dtype=np.float64, a=0, b=1)

        rand_u = np.random.rand(N_var,msg_at_time)

        # Create OpenCL buffers

        rand_u_buff = cl_array.to_device(self.queue,rand_u.astype(np.float64) )



        self.quantize(self.queue, (N_var,msg_at_time), None, self.cardinality_T+1, rand_u_buff.data,
                      self.limit_buff.data, self.cluster_buff.data)


        self.queue.finish()

        if self.return_buffer_only:
            return self.cluster_buff
        else:
            clusters = self.cluster_buff.get()
            return clusters

    def quantize_direct_OpenCL_LLR(self,N_var,msg_at_time):
        """ Returns the LLRs of the sampled cluster indices. These indices correspond to the quantized outputs which
        are found directly on the graphic card using the inversion method. """

        rand_u = np.random.rand(N_var,msg_at_time)

        # Create OpenCL buffers
        rand_u_buff = cl_array.to_device(self.queue,rand_u.astype(np.float64) )

        self.quantize_LLR(self.queue, (N_var,msg_at_time), None, self.cardinality_T+1, rand_u_buff.data,
                      self.limit_buff.data, self.LLR_values_buff.data, self.LLR_buff.data)

        self.queue.finish()

        if self.return_buffer_only:
            return self.LLR_buff
        else:
            LLRs = self.LLR_buff.get()
            return LLRs

