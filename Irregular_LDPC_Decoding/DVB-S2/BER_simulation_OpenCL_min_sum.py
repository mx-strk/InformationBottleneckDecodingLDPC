import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import numpy as np
from AWGN_Channel_Transmission.AWGN_Quantizer_BPSK import AWGN_Channel_Quantizer
from AWGN_Channel_Transmission.LDPC_Transmitter import LDPC_BPSK_Transmitter as transmitter
from Continous_LDPC_Decoding.min_sum_decoder_irreg import Min_Sum_Decoder_class_irregular as LDPC_Decoder
import os.path
import matplotlib as mpl
import time
import scipy.io as sio

mpl.use("pgf")
pgf_with_pdflatex = {
    "pgf.texsystem": "pdflatex",
    "text.usetex": True,
"font.family": "serif"
}
mpl.rcParams.update(pgf_with_pdflatex)
import matplotlib.pyplot as plt


__author__ = "Maximilian Stark"
__copyright__ = "2016, Institute of Communications, University of Technology Hamburg"
__credits__ = ["Maximilian Stark"]
__version__ = "1.0"
__email__ = "maximilian.stark@tuhh.de"
__status__ = "Production"
__name__ = "Simulation Env"
__doc__ = """This script sets up a proper simulation environment to analyse a min-sum decoder.
          The BER performance of the chosen decoder can be stored and compared."""

# Choose the correct context
#os.environ['PYOPENCL_CTX'] = '1' # CPU
os.environ['PYOPENCL_CTX'] = '0' # GraKA
os.environ['PYOPENCL_COMPILER_OUTPUT']='1'
np.seterr(all='raise')

# Load stored data
filepath = os.path.join(os.path.dirname(__file__),'../../LDPC_codes/irregular_codes/DVB_S2_0.5.npz')


card_vec = np.array([16,32])

# Human choice
AD_max_abs = 3
cardinality_Y_channel = 2000
cardinality_T_channel = card_vec[sys.argv[1]]
cardinality_T_decoder_ops = card_vec[sys.argv[1]]
msg_at_time = 2
min_errors = 7000

decoder_name = 'minsum_'+str(cardinality_T_decoder_ops)

print("###############")
print(decoder_name)
print("###############")


timestr = time.strftime("%y%m%d-%H%M")

filename = os.path.splitext(decoder_name)[0].replace('.','').replace('_config','')+'_'+timestr
pathname = os.path.join(os.path.dirname(__file__),'BER_Results', filename.replace('.', ''))
os.makedirs(os.path.dirname(os.path.join(pathname,' ')), exist_ok=True)




imax = 50


#sets the start EbN0_dB value
EbN0_dB_max_value = 2.5

#sets the start EbN0_dB value
EbN0_dB_max_value = 1.2

#simulation runs until this BER is achieved
target_error_rate=1e-6
BER_go_on_in_smaller_steps=1e-5

#in steps of size..
EbN0_dB_normal_stepwidth=0.1
EbN0_dB_small_stepwidth=0.1

# start EbN0 simulation
EbN0_dB = 0
EbN0_dB_ind = 0
BER_vector = np.array([0.])
EbN0_dB_vector = np.array([EbN0_dB])
ready = False
NR_BLOCKS_PER_CONTROL_MSG = 100

transi = transmitter(filepath, msg_at_time)
decodi = LDPC_Decoder(filepath, imax, cardinality_T_channel, msg_at_time)

N_var = transi.codeword_len
while not ready:
    EbN0_dB_ind += EbN0_dB_ind
    EbN0_dB = EbN0_dB_vector[-1]

    sigma_n2 = 10**(-EbN0_dB/10) / (2*transi.R_c)

    quanti = AWGN_Channel_Quantizer(sigma_n2, AD_max_abs, cardinality_T_channel, cardinality_Y_channel)
    quanti.init_OpenCL_quanti(N_var,msg_at_time,return_buffer_only=True)
    decodi.init_OpenCL_decoding(msg_at_time,quanti.context)


    errors = 0
    transmitted_blocks = 0
    # transmit
    start = time.time()
    while errors < min_errors:


        rec_data_quantized = quanti.quantize_direct_OpenCL_LLR(N_var, msg_at_time)
        decoded_mat = decodi.decode_OpenCL_min_sum(rec_data_quantized,buffer_in=True,return_buffer=True)

        errors += decodi.return_errors_all_zero(decoded_mat)
        transmitted_blocks += + msg_at_time



        if np.mod(transmitted_blocks, NR_BLOCKS_PER_CONTROL_MSG) == 0:
            time_so_far = time.time()-start
            time_per_error = (time_so_far / (errors+1)) #+1 to avoid devide by 0 errors
            estim_minutes_left = ((min_errors * time_per_error) - time_so_far) / 60

            print('EbN0_dB=', EbN0_dB, ', '
                  'errors=', errors,
                  ' elapsed time this run=', time_so_far,
                  ' BER_estimate=','{:.2e}'.format( (errors / (transi.R_c*transmitted_blocks * N_var))),
                  ' datarate_Bps =', '{:.2e}'.format(  (transi.R_c*transmitted_blocks * N_var) / time_so_far),
                  ' estim_minutes_left=',estim_minutes_left)


    end = time.time()

    BER_vector[-1] = errors / (transi.R_c*transmitted_blocks * N_var)
    spent = end-start
    datarate_Bps = (transi.R_c*transmitted_blocks * N_var) / spent
    print(EbN0_dB_vector[-1], '{:.2e}'.format(BER_vector[-1]), ' Bitrate:','{:.2e}'.format(datarate_Bps) )
    np.savez(os.path.join(pathname, 'BER_results'), EbN0_dB_vector=EbN0_dB_vector, BER_vector=BER_vector)


    if (BER_vector[-1] > target_error_rate) and (EbN0_dB < EbN0_dB_max_value):
        if BER_vector[-1] < BER_go_on_in_smaller_steps:
            EbN0_dB_vector = np.append(EbN0_dB_vector, EbN0_dB_vector[-1] + EbN0_dB_small_stepwidth)
        else:
            EbN0_dB_vector = np.append(EbN0_dB_vector, EbN0_dB_vector[-1] + EbN0_dB_normal_stepwidth)

        BER_vector = np.append(BER_vector, 0)
    else:
        ready = True



#Plot
plt.figure()
plt.semilogy(EbN0_dB_vector,BER_vector)
plt.xlabel('Eb/N0')
plt.ylabel('Bit Error Rate ')
plt.grid(True)
plt.show()


np.savez(os.path.join(pathname,'BER_results'), EbN0_dB_vector=EbN0_dB_vector, BER_vector=BER_vector)

plt.savefig(os.path.join(pathname,'BER_figure.pgf'))
plt.savefig(os.path.join(pathname,'BER_figure.pdf'))

res_dict = {"EbN0_dB_vector" : EbN0_dB_vector, "BER_vector":BER_vector
    , "decoder_name":decoder_name}

sio.savemat(os.path.join(pathname,'BER_results.mat'),res_dict)
