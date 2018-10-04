import matplotlib.pyplot as plt
import numpy as np

from AWGN_Channel_Transmission.AWGN_Discrete_Density_Evolution import AWGN_Discrete_Density_Evolution_class as DDE

__author__ = "Maximilian Stark"
__copyright__ = "2016, Institute of Communications, University of Technology Hamburg"
__credits__ = ["Maximilian Stark"]
__version__ = "1.0"
__email__ = "maximilian.stark@tuhh.de"
__status__ = "Production"
__name__ = "Decoder Generation"
__doc__ = """This script generates a discrete decoder for the desired design-Eb/N0."""
# set noise level for DE

EbN0_dB_mapping_gen = 1.25


# set quantizer limits
AD_Max_abs = 3


cardinality_Y_channel = 2000
cardinality_T_channel = 16
cardinality_T_decoder_ops = 16
i_max = 250
nror = 5

# set code related parameters
d_v = 3
d_c = 6

R_c = 1-d_v/d_c    # code rate

sigma_n2 = 10**(-EbN0_dB_mapping_gen/10) / (2*R_c)

# generate decoder config
DDE_inst = DDE(sigma_n2, AD_Max_abs, cardinality_Y_channel, cardinality_T_channel,
               cardinality_T_decoder_ops, d_v, d_c, i_max, nror )
DDE_inst.run_discrete_density_evolution()
#DDE_inst.save_config()

# generate trajectory

x_vec = np.zeros(2*i_max-1)
y_vec = np.zeros(2*i_max-1)

x_vec[0] = 0
y_vec[0] = DDE_inst.DDE_inst_data['ext_mi_varnode_in_iter'][0]


for i in range(1,i_max):
    x_vec[2*i-1] = DDE_inst.DDE_inst_data['ext_mi_checknode_in_iter'][i-1]
    y_vec[2*i-1] = y_vec[2*i-2]

    x_vec[2 * i] = x_vec[2*i-1]
    y_vec[2 * i] = DDE_inst.DDE_inst_data['ext_mi_varnode_in_iter'][i]

plt.plot(x_vec,y_vec)

plt.show()
