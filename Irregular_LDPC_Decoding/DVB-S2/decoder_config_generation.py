import matplotlib.pyplot as plt

from AWGN_Channel_Transmission.AWGN_Discrete_Density_Evolution import \
    AWGN_Discrete_Density_Evolution_class_irregular as DDE_irregular
from Discrete_LDPC_decoding.Information_Matching import *

__author__ = "Maximilian Stark"
__copyright__ = "2016, Institute of Communications, University of Technology Hamburg"
__credits__ = ["Maximilian Stark"]
__version__ = "1.0"
__email__ = "maximilian.stark@tuhh.de"
__status__ = "Production"
__name__ = "Decoder Generation"
__doc__ = """This script generates a discrete decoder for the desired design-Eb/N0."""



# set noise level for DE
EbN0_dB_mapping_gen = 0.7
for EbN0_dB_mapping_gen in np.array([0.6,0.7,0.8,0.9,1.0]):
    # set quantizer limits
    AD_Max_abs = 3
    plt.figure()

    cardinality_Y_channel = 2000
    cardinality_T_channel = 16
    cardinality_T_decoder_ops = 16
    i_max = 50
    nror = 10

    #   1 2 3 4 5 6 7
    d_c_dist = np.array([0,0,0,0,0,1,32399]) / 32400
            #   1 2 3 4 5 6 7 8
    d_v_dist = np.array([1,32399,19440,0,0,0,0,12960])/64800


    lambda_vec = convert_node_to_edge_degree(d_v_dist)
    rho_vec = convert_node_to_edge_degree(d_c_dist)

    #R_c = 1-d_v/d_c    # code rate
    R_c = 1 - (d_v_dist*(np.arange(d_v_dist.shape[0])+1)).sum() /  (d_c_dist*(np.arange(d_c_dist.shape[0])+1)).sum()  # code rate

    sigma_n2 = 10**(-EbN0_dB_mapping_gen/10) / (2*R_c)
    steps = 5

    config = 'cas'
        # generate decoder config
    DDE_inst = DDE_irregular(sigma_n2, AD_Max_abs, cardinality_Y_channel, cardinality_T_channel,
                             cardinality_T_decoder_ops, lambda_vec, rho_vec, i_max, nror , match = True)

    DDE_inst.run_discrete_density_evolution()
    DDE_inst.save_config(config)
    plt.plot(DDE_inst.DDE_inst_data['MI_T_dvm1_v_X_dvm1_v'],label='match')


    # DDE_inst = DDE_irregular(sigma_n2, AD_Max_abs, cardinality_Y_channel, cardinality_T_channel,
    #                          cardinality_T_decoder_ops, lambda_vec, rho_vec, i_max, nror , match = False)
    #
    # DDE_inst.run_discrete_density_evolution()
    # DDE_inst.save_config('adapt_no_match')
    # plt.plot(DDE_inst.DDE_inst_data['MI_T_dvm1_v_X_dvm1_v'],label='no match')
    # plt.legend(loc=4)
  