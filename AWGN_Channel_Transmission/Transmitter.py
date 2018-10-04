import numpy as np

__author__ = "Maximilian Stark"
__copyright__ = "10.08.2016, Institute of Communications, University of Technology Hamburg"
__credits__ = ["Maximilian Stark"]
__version__ = "1.0"
__email__ = "maximilian.stark@tuhh.de"
__status__ = "Production"
__name__ = "LDPC encoded BPSK Transmitter"
__doc__ = """This module contains classes which can generate a random sequence of symbols for BPSK, QAM, MPSK.
            These symbols are encoded using a given LDPC code parity check matrix."""


class BPSK_Transmitter:
    def __init__(self, sequence_len,  msg_at_time=1):

        self.data_len = int(sequence_len)

        self.last_transmitted_bits = []
        self.msg_at_time = msg_at_time

    def transmit(self):

        uncoded_msgs = np.random.randint(0,2, (self.data_len, self.msg_at_time))

        self.last_transmitted_bits = uncoded_msgs

        data = self.BPSK_mapping(uncoded_msgs)

        return data

    def BPSK_mapping(self, X):

        data = np.ones((self.codeword_len, self.msg_at_time))
        data[X == 1] = -1

        return data


class QAM_Transmitter(BPSK_Transmitter):
    def __init__(self, sequence_len, encoding_table, sqrt_M, msg_at_time=1):

        BPSK_Transmitter.__init__(self,sequence_len, msg_at_time)

        self.encoding_table = encoding_table
        self.sqrt_M = sqrt_M
        self.num_bits = int(np.log2(sqrt_M) * 2)

        self.amplitude_values = np.zeros(self.sqrt_M)
        self.determine_amplitudes_for_encoding_values()

    def transmit(self):
        uncoded_msgs = np.random.randint(0, 2, (self.data_len, self.msg_at_time))

        data =  np.zeros((int(self.data_len/self.num_bits), self.msg_at_time),dtype=complex)

        for i in range(self.msg_at_time):
            data[:, i] = self.QAM_mapping(uncoded_msgs[:,i])[:,0]

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


class MPSK_Tranmitter(BPSK_Transmitter):
    def __init__(self, sequence_len, encoding_table, M, msg_at_time=1):
        BPSK_Transmitter.__init__(self, sequence_len, msg_at_time)

        self.encoding_table = encoding_table
        self.M = M
        self.num_bits = int(np.log2(M))

        self.phase_values = np.zeros(self.M, dtype=complex)
        self.determine_phase_for_encoding_values()

    def transmit(self):
        uncoded_msgs = np.random.randint(0, 2, (self.data_len, self.msg_at_time))

        data = np.zeros((int(self.data_len / self.num_bits), self.msg_at_time), dtype=complex)

        for i in range(self.msg_at_time):
            data[:, i] = self.MPSK_mapping(uncoded_msgs[:, i])[:, 0]

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
