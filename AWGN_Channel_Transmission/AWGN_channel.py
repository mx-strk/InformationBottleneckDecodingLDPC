import numpy as np

__author__ = "Maximilian Stark"
__copyright__ = "09.08.2016, Institute of Communications, University of Technology Hamburg"
__credits__ = ["Maximilian Stark"]
__version__ = "1.0"
__email__ = "maximilian.stark@tuhh.de"
__status__ = "Production"
__name__ = "AWGN Channel"
__doc__ = """Module holds the AWGN_channel class which can be used for simulation of an AWGN channel with real or
             complex noise."""

class AWGN_channel:
    """ Class implements an additive white Gaussian noise channel

    The added noise can either be real or complex depending on the arguments of the constructor.
    The default value is real.

    Attributes:
        sigma_n2: a double setting noise variance
        complex: a boolean value indicating if noise is complex or not
    """
    def __init__(self, sigma_n2_, complex =False):
        """Inits the AWGN_channel class
        Args:
            sigma_n2_: noise variance specified by user
            complex: default is false, indicating if noise is complex
        """
        self.sigma_n2 = sigma_n2_
        self.complex = complex

    def transmission(self, input):
        """Performs the transmission of an input stream over an AWGN channel
        Args:
            input: sequence of symbols as numpy array or scalar
        Returns:
            output: summation of noise and input
        """
        if self.complex:
            noise = np.sqrt(self.sigma_n2/2) * np.random.randn(input.shape[0], input.shape[1]) + \
                    1j * np.sqrt(self.sigma_n2/2) * np.random.randn(input.shape[0], input.shape[1])

        else:
            noise = np.sqrt(self.sigma_n2) * np.random.randn(input.shape[0],input.shape[1])

        output = input + noise

        return output
