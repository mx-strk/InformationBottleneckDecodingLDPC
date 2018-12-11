# Decoding of LDPC Codes Using the Information Bottleneck Method in Python

### Introduction

This source code is intended to reproduce the results published in [LB18,LBT+18] and [SLB18, Sta18]. In these works, it is explained how the information bottleneck method can be used to design decoders for regular and irregular LDPC codes.


### Requirements

1. Download and install Python 3.6 (we recommend Anaconda)
2. Download and install the [ib_base package](https://collaborating.tuhh.de/cip3725/ib_base)
2. Clone the git repository
4. Installation requires the following packages:
  * numpy
  * [IB_Base](https://collaborating.tuhh.de/cip3725/ib_base)
  * [PyOpenCl.2018](https://documen.tician.de/pyopencl/misc.html)
  * mako
  * [progressbar](https://pypi.org/project/progressbar2/)


### Documentation

#### Generate a Decoder
To construct a decoder, make use of the scripts `decoder_config_generation.py` in the different folders. Here, you can enter:
- the maximum number of decoding iterations $`i_{max}`$
- the cardinality of the exchanged messages $`|\mathcal{T}|`$  
- the the degree distributions $`\mathbf{d}_c`$ and $`\mathbf{d}_v`$
- the design-$`E_b/N_0`$ for which you want to construct the decoder

#### Running a BER Simulation
You can run either benchmark simulations for the belief-propagation decoding, belief-propagation decoding with an channel output quantizer or min-sum decoding. Please make sure that you have OpenCL set up correctly.

###### Benchmark simulations
To run the benchmark simulations just use `BER_simulation_OpenCL_min_sum.py` or `BER_simulation_OpenCL_quant_BP.py` in the respective folders.

###### Information bottleneck decoding
To run BER simulations just use `BER_simulation_OpenCL_enc.py` in the respective folders. The ending "enc" indicates, that the transmission chain uses the appropriate LDPC encoder, not the all-zeros codeword is transmitted.  

**Note:** Make sure that you have an IB decoder generated before running a simulation.


A detailed documentation of all provided functions and a more complete test suite will be available soon.

### Citation

The code is distributed under the MIT license. When using the code for your research please cite our work.

### References

[SLB18] M. Stark, J. Lewandowsky, G. Bauch. "Information-Bottleneck Decoding of High-Rate Irregular LDPC Codes for Optical Communication using Message Alignment“. Applied Sciences. 2018; 8(10):1884. 

[SLB18a] M. Stark, J. Lewandowsky, and G. Bauch, “Information-Optimum LDPC Decoders with Message Alignment for Irregular Codes,” in 2018 IEEE Global Communications Conference (Globecom2018), Abu Dhabi, United Arab Emirates, 2018.

[LB18] J. Lewandowsky and G. Bauch, “Information-Optimum LDPC Decoders Based on the Information Bottleneck Method,” IEEE Access, vol. 6, pp. 4054–4071, 2018. https://ieeexplore.ieee.org/document/8268118

[LBT+18] J. Lewandowsky, G. Bauch, M. Tschauner, and P. Oppermann, “Design and Evaluation of Information Bottleneck LDPC Decoders for Software Defined Radios,” in Proc. 12th International Conference on Signal Processing and Communication Systems (ICSPCS), Australia, 2018.

