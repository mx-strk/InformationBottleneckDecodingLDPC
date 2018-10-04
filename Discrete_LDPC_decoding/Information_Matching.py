import numpy as np
from information_bottleneck.tools.inf_theory_tools import kl_divergence
__author__ = "Maximilian Stark"
__copyright__ = "05.11.2016, Institute of Communications, University of Technology Hamburg"
__credits__ = ["Maximilian Stark","Jan Lewandowsky"]
__version__ = "1.0"
__email__ = "maximilian.stark@tuhh.de"
__status__ = "Production"
__name__ = "Information Matching"
__doc__ = """This module contains functions needed to construct discrete decoders for irregular LDPC codes.
          Especially, the information matching algorithm is of crucial importance because it allows to resolve the
          existing conflicts of opinion."""


def convert_node_to_edge_degree(node_distribution):
    max = node_distribution.shape[0]
    values = np.arange(max)+1
    edge_distribution = node_distribution * values / (node_distribution * values).sum()

    return edge_distribution


def convert_node_to_edge_degree_from_H(H, axis):
    degree_checknode_nr = H.sum(axis)
    max = np.max(degree_checknode_nr)
    values = np.arange(max) + 1
    edge_distribution = np.zeros(max)
    for i,k in enumerate(values):
        edge_distribution[i] = ((degree_checknode_nr == k).sum() * k / (H).sum(axis))

    return edge_distribution


def information_matching_v2(cardinality_Y,p_x_and_t0,p_x_and_z1):
    """ This function implements the information matching algorithm
    The function we need to minimize is
    D_KL ( p(x|t0) || p(x|Z1=z0) ) forall t_0 = z_0
    This script is an implementation of an optimization algorithm which solves the following problem:

    We have two sources z1 and t0,
    where X -> Z1
    and   X -> T0

    we are interested in a transformation z0=f(t0) (equivalently: deterministic p(z0|t0)), such that
    I(X;Z0) \approx I(X;Y0) and
    p(x|z0) \approx p(x|t0) for z0=y0


    param:
    cardinality_Y - number of cluster to optimize
    p_x_given_t0 - distribution to transform
    p_x_given_z1 - reference distribution for transformation

    return:

    """
    z0_stars = np.zeros(cardinality_Y).astype(np.int)
    p_x_given_t0 = p_x_and_t0 / p_x_and_t0.sum(1)[:, np.newaxis]
    p_t0 = p_x_and_t0.sum(1)
    p_x_given_z1 = p_x_and_z1 / p_x_and_z1.sum(1)[:, np.newaxis]

    for t0 in range( int(cardinality_Y) ):
        z0_stars[t0] = np.argmin(kl_divergence(p_x_given_t0[t0,:],p_x_given_z1) )


    p_star_z0_given_t0 = np.zeros((cardinality_Y, cardinality_Y))
    for j, z0_star in enumerate(z0_stars):
        p_star_z0_given_t0[j, z0_star] = 1

    p_star_z0 = np.zeros(cardinality_Y)
    for t0, z0 in enumerate(z0_stars):
        p_star_z0[z0] += p_t0[t0]

    p_x_given_z0_stars = 1/(p_star_z0[:,np.newaxis]+1e-80) * np.dot( p_star_z0_given_t0.transpose(), p_x_and_t0)
    p_x_and_z0_stars = p_x_given_z0_stars * p_star_z0[:,np.newaxis]

    return  p_x_given_z0_stars, p_x_and_z0_stars,p_star_z0, z0_stars , p_star_z0_given_t0

