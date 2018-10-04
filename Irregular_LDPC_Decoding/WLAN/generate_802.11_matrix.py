"""This script generates a parity check matrix for irregular LDPC codes defined in the WLAN 802.11 2012 standart."""

import numpy as np
import scipy.io as sio


ind_matrix =np.array([[40, -1, -1, -1, 22, -1, 49, 23, 43, -1, -1, -1, 1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                        [50, 1, -1, -1, 48, 35, -1, -1, 13, -1, 30, -1, -1, 0 , 0, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                        [39, 50, -1, -1, 4, -1, 2, -1, -1, -1, -1, 49, -1, -1, 0 , 0, -1, -1, -1, -1, -1, -1, -1, -1],
                        [33, -1, -1, 38, 37, -1, -1, 4, 1, -1, -1, -1, -1, -1, -1, 0 , 0, -1, -1, -1, -1, -1, -1, -1],
                        [45, -1, -1, -1, 0, 22, -1, -1, 20, 42, -1, -1,-1, -1, -1, -1, 0 , 0, -1, -1, -1, -1, -1, -1],
                        [51, -1, -1, 48, 35, -1, -1, -1, 44, -1, 18, -1,-1, -1, -1, -1, -1, 0 , 0, -1, -1, -1, -1, -1],
                        [47, 11, -1, -1, -1, 17, -1, -1, 51, -1, -1, -1, 0, -1, -1, -1, -1, -1, 0 , 0, -1, -1, -1, -1],
                        [5, -1, 25, -1, 6, -1, 45, -1, 13, 40, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0 , 0, -1, -1, -1],
                        [33, -1, -1, 34, 24, -1, -1, -1, 23, -1, -1, 46, -1, -1, -1, -1, -1, -1, -1, -1, 0 , 0, -1, -1],
                        [1, -1, 27, -1, 1, -1, -1, -1, 38, -1, 44, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0 , 0, -1],
                        [-1, 18, -1, -1, 23, -1, -1, 8, 0, 35, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0 , 0],
                        [49, -1, 17, -1, 30, -1, -1, -1, 34, -1, -1, 19, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0],
                    ])

Z=54
H= np.empty(Z*24)[np.newaxis,:]
ys = np.array([], dtype=np.int64).reshape(0, 5)
for i in range(ind_matrix.shape[0]):
    H_row = np.empty((Z, Z))
    for j in range(ind_matrix.shape[1]):
        if ind_matrix[i,j]>=0:
            a= np.roll(np.eye(Z),ind_matrix[i,j],axis=1)
        else:
            a=np.zeros((Z,Z))
        H_row=np.concatenate((H_row,a),axis=1)
    H=np.concatenate((H,H_row[:,-1296:]),axis=0)

H= H[1:,:]
print(np.unique(H.sum(1)))
print((H.sum(1)==8).sum())
print(np.unique(H.sum(0)))

dictionary={}
dictionary['H_mat']=H
sio.savemat('WLAN_H.mat',dictionary)

np.save('WLAN_H', H)