from sys import argv, exit
import math
import numpy as np
import os
from scipy.spatial.transform import Rotation as R



def readG2o(fileName):
	f = open(fileName, 'r')
	A = f.readlines()
	f.close()

	X = []
	Y = []
	Z = []
	Qx = []
	Qy = []
	Qz = []
	Qw = []

	for line in A:
		if "VERTEX_SE3:QUAT" in line:			
			if(len(line.split(' ')) == 10):
				(ver, ind, x, y, z, qx, qy, qz, qw, newline) = line.split(' ')
			elif(len(line.split(' ')) == 9):
				(ver, ind, x, y, z, qx, qy, qz, qw) = line.split(' ')

			X.append(float(x))
			Y.append(float(y))
			Z.append(float(z))
			Qx.append(float(qx))
			Qy.append(float(qy))
			Qz.append(float(qz))

			if(len(line.split(' ')) == 10):
				Qw.append(float(qw))
			elif(len(line.split(' ')) == 9):
				Qw.append(float(qw.rstrip('\n')))

	return (X, Y, Z, Qx, Qy, Qz, Qw)


def convert2Kitti(poses, name):
    A = np.zeros((poses.shape[0], 12))

    for i in range(poses.shape[0]):
        Rot = np.array(R.from_quat([poses[i,3], poses[i,4], poses[i,5], poses[i,6]]).as_matrix())
        t = poses[i,0:3].reshape((-1,1))
        T = np.hstack((Rot, t)).reshape((1,12))
        A[i,:] = T[:]

    np.savetxt(name , A, delimiter=' ')

if __name__=='__main__':
    X, Y, Z, Qx, Qy, Qz, Qw = readG2o(argv[1])
    poses = np.zeros((len(X), 7))
    poses[:,0] = X
    poses[:,1] = Y
    poses[:,2] = Z
    poses[:,3] = Qx
    poses[:,4] = Qy
    poses[:,5] = Qz
    poses[:,6] = Qw
    gtKitti = convert2Kitti(poses, argv[2])