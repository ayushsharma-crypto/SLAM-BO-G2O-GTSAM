from sys import argv, exit
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


def readTxt(filename):
	f = open(filename, 'r')
	A = f.readlines()
	f.close()

	X = []
	Y = []
	THETA = []

	for i, line in enumerate(A):
		if(i % 1 == 0):
			(x, y, theta) = line.split()
			X.append(float(x))
			Y.append(float(y))
			THETA.append(math.radians(float(theta.rstrip('\n'))))

	return X, Y, THETA


def readG2o(fileName):
	f = open(fileName, 'r')
	A = f.readlines()
	f.close()

	X = []
	Y = []
	THETA = []

	for line in A:
		if "VERTEX_SE2" in line:
			(ver, ind, x, y, theta) = line.split()
			X.append(float(x))
			Y.append(float(y))
			THETA.append(float(theta.rstrip('\n')))


	return (X, Y, THETA)



def readG2oSE3(fileName):
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

def convert(X, Y, THETA):
	A = np.zeros((len(X), 12))

	for i in range(len(X)):
		T = np.identity(4)
		T[0, 3] = X[i]
		T[1, 3] = Y[i]
		T[0:3, 0:3] = np.array([[math.cos(THETA[i]), -math.sin(THETA[i]), 0], [math.sin(THETA[i]), math.cos(THETA[i]), 0], [0, 0, 1]])
		
		A[i] = T[0:3, :].reshape(1, 12)
		
	return A

def convertSE3(X, Y, Z, Qx, Qy, Qz, Qw):
	A = np.zeros((len(X), 12))

	for i in range(len(X)):
		T = np.identity(4)
		T[0, 3] = X[i]
		T[1, 3] = Y[i]
		T[2, 3] = Z[i]
		T[0:3, 0:3] = R.from_quat([Qx[i],Qy[i],Qz[i],Qw[i]]).as_matrix()
		
		A[i] = T[0:3, :].reshape(1, 12)
		
	return A




if __name__ == '__main__':

	if int(argv[3])==1:
		X, Y, Z, Qx, Qy, Qz, Qw = readG2oSE3(argv[1])
		A = convertSE3(X, Y, Z, Qx, Qy, Qz, Qw)
	else:
		(X, Y, THETA) = readG2o(argv[1])
		A = convert(X, Y, THETA)
	
	file_name = argv[2]
	np.savetxt(file_name, A, delimiter=' ')
	print(f"saved '{file_name}' from '{argv[1]}'")
