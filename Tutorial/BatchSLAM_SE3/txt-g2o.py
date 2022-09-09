import os
import numpy as np
from sys import argv
import math

from scipy.spatial.transform import Rotation as R

def writeG2O(X, Y, Z, Qx, Qy, Qz, Qw, file):
	g2o = open(file, 'w')

	sp = ' '

	for i, (x, y, z, qx, qy, qz, qw) in enumerate(zip(X, Y, Z, Qx, Qy, Qz, Qw)):
		line = "VERTEX_SE3:QUAT " + str(i) + sp + str(x) + sp + str(y) + sp + str(z) + sp + str(qx) + sp + str(qy) + sp + str(qz) + sp + str(qw) + '\n'
		g2o.write(line)

	# Odometry
	# T1_w : 1 with respect to world
	g2o.write("\n\n\n# Odometry constraints\n\n\n\n")
	info = '20 0 0 0 0 0 20 0 0 0 0 20 0 0 0 20 0 0 20 0 20'

	for i in range(1, len(X)):
		p1 = (X[i-1], Y[i-1], Z[i-1], Qx[i-1], Qy[i-1], Qz[i-1], Qw[i-1])
		p2 = (X[i], Y[i], Z[i], Qx[i], Qy[i], Qz[i], Qw[i])

		R1_w = R.from_quat([p1[3], p1[4], p1[5], p1[6]]).as_matrix()
		R2_w = R.from_quat([p2[3], p2[4], p2[5], p2[6]]).as_matrix()

		T1_w = np.identity(4)
		T2_w = np.identity(4)

		T1_w[0:3, 0:3] = R1_w
		T2_w[0:3, 0:3] = R2_w

		T1_w[0, 3] = p1[0] 
		T1_w[1, 3] = p1[1]
		T1_w[2, 3] = p1[2]

		T2_w[0, 3] = p2[0]
		T2_w[1, 3] = p2[1]
		T2_w[2, 3] = p2[2]

		T2_1 = np.dot(np.linalg.inv(T1_w), T2_w)

		dx, dy, dz = T2_1[0, 3], T2_1[1, 3], T2_1[2, 3]
		dqx, dqy, dqz, dqw = list(R.from_matrix(T2_1[0:3, 0:3]).as_quat())
		
		line = "EDGE_SE3:QUAT " + str(i-1) + sp + str(i) + sp + str(dx) + sp + str(dy) + sp + str(dz) + sp + str(dqx) + sp + str(dqy) + sp + str(dqz) + sp + str(dqw) + sp +  info + '\n'
		g2o.write(line)

	# # LC Constraints
	# g2o.write("\n\n\n# Loop constraints\n\n\n\n")
	# info = '40 0 0 0 0 0 40 0 0 0 0 40 0 0 0 40 0 0 40 0 40'

	# for i in range(len(src)):
	# 	line = "EDGE_SE3:QUAT " + str(trg[i]) + sp + str(src[i]) + sp + str(trans[i][0]) + sp + str(trans[i][1]) + sp + str(trans[i][2]) + sp + str(trans[i][3]) + sp + str(trans[i][4]) + sp + \
	# 	str(trans[i][5]) + sp + str(trans[i][6]) + sp +  info + '\n'
	# 	g2o.write(line)


	g2o.write("FIX 0\n")
	g2o.close()




def get_input(noisy_input):
    with open(noisy_input, 'r') as data_file:
        lines = data_file.readlines()
        xN = []
        yN = []
        zN = []
        QxN = []
        QyN = []
        QzN = []
        QwN = []
        ldmk = []
        measurement = []
    for idx, line in enumerate(lines):
        line = line.strip().split(' ')
        if idx==0:
            (pose_count, ldmk_count, obs_count) = line
            pose_count, ldmk_count, obs_count = int(pose_count), int(ldmk_count), int(obs_count)
            xN = np.empty((pose_count))
            yN = np.empty((pose_count))
            zN = np.empty((pose_count))
            QxN = np.empty((pose_count))
            QyN = np.empty((pose_count))
            QzN = np.empty((pose_count))
            QwN = np.empty((pose_count))
            ldmk = np.empty((ldmk_count,3))
            measurement = np.empty((obs_count,4))
        elif idx<pose_count+1:
            (x, y, z, Qx, Qy, Qz, Qw) = line
            xN[idx-1], yN[idx-1], zN[idx-1], QxN[idx-1], QyN[idx-1], QzN[idx-1], QwN[idx-1] = float(x), float(y), float(z), float(Qx), float(Qy), float(Qz), float(Qw) 
        elif idx<pose_count+ldmk_count+1:
            (x, y, z) = line
            ldmk[idx-pose_count-1, :] = np.asarray([float(x), float(y), float(z)])
        else:
            (ps, pt, u, v) = line
            measurement[idx-pose_count-ldmk_count-1,:] = np.array([float(ps), float(pt), float(u), float(v)])
    
    return xN, yN, zN, QxN, QyN, QzN, QwN, ldmk, measurement, pose_count, ldmk_count, obs_count



if __name__ == '__main__':
	dirc = os.path.dirname(argv[1])
	xN, yN, zN, QxN, QyN, QzN, QwN , noise_ldmk, range_bearing, pose_count, ldmk_count, obs_count = get_input(argv[1])
	writeG2O(xN, yN, zN, QxN, QyN, QzN, QwN, argv[2])
