from sys import argv
import numpy as np
import math
import gtsam
from tqdm import tqdm as tm
from scipy.spatial.transform import Rotation as R


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

def pose_symbol(i):
    	return gtsam.symbol_shorthand.X(int(i))

def ldmk_symbol(i):
    	return gtsam.symbol_shorthand.L(int(i))

def get_trans(p1, p2):
    
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
    T2_w[2, 3] = p2[2]  ;\
	T2_1 = np.dot(np.linalg.inv(T1_w), T2_w);\
	return T2_1

def compute_ds(T2_1):
	dx = T2_1[0][3];\
	dy = T2_1[1][3];\
	dz = T2_1[2][3];\
	dqx, dqy, dqz, dqw = list(R.from_matrix(T2_1[:3,:3]).as_quat());\
    return dx, dy, dz, dqx, dqy, dqz, dqw

def get_between_factor(p1, p2):
    T2_1 = get_trans(p1, p2);\
	dx, dy, dz, dqx, dqy, dqz, dqw = compute_ds(T2_1);\
	return gtsam.Pose3(make_T_from_xyz(dx, dy, dz, dqx, dqy, dqz, dqw))

def make_T_from_xyz(xN, yN, zN, QxN, QyN, QzN, QwN):
		r = R.from_quat([QxN,QyN, QzN, QwN]).as_matrix()
		t = np.array([xN, yN, zN])
		T = np.eye(4)
		T[:3,:3] = r
		T[:3,3] = t
		return T

def gtsam_optimisation(xN, yN, zN, QxN, QyN, QzN, QwN, noise_ldmk, range_bearing, pose_sigma, print_log=False):
	graph = gtsam.NonlinearFactorGraph()
	initial = gtsam.Values()
	priorModel = gtsam.noiseModel.Diagonal.Sigmas(pose_sigma)
	K = gtsam.Cal3_S2(565.6008952774197, 565.6008952774197, 1000.0 ,320.5, 240.5 )


	if print_log:
		print("Adding poses...")
        
	for idx in range(len(xN)):
		PT = make_T_from_xyz(xN[idx], yN[idx], zN[idx], QxN[idx],QyN[idx], QzN[idx], QwN[idx]);\
        initial.insert(
			pose_symbol(idx), 
			gtsam.Pose3(PT)
		)

	PT = make_T_from_xyz(xN[0], yN[0], zN[0], QxN[0],QyN[0], QzN[0], QwN[0])
	graph.add(gtsam.PriorFactorPose3(pose_symbol(0), gtsam.Pose3(PT), priorModel))


	if print_log:
		print("Adding odom edges...")
        

	for idx in range(len(xN)-1):
			graph.add(gtsam.BetweenFactorPose3(
				pose_symbol(idx),
				pose_symbol(idx+1),
				get_between_factor(
                    [xN[idx], yN[idx], zN[idx], QxN[idx], QyN[idx], QzN[idx], QwN[idx]],
                    [xN[idx+1], yN[idx+1], zN[idx+1], QxN[idx+1], QyN[idx+1], QzN[idx+1], QwN[idx+1]]
                ),
				priorModel
			))
    
	if print_log:
		print("Adding ldmk...")

	for idx in range(len(noise_ldmk)):
		initial.insert(
			ldmk_symbol(idx), 
			gtsam.Point3(*noise_ldmk[idx])
		)


	if print_log:
		print("observation edges...")


	for [ps, pt, u, v] in range_bearing:
		observation_edge = gtsam.GenericProjectionFactorCal3_S2(
                np.array([u,v]),
				gtsam.noiseModel.Isotropic.Sigma(2, 1000.0),
                pose_symbol(ps),
                ldmk_symbol(pt),
                K,
			)
		graph.add(observation_edge)
        

	# print("GRAPH START : \n\n", graph)
	
	params = gtsam.LevenbergMarquardtParams()
	if print_log:
		params.setVerbosity("Termination")  # this will show info about stopping conds
	params.setMaxIterations(500)
	optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)

	result = optimizer.optimize()

	if print_log:
		print("Optimization complete")
		print("initial error = ", graph.error(initial))
		print("final error = ", graph.error(result))
	# marginals = gtsam.Marginals(graph, result)


	final_x = []
	final_y = [] 
	final_z = [] 
	final_qx = [] 
	final_qy = [] 
	final_qz = [] 
	final_qw = [] 
	final_ldmk = [] 
	for idx in range(len(xN)):
		T = result.atPose3(pose_symbol(idx)).matrix()
		final_x.append(T[0,3])
		final_y.append(T[1,3])
		final_z.append(T[2,3])
		quat = R.from_matrix(T[:3,:3]).as_quat()
		final_qx.append(quat[0])
		final_qy.append(quat[1])
		final_qz.append(quat[2])
		final_qw.append(quat[3])
		# print("X",idx ," -> ", marginals.marginalCovariance(pose_symbol(idx)))
        
	for idx in range(len(noise_ldmk)):
		final_ldmk.append(result.atPoint3(ldmk_symbol(idx)))
		# print("L",idx ," -> ", marginals.marginalCovariance(ldmk_symbol(idx)))

	return final_x, final_y, final_z, final_qx, final_qy, final_qz, final_qw, final_ldmk

def write_initialisation(filepath, xN, yN, zN, QxN, QyN, QzN, QwN, LDMK, measurements):
    with open(filepath, 'w') as data_file:
        line  = str(len(xN)) + ' ' + str(len(LDMK)) + ' ' + str(len(measurements)) + '\n';
        data_file.write(line);
        for i in range(len(xN)):
            line = str(xN[i]) + ' ' + str(yN[i]) + ' ' + str(zN[i]) + ' ' + str(QxN[i]) + ' ' + str(QyN[i]) + ' ' + str(QzN[i]) + ' ' + str(QwN[i]) + '\n';
            data_file.write(line);
        for i in range(len(LDMK)):
            line = str(LDMK[i][0]) + ' ' + str(LDMK[i][1]) + ' ' + str(LDMK[i][2]) + '\n';
            data_file.write(line);
        for i in range(len(measurements)):
            line = str(measurements[i][0]) + ' ' + str(measurements[i][1]) + ' ' + str(measurements[i][2]) + ' ' + str(measurements[i][3]) + '\n';
            data_file.write(line);




if __name__=="__main__":
    xN, yN, zN, QxN, QyN, QzN, QwN , noise_ldmk, range_bearing, pose_count, ldmk_count, obs_count = get_input(argv[1])


    pose_sigma = np.array([20]*6)

    final_x, final_y, final_z, final_qx, final_qy, final_qz, final_qw, final_ldmk = gtsam_optimisation(xN, yN, zN, QxN, QyN, QzN, QwN , noise_ldmk, range_bearing, pose_sigma, True)

    write_initialisation(argv[2], final_x, final_y, final_z, final_qx, final_qy, final_qz, final_qw, final_ldmk, range_bearing)