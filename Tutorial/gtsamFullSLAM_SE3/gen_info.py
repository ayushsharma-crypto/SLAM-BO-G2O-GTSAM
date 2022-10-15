from sys import argv
import numpy as np
import matplotlib.pyplot as plt
import bz2
from tqdm import tqdm as tm
from scipy.spatial.transform import Rotation as R
import gtsam 

def get_traj():
    X, Y, Z = [50], [50], [50]
    length = 10
    poses = 20
    step = float(length)/poses
    for idx in range(5):
        for _ in range(1,poses):
            X.append(X[len(X)-1])
            Y.append(Y[len(Y)-1])
            Z.append(Z[len(Z)-1])
            if idx==0:
                X[len(X)-1] = X[len(X)-1] + step
            elif idx==1:
                Y[len(Y)-1] = Y[len(Y)-1] + step
            elif idx==2:
                Z[len(Z)-1] = Z[len(Z)-1] - step
            elif idx==3:
                X[len(X)-1] = X[len(X)-1] - step
            elif idx==4:
                Z[len(Z)-1] = Z[len(Z)-1] + step
    
    LT = [np.eye(4)]
    LT[0][:3,:3] = R.from_euler('y',90, True).as_matrix()
    LT[0][:3,3] = np.array([X[0],Y[0],Z[0]])
    for idx in range(5):
        for _ in range(1, poses):
            lt_idx = len(LT)-1
            LT.append(np.array(LT[len(LT)-1]))
            lt_idx = len(LT)-1
            LT[lt_idx][:3,3] = np.array([X[lt_idx],Y[lt_idx],Z[lt_idx]])
        rot = np.eye(4)
        if idx==0:
            axis = 'x'
            deg = 270
        elif idx==1:
            axis='y'
            deg = 90
        elif idx==2:
            axis='x'
            deg = 270
        elif idx==3:
            axis='x'
            deg = 270
        elif idx==4:
            axis='y'
            deg = 90 
        rot[:3,:3] = R.from_euler(axis,deg, True).as_matrix()
        LT[len(LT)-1] = LT[len(LT)-1] @ rot
    
    Qx, Qy, Qz, Qw = [], [], [], []
    for T in LT:
        [qx, qy, qz, qw] = R.from_matrix(T[:3,:3]).as_quat()
        Qx.append(qx)
        Qy.append(qy)
        Qz.append(qz)
        Qw.append(qw)
    
    val = 2
    LDMK =  [
        [X[(poses-1)]+np.random.uniform(val,1.5*val), Y[(poses-1)]+np.random.uniform(-val,val), Z[(poses-1)]+np.random.uniform(-val,val)],
        [X[2*(poses-1)]+np.random.uniform(-val,val), Y[2*(poses-1)]+np.random.uniform(val,1.5*val), Z[2*(poses-1)]+np.random.uniform(-val,val)],
        [X[3*(poses-1)]+np.random.uniform(-val,val), Y[3*(poses-1)]+np.random.uniform(-val,val), Z[3*(poses-1)]+np.random.uniform(-1.5*val,-val)],
        [X[4*(poses-1)]+np.random.uniform(-1.5*val,-val), Y[4*(poses-1)]+np.random.uniform(-val,val), Z[4*(poses-1)]+np.random.uniform(-val,val)],
        [X[5*(poses-1)]+np.random.uniform(-val,val), Y[5*(poses-1)]+np.random.uniform(-val,val), Z[5*(poses-1)]+np.random.uniform(val,1.5*val)],
        # [X[6*(poses-1)]+np.random.uniform(-val,val), Y[6*(poses-1)]+np.random.uniform(-1.5*val,-val), Z[6*(poses-1)]+np.random.uniform(-val,val)],
    ]


    K = gtsam.Cal3_S2(565.6008952774197, 565.6008952774197, 1000.0 ,320.5, 240.5 )
    measurement = []
    for idx in range(len(X)):
        if idx < (poses-1):
            lidx = 0
        elif idx<2*(poses-1):
            lidx=1
        elif idx<3*(poses-1):
            lidx=2
        elif idx<4*(poses-1):
            lidx=3
        elif idx<5*(poses-1):
            lidx=4
        else:
            break
            
        camera = gtsam.PinholeCameraCal3_S2(gtsam.Pose3(LT[idx]),K)
        point = LDMK[lidx]
        measurement.append([idx, lidx, *camera.project(point)])
    return X, Y, Z, Qx, Qy, Qz, Qw, LDMK, measurement

def add_noise_ldmk(ldmk):
    noise_ldmk = []
    for idx, L in enumerate(ldmk):
        noise_ldmk.append([
            L[0]+ np.random.uniform(-4,4),
            L[1]+ np.random.uniform(-4,4),
            L[2]+ np.random.uniform(-4,4),
        ])
    return noise_ldmk

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

def read_bal_data(file_name):
    with bz2.open(file_name, "rt") as file:
        n_cameras, n_points, n_observations = map(
            int, file.readline().split())

        camera_indices = np.empty(n_observations, dtype=int)
        point_indices = np.empty(n_observations, dtype=int)
        points_2d = np.empty((n_observations, 2))

        for i in range(n_observations):
            camera_index, point_index, x, y = file.readline().split()
            camera_indices[i] = int(camera_index)
            point_indices[i] = int(point_index)
            points_2d[i] = [float(x), float(y)]

        camera_params = np.empty(n_cameras * 9)
        for i in range(n_cameras * 9):
            camera_params[i] = float(file.readline())
        camera_params = camera_params.reshape((n_cameras, -1))

        points_3d = np.empty(n_points * 3)
        for i in range(n_points * 3):
            points_3d[i] = float(file.readline())
        points_3d = points_3d.reshape((n_points, -1))

    return camera_params, points_3d, camera_indices, point_indices, points_2d

def rotation_matrix(rot_vec):
    theta = np.linalg.norm(rot_vec)
    with np.errstate(invalid='ignore'):
        v = rot_vec / theta
        v = np.nan_to_num(v)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    [v1, v2, v3] = v
    cross_product_mat = np.array([
        [0, -v3, v2],
        [v3, 0, -v1],
        [-v2, v1, 0],
    ])
    rot = cos_theta * np.eye(3)
    rot = rot + sin_theta*cross_product_mat
    rot = rot + (1-cos_theta)*(np.outer(v,v))
    return rot
    
def get_poses(r_list, t_list):
    X, Y, Z, Qx, Qy, Qz, Qw = [], [], [], [], [], [], []
    for idx in range(len(r_list)):
        rot_vec = r_list[idx]
        t_vec = t_list[idx]
        X.append(t_vec[0])
        Y.append(t_vec[1])
        Z.append(t_vec[2])
        # R = rotation_matrix(rot_vec)
        [qx, qy, qz, qw] = R.from_rotvec(rot_vec).as_quat()
        Qx.append(qx)
        Qy.append(qy)
        Qz.append(qz)
        Qw.append(qw)
    return X, Y, Z, Qx, Qy, Qz, Qw

def addNoise(X, Y, Z, Qx, Qy, Qz, Qw):
	XN = np.zeros(len(X)); YN = np.zeros(len(Y)); ZN = np.zeros(len(Z))
	QxN = np.zeros(len(Qx)); QyN = np.zeros(len(Qy)); QzN = np.zeros(len(Qz)); QwN = np.zeros(len(Qw))

	XN[0] = X[0]; YN[0] = Y[0]; ZN[0] = Z[0]; QxN[0] = Qx[0]; QyN[0] = Qy[0]; QzN[0] = Qz[0]; QwN[0] = Qw[0]

	for i in range(1, len(X)):
		# Get T2_1
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
		dyaw, dpitch, droll = list(R.from_matrix(T2_1[0:3, 0:3]).as_euler('zyx'))
		
		# Add noise
		if(i<5):
			xNoise = 0; yNoise = 0; zNoise = 0; rollNoise = 0; pitchNoise = 0; yawNoise = 0
		else:
			xNoise = 0; yNoise = 0; zNoise = 0; rollNoise = 0.005; pitchNoise = 0.005; yawNoise = 0.005

		dx += xNoise; dy += yNoise; dz += zNoise
		dyaw += yawNoise; dpitch += pitchNoise; droll += rollNoise

		# Convert to T2_1'
		R2_1N = R.from_euler('zyx', [dyaw, dpitch, droll]).as_matrix()
		
		T2_1N = np.identity(4)
		T2_1N[0:3, 0:3] = R2_1N

		T2_1N[0, 3] = dx
		T2_1N[1, 3] = dy
		T2_1N[2, 3] = dz

		# Get T2_w' = T1_w' . T2_1'
		p1 = (XN[i-1], YN[i-1], ZN[i-1], QxN[i-1], QyN[i-1], QzN[i-1], QwN[i-1])
		R1_wN = R.from_quat([p1[3], p1[4], p1[5], p1[6]]).as_matrix()
		
		T1_wN = np.identity(4)
		T1_wN[0:3, 0:3] = R1_wN

		T1_wN[0, 3] = p1[0] 
		T1_wN[1, 3] = p1[1]
		T1_wN[2, 3] = p1[2]

		T2_wN = np.dot(T1_wN, T2_1N)

		# Get x2', y2', z2', qx2', qy2', qz2', qw2'
		x2N, y2N, z2N = T2_wN[0, 3], T2_wN[1, 3], T2_wN[2, 3]
		qx2N, qy2N, qz2N, qw2N = list(R.from_matrix(T2_wN[0:3, 0:3]).as_quat())

		XN[i] = x2N; YN[i] = y2N; ZN[i] = z2N
		QxN[i] = qx2N; QyN[i] = qy2N; QzN[i] = qz2N; QwN[i] = qw2N

	return (XN, YN, ZN, QxN, QyN, QzN, QwN)



if __name__=="__main__":
    
    X, Y, Z, Qx, Qy, Qz, Qw, LDMK, measurement = get_traj()
    xN, yN, zN, QxN, QyN, QzN, QwN = addNoise(X, Y, Z, Qx, Qy, Qz, Qw)
    noise_ldmk = add_noise_ldmk(LDMK)
    
    write_initialisation(argv[1], X, Y, Z, Qx, Qy, Qz, Qw, LDMK, measurement)
    write_initialisation(argv[2], xN, yN, zN, QxN, QyN, QzN, QwN , noise_ldmk, measurement)