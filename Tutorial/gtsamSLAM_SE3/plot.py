from sys import argv
import numpy as np
import matplotlib.pyplot as plt


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



def plot(X,Y, Z, LDMK, xN, yN, zN, noise_ldmk, xO, yO, zO, opt_ldmk):
    
    fig = plt.figure()
    ax = plt.axes(projection="3d")

    LDMK = np.array(LDMK)
    noise_ldmk = np.array(noise_ldmk)
    opt_ldmk = np.array(opt_ldmk)
    ax.scatter(X, Y, Z, s=10, c="g")
    ax.scatter(xN,yN, zN,s=10, c='r')
    ax.scatter(xO, yO, zO, s=10, c='b')

    ax.scatter3D(LDMK[:,0], LDMK[:,1],  LDMK[:,2], s=[70 for _ in range(len(LDMK))], c= 'g', marker='x', label="ground truth landmarks")
    ax.scatter(noise_ldmk[:,0], noise_ldmk[:,1], noise_ldmk[:,2], s=[60 for _ in range(len(LDMK))], marker='.' ,c='r', label="noisy landmarks")
    ax.scatter(opt_ldmk[:,0], opt_ldmk[:,1], opt_ldmk[:,2], s=[50 for _ in range(len(LDMK))], marker="^", c='b', label="optimised landmarks")


    ax.plot3D(X,Y, Z, 'g-', label="ground truth traj.")
    ax.plot(xN, yN, zN, 'r-', label="noisy traj.")
    ax.plot(xO, yO, zO, 'b-', label="optimised traj.")


    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.legend(loc="upper left")
    plt.show()


if __name__=="__main__":
    
    X, Y, Z, QX, QY, QZ, QW, LDMK, MSg, pose_count, ldmk_count, obs_count = get_input(argv[1])
    xN, yN, zN, QxN, QyN, QzN, QwN, noise_ldmk, MSn, pose_count, ldmk_count, obs_count = get_input(argv[2])
    xO, yO, zO, QxO, QyO, QzO, QwO, opt_ldmk, MSo, pose_count, ldmk_count, obs_count = get_input(argv[3])
    # last_idx = int(argv[4])
    # plot(X[:last_idx],Y[:last_idx], Z[:last_idx], LDMK, xN, yN, zN, noise_ldmk, xO, yO, zO, opt_ldmk)
    plot(X,Y, Z, LDMK, xN, yN, zN, noise_ldmk, xO, yO, zO, opt_ldmk)