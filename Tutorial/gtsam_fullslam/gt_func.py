
import math
import numpy as np
import matplotlib.pyplot as plt

def getTheta(X ,Y):
	THETA = [None]*len(X)
	for i in range(1, len(X)-1):
		if(X[i+1] == X[i-1]):
			if (Y[i+1]>Y[i-1]):
				THETA[i] = math.pi/2
			else:
				THETA[i] = 3*math.pi/2
			continue

		THETA[i] = math.atan((Y[i+1]-Y[i-1])/(X[i+1]-X[i-1]))

		if(X[i+1]-X[i-1] < 0):
			THETA[i] += math.pi 

	if X[1]==X[0]:
		if Y[1] > Y[0]:
			THETA[0] = math.pi/2
		else:
			THETA[0] = 3*math.pi/2
	else:
		THETA[0] = math.atan((Y[1]-Y[0])/(X[1]-X[0]))

	if X[-1] == X[len(Y)-2]:
		if Y[1] > Y[0]:
			THETA[-1] = math.pi/2
		else:
			THETA[-1] = 3*math.pi/2
	else:
		THETA[-1] = math.atan((Y[-1]-Y[len(Y)-2])/(X[-1]-X[len(Y)-2]))

	return THETA


def gt_lshape():
    X, Y = [-10], [3]
    leng = 9.0 
    num = 20
    step = float(leng)/num
    for i in range(1,num):
        X.append(X[i-1]+step)
        Y.append(Y[i-1])
    for i in range(1,num):
        X.append(X[len(X)-1])
        Y.append(Y[len(Y)-1]+step)
    THETA = getTheta(X, Y)
    pose = np.zeros((len(X), 3))
    pose[:,0] = X
    pose[:,1] = Y
    pose[:,2] = THETA
    return X, Y, THETA, pose

def make_landmarks(X, Y, batch_size, hop):
	assert len(X)==len(Y)
	assert batch_size <= len(X)
	assert batch_size >= hop
	limit = 2
	landmarks = []
	i = 0
	while i+batch_size<=len(X):
		curr_X = X[i:i+batch_size]
		curr_Y = Y[i:i+batch_size]
		val = np.random.uniform(-limit,limit)
		landmarks.append([np.mean(curr_X)+val, np.mean(curr_Y)+val])
		i+=hop
	if i<len(X):
		curr_X = X[i:]
		curr_Y = Y[i:]
		val = np.random.uniform(-limit,limit)
		landmarks.append([np.mean(curr_X)+val, np.mean(curr_Y)+val])
	return np.array(landmarks)


def get_bearing(X, L):
	return  math.atan2(-L[1]+X[1],-L[0]+X[0],)


def get_range(X, L):
	return np.sqrt((X[1]-L[1])**2+(X[0]-L[0])**2)


def get_pose_landmark_edge(X, Y, LDMK, batch_size, hop):
    i = 0 
    j = 0
    range_bearing = []
    while i+batch_size<=len(X):
        curr_X = X[i:i+batch_size]
        curr_Y = Y[i:i+batch_size]
        for idx in range(batch_size):
            range_bearing.append([
                get_range(LDMK[j], [curr_X[idx], curr_Y[idx]]),
                get_bearing(LDMK[j], [curr_X[idx], curr_Y[idx]]),
            ])
        j=j+1
        i+=hop
    if i<len(X):
        curr_X = X[i:]
        curr_Y = Y[i:]
        for idx in range(len(X)-i):
            range_bearing.append([
                get_range(LDMK[j], [curr_X[idx], curr_Y[idx]]),
                get_bearing(LDMK[j], [curr_X[idx], curr_Y[idx]]),
            ])
    return np.array(range_bearing)

def plot_pose_landmark(X,Y, LDMK, batch_size, hop):
    ax = plt.subplot(111)
    ax.plot(X,Y, 'k-')
    ax.plot(X,Y, 'go')
    for L in LDMK:
        ax.plot(L[0], L[1], 'bo')

    i = 0 
    j = 0
    while i+batch_size<=len(X):
        curr_X = X[i:i+batch_size]
        curr_Y = Y[i:i+batch_size]
        for idx in range(batch_size):
            ax.plot([curr_X[idx], LDMK[j][0]],[curr_Y[idx], LDMK[j][1]], 'r-')
        j=j+1
        i+=hop
    if i<len(X):
        curr_X = X[i:]
        curr_Y = Y[i:]
        for idx in range(len(X)-i):
            ax.plot([curr_X[idx], LDMK[j][0]],[curr_Y[idx], LDMK[j][1]], 'r-')

    plt.show()


if __name__=="__main__":
    X, Y, THETA, pose = gt_lshape()
    batch_size, hop = 5, 3
    LDMK = make_landmarks(X, Y, batch_size, hop)
    range_bearing = get_pose_landmark_edge(X, Y, LDMK, batch_size, hop)
    plot_pose_landmark(X, Y, LDMK, batch_size, hop)