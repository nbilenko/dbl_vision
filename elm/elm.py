import numpy as np
import sklearn

def run_elm(nH = 50, xval = 10, fraction = 0.1, features = "random", act = "sigmoid", rcond = 1e-15):
	if act == "sigmoid":
		activation = lambda proj: 1.0 / (1.0 + np.exp(- proj))
	elif act == "rbf":
		activation = lambda proj: np.exp(- pow(proj, 2.0))
	elif act == "tanh":
		activation = lambda proj: np.tanh(proj)
	else:
		print "Wrong activation function specification"

	if features == "random":
		data, vdata, targets, vtargets = load_data()
		nTR, nF = data.shape
		w = 2.0 * np.random.rand(nF, nH) - 1.0
		b = 2.0 * np.random.rand(nH) - 1.0
		proj = np.dot(data, w) + b
		vproj = np.dot(vdata, w) + b
	elif features == "hog":
		data, vdata, targets, vtargets = load_data(delays = [3])
		nTR, nF = data.shape
		proj = get_hog_proj(data)
		vproj = get_hog_proj(vdata)
	else:
		print "Wrong feature specification"

	H = activation(proj)
	H_v = activation(vproj)
	chunklen = 10
	nchunks = int(fraction*nTR/chunklen)
	allinds = range(nTR)
	indchunks = zip(*[iter(allinds)]*chunklen)

	Bs = []
	for xv in range(xval):
		np.random.shuffle(indchunks)
		heldinds = [ind for chunk in indchunks[:nchunks] for ind in chunk]
		notheldinds = list(set(allinds)-set(heldinds))
		d = data[notheldinds]
		t = targets[notheldinds]
		if features == "random":
			iproj = np.dot(d, w) + b
		elif features == "hog":
			iproj = get_hog_proj(d)
		Bs.append(train(iproj, t, activation))
	B = np.array(Bs).mean(0)

	tpred = np.dot(H, B)
	tcs = rowcorr(tpred.T, targets.T)
	print('Training average correlation: %.04f' % tcs.mean())

	## validation misfit
	vpred = np.dot(H_v, B)
	vcs = rowcorr(vpred.T, vtargets.T)
	print('Validation average correlation: %04f' % vcs.mean())
	return tcs, vcs

def get_hog_proj(data):
	from skimage.feature import hog
	features = []
	imsize = (int(np.sqrt(data.shape[1])),int(np.sqrt(data.shape[1]))) 
	for i in range(data.shape[0]):
		features.append(hog(data[i].reshape(imsize)))
	features = np.array(features)
	return features

def train(proj, t, activation, rcond = 1e-15):
	# hidden-layer activations
	H = activation(proj)

	# pseudo-inverse of activations
	H_dagger = np.linalg.pinv(H, rcond)

	# solve for output weight matrix
	B = np.dot(H_dagger, t)
	return B

def rowcorr(a, b):
	'''Correlations between corresponding matrix rows
	'''
	cs = np.zeros((a.shape[0]))
        for idx in range(a.shape[0]):
            cs[idx] = np.corrcoef(a[idx], b[idx])[0,1]
	return cs

def load_data(rbfsize = 5, delays = [2, 3, 4]):
	tstim = add_delays(np.load("../../data/stim-t.npy"), delays)
	vstim = add_delays(np.load("../../data/stim-v.npy"), delays)
	tdata = np.load("../../data/evdata-t-%d.npy" % rbfsize)
	vdata = np.load("../../data/evdata-v-%d.npy" % rbfsize)
	return tstim, vstim, tdata, vdata

def add_delays(data, delays=[2, 3, 4], method='insert'):
    '''Adds delays to data
        Inputs:
        data (timepoints x features)
        delays(optional) = list of delays, default = [2, 3, 4]
        method(optional) = 'insert' or 'concat' (whether to insert the delays per feature or concatenate)

        Outputs:
        delayed_data (timepoints x features)
    '''

    nT = data.shape[0]
    nF = data.shape[1]
    nD = len(delays)

    delayed_data = np.zeros((nT, nF*nD))

    for di, d in enumerate(delays):
        delayed_data[d:, di:nF*nD:nD] = data[:-d]
        delayed_data[:d, di:nF*nD:nD] = data[-d:]

    return delayed_data