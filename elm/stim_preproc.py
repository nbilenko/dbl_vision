import movie_utils
fname = "/auto/k6/nbilenko/preproc_data/movie/stim_%s_0%d.mat"
outfile = "/auto/k6/nbilenko/preproc_data/movie/stim_%s_0%d.npy"
sessions = 3
imshape = (128, 128, 3)

for sess in range(1, sessions+1):
	for stim_type in ("trn", "val"):
		stim = movie_utils.load_table_file(tfname % (stim_type, sess))["data"]
		images = []
		for imnum in range(stim.shape[1]):
			images.append(stim[:, imnum].reshape((128, 128, 3), order ="F"))
		images = np.array(images).reshape(stim.shape[1], np.product(imshape))
		np.save(outfile % (stim_type, sess), images)