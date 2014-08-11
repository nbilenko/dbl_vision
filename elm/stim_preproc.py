from matplotlib.pyplot import imread
from scipy.misc import imresize
import numpy as np

stimdir = "/auto/k2/stimuli/movies/"
imname = "im%07d.png"
outdir = "/auto/k6/nbilenko/preproc_data/movie/"

runs = {"trn": [stimdir+"trn%03d/" % i for i in range(1, 13)], "val": [stimdir+"val%03d_3min/" % i for i in range(1, 4)]}
stim_nums = {"trn": 9001, "val": 2701}
imsize = (128, 128, 3)

for stim_type in ("trn", "val"):
	for ri, run in enumerate(runs[stim_type]):
		images = []
		for imnum in range(stim_nums[stim_type]):
			images.append(imresize(imread(run+imname % imnum), 0.25).reshape(np.product(imsize)))
		images = np.array(images)
		np.save(outdir+"%s%03d_stim.npy" % (stim_type, ri+1), images)


# import movie_utils
# fname = "/auto/k6/nbilenko/preproc_data/movie/stim_%s_0%d.mat"
# outfile = "/auto/k6/nbilenko/preproc_data/movie/stim_%s_0%d.npy"
# sessions = 3
# imshape = (128, 128, 3)

# for sess in range(1, sessions+1):
# 	for stim_type in ("trn", "val"):
# 		stim = movie_utils.load_table_file(

# 			fname % (stim_type, sess))["data"]
# 		images = []
# 		for imnum in range(stim.shape[1]):
# 			images.append(stim[:, imnum].reshape((128, 128, 3), order ="F"))
# 		images = np.array(images).reshape(stim.shape[1], np.product(imshape))
# 		np.save(outfile % (stim_type, sess), images)