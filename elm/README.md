# Brain data resampling w/ Gaussian RBFs

Resample voxel volume data onto the unstructured Brainlove nodes using
collocated Gaussian RBFs as the weighting function.

## Usage

The script takes, as input:
* the (pytables) hdf5 input volume; shape: (nz, ny, nx, nt)
* the flattened node-to-voxel location mask; shape (nnodes, nx * ny * nz)
* the correlation length of the RBF, i.e. the std. dev. of the Gaussian, in
  units of voxel widths

For example:

    ./resample_brain.py --voxel-data MNI-v-movies.hf5 --mask mask.npy --corr-len 10

The result is written to the current directory as an npy file. For this above
example, one would have a file named:

    node-data.corr-len-10.MNI-v-movies.npy
