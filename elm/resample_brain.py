#!/usr/bin/env python

import argparse
import numpy as np
import tables
from os.path import basename, splitext

##def draw_slices(volume, nr, nc, fname=None):
##    import itertools
##    import matplotlib.pyplot as plt
##    fig = plt.figure()
##    tmin = volume.min()
##    tmax = volume.max()
##    ###
##    ledges = np.linspace(0, 1, nc+1)[:-1]
##    bedges = np.linspace(1, 0, nr+1)[1:]
##    width = 1/float(nc)
##    height = 1/float(nr)
##    ###
##    bottoms,lefts = zip(*list(itertools.product(bedges, ledges)))
##    ###
##    for ni,sl in enumerate(np.split(volume, len(volume))):
##        ax = fig.add_axes((lefts[ni], bottoms[ni], width, height))
##        ax.imshow(sl.squeeze(), vmin=tmin, vmax=tmax, interpolation="nearest")
##        ax.set_xticks([])
##        ax.set_yticks([])
##    if not fname is None:
##        plt.savefig(fname)
##        plt.close()

def define_weights(mask, volume_shape, corr_len):
    # define the (fake) coordinate system
    nx, ny, nz = volume_shape
    x, y, z = np.mgrid[0:nx, 0:ny, 0:nz]
    x = x.astype(float)
    y = y.astype(float)
    z = z.astype(float)
    # generate weight matrix ...
    W = np.zeros(mask.shape)
    nnodes, nvoxels = mask.shape
    for n in range(nnodes):
        # locate voxel and evaluate square distance to others
        i, j, k = np.nonzero(mask[n,:].reshape(volume_shape))
        s = (x - i) ** 2 + (y - j) ** 2 + (z - k) ** 2
        # compute normalized weights for this node (row of W)
        w  = np.exp(- s / (2 * corr_len ** 2))
        w /= w.sum()
        W[n,:] = w.ravel()
    return W

def load_voxel_data(fname_voxel_data):
    with tables.open_file(fname_voxel_data) as f:
        data = f.root.data.read()
    return np.transpose(data, axes=(3,2,1,0))

def resample_brain_data(fname_voxel_data, fname_mask, corr_len, fname_node_data):
    # load input data: (nt, nxvoxels, nyvoxels, nzvoxels)
    voxel_data = load_voxel_data(fname_voxel_data)
    nt, nx, ny, nz = voxel_data.shape
    print('voxel volume is (%i,%i,%i) over %i time samples' % (
        nx, ny, nz, nt))
    # load the mask: (nnodes, nvoxels)
    mask = np.load(fname_mask)
    nnodes, nvoxels = mask.shape
    assert nvoxels == nx * ny * nz
    # compute the weight matrix (nnodes, nvoxels)
    W = define_weights(mask, (nx, ny, nz), corr_len)
    # resample and save; node data will be: (nt, nnodes)
    voxel_data_flat = voxel_data.reshape((nt, nvoxels))
    node_data = np.dot(voxel_data_flat, W.T)
    np.save(fname_node_data, node_data)

##    # plot the brain data for an arbitrary time point and the weights for an
##    # also-arbitrary couple of nodes
##    t_plot = 20
##    draw_slices(voxel_data[t_plot,:,:,:], 10, 10, 'brain.t_%i.png' % (t_plot))
##    for n in range(0, nnodes, 30):
##        print n
##        w_n = W[n,:].reshape((nx,ny,nz))
##        draw_slices(w_n, 10, 10,
##            'weights.corrlen_%2.2i.node_%3.3i.png' % (corr_len, n))

def main():
    p = argparse.ArgumentParser(
        description='resample voxel data onto a known mask with gaussian smoothing')
    p.add_argument('--voxel-data', type=str, required=True,
            metavar='voxel-data-file',
            help='numpy file containing voxel data (nt, nxvoxels, nyvoxels, nzvoxels)')
    p.add_argument('--mask', type=str, required=True,
            metavar='mask-file',
            help='numpy file containing the mask (nnodes, nvoxels)')
    p.add_argument('--corr-len', type=float, required=True,
            metavar='voxel-corr-len',
            help='correlation length for smoothing (units: voxel widths)')
    args = p.parse_args()
    fname, _ = splitext(args.voxel_data)
    outfile = 'node-data.corr-len-%i.%s' % (args.corr_len, basename(fname))
    resample_brain_data(args.voxel_data, args.mask, args.corr_len, outfile)

if __name__ == '__main__':
    main()
