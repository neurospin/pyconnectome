##########################################################################
# NSAp - Copyright (C) CEA, 2017
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Common tools to filter tractograms.
"""

# System import
from __future__ import print_function
import os

# Third party import
import nibabel
from nibabel.streamlines.tractogram import Tractogram
import numpy
from dipy.viz.colormap import line_colors
from dipy.viz import fvtk
from dipy.core.gradients import gradient_table
from dipy.tracking.distances import approx_polygon_track
from dipy.tracking.utils import length
import dipy.core.optimize as opt
import dipy.tracking.life as dpilife
import matplotlib.pyplot as plt
import matplotlib


def life(dwifile, bvecsfile, bvalsfile, tractogramfile, outdir,
         display_tracks=False, verbose=0):
    """ Linear fascicle evaluation (LiFE)
    Evaluating the results of tractography algorithms is one of the biggest
    challenges for diffusion MRI. One proposal for evaluation of tractography
    results is to use a forward model that predicts the signal from each of a
    set of streamlines, and then fit a linear model to these simultaneous
    prediction.

    Parameters
    ----------
    dwifile: str
        the path to the diffusion dataset.
    bvecsfile: str
        the path to the diffusion b-vectors.
    bvalsfile: str
        the path to the diffusion b-values.
    tractogramfile: str
        the path to the tractogram.
    outdir: str
        the destination folder.
    display_tracks: bool, default False
        if True render the tracks.
    verbose: int, default 0
        the verbosity level.

    Returns
    -------
    life_weights_file: str
        a file containing the fiber track weights.
    life_weights_snap: str
        a snap with the distrubution of weights.
    spatial_error_file: str
        the model root mean square error.
    tracks_snap: str
        a snap with the tracks.
    """
    # Load diffusion data and tractogram
    bvecs = numpy.loadtxt(bvecsfile)
    bvals = numpy.loadtxt(bvalsfile)
    gtab = gradient_table(bvals, bvecs)
    im = nibabel.load(dwifile)
    data = im.get_data()
    trk = nibabel.streamlines.load(tractogramfile)
    if verbose > 0:
        print("[info] Diffusion shape: {0}".format(data.shape))
        print("[info] Number of tracks: {0}".format(len(trk.streamlines)))

    # Express the tractogram in voxel coordiantes
    inv_affine = numpy.linalg.inv(trk.affine)
    trk = [
        numpy.dot(
            numpy.concatenate(
                (
                    streamline,
                    numpy.ones(
                        (len(streamline), 1)
                    )
                ), axis=1
            ),
            inv_affine) for streamline in trk.streamlines]
    trk = [track[..., :3] for track in trk]

    # Create a viewer
    tracks_snap = None
    if display_tracks:
        nb_tracks = len(trk)
        if nb_tracks < 5000:
            downsampling = 1
        else:
            downsampling = nb_tracks // 5000
        tracks_snap = os.path.join(outdir, "tracks.png")
        streamlines_actor = fvtk.line(trk[::downsampling],
                                      line_colors(trk[::downsampling]))
        vol_actor = fvtk.slicer(data[..., 0])

        vol_actor.display(data.shape[0] // 2, None, None)
        ren = fvtk.ren()
        fvtk.add(ren, streamlines_actor)
        fvtk.add(ren, vol_actor)
        fvtk.record(ren, n_frames=1, out_path=tracks_snap, size=(800, 800))
        if verbose > 1:
            fvtk.show(ren)

    # Fit the Life model and save the associated weights
    fiber_model = dpilife.FiberModel(gtab)
    fiber_fit = fiber_model.fit(data, trk, affine=numpy.eye(4))
    life_weights = fiber_fit.beta
    life_weights_file = os.path.join(outdir, "life_weights.txt")
    numpy.savetxt(life_weights_file, life_weights)
    life_weights_snap = os.path.join(outdir, "life_weights.png")
    fig, ax = plt.subplots(1)
    ax.hist(life_weights, bins=100, histtype="step")
    ax.set_xlabel("Fiber weights")
    ax.set_ylabel("# Fibers")
    fig.savefig(life_weights_snap)

    # Invert the model and predict back either the data that was used to fit
    # the model: compute the prediction error of the diffusion-weighted data
    # and calculate the root of the mean squared error.
    model_predict = fiber_fit.predict()
    model_error = model_predict - fiber_fit.data
    model_rmse = numpy.sqrt(numpy.mean(model_error[:, 10:] ** 2, -1))
    data_rmse = numpy.ones(data.shape[:3]) * numpy.nan
    data_rmse[fiber_fit.vox_coords[:, 0],
              fiber_fit.vox_coords[:, 1],
              fiber_fit.vox_coords[:, 2]] = model_rmse
    model_error_file = os.path.join(outdir, "model_rmse.nii.gz")
    error_im = nibabel.Nifti1Image(data_rmse, im.affine)
    nibabel.save(error_im, model_error_file)

    # As a baseline against which we can compare, we assume that the weight
    # for each streamline is equal to zero. This produces the naive prediction
    # of the mean of the signal in each voxel.
    life_weights_baseline = numpy.zeros(life_weights.shape[0])
    pred_weighted = numpy.reshape(
        opt.spdot(fiber_fit.life_matrix, life_weights_baseline),
        (fiber_fit.vox_coords.shape[0], numpy.sum(~gtab.b0s_mask)))
    mean_pred = numpy.empty(
        (fiber_fit.vox_coords.shape[0], gtab.bvals.shape[0]))
    S0 = fiber_fit.b0_signal

    # Since the fitting is done in the demeaned S/S0 domain, we need to add
    # back the mean and then multiply by S0 in every voxels.
    mean_pred[..., gtab.b0s_mask] = S0[:, None]
    mean_pred[..., ~gtab.b0s_mask] = (
            (pred_weighted + fiber_fit.mean_signal[:, None]) * S0[:, None])
    mean_error = mean_pred - fiber_fit.data
    mean_rmse = numpy.sqrt(numpy.mean(mean_error ** 2, -1))
    data_rmse = numpy.ones(data.shape[:3]) * numpy.nan
    data_rmse[fiber_fit.vox_coords[:, 0],
              fiber_fit.vox_coords[:, 1],
              fiber_fit.vox_coords[:, 2]] = mean_rmse
    mean_error_file = os.path.join(outdir, "mean_rmse.nii.gz")
    error_im = nibabel.Nifti1Image(data_rmse, im.affine)
    nibabel.save(error_im, mean_error_file)

    # Compute the improvment array
    data_rmse = numpy.ones(data.shape[:3]) * numpy.nan
    data_rmse[fiber_fit.vox_coords[:, 0],
              fiber_fit.vox_coords[:, 1],
              fiber_fit.vox_coords[:, 2]] = mean_rmse - model_rmse
    improvment_error_file = os.path.join(outdir, "improvment_rmse.nii.gz")
    error_im = nibabel.Nifti1Image(data_rmse, im.affine)
    nibabel.save(error_im, improvment_error_file)

    return (life_weights_file, life_weights_snap, model_error_file,
            mean_error_file, improvment_error_file, tracks_snap)


def lossy_compression_of_tractogram(tractogramfile, outdir, rate=0.392,
                                    search_optimal_rate=False,
                                    weightsfile=None, weights_thr=0.,
                                    max_search_dist=2.2, verbose=0):
    """ Reduce the number of points of the track by keeping
    intact the start and endpoints of the track and trying to remove
    as many points as possible without distorting much the shape of
    the track, ie. more points in curvy regions and less points in less curvy
    regions.

    Parameters
    ----------
    tractogramfile: str
        the path to the tractogram.
    outdir: str
        the destination folder.
    rate: float, default 0.392
        the compression rate, ie. smoothing parameter (<0.392 smoother,
        >0.392 rougher).
    search_optimal_rate: bool, default False
        determine the optimal compression rate.
    weightsfile: str, default None
        use these weights to remove unsignificant streamlines.
    weights_thr: float, default 0.
        the threshold used to identify unsignificant streamlines.
    max_search_dist: float, default 2.2
        the maximum distance between the initial and downsampled streamlines
        allowed during the best rate search.
    verbose: int, default 0
        the verbosity level.

    Returns
    -------
    compressed_tractogramfile: str
        the compressed tractogram.
    nb_points_file: str
        the compression result compared to the original sampling.
    """
    # Load the tractogram
    trk = nibabel.streamlines.load(tractogramfile)
    if verbose > 0:
        print("[info] Number of tracks: {0}".format(len(trk.streamlines)))

    # Keep only significant streamlines
    tracks = trk.streamlines
    if weightsfile is not None:
        weights = numpy.loadtxt(weightsfile)
        keep_indices = numpy.where(weights > weights_thr)[0]
        tracks = list(numpy.array(tracks)[keep_indices])
        weights = weights[numpy.where(keep_indices)[0]]
        if verbose > 0:
            print("[info] Number of significant tracks: {0}".format(
                len(tracks)))
    else:
        weights = None

    # Compress tractogram
    # > dynamic compression rate
    if search_optimal_rate:
        rate = "dynamic"
        ref_lengths = list(length(tracks))
        rates = numpy.linspace(1, 0, 21)
        opt_lengths = numpy.zeros((len(rates), len(tracks)))
        for idx, optrate in enumerate(rates):
            if verbose > 0:
                print("[info] Grid search at rate '{0}'.".format(optrate))
            decimated_tracks = [
                approx_polygon_track(t, optrate) for t in tracks]
            opt_lengths[idx] = list(length(decimated_tracks))
            opt_lengths[idx] -= ref_lengths
        opt_lengths = numpy.abs(opt_lengths)
        if verbose > 2:
            print("[debug] Optimal lengths: {0}".format(opt_lengths))
        opt_lengths[numpy.where(opt_lengths > max_search_dist)] = 0
        opt_rate_indices = numpy.argmax(opt_lengths, axis=0)
        if verbose > 2:
            print("[debug] Optimal rate indices: {0}".format(opt_rate_indices))
        tracks = [approx_polygon_track(t, rates[i])
                  for t, i in zip(tracks, opt_rate_indices)]
    # > static compression rate
    else:
        tracks = [approx_polygon_track(t, rate) for t in tracks]
    compressed_tractogramfile = os.path.join(
        outdir, "compressed_tractogram.trk")
    compressed_trk = Tractogram(
        streamlines=tracks,
        affine_to_rasmm=trk.affine)
    nibabel.streamlines.save(compressed_trk, compressed_tractogramfile)

    # Summary graph
    n_pts_initial = [len(t) for t in trk.streamlines]
    n_pts_compressed = [len(t) for t in compressed_trk.streamlines]
    nb_points_file = os.path.join(outdir, "nb_points.png")
    fig, ax = plt.subplots(1)
    ax.hist(n_pts_initial, color="r", histtype="step", label="initial")
    ax.hist(n_pts_compressed, color="b", histtype="step",
            label="compressed ({0})".format(rate))
    ax.set_xlabel("Number of points")
    ax.set_ylabel("Count")
    plt.legend()
    plt.savefig(nb_points_file)

    return compressed_tractogramfile, nb_points_file
