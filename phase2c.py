#!/usr/bin/env python

import os
os.environ['OMP_WAIT_POLICY'] = 'passive'
import mkl
mkl.set_num_threads(22)
vml_threads = mkl.domain_get_max_threads('vml')
import numpy
import numexpr
numexpr.set_num_threads(14)
import numba
numba.set_num_threads(8)
import numpy as np
import argparse, os, shutil, datetime, collections, itertools, sys, signal, socket, time, gc, math
import h5py
import scipy.signal as ss
import fast_histogram
from accum import *
import idi.reconstruction as recon
import scipy.ndimage as snd
import skimage.morphology as skm
from datasetreader import *
from dataclasses import dataclass
from h5util import *

mkl.domain_set_num_threads(vml_threads, 'vml')

terminated = 0


class correlator:
    def __init__(self, qs, maskout, meandata):
        """
        3d fft correlator
        qs: q vectors of data points, array of shape (T,P,3) with T: non overlapping tiles, P: pixels per tile
        maskout: points on detector that will not be used, array of shape (T,P)
        meandata: mean of data values used for normalisation, array of shape (T,P)
        resolution is fixed at dq=1
        """
        q = _atleastnd(qs, 3)
        for a, b in itertools.combinations(range(q.shape[0]), 2):
            if len(np.intersect1d(q[a, :, :], q[b, :, :])):
                raise ValueError('qs are asummed to be unique in first dimension!')
        if not q.shape[2] == 3:
            raise ValueError('q should qx,qy,qz in last axis')
        mask = _atleastnd(maskout, 2)
        mean = _atleastnd(meandata, 2)
        if not q.shape[:2] == mask.shape == mean.shape:
            print(q.shape, mask.shape, mean.shape)
            raise ValueError('shape missmatch')

        q = np.rint(q - np.min(q[~maskout], axis=(0))).astype(int)
        q[mask] = np.max(q, axis=(0, 1)) + 2
        qlen = np.array([_fastlen(2 * (np.max(k) + 1)) for k in q[~mask].T])
        self.q = q
        self.mask = np.copy(mask)
        tmp = np.zeros((qlen[0], qlen[1], qlen[2] + 2))
        np.subtract(tmp, 0, out=tmp)
        self.tmp = tmp
        accum = np.zeros_like(tmp)
        np.subtract(accum, 0, out=accum)
        self.accum = accum
        assemblenorm = _getnorm(q, mask)
        with np.errstate(divide='ignore'):
            self.invmean = 1 / (mean * assemblenorm)
        self.invmean[~np.isfinite(self.invmean)] = 0
        # self.data=np.zeros_like(mean)
        self.N = 0
        self.finished = False

    def suspend(self):
        """
        free tmp buffer
        """
        self.tmp = None

    def add(self, data):
        """
        does correlation of data and adds to internal accumulator
        data: array of shape T,P with T: non overlapping tiles, P pixel per Tile
        """
        d = _atleastnd(data, 2)
        if not d.shape == self.q.shape[:-1]:
            print(d.shape, self.q.shape, flush=True)
            raise ValueError('shape missmatch')
        if self.finished:
            raise GeneratorExit('already finished')
        if self.tmp is None:
            self.tmp = np.zeros_like(self.accum)
        # zero(self.data)
        _zero(self.tmp)
        d = d * self.invmean
        d[self.mask] = 0
        _addat(self.tmp, self.q, d)
        err = recon.ft.autocorrelate3.autocorrelate3(self.tmp)
        if err:
            raise RuntimeError(f'cython autocorrelations failed with error code {err}')
        np.add(self.accum, self.tmp, out=self.accum)
        self.N += 1

    def result(self, finish=False):
        """
        returns result of accumulated correlations
        finish: free accumulator and buffer.
        """
        _zero(self.tmp)
        assemblenorm = _getnorm(self.q, self.mask)
        _addat(self.tmp, self.q, np.sqrt(self.N) * np.array(~self.mask, dtype=np.float64) / assemblenorm)
        err = recon.ft.autocorrelate3.autocorrelate3(self.tmp)
        if err:
            raise RuntimeError(f'cython autocorrelations failed with error code {err}')

        res = numexpr.evaluate('where((norm<(100*N)), nan,accum/norm)', local_dict={'N': self.N, 'norm': self.tmp, 'nan': np.nan, 'accum': self.accum})
        if finish:
            self.finished = True
            self.accum = None
            self.tmp = None
        res = recon.ft.unwrap(res)
        return res


@numba.njit(parallel=True)
def _addat(array, ind, input):
    """
    sets array to input at ind, parallelä
    """
    nproc = ind.shape[0]
    nel = ind.shape[1]
    for i in numba.prange(nproc):
        tinp = input[i, ...]
        tind = ind[i, ...]
        for j in range(nel):
            array[tind[j, 0], tind[j, 1], tind[j, 2]] += tinp[j]
    return


@numba.njit(parallel=True)
def _zero(array):
    """
    set array to zero
    """
    a = array.ravel()
    for i in numba.prange(len(a)):
        a[i] = 0
    return True


def _getnorm(q, mask):
    """
    returns amount of pixels with same q
    """
    maxq = np.max(q.reshape(-1, 3), axis=0)
    hist = np.histogramdd(q.reshape(-1, 3), bins=maxq + 1, range=[[-0.5, mq + 0.5] for mq in maxq], weights=~mask.ravel())[0]
    ret = hist[q.reshape(-1, 3)[:, 0], q.reshape(-1, 3)[:, 1], q.reshape(-1, 3)[:, 2]].reshape(q.shape[:2])
    ret[mask] = 1
    return ret


def _fastlen(x, factors=(2, 3, 5, 7, 11)):
    """
    return N>=x conisting only of the prime factors given as factors
    """
    lens = np.unique([np.product(np.array(factors) ** np.array(i)) for i in itertools.product(*(range(int(1 + np.log(x) / np.log(k))) for k in factors))])
    return lens[np.argmax(x <= lens)]


def _atleastnd(array, n):
    """
    adds dimensions of length 1 in front of array to get to n dimensions
    """
    return array[tuple((n - array.ndim) * [None] + [...])]


def jobsplit(shotok, nimages, job):
    totalimages = np.sum(shotok)
    cum = np.cumsum(shotok)
    nimagesjob = math.floor(totalimages / math.floor(totalimages / nimages))
    start = nimagesjob * job
    if start + nimagesjob > totalimages:
        raise StopIteration
    end = nimagesjob * (job + 1)
    startid = np.argmax(cum == start) + int(start > 0)
    endid = np.argmax(cum == (end + 1))
    if (totalimages - end) < nimagesjob or end == 0:
        endid = len(shotok)
        end = np.max(cum)
    if not shotok[startid]:
        startid += 1

    return (startid, endid, start, end)


def isdir(string):
    if os.path.isdir(string):
        return os.path.abspath(string)
    else:
        raise NotADirectoryError(string)


def isfile(string):
    if os.path.isfile(string):
        return os.path.abspath(string)
    else:
        raise FileNotFoundError(string)


import lmfit


def fitcenter(mean, mask, q, maxq=750):
    points = np.copy(q[..., 1:].reshape(-1, 2).T)
    data = np.copy(mean.ravel())
    fitmask = np.copy(mask)
    bigq = np.logical_or((np.abs(q[..., 1]) > maxq), (np.abs(q[..., 2]) > maxq)).reshape(8, 1024, 512)
    fitmask[bigq] = False
    fitmask = fitmask.ravel()
    data[~fitmask] = np.nan

    def cubic2d(xy, amp, xo, yo, offset, c):
        x, y = xy
        val = amp * ((x + xo) ** 2 + (y + yo) ** 2) ** c + offset
        return np.ravel(val)

    lmfit_model = lmfit.Model(cubic2d)
    lmfit_result2 = lmfit_model.fit(np.nan_to_num(data), xy=points, amp=-0.005, xo=0, yo=0, offset=np.nanmax(data[fitmask]), c=0.8, weights=fitmask.astype(float))
    return np.rint((lmfit_result2.params['yo'].value, lmfit_result2.params['xo']))


from scipy.spatial.transform import Rotation


def qs(det, tiles, globaloffset=[0, 0, 2800]):
    points = []
    for tile in tiles:
        offset = np.array(det[tile].attrs['detector_coordinate_in_micro_meter']) / 50
        offset = np.array((-offset[1] + globaloffset[0], offset[0] + globaloffset[1], globaloffset[2]))
        y, x, z = np.meshgrid(np.arange(512), np.arange(1024), 0)
        rotangle = ((np.array(det[tile].attrs['detector_rotation_angle_in_degree']))) / 180 * np.pi
        rot = Rotation.from_euler('xyz', [0, 0, rotangle])
        v = np.array([x.ravel(), y.ravel(), z.ravel()]).T
        points.append(rot.apply(v) + offset)
    points = np.array(points)
    d = np.sqrt(np.sum(points ** 2, axis=(-1)))
    q = (points / d[..., None]) * globaloffset[2]
    q = np.array(q[..., ::-1], order='C')
    return q


@numba.njit(parallel=True)
def correct(data, limit, block, mask):
    mblocks = int(mask.size // block)
    nblocks = int(data.size // block)
    maskview = mask.reshape((mblocks, block))
    dataview = data.reshape((nblocks, block))
    for i in numba.prange(nblocks):
        currentmask = maskview[i % mblocks, :]
        tmp = np.empty(block)
        n = 0
        for j in range(block):
            pixel = dataview[i, j]
            if currentmask[j] != 0 and pixel < limit and pixel > -limit:
                tmp[n] = pixel
                n += 1
        if n > 0:
            dataview[i, :] -= currentmask * numba.np.arraymath._median_inner(tmp, n)


def correctionmask(absfft, Ncorrect, threshold=0.3):
    """
    gets mask for correction by finding affected columns
    """
    ftrange = (100, 300)
    backgroundradius = 10
    maskfactor = 2
    minpixel = 3
    ft = np.array(absfft, dtype=np.float64)
    # remove background of fft
    ft -= snd.grey_opening(ft, structure=skm.disk(backgroundradius))
    # get a mask of significant area
    ftm = np.abs(ft - np.mean(ft[50:])) > (maskfactor * np.std(ft[50:]))
    # clean up mask
    ftm = skm.binary_closing(ftm, np.ones((1, 20)))
    ftm = skm.binary_opening(ftm, np.ones((1, 15)))
    ftm[: ftrange[0]] = False
    ftm[ftrange[1] :] = False
    ftm = skm.binary_dilation(ftm, np.ones((1, 2 * ft.shape[1])))
    # normalised mean of this area for each column
    ft[~ftm] = np.nan
    w = np.nanmean(ft, axis=0)
    w = np.reshape(w, (w.shape[0] // Ncorrect, Ncorrect))
    w -= np.min(w, axis=-1)[:, None]
    w /= np.max(w, axis=-1)[:, None]
    # threshold to find mask
    mask = w > threshold
    # clean up mask
    mask = skm.binary_opening(mask, np.ones((1, 2)))
    mask[np.sum(mask, axis=1) < minpixel, :] = False  # blocks with few pixels are not useful for median correction
    return mask.ravel()


def meanstdmask(means, stds, meanthres=4, stdthres=4, opening_radius=5, border=2, initial_mask=100):
    mask = []
    initial_mask = np.array(initial_mask)
    if initial_mask.size == 1:
        initial_thres = initial_mask
        initial_mask = []
        for mean, std in zip(means, stds):
            initial_mask.append(
                ~np.logical_or(
                    np.abs(mean - np.mean(mean[mean > initial_thres])) > (np.std(mean[mean > initial_thres])), np.abs(std - np.mean(std[mean > initial_thres])) > (np.std(std[mean > initial_thres]))
                )
            )

    for mean, std, cmask in zip(means, stds, initial_mask):
        cmask = np.logical_or(np.abs(mean - np.mean(mean[cmask])) > (meanthres * np.std(mean[cmask])), np.abs(std - np.mean(std[cmask])) > (stdthres * np.std(std[cmask])))
        cmask = skm.binary_dilation(cmask, skm.disk(1))
        cmask = skm.binary_opening(~cmask, skm.disk(opening_radius))
        cmask[:border, :] = False
        cmask[-border:, :] = False
        cmask[:, :border] = False
        cmask[:, -border:] = False
        mask.append(cmask)
    mask = np.array(mask)
    return mask


def get_thresholds(kalpha, deltaelow, deltaehigh, maxphotons, nscatter, scatter):
    thresholds = tuple(
        [
            (float(n), float(s), n * kalpha + s * scatter - deltaelow, n * kalpha + s * scatter + deltaehigh, s * scatter)
            for s in range(nscatter, -1, -1)
            for n in range(maxphotons - s + 1)
            if not (n == 0 and s == 0)
        ]
    )
    return thresholds


@numba.njit(parallel=True)
def getstats(img, thresholds):
    number = np.zeros(img.shape, np.float64)
    ev = np.zeros(img.shape, np.float64)
    scatter = np.zeros(img.shape, np.float64)
    for n, s, low, high, evs in thresholds:
        for i in numba.prange(img.shape[0]):
            for j in range(img.shape[1]):
                for k in range(img.shape[2]):
                    if (low < img[i, j, k]) and (img[i, j, k] < high):
                        scatter[i, j, k] = s
                        number[i, j, k] = n
                        ev[i, j, k] = img[i, j, k] - evs
    return ev, number, scatter


def enumerate_detector(det, thresholds, shot_ok=None, tiles=None, nimages=np.inf, startimg=0, stopimg=np.inf, correction=False, progress=True):
    """
    yields
    (ind_filtered, ind_orig, data, ev, number, scatter)
    """
    global terminated
    Ncorrect = 64
    correctionphotonthres = 3000
    if not isinstance(det, h5py.Group):
        raise TypeError('det should be a h5 group')
    if tiles is None:
        tiles = [k for k in det.keys() if 'tile' in k]
    else:
        newtiles = []
        for t in tiles:
            if t in det:
                newtiles.append(t)
            elif f'tile{t}' in det:
                newtiles.append(f'tile{t}')
            else:
                raise KeyError(f'tile {t} not found')
        tiles = newtiles
    datanames = [(f'{det.name}/{t}/data') for t in tiles]
    filename = det.file.filename

    nshots = det[f'{tiles[0]}/data'].shape[0]
    startimg = int(np.clip(startimg, 0, nshots))
    stopimg = int(np.clip(stopimg, startimg, nshots))
    tileshape = det[f'{tiles[0]}/data'].shape[1:]
    correctmask = [correctionmask(det[t]['absfft0/mean'], Ncorrect) for t in tiles]
    if shot_ok is None:
        shot_ok = np.ones(nshots, np.bool)
    ind_filtered = 0
    data = np.zeros((len(tiles), *tileshape))
    willread = np.copy(shot_ok)
    willread[:startimg] = False
    willread[stopimg:] = False
    with datasetreader(datanames, filename, sizecache=10, willread=willread) as reader:

        for ind_orig in range(startimg, stopimg):
            if not shot_ok[ind_orig]:
                continue
            if ind_filtered >= nimages or terminated != 0:
                return
            if progress and ind_filtered % 10 == 0:
                print(ind_filtered, end=' ', flush=True)

            for it, t in enumerate(tiles):
                cdat = np.array(reader[ind_orig, it], dtype=np.float, order='C')
                if correction:
                    correct(cdat, correctionphotonthres, Ncorrect, correctmask[it])
                data[it, ...] = cdat
            ev, number, scatter = getstats(data, thresholds)

            yield (ind_filtered, ind_orig, data, ev, number, scatter)

            ind_filtered += 1


@dataclass
class detectorinfo3_t:
    name: str
    inputname: str
    tiles: list
    maxphotons: int
    scatterphotons: int
    skipphotons: int
    deltaelow: float
    deltaehigh: float
    correction: bool


def main():
    ##################################
    def sigterm_handler(signo, stack_frame):
        global terminated
        if terminated > 2:
            print('will exit immediatly!')
            sys.exit(1)
        terminated += 1
        print('\n!!!terminated!!!', flush=True)
        print(f'Signal {signo}', flush=True)
        overwritedata(outfile, 'meta/interrupted', 'interrupted')

    signal.signal(signal.SIGTERM, sigterm_handler)
    signal.signal(signal.SIGUSR1, sigterm_handler)
    signal.signal(signal.SIGINT, sigterm_handler)
    ##################################

    print(f'Phase2 single crystal {args.name} run using inputfile \n {args.inputfile}')
    print(f'started {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} on {socket.gethostname()}')
    print()

    # inputfile
    try:
        inputfile = h5py.File(args.inputfile, 'r')
    except Exception as e:
        print('could not open inputfile: ', end='')
        print(e, flush=True)
        sys.exit(1)
    if os.path.isfile(args.outfile):
        print(f'{args.outfile} exists. will not overwrite outputfile!')
        raise FileExistsError

    try:
        det = inputfile[f'detectors/{detinfo.inputname}']
    except Exception as e:
        print(f'   data for {detinfo.name} not found!')
        print(e, flush=True)
        sys.exit(1)
    if detinfo.tiles is None:
        detinfo.tiles = [tile for tile in det.keys() if 'tile' in tile]
    else:
        detinfo.tiles = [f'tile{t}' for t in detargs.tiles]
    for t in detinfo.tiles:
        try:
            det[t]['data']
        except Exception as e:
            print(f'   data for {detinfo.name}/{t} not found!')
            print(e, flush=True)
            sys.exit(1)

    with h5py.File(args.outfile, 'w') as outfile:

        # loading some vars
        kalpha = np.array(inputfile['meta/Kalpha'])[0]
        pulsebeam = np.array(inputfile['pulse_energy_beam_joule'])
        pulsehutch = np.array(inputfile['pulse_energy_hutch_joule'])
        samplex = np.array(inputfile['motors/sampleX'])
        samplez = np.array(inputfile['motors/sampleZ']) + np.array(inputfile['motors/profZ'])
        photonEnergy = np.array(inputfile['photonEnergy'])

        # filtering
        print('filtering...', flush=True)
        try:

            pulse_ok = pulsebeam > (args.pulsethres * 1e-6)  # given threshold
            pulse_ok[pulse_ok] = pulsebeam[pulse_ok] < (np.nanmean(pulsebeam[pulse_ok]) + 4 * np.nanstd(pulsebeam[pulse_ok]))  # discard more than 4std to high
            pulse_ok[pulse_ok] = np.abs((pulsehutch[pulse_ok] / pulsebeam[pulse_ok]) - np.nanmean(pulsehutch[pulse_ok] / pulsebeam[pulse_ok])) < 4 * np.nanstd(
                pulsehutch[pulse_ok] / pulsebeam[pulse_ok]
            )  # discard abnormal transmission
            samplex_ok = np.logical_and(samplex < (np.max(samplex) - 0.0002), samplex > np.min(samplex) + 0.0002)  # cut left and right a bit
            energy_ok = np.abs(np.nan_to_num(photonEnergy) - np.nanmedian(photonEnergy)) < (2 * np.nanstd(photonEnergy))  # discard unusual energies

            shot_ok = np.logical_and.reduce((samplex_ok, pulse_ok, energy_ok))

            # filter on the detector intensity
            intensity_ok = np.ones_like(shot_ok, bool)
            edges_ok = np.ones_like(shot_ok, bool)
            binsX = 100  # for binning the intensity - the sample will be cut into equally sized squares, discard squares with unusual intensity
            binsZ = int(binsX * (samplez.max() - samplez.min()) / (samplex.max() - samplex.min()))
            indSampleX = np.searchsorted(np.linspace(samplex.min(), samplex.max(), binsX + 1)[:-1], samplex, side='right') - 1
            indSampleZ = np.searchsorted(np.linspace(samplez.min(), samplez.max(), binsZ + 1)[:-1], samplez, side='right') - 1
            ind = indSampleX + binsX * indSampleZ
            n = np.bincount(ind)
            intensity = sum(np.array(det[f'{tile}/intensity']) for tile in detinfo.tiles)
            edgeintensities = np.hstack([np.array(det[f'{tile}/edgeintensity'] / intensity[:, None]) for tile in detinfo.tiles])
            edges_ok[np.any((edgeintensities - np.mean(edgeintensities[shot_ok, :], axis=0)) > (3 * np.std(edgeintensities[shot_ok, :], axis=0)), axis=1)] = False
            intensity_f = intensity[shot_ok] / pulsebeam[shot_ok]
            intthresl = np.clip(np.mean(intensity_f) - 2 * np.std(intensity_f), np.nanpercentile(intensity_f, 0.1), np.nanpercentile(intensity_f, 5))  # clammped thresholds for low and high
            intthresh = np.clip(np.mean(intensity_f) + 4 * np.std(intensity_f), np.nanpercentile(intensity_f, 99.9), np.inf)

            with np.errstate(divide='ignore', invalid='ignore'):
                intensity_ok[(intensity / pulsebeam) > intthresh] = False
                intensity_ok[(intensity / pulsebeam) < intthresl] = False
                binnedint = np.zeros_like(n)
                binnedint[n > 0] = (np.bincount(ind, np.nan_to_num(intensity / pulsehutch)) / n)[n > 0]
            bad = np.logical_or(binnedint > (np.nanmean(binnedint[n > 0]) + 2 * np.nanstd(binnedint[n > 0])), binnedint < (np.nanmean(binnedint[n > 0]) - 2 * np.nanstd(binnedint[n > 0])))
            badbin = np.linspace(0, len(binnedint) - 1, len(binnedint))[bad]
            badshot = np.in1d(ind, badbin)
            intensity_ok[badshot] = False
            shot_ok = np.logical_and.reduce((samplex_ok, pulse_ok, energy_ok, intensity_ok, edges_ok))
            print(f'will keep {np.sum(shot_ok)} of {len(shot_ok)} shots')

            (ind_orig_start, ind_orig_stop, ind_filtered_start, ind_filtered_stop) = jobsplit(shot_ok, args.nimages, args.job)
            print(f'this job #{args.job} will do shot {ind_orig_start}-{ind_orig_stop}, containing {ind_filtered_stop-ind_filtered_start} shots')
            overwritedata(outfile, f'filtering/job_ind_orig', np.array([ind_orig_start, ind_orig_stop]))
            overwritedata(outfile, f'filtering/job_ind_filtered', np.array([ind_filtered_start, ind_filtered_stop]))

            overwritedata(outfile, f'filtering/pulse_ok', np.array(pulse_ok))
            overwritedata(outfile, f'filtering/samplex_ok', np.array(samplex_ok))
            overwritedata(outfile, f'filtering/energy_ok', np.array(energy_ok))
            overwritedata(outfile, f'filtering/intensity_ok', np.array(intensity_ok))
            overwritedata(outfile, f'filtering/edges_ok', np.array(edges_ok))
            overwritedata(outfile, f'filtering/shot_ok', np.array(shot_ok))

        except Exception as e:
            print('   error in filtering')
            raise e

        print('saving meta data', flush=True)
        dictmeta = {
            'name': inputfile['meta/name'],
            'sample': inputfile['meta/sample'],
            'runs': inputfile['meta/runs'],
            'inputfiles': args.inputfile,
            'Kalpha': float(kalpha),
            'nimages': args.nimages,
            'focus_y': inputfile['meta/focus_y'],
            'interrupted': 'ok (not interrupted)',
            'detz': args.detz,
        }

        for k, v in dictmeta.items():
            overwritedata(outfile, f'meta/{k}', np.array(v))
        print('writing settings:')
        for k, v in vars(args).items():
            if k == 'det':
                v = np.array([args.det])
            if v is None:
                v = 'None'
            print('  ', k, v)
            overwritedata(outfile, f'meta/arguments/{k}', np.array(v))

        print('doing initial mask')
        mask = meanstdmask([np.array(det[t]['mean']) for t in detinfo.tiles], [np.array(det[t]['std']) for t in detinfo.tiles])
        print(f'   will keep {np.sum(mask)} of {mask.size} pixels')
        overwritedata(outfile, f'{detinfo.name}/initalmask', mask)

        print('finding offset')
        q = qs(det, detinfo.tiles, [0, 0, args.detz])
        offset = fitcenter(np.array([np.array(det[tile]['mean']) for tile in detinfo.tiles]), mask, q, maxq=750)
        q = qs(det, detinfo.tiles, [offset[0], offset[1], args.detz])
        print(f'  found offset {offset}')
        overwritedata(outfile, f'{detinfo.name}/qs', q)
        overwritedata(outfile, f'{detinfo.name}/offset', np.array([offset[0], offset[1], args.detz]))

        print('doing photon stats. ', end='')
        shot_ok = np.array(outfile['filtering/shot_ok'])
        intensities = []
        max_intensities = []
        shot_skipped = np.zeros_like(shot_ok)
        skipped = 0

        accums = collections.defaultdict(lambda: accumulator(False))

        thresholds = get_thresholds(kalpha, detinfo.deltaelow, detinfo.deltaehigh, detinfo.maxphotons, detinfo.scatterphotons, np.nanmean(photonEnergy[shot_ok]))
        overwritedata(outfile, f'{detinfo.name}/thresholds', np.array(thresholds))

        for (ind_filtered, ind_orig, img, ev, photons, scatterphotons) in enumerate_detector(
            det, thresholds=thresholds, shot_ok=shot_ok, tiles=detinfo.tiles, correction=detinfo.correction, startimg=ind_orig_start, stopimg=ind_orig_stop
        ):
            if np.any(img[mask] > ((detinfo.skipphotons + 0.5) * kalpha)):
                print(f' s{int(np.rint(np.max(img[mask])/kalpha))} ', end='')
                shot_skipped[ind_orig] = True
                skipped += 1
            else:
                accums[f'{detinfo.name}/ev_photons'].add(ev)
                accums[f'{detinfo.name}/photons'].add(photons)
                accums[f'{detinfo.name}/scatterphotons'].add(scatterphotons)
                intensities.append(np.sum(photons[mask]))
                max_intensities.append(np.nanmax(img[mask]))
                img[img < detinfo.deltaelow * 0.5] = 0
                img[img > (detinfo.maxphotons * kalpha + detinfo.deltaehigh)] = 0
                img = img.astype(float)
                accums[f'{detinfo.name}/ev'].add(img)

        if len(intensities) > 0:
            intensities = np.array(intensities)
            overwritedata(outfile, f'{detinfo.name}/intensity', intensities)

        if len(max_intensities) > 0:
            max_intensities = np.array(max_intensities)
            overwritedata(outfile, f'{detinfo.name}/max_intensity', max_intensities)

        overwritedata(outfile, f'filtering/shot_skipped_{detinfo.name}', shot_skipped)

        if not terminated:
            print(f'{ind_filtered + 1}, skipped {np.sum(shot_skipped)}')
            shot_ok[shot_skipped] = False

            print('doing batch specific mask')
            mask = np.logical_and(mask, meanstdmask(accums[f'{detinfo.name}/photons'].mean, accums[f'{detinfo.name}/photons'].std, initial_mask=mask, meanthres=3, stdthres=3))
            overwritedata(outfile, f'{detinfo.name}/mask', mask)
            print(f'      will keep {np.sum(mask)} pixels on {detinfo.name}')

        if args.shotnormalisation and not terminated:
            print('doing shot total intensity normalisation. ', end='')
            intensity_norm = np.mean(intensities) / intensities
            for (ind_filtered, ind_orig, img, ev, photons, scatterphotons) in enumerate_detector(
                det, thresholds=thresholds, shot_ok=shot_ok, tiles=detinfo.tiles, correction=detinfo.correction, startimg=ind_orig_start, stopimg=ind_orig_stop
            ):
                cintensity_norm = intensity_norm[ind_filtered]
                accums[f'{detinfo.name}/ev_photons_shotnorm'].add(ev * cintensity_norm)
                accums[f'{detinfo.name}/photons_shotnorm'].add(photons * cintensity_norm)
                img[img < detinfo.deltaelow * 0.5] = 0
                img[img > (detinfo.maxphotons * kalpha + detinfo.deltaehigh)] = 0
                img = img.astype(float)
                accums[f'{detinfo.name}/ev_shotnorm'].add(img * cintensity_norm)
        print('')

        if not terminated:
            print('starting 3d correlations.', end='')
            mean = accums[f'{detinfo.name}/photons_shotnorm'].mean if args.shotnormalisation else accums[f'{detinfo.name}/photons'].mean
            cor = None
            for (ind_filtered, ind_orig, img, ev, photons, scatterphotons) in enumerate_detector(
                det, thresholds=thresholds, shot_ok=shot_ok, tiles=detinfo.tiles, correction=detinfo.correction, startimg=ind_orig_start, stopimg=ind_orig_stop
            ):
                cimg = photons * intensity_norm[ind_filtered] if args.shotnormalisation else photons
                if cor is None:
                    oldtime=time.time()
                    print(' allocating memory for correlations. ', end='', flush=True)
                    cor = correlator(q, ~mask.reshape(mask.shape[0], -1), mean.reshape(mean.shape[0], -1))
                    
                    print('done.\ndoing correlations. ', end=' ')
                if ind_filtered%10==0:
                    dt=time.time()-oldtime
                    oldtime=time.time()
                    print(f'({dt:.1f}s)',end=' ', flush=True)
                    
                cor.add(cimg.reshape(cimg.shape[0], -1))
        print('')
        print('saving correlation. ', flush=True, end='')
        try:
            result = cor.result(True)
            print('.', flush=True)
            print(f'   (shape {result.shape}, value range ~ {np.nanmin(result[:5,...])}-{np.nanmax(result[:5,...])}, mean ~{np.nanmean(result[:10,500:-500,500:-500])}) ')
            overwritedata(outfile, f'{detinfo.name}/correlation', result, chunks=(1, int(result.shape[1] // 2), int(result.shape[2] // 2)))
            print('   done.', flush=True)

        except Exception as e:
            print(' failed.')
            print(e)
        cor = None
        gc.collect()
        print('saving other accumulated data', flush=True)
        for k, v in accums.items():
            print('  ', k, v.mean.shape, np.mean(v.n))
            overwritedata(outfile, f'{k}/mean', np.array(v.mean, np.float64))
            overwritedata(outfile, f'{k}/std', np.array(v.std, np.float64))
            overwritedata(outfile, f'{k}/n', np.array(v.n, np.int64))
        if not terminated:
            print('copying', flush=True)
            mask = np.copy(shot_ok)
            mask[:ind_orig_start] = False
            mask[ind_orig_stop:] = False
            param = outfile.create_group('parameters')
            copymasked(inputfile['attenuator'], param, mask)
            copymasked(inputfile['motors'], param, mask)
            copymasked(inputfile['photonEnergy'], param, mask)
            copymasked(inputfile['pulse_energy_beam_joule'], param, mask)
            copymasked(inputfile['pulse_energy_hutch_joule'], param, mask)
            copymasked(inputfile['tag_number_list'], param, mask)
            overwritedata(outfile, 'meta/interrupted', 'done')
            print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='sacla 2020 analysis phase2')
    parser.add_argument('inputfile', metavar='inputfile', type=isfile, help='the hdf5 inputfiles to process')
    parser.add_argument('outfile', default=None, metavar='outputfilename', type=str, help='where to save the output')
    parser.add_argument('--name', default='', dest='name', metavar='str', type=str, help='name to show in log')

    parser.add_argument(
        '--det', default=None, dest='det', type=str, help='detector (name inputname +tiles +correct +maxphotons +scatterphotons +skipphotons +deltaelow +deltaehigh), default: octal +correct'
    )
    parser.add_argument('--detz', default=2800, dest='detz', type=int, help='detector z distance in pixels')

    arg_correlation = parser.add_argument_group('correlation normalisation')
    arg_correlation.add_argument('--no-shotnormalisation', default=True, action='store_false', dest='shotnormalisation', help='SKIP normalisation of shot (fluorescence) intensity')
    arg_correlation.add_argument('--no-pixelnormalisation', default=True, action='store_false', dest='pixelnormalisation', help='SKIP normalisation of pixel intensity')

    arg_filtering = parser.add_argument_group('filtering')
    arg_filtering.add_argument('--pulsethres', default=400, dest='pulsethres', metavar='uJ', type=float, help='threshold energy beam')
    parser.add_argument('--nimages', default=np.inf, dest='nimages', metavar='N', type=float, help='number of images to do (will do approx this number)')
    parser.add_argument('--job', default=0, dest='job', metavar='N', type=float, help='do the Nth block of Nimages in the file')

    args = parser.parse_args()

    detparser = argparse.ArgumentParser('--det', prefix_chars='+', add_help=False)
    detparser.add_argument('name')
    detparser.add_argument('inputname')
    detparser.add_argument('+tiles', type=int, nargs='+', default=None)
    detparser.add_argument('+correct', default=False, dest='correction', action='store_true')
    detparser.add_argument('+maxphotons', type=int, default=5)
    detparser.add_argument('+scatterphotons', type=int, default=0)
    detparser.add_argument('+skipphotons', type=int, default=10)
    detparser.add_argument('+deltaehigh', type=float, default=3000)
    detparser.add_argument('+deltaelow', type=float, default=4000)

    if args.det is None:
        args.det = ['octal', 'octal', '+correct']
    detargs = detparser.parse_args(args.det)
    detinfo = detectorinfo3_t(
        detargs.name, detargs.inputname, detargs.tiles, detargs.maxphotons, detargs.scatterphotons, detargs.skipphotons, detargs.deltaelow, detargs.deltaehigh, detargs.correction
    )

    main()
    print(f'done at {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
