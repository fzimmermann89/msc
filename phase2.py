#!/usr/bin/env python

import os
os.environ['OMP_NUM_THREADS'] = '28'
os.environ['OPENBLAS_NUM_THREADS'] = '28'
os.environ['MKL_NUM_THREADS'] = '16'
os.environ['VECLIB_MAXIMUM_THREADS'] = '12'
os.environ['NUMEXPR_NUM_THREADS'] = '14'
os.environ['NUMBA_NUM_THREADS'] = '14'
os.environ['OMP_DYNAMIC'] = 'FALSE' 
import mkl
mkl.set_num_threads(16) #12
mkl.set_dynamic(False)
import numpy
import numexpr 
numexpr.set_num_threads(14) #6
numexpr.set_vml_num_threads(1) #1
import numba
numba.set_num_threads(10) #6
import gc

import numpy as np
import argparse, os, shutil, datetime, collections, itertools, sys, signal, socket
import h5py
import scipy.signal as ss
import fast_histogram
from accum import *
from idi.reconstruction.simple import *
from correlator import *
import numba
import scipy.ndimage as snd
import skimage.morphology as skm
from datasetreader import *
from dataclasses import dataclass

from h5util import *
terminated = 0


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

@dataclass
class detectorinfo2_t:
    inputname: str
    tiles: list
    stats: bool
    correlations: bool
    maxphotons: bool
    scatterphotons: int
    skipphotons: int
    deltaelow: float
    deltaehigh: float
    correction: bool

@numba.njit(parallel=True)
def _correct(data, limit, block, mask, rot, out, o0, o1, startx, starty):
    s0,s1 = data.shape
    mblocks = mask.size // block
    maskview = mask.reshape((mblocks, block))
    nblocks = (s0 * s1) // block
    blocksperrow=s1 // block
    data = data.reshape((nblocks, block))
    for i in numba.prange(nblocks):
        currentmask = maskview[i % mblocks, :]
        tmp = np.empty(block)
        n = 0
        for j in range(block):
            pixel = data[i,j]
            if currentmask[j]!= 0 and pixel < limit and pixel > -limit:
                tmp[n] = pixel
                n += 1
        if n>0:
            row = data[i, :] - currentmask * numba.np.arraymath._median_inner(tmp, n)
        else:
            row = data[i, :]
        if rot==0:
            s=i*block+i//blocksperrow*(o1-s1)+(startx*o1+starty)
            out[s:s+block] = row
        elif rot==2:
            s=(nblocks-i)*block+(s1-o1)*(i//blocksperrow+1-s0)+(startx*o1+starty)
            out[s-block:s] = row[::-1]
        elif rot==1:
            s=(o1*(blocksperrow-1-i%blocksperrow)*block)+i//blocksperrow+(startx*o1+starty)
            out[s:s+o1*block:o1] = row[::-1]
        elif rot==3:
            s=i%blocksperrow*(block*o1)-i//blocksperrow+(s0-1+startx*o1+starty)
            out[s:s+o1*block:o1] = row

@numba.jit(nopython=True)
def correct(data, limit=2000, block=64, mask=None, rot=0, out=None, startx=0, starty=0):
    if rot>3 or rot<0: rot=rot%4
    s0,s1=data.shape
    outisnone=out is None
    if outisnone:
        if startx!=0 or starty!=0: raise ValueError('for using startx and starty, supply out')
        outflat=np.empty(s0*s1)
        if rot==0 or rot==2:
            o0, o1 = s0, s1
        else:
            o0, o1 = s1, s0
    else:
        o0,o1=out.shape
        if (startx<0  or starty<0): 
            raise ValueError('startx and starty must be postive')   
        if (rot==0 or rot==2): 
            if (startx+s0>o0 or starty+s1>o1):
                raise ValueError('out too small, will not fit')   
        elif (startx+s1>o0 or starty+s0>o1):
            raise ValueError('out to small, will not fit')   
        outflat=out.ravel()
    if mask is None: mask = np.ones(data.shape[1], dtype=numba.boolean)
    if mask.size != s1: raise ValueError('mask should be of length data.shape[1]')
    if s1 % block != 0: raise ValueError('block doesnt divide data.shape[1]')
    _correct(data, limit, block, mask, rot, outflat, o0, o1, startx, starty)
    if outisnone:
        out=outflat.reshape(o0,o1)

    return out





@numba.njit(parallel=True)
def place(data,times, out, startx,starty):
    if times>3 or times<0: times=times%4
    s0,s1=data.shape
    o0,o1=out.shape
    if (startx<0  or starty<0): 
        raise ValueError('startx and starty must be postive')   
    if (times==0 or times==2): 
        if (startx+s0>o0 or starty+s1>o1):
            raise ValueError('out too small, will not fit')   
    elif (startx+s1>o0 or starty+s0>o1):
        raise ValueError('out to small, will not fit')   

    if times==0:
        for i in numba.prange(s0):
            for j in range(s1):
                out[startx+i,starty+j]=data[i,j]
        return
    elif times==1:
        for j in numba.prange(s0):
            for i in range(s1):
                out[startx+i,starty+j]=data[j,s1-i-1]            
        return
    elif times==2:
        for i in numba.prange(s0):
            for j in range(s1):
                out[startx+i,starty+j]=data[s0-i-1,s1-j-1]           
        return
    elif times==3:
        for j in numba.prange(s0):
            for i in range(s1):
                out[startx+i,starty+j]=data[s0-j-1,i]              
        return
    


@numba.njit(parallel=True)
def getphotons(img, thresholds):
        number = np.zeros(img.shape, np.float64)
        for n,s,low,high,evs in thresholds:
            for i in numba.prange(img.shape[0]):
                for j in numba.prange(img.shape[1]):
                    if (low < img[i, j]) and (img[i, j] < high):
                        number[i, j] = n
        return number
    
@numba.njit(parallel=True)
def getstats(img, thresholds):
    number = np.zeros(img.shape, np.float64)
    ev = np.zeros(img.shape, np.float64)
    scatter = np.zeros(img.shape, np.float64)
    for n, s, low, high, evs in thresholds:
        for i in numba.prange(img.shape[0]):
            for j in numba.prange(img.shape[1]):
                if (low < img[i, j]) and (img[i, j] < high):
                    scatter[i, j] = s
                    number[i, j] = n
                    ev[i, j] = img[i, j] - evs
    return ev, number, scatter

def get_thresholds(kalpha, deltaelow, deltaehigh, maxphotons, nscatter, scatter):
    thresholds = tuple([
        (
            float(n),
            float(s),
            n * kalpha + s * scatter - deltaelow,
            n * kalpha + s * scatter + deltaehigh,
            s*scatter
        )
        for s in range(nscatter+1,-1,-1) 
        for n in range(maxphotons-s+1)
        if not (n==0 and s==0)
    ])
    return thresholds



def correctionmask(absfft, Ncorrect, threshold=0.3):
    '''
    gets mask for correction by finding affected columns
    '''
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

def fastlen(length):
    fast = [2,     4,     6,     8,    10,    12,    16,    18,    20,
           24,    30,    32,    36,    40,    48,    50,    54,    60,
           64,    72,    80,    90,    96,   100,   108,   120,   128,
          144,   150,   160,   162,   180,   192,   200,   216,   240,
          250,   256,   270,   288,   300,   320,   324,   360,   384,
          400,   432,   450,   480,   486,   500,   512,   540,   576,
          600,   640,   648,   720,   750,   768,   800,   810,   864,
          900,   960,   972,  1000,  1024,  1080,  1152,  1200,  1250,
         1280,  1296,  1350,  1440,  1458,  1500,  1536,  1600,  1620,
         1728,  1800,  1920,  1944,  2000,  2048,  2160,  2250,  2304,
         2400,  2430,  2500,  2560,  2592,  2700,  2880,  2916,  3000,
         3072,  3200,  3240,  3456,  3600,  3750,  3840,  3888,  4000,
         4050,  4096,  4320,  4374,  4500,  4608,  4800,  4860,  5000,
         5120,  5184,  5400,  5760,  5832,  6000,  6144,  6250,  6400,
         6480,  6750,  6912,  7200,  7290,  7500,  7680,  7776,  8000,
         8100,  8192,  8640,  8748,  9000,  9216,  9600,  9720, 10000]
    for l in fast:
        if l>=length:
            return l
    return length


def roi(data):
    return data[data.shape[0] // 2 - args.roi : data.shape[0] // 2 + args.roi, data.shape[1] // 2 - args.roi : data.shape[1] // 2 + args.roi]

#@profile
def enumerate_detector(det, thresholds, shot_ok=None, tiles=None, nimages=np.inf, stats=True, correction=False, progress=True):
    Ncorrect = 64
    correctionphotonthres = 3000
    if not isinstance(det, h5py.Group):
        raise TypeError('det should be a h5 group')
    if tiles is None:
        tiles = [k for k in det.keys() if 'tile' in k]
    else:
        newtiles = []
        for t in tiles:
            if t in det: newtiles.append(t)
            elif f'tile{t}' in det: newtiles.append(f'tile{t}')
            else: raise KeyError(f'tile {t} not found')
        tiles = newtiles
    multitiles=not (len(tiles)==1 and 'data' in det[tiles[0]])
    mincorners = []
    maxcorners = []
    rots = []
    datanames = []
    filename = det.file.filename
    nshots = det[f'{tiles[0]}/data'].shape[0]
    correctmask = []
    for t in tiles:
        d = det[t]
        offset = np.rint(d.attrs['detector_tile_position_in_pixels'])
        rot = int(d.attrs['detector_rotation_steps'][0])
        rots.append(rot)
        n, a, b = d['data'].shape
        if n != nshots:
            raise ValueError('tiles should have same number of shots')
        shape = ((a, b), (-b, a), (-a, -b), (b, -a))[rot % 4]
        corners = (offset, (shape + offset))
        mincorners.append(np.min(corners, axis=0))
        maxcorners.append(np.max(corners, axis=0))
        datanames.append(f'{d.name}/data')
        if correction:
            correctmask.append(correctionmask(det[t]['absfft0/mean'], Ncorrect))

    globaloffset = np.floor(np.min(mincorners, axis=0)).astype(int)
    extent = [fastlen(x) for x in (np.ceil(np.max(maxcorners, axis=0)) - globaloffset)]
    startx, starty = [list(s) for s in (np.floor(mincorners - globaloffset).astype(int)).T]

    if shot_ok is None:
        shot_ok = np.ones(nshots, np.bool)
    assembled = np.zeros(extent, np.float64)
    global terminated
    ind_filtered = 0
    with datasetreader(datanames, filename, willread = shot_ok) if multitiles else arrayreader(det[tiles[0]]['data']) as reader:
        #print('ready')
        #mkl.set_num_threads(14)
        for ind_orig in range(nshots):
            if not shot_ok[ind_orig]:
                continue
            if ind_filtered >= nimages or terminated != 0:
                return
            if progress and ind_filtered % 100 == 0:
                print(ind_filtered, end=' ', flush=True)

            for t in range(len(tiles)):
                if multitiles:
                    tile = np.asarray(reader[ind_orig, t], order='C', dtype=np.float64)
                    if correction:
                        correct(tile, correctionphotonthres, Ncorrect, correctmask[t], rots[t], assembled, startx[t], starty[t])
                    else:
                        place(tile, rots[t], assembled, startx[t], starty[t])
                else:
                    if correction:
                        tile = np.asarray(reader[ind_orig], order='C', dtype=np.float64)
                        correct(tile, correctionphotonthres, Ncorrect, correctmask[t], rots[t], assembled, startx[t], starty[t])
                    else:
                        assembled = np.asarray(np.rot90(reader[ind_orig],rots[t]), order='C', dtype=np.float64)
            if stats:
                ev, number, scatter = getstats(assembled, thresholds)
                yield (ind_filtered, ind_orig, np.copy(assembled), ev, number, scatter)
            else:
                number = getphotons(assembled, thresholds)
                yield (ind_filtered, ind_orig, np.copy(assembled), None, number, None)

            ind_filtered += 1


       
#@profile
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
    
    print(f'Phase2 {args.name} run using inputfile \n {args.inputfile}')
    print(f'started {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} on {socket.gethostname()}')
    print('environment:')
    print(os.environ)
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
        
    for detname, detinfo in dets.items():
        if detinfo.tiles is None:
            detinfo.tiles = [tile for tile in inputfile[f'detectors/{detinfo.inputname}'].keys() if 'tile' in tile]
        else:
            detinfo.tiles = [f'tile{t}' for t in detargs.tiles]
            
        
    with h5py.File(args.outfile, 'w') as outfile:

        # maskfile
        if args.maskfile is not None:
            print(f'using mask file {args.maskfile}')
            try:
                maskfile = h5py.File(args.maskfile, 'r')
                for detname, detinfo in dets.items():
                    if detinfo.stats or detinfo.correlations:
                        print(f'   will keep {np.sum(maskfile[detname])} pixels on {detname}')
            except Exception as e:
                print('   could not open mask:', end='')
                print(e, flush=True)
                sys.exit(1)

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
            pulse_ok[pulse_ok] = np.abs((pulsehutch[pulse_ok] / pulsebeam[pulse_ok]) - np.nanmean(pulsehutch[pulse_ok] / pulsebeam[pulse_ok])) < 4 * np.nanstd(pulsehutch[pulse_ok] / pulsebeam[pulse_ok])  # discard abnormal transmission

            samplex_ok = np.logical_and(samplex < (np.max(samplex) - 0.0005), samplex > np.min(samplex) + 0.0005)  # cut left and right a bit
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
            
            for detname, detinfo in dets.items():
                intensity = sum(np.array(inputfile[f'detectors/{detinfo.inputname}/{tile}/intensity']) for tile in detinfo.tiles)
                edgeintensities = np.hstack([np.array(inputfile[f'detectors/{detinfo.inputname}/{tile}/edgeintensity']/intensity[:,None]) for tile in detinfo.tiles])
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
            if args.nimages < np.sum(shot_ok):
                print(f'limited to {int(args.nimages)} images')
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
        }

        for k, v in dictmeta.items():
            overwritedata(outfile, f'meta/{k}', np.array(v))
        print('writing settings:')
        for k, v in vars(args).items():
            if k == 'det':
                v = np.array([' '.join(x) for x in args.det])
            if v is None:
                v = 'None'
            print('  ', k, v)
            overwritedata(outfile, f'meta/arguments/{k}', np.array(v))

        accums = collections.defaultdict(lambda: accumulator(False))

        for detname, detinfo in dets.items():
            
            #stats thread settings
#             mkl.set_num_threads(10)
#             numexpr.set_num_threads(6)
#             numba.set_num_threads(8)


            thresholds = get_thresholds(kalpha, detinfo.deltaelow, detinfo.deltaehigh, detinfo.maxphotons, detinfo.scatterphotons, np.nanmean(photonEnergy[shot_ok]))
            overwritedata(outfile, f'{detname}/thresholds', np.array(thresholds))
            print(f'doing detector {detname}: ')
            if not (detinfo.stats or detinfo.correlations):
                print('   skipping!', flush=True)
                continue
            try:
                det = inputfile[f'detectors/{detname}']
                for tile in detinfo.tiles:
                        inputfile[f'detectors/{detname}/{tile}/data']
            except Exception as e:
                print('  ', e)
                print(f'   data for {detname}/{detinfo.tile} not found. skipping detector!')
                continue

            # load mask
            if args.maskfile is not None:
                ##TODO
                mask = np.array(maskfile[f'{detname}'])
            else:
                mask = None
            shot_ok = np.array(outfile['filtering/shot_ok'])
            if detinfo.stats or detinfo.correlations:
                print('   doing photon stats. ', end='')
                intensities = []
                max_intensities = []
                shot_skipped = np.zeros_like(shot_ok)
                skipped = 0
                for (ind_filtered, ind_orig, img, ev, photons, scatterphotons) in enumerate_detector(det, thresholds=thresholds, shot_ok=shot_ok, tiles=detinfo.tiles, nimages=np.inf, stats=True, correction=detinfo.correction):
                    if mask is None:
                        mask = np.ones(img.shape, bool)
                    if ind_filtered - skipped >= args.nimages:
                        break
                    if np.any(img[mask] > ((detinfo.skipphotons + 0.5) * kalpha)):
                        
                        shot_skipped[ind_orig] = True
                        skipped += 1
                    else:
                        accums[f'{detname}/ev_photons'].add(ev)
                        accums[f'{detname}/photons'].add(photons)
                        accums[f'{detname}/scatterphotons'].add(scatterphotons)

                        intensities.append(np.sum(photons[mask]))
                        max_intensities.append(np.nanmax(img[mask]))
                        img[img < detinfo.deltaelow * 0.5] = 0
                        img[img > (detinfo.maxphotons * kalpha + detinfo.deltaehigh)] = 0
                        img = img.astype(float)
                        accums[f'{detname}/ev'].add(img)

                    # appenddata(outfile,f'{detname}/testev',ev[None,...])
                    # appenddata(outfile,f'{detname}/testphotons',photons[None,...])

                if len(intensities) > 0:
                    intensities = np.array(intensities)
                    overwritedata(outfile, f'{detname}/intensity', intensities)

                if len(max_intensities) > 0:
                    max_intensities = np.array(max_intensities)
                    overwritedata(outfile, f'{detname}/max_intensity', max_intensities)

                overwritedata(outfile, f'filtering/shot_skipped_{detname}', shot_skipped)

                if not terminated:
                    print(f'{ind_filtered + 1}, skipped {np.sum(shot_skipped)}')
                    print('   doing mask')

                    # ignore empty pixels
                    mask = np.logical_and(mask, accums[f'{detname}/ev_photons'].mean > 1e-10)
                    mask = np.logical_and(mask, accums[f'{detname}/ev'].mean > 1e-10)
                    mask = np.logical_and(mask, accums[f'{detname}/photons'].mean > 1e-10)
                    print(f'      will keep {np.sum(mask)} pixels on {detname}')

            overwritedata(outfile, f'{detname}/mask', mask)
            shot_ok[shot_skipped] = False
            gc.collect()
            if detinfo.correlations and args.shotintensitynormalised:
                #normalisation thread settings
#                 mkl.set_num_threads(12)
#                 numexpr.set_num_threads(7)
#                 numba.set_num_threads(7)
                print('   doing normalisation:', end=' ', flush=True)
                for (ind_filtered, ind_orig, img, ev, photons, _) in enumerate_detector(det, thresholds=thresholds, shot_ok=shot_ok, tiles=detinfo.tiles, nimages=args.nimages, stats=args.intv, correction=detinfo.correction):
                    if args.intv: 
                        accums[f'{detname}/ev_photons_shotintensitynormalised'].add(ev / intensities[ind_filtered] * np.mean(intensities))
                    if args.disc:
                        accums[f'{detname}/photons_shotintensitynormalised'].add(photons / intensities[ind_filtered] * np.mean(intensities))
                    if args.cont:
                        img[img < detinfo.deltaelow * 0.5] = 0
                        img[img > (detinfo.maxphotons * kalpha + detinfo.deltaehigh)] = 0
                        img = img.astype(float)
                        accums[f'{detname}/ev_shotintensitynormalised'].add(img / intensities[ind_filtered] * np.mean(intensities))

                if not terminated:
                    print(ind_filtered + 1)
            gc.collect()
            # correlations
            if detinfo.correlations:
                #corr thread settings
#                 numexpr.set_num_threads(8)
#                 mkl.set_num_threads(12)
#                 numba.set_num_threads(8)
                
                norm = corr(mask.astype(float))
                invroinorm=np.nan_to_num(1/roi(norm))
                print('   doing correlation:', end=' ', flush=True)

                if args.singleshotcorr: 
                    tmpcorrelator = correlator(mask) #TODO: rewrite correlator and seperate with internal accum and without..

                for (ind_filtered, ind_orig, img, ev, photons, _) in enumerate_detector(det, thresholds=thresholds, shot_ok=shot_ok, nimages=args.nimages, stats=args.intv, correction=detinfo.correction):
                    with np.errstate(all='ignore'):
#                         a=0
#                         t=np.random.rand(2000,2000)
#                         for i in range(100):
#                             b=np.fft.fft2(t)
#                             b*=2
#                             b=np.fft.ifft2(b)
#                             a=a+np.sum(np.abs(b))
                        params = []
                        if args.cont:
                            img[img < detinfo.deltaelow * 0.5] = 0
                            img[img > (detinfo.maxphotons * kalpha + detinfo.deltaehigh)] = 0
                            img = img.astype(float)
                            params.append((img, accums[f'{detname}/ev'].result, accums[f'{detname}/ev_shotintensitynormalised'].result, 'continuous'))
                        if args.intv:
                            params.append((ev, accums[f'{detname}/ev_photons'].result, accums[f'{detname}/ev_photons_shotintensitynormalised'].result, 'interval'))
                        if args.disc:
                            params.append((photons, accums[f'{detname}/photons'].result, accums[f'{detname}/photons_shotintensitynormalised'].result, 'discrete'))

                        for img, accum, accum_shotintensitynormalised, name in params:
                            img[~mask] = 0
                            img_norm = img * accum.invmean
                            img_norm[~mask] = 0
                            if args.shotintensitynormalised:
                                img_shotintensitynormalised = img / intensities[ind_filtered] * np.mean(intensities)
                                img_shotintensitynormalised = np.nan_to_num(img_shotintensitynormalised, copy=False)

                            if args.g2:
                                if args.nonshotintensitynormalised:
                                    # g2
                                    g2 = corr(img_norm)
                                    accums[f'{detname}/correlations/{name}/g2'].add(g2)
                                    appenddata(outfile, f'{detname}/correlations/{name}/g2/all', (roi(g2) * invroinorm)[None, ...])

                                if args.shotintensitynormalised:
                                    # g2 shotintensitynormalised
                                    img_g2_shotintensitynormalised = img_shotintensitynormalised * accum_shotintensitynormalised.invmean
                                    img_g2_shotintensitynormalised[~mask] = 0
                                    g2_intensitynormalised = corr(img_g2_shotintensitynormalised)
                                    accums[f'{detname}/correlations/{name}/g2_shotintensitynormalised'].add(g2_intensitynormalised)
                                    appenddata(outfile, f'{detname}/correlations/{name}/g2_shotintensitynormalised/all', (roi(g2_intensitynormalised) * invroinorm)[None, ...])

                            if args.singleshotcorr:
                                if args.nonpixelmeannormalised:
                                    # correlation based on masked zncc paper
                                    cur_cor = tmpcorrelator.corr(img)
                                    accums[f'{detname}/correlations/{name}/singleshotcorr'].add(cur_cor)
                                    appenddata(outfile, f'{detname}/correlations/{name}/singleshotcorr/all', (roi(cur_cor))[None, ...])

                                if args.pixelmeannormalised:
                                    # correlation based on masked zncc paper, normalised by pixelmean
                                    cur_cor_norm = tmpcorrelator.corr(img_norm)
                                    accums[f'{detname}/correlations/{name}/singleshotcorr_pixelmeannormalised'].add(cur_cor_norm)
                                    appenddata(outfile, f'{detname}/correlations/{name}/singleshotcorr_pixelmeannormalised/all', (roi(cur_cor_norm))[None, ...])

                            if args.pearson:
                                if args.shotintensitynormalised:
                                    # pearson correlation coeff shotintensity normalised
                                    img_pearson_shotintensitynormalised = (img_shotintensitynormalised - accum_shotintensitynormalised.mean) 
                                    img_pearson_shotintensitynormalised *= accum_shotintensitynormalised.invstd
                                    img_pearson_shotintensitynormalised[~mask] = 0
                                    pearson_shotintensitynormalised = corr(img_pearson_shotintensitynormalised)
                                    accums[f'{detname}/correlations/{name}/pearson_shotintensitynormalised'].add(pearson_shotintensitynormalised)
                                    appenddata(outfile, f'{detname}/correlations/{name}/pearson_shotintensitynormalised/all', (roi(pearson_shotintensitynormalised) * invroinorm)[None, ...])

                                if args.nonshotintensitynormalised:
                                    # pearson correlation coeff
                                    img_pearson = (img - accum.mean) 
                                    img_pearson *= accum.invstd
                                    img_pearson[~mask] = 0
                                    pearson = corr(img_pearson)
                                    accums[f'{detname}/correlations/{name}/pearson'].add(pearson)
                                    appenddata(outfile, f'{detname}/correlations/{name}/pearson/all', (roi(pearson) * invroinorm)[None, ...])

                            if args.pearsonimagenorm:
                                # pearson correlation coeff, but use mean and std current image, not along time
                                img_imagenorm = (img_norm - np.nanmean(img_norm[mask])) 
                                img_imagenorm /= np.nanstd(img_norm[mask])
                                img_imagenorm = np.nan_to_num(img_imagenorm, copy=False)
                                img_imagenorm[~mask] = 0
                                pearson_imagenorm = corr(img_imagenorm)
                                accums[f'{detname}/correlations/{name}/pearson_imagenorm'].add(pearson_imagenorm)
                                appenddata(outfile, f'{detname}/correlations/{name}/pearson_imagenorm/all', (roi(pearson_imagenorm) * invroinorm)[None, ...])

                if not terminated:
                    print(ind_filtered + 1)
                else:
                    print()

        print('saving accumulated data', flush=True)
        for k, v in accums.items():
            print('  ', k, v.mean.shape, np.mean(v.n))
            with np.errstate(all='ignore'):
                if 'correlations' in k and not 'singleshotcorr' in k:
                    detname=k.split('/')[0]
                    norm = corr(np.array(outfile[f'{detname}/mask'], np.float64))
                    overwritedata(outfile, f'{k}/mean', np.array(v.mean / norm, np.float64))
                    overwritedata(outfile, f'{k}/std', np.array(v.std / norm, np.float64))
                else:
                    overwritedata(outfile, f'{k}/mean', np.array(v.mean, np.float64))
                    overwritedata(outfile, f'{k}/std', np.array(v.std, np.float64))
                overwritedata(outfile, f'{k}/n', np.array(v.n, np.int64))
        if not terminated:
            print('copying', flush=True)
            mask = np.copy(shot_ok)
            maximg = np.searchsorted(np.cumsum(np.logical_and.reduce([shot_ok] + [~np.array(outfile['filtering'][k], bool) for k in outfile['filtering'].keys() if 'skipped' in k])), args.nimages + 1)
            mask[maximg:] = False
            # mask[:]=True #TESTING
            param = outfile.create_group('parameters')
            copymasked(inputfile['attenuator'], param, mask)
            copymasked(inputfile['motors'], param, mask)
            copymasked(inputfile['photonEnergy'], param, mask)
            copymasked(inputfile['pulse_energy_beam_joule'], param, mask)
            copymasked(inputfile['pulse_energy_hutch_joule'], param, mask)
            copymasked(inputfile['tag_number_list'], param, mask)
            overwritedata(outfile, 'meta/interrupted', 'done')
            print()
        
        if args.maskfile is not None:
            maskfile.close()
    


        
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='sacla 2020 analysis phase2')
    parser.add_argument('inputfile', metavar='inputfile', type=isfile, help='the hdf5 inputfiles to process')
    parser.add_argument('outfile', default=None, metavar='outputfilename', type=str, help='where to save the output')
    parser.add_argument('--maskfile', default=None, dest='maskfile', metavar='maskfile', type=isfile, help='mask to use')
    parser.add_argument('--name', default='', dest='name', metavar='str', type=str, help='name to show in log')
    parser.add_argument('--det', default=None, dest='det', type=str, nargs='+', action='append', help='detectors (name cor/stat/both/none [+maxphotons N] [+scatterphotons N] [+skipphotons N] [+deltaelow N] [+deltaehigh N]). default: dual both. can be used multiple times.')

    arg_filtering = parser.add_argument_group('filtering')
    arg_filtering.add_argument('--pulsethres', default=400, dest='pulsethres', metavar='uJ', type=float, help='threshold energy beam')
    arg_filtering.add_argument('--nimages', default=np.inf, dest='nimages', metavar='N', type=float, help='number of images to do')

    arg_photons = parser.add_argument_group('default photon parameters if not set with --det')
    arg_photons.add_argument('--maxphotons', default=4, dest='maxphotons', metavar='N', type=int, help='max fluorescence photons to consider in one pixel')
    arg_photons.add_argument('--scatterphotons', default=1, dest='scatterphotons', metavar='N', type=int, help='max scatter photons to consider in one pixel')
    arg_photons.add_argument('--skipphotons', default=None, dest='skipphotons', metavar='N', type=int, help='skip images that have more than N*kalpha energy somewhere in them, default maxphotons')
    arg_photons.add_argument('--deltaehigh', default=1000, dest='deltaehigh', metavar='eV', type=float, help='how far ABOVE n*kalpha consider a fluorescence photon')
    arg_photons.add_argument('--deltaelow', default=1000, dest='deltaelow', metavar='eV', type=float, help='how far BELOW n*kalpha consider a fluorescence photon')
    arg_photons.add_argument('--correct', default=False, dest='correction', action='store_true', help='do masked median correction')


    arg_correlation = parser.add_argument_group('correlations')
    arg_correlation.add_argument('--disc', default=False, dest='disc', action='store_true', help='discrete photon variants (use number of fluorescence photons)')
    arg_correlation.add_argument('--cont', default=False, dest='cont', action='store_true', help='continuous variant (use ev values that are in 0.2*deltaelow..maxphotons*kalpha+deltaehigh)')
    arg_correlation.add_argument('--intv', default=False, dest='intv', action='store_true', help='interval variant (use ev values that are in the range around the kalpha peak)')
    arg_correlation.add_argument('--roi', default=50, dest='roi', metavar='pixel', type=int, help='roi size around center')
    arg_correlation.add_argument('--no-shotintensitynormalised', default=True, action='store_false', dest='shotintensitynormalised', help='SKIP shotintensitynormalised variants')
    arg_correlation.add_argument('--no-pixelmeannormalised', default=True, action='store_false', dest='pixelmeannormalised', help='SKIP pixelmeannormalised variants')
    arg_correlation.add_argument('--no-g2', default=True, action='store_false', dest='g2', help='SKIP g2 correlation')
    arg_correlation.add_argument('--nonshotintensitynormalised', default=False, action='store_true', dest='nonshotintensitynormalised', help='DO non-shotintensitynormalised variants')
    arg_correlation.add_argument('--nonpixelmeannormalised', default=False, action='store_true', dest='nonpixelmeannormalised', help='DO non-pixelmeannormalised variants')
    arg_correlation.add_argument('--singleshotcorr', default=False, action='store_true', dest='singleshotcorr', help='DO zncc fft paper based correlation')
    arg_correlation.add_argument('--pearson', default=False, action='store_true', dest='pearson', help='DO pearson correlation coeficent')
    arg_correlation.add_argument('--pearsonimagenorm', default=False, action='store_true', dest='pearsonimagenorm', help='DO pearson correlation with normalisation out of each image')

    args = parser.parse_args()
    if args.skipphotons is None:
        args.skipphotons = args.maxphotons
    if not (args.cont or args.disc or args.intv):
        parser.error('should at least one of cont, intv or disc')
    if (args.pearson or args.g2) and not (args.shotintensitynormalised or args.nonshotintensitynormalised):
        parser.error('should at least to one of non- and shotintensitynormalised')
    if (args.singleshotcorr) and not (args.pixelmeannormalised or args.nonpixelmeannormalised):
        parser.error('should at least to one of non- and pixelmeannormalised')
    if not (args.g2 or args.singleshotcorr or args.pearson or args.pearsonimagenorm):
        parser.error('no correlation method enabled')

    #lambda args:[int(i) for i in args.split(',')],
    detparser = argparse.ArgumentParser('--det', prefix_chars='+', add_help=False)
    detparser.add_argument('name')
    detparser.add_argument('inputname')
    detparser.add_argument('what', default='both', nargs='+', metavar='both/stats/corr/none')
    detparser.add_argument('+tiles', type=int, nargs='+', default=None)
    detparser.add_argument('+correct', default=args.correction, dest='correction', action='store_true')
    detparser.add_argument('+maxphotons', type=int, default=args.maxphotons)
    detparser.add_argument('+scatterphotons', type=int, default=args.scatterphotons)
    detparser.add_argument('+skipphotons', type=int, default=args.skipphotons)
    detparser.add_argument('+deltaehigh', type=float, default=args.deltaehigh)
    detparser.add_argument('+deltaelow', type=float, default=args.deltaelow)

    #detectorinfo2_t = collections.namedtuple('detectorinfo2', ['inputname','tiles','photons', 'correlations', 'maxphotons', 'scatterphotons', 'skipphotons', 'deltaelow', 'deltaehigh', 'correction'])

    
    
    if args.det is None:
        #TODO
        args.det = [['dual', 'both']]
    dets = {}
    for d in args.det:
        detargs = detparser.parse_args(d)
        dets[detargs.name] = detectorinfo2_t(
            detargs.inputname,
            detargs.tiles,
            any(x in ' '.join(detargs.what).lower() for x in ['corr', 'both', 'stat']),
            any(x in ' '.join(detargs.what).lower() for x in ['corr', 'both']),
            detargs.maxphotons,
            detargs.scatterphotons,
            detargs.skipphotons,
            detargs.deltaelow,
            detargs.deltaehigh,
            detargs.correction
        )

    main()
    print(f'done at {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
      