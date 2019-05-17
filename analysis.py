import sacla
import argparse, os
import numpy as np
import idi.reconstruction as recon
from idi.util import *
from funchelper import *
import scipy.ndimage as snd
import os, shutil
import datetime

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

def diffdist(*args):
    accum = 0
    for arg in args:
        accum += np.diff(arg) ** 2
    return np.sqrt(accum)

def intensities(detector):
    @asgen
    def intensity(img):
        return np.sum(img)

    return detector.absolute_gain * 3.65 * np.array(list(intensity(detector)))

def getbg(detector):
    accum = accumulator()
    for img in detector:
        dat = np.array(img) * detector.absolute_gain * 3.65
        hits = dat > 2000
        empty = ~(snd.morphology.binary_dilation(hits, snd.morphology.generate_binary_structure(2, 2)))
        accum.add(dat * empty.astype(float), empty)
    return accum.mean

def photonize(img, energy, gain=1, bg=0):
    return np.rint(((np.squeeze(np.array(img)) * gain * 3.65) - bg) / energy)

def photonsstats(detector, bg, energy, thres=10):
    accum = accumulator()
    photonsum = []
    maxphotons = 0
    for n, img in enumerate(detector):

        photons = photonize(img, energy, detector.absolute_gain, bg)
        ps = np.sum(photons)
        if ps > thres:
            accum.add(photons)
            maxphotons = np.maximum(maxphotons, photons)
        photonsum.append(ps)
    return (accum.mean, accum.std, maxphotons, np.array(photonsum))


parser = argparse.ArgumentParser(description='sacla 2019 analysis')
parser.add_argument('inputfile', metavar='inputfile', type=isfile, help='the hdf5 inputfile to process')
parser.add_argument('--outpath', default=None, metavar='path', type=isdir, help='where to save the output (default work dir)')
parser.add_argument('--workpath', default=None, metavar='path', type=isdir, help='the work dir (default input file dir)')
parser.add_argument('--run', default='', dest='run', type=str, help='run info/number to store as reference')
parser.add_argument('--simple', dest='simple', action='store_true', help='do simple ft correlation')
parser.add_argument('--ft3d', dest='ft3d', action='store_true', help='do 3d ft correlation')
parser.add_argument('--direct', dest='direct', action='store_true', help='do 3d direct correlation (slow!)')
parser.add_argument('--directrad', dest='directrad', type=int, nargs='?', default=False, const=-1, metavar='QMAX', help='do radial direct correlation')
parser.add_argument('--detector', dest='detector', type=str, default='detector_2d_3', metavar='DETECTORNAME', help='name of detector')
parser.add_argument('-e', dest='energy', type=float, default=6450, metavar='ENERGY in ev', help='photon energy')
parser.add_argument('-z', dest='z', type=float, default=10, metavar='DISTANCE in cm', help='detector distance')
parser.add_argument('--threshold', dest='photonsthreshold', type=int, default=50, metavar='THRESHOLD in photons', help='min. photons in image to keep it')
parser.add_argument('--pixelsize', dest='pixelsize', type=float, default=0.1, metavar='PIXELSIZE in um', help='detector pixelzie')
parser.add_argument('--maximg', dest='maximg', type=int, default=-1, metavar='MAXIMG', help='detector pixelsize')
parser.add_argument('--allimg', dest='allimg', action='store_true', help='store all photonized images in result')

args = parser.parse_args()
if args.workpath is None:
    args.workpath = os.path.dirname(args.inputfile)
if args.outpath is None:
    args.outpath = args.workpath
workfile = os.path.join(args.workpath, os.path.basename(args.inputfile))
if os.path.isfile(workfile):
    print(f' File {workfile} exists, not copying to workdir.')
else:
    print(f' copying input to {workfile}')
    shutil.copy(args.inputfile, workfile)
outfile=os.path.join(args.outpath,datetime.datetime.now().strftime(f'{os.path.splitext(os.path.basename(args.inputfile))[0]}-%y%m%d-%H%M%S.npz'))

run = sacla.saclarun(workfile, settings=sacla.Tais2019)
print(f'{len(run)} images in input')
detector = getattr(run, args.detector)
energy = args.energy
z = args.z * 1e-2 / args.pixelsize * 1e-6
nmax = np.inf if args.maximg == -1 else args.maximg
print(vars(args))
print('init done', flush=True)

# filter by distance between shots
setdist = np.percentile(diffdist(run.sampleX), 75)
mindist = setdist * 0.7
distok = np.concatenate(([0], diffdist(run.sampleX, run.sampleZ))) > mindist
shots = run[distok]
detector = getattr(shots, args.detector)
print(f'distance done, {len(shots)} remaining')

#background
bg = getbg(detector)
print('background done', flush=True)

#photons statistics
meanphotons, stdphotons, maxphotons, photonsum = photonsstats(detector, bg, energy, args.photonsthreshold)
intok = photonsum > args.photonsthreshold
nphotonsmin = np.rint(np.percentile(photonsum[intok], 1))
nphotonsmax = np.rint(np.percentile(photonsum[intok], 99))

intok = np.logical_and.reduce((intok, nphotonsmin < photonsum, photonsum < nphotonsmax))
mask = meanphotons > (0.1 * np.mean(meanphotons))
shots = shots[intok]
detector = getattr(shots, args.detector)
print(f'found photons statistics. intensity filter done, keep >{nphotonsmin} && <{nphotonsmax}. {len(shots)} remaining')
accum = {'simple': accumulator(), 'ft3d': accumulator(), 'direct': accumulator(), 'directrad': accumulator()}

print('start recon...', flush=True)
allimg=[]
for n, img in enumerate(detector):
    if n >= nmax:
        break
    photons = photonize(img, energy, detector.absolute_gain, bg) / meanphotons
    photons[~mask] = 0
    if args.allimg:
        allimg.append(np.array(photons))
    if args.simple:
        accum['simple'].add(recon.simple.corr(photons))
    if args.ft3d:
        accum['ft3d'].add(recon.ft.corr(photons, z))
    if args.direct:
        accum['direct'].add(recon.direct.corr(photons, z))
    if args.directrad:
        accum['directrad'].add(recon.newrad.corr(photons, z, qmax))
    if n == 0:
        for a in accum:
            print(a, accum[a].shape)
    if n % 10 == 0:
        print(n, end=' ',flush=True)

allimg=np.array(allimg)
print()
print(f'start saving to {outfile}')
tosave = vars(args)
tosave.update(
    {
        'workfile': workfile,
        'outfile': outfile,
        'mask': mask,
        'meanphotons': meanphotons,
        'stdphotons': stdphotons,
        'maxphotons': maxphotons,
        'nphotonsmax': nphotonsmax,
        'nphotonsmin': nphotonsmin,
        'photonsum': photonsum,
        'bg': bg,
        'mindist': mindist,
        'allimg':allimg
    }
)
tosave.update({f'{k}_mean': v.mean for k, v in accum.items()})
tosave.update({f'{k}_std': v.std for k, v in accum.items()})
np.savez_compressed(outfile, **tosave)
print('done!')
