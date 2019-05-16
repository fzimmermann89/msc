import sacla
import argparse, os
import numpy as np
import idi.reconstruction as recon
from idi.util import *
from funchelper import *
import scipy.ndimage as snd

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
    accum=0
    for arg in args: accum+=np.diff(arg)**2
    return np.sqrt(accum)

def intensities(detector):
    @asgen
    def intensity(img):
        return np.sum(img)
    return detector.absolute_gain*3.65*np.array(list(intensity(detector)))

parser = argparse.ArgumentParser(description='sacla 2019 analysis')
parser.add_argument('inputfile', metavar='inputfile', type=isfile,
                    help='the hdf5 inputfile to process')
parser.add_argument('--outpath', default=None, metavar='path', type=isdir,
                    help='the hdf5 inputfile to process')
parser.add_argument('--run', default='', dest='run', type=str,
                    help='run info/number to store as reference')
parser.add_argument('--simple', dest='simple', action='store_true',
                    help='do simple ft correlation')
parser.add_argument('--ft3d', dest='ft3d', action='store_true',
                    help='do 3d ft correlation')
parser.add_argument('--direct', dest='direct', action='store_true',
                    help='do 3d direct correlation (slow!)')
parser.add_argument('--directrad', dest='directrad', type=int, nargs='?',
                    default=False, const=-1, metavar='QMAX',
                    help='do radial direct correlation')
parser.add_argument('--detector', dest='detector', type=str,
                    default='detector_2d_3', metavar='DETECTORNAME',
                    help='name of detector')
parser.add_argument('-e', dest='energy', type=float,
                    default=6400, metavar='ENERGY in ev',
                    help='photon energy')
parser.add_argument('-z', dest='z', type=float,
                    default=10, metavar='DISTANCE in cm',
                    help='detector distance')
parser.add_argument('--pixelsize', dest='pixelsize', type=float,
                    default=0.1, metavar='PIXELSIZE in um',
                    help='detector pixelzie')
parser.add_argument('--maximg', dest='maximg', type=int,
                    default=0, metavar='MAXIMG',
                    help='detector pixelzie')
args = parser.parse_args()



if args.outpath is None: args.outpath=os.path.dirname(args.inputfile)    
run=sacla.saclarun(args.inputfile,settings=sacla.Tais2019)
detector=getattr(run,args.detector)        
energy=args.energy
z=args.z*1e-2/args.pixelsize*1e-6
nmax=np.inf if args.maximg == 0 else args.maximg
print(vars(args))

#filter by distance between shots and intensity
setdist=np.percentile(diffdist(run.sampleX),75)
mindist=setdist*0.7
distok=np.concatenate(([0],diffdist(run.sampleX,run.sampleZ)))>mindist
intensity=intensities(detector)
intok=np.logical_and(intensity>np.percentile(intensity,5),intensity<np.percentile(intensity,95))
shots=run[np.logical_and(intok,distok)]
detector=getattr(shots,args.detector)



def getbg(detector):
    accum=accumulator()
    for img in detector:
        dat=np.array(img)*run.detector.absolute_gain*3.65
        hits=dat>2000
        empty=~(snd.morphology.binary_dilation(hits,snd.morphology.generate_binary_structure(2, 2)))
        count+=empty
        accum.add(dat*empty.astype(float), empty)
    return accum.mean
bg=0#getbg(detector)

def photonize(img, energy, gain=1, bg=0):
    return np.rint(((np.squeeze(np.array(img))*gain*3.65)-bg)/energy)
def photonsstats(detector,bg,energy,maxthres=10):
    accum=accumulator()
    maxphotons=0
    for n,img in enumerate(detector):   
        
        photons=photonize(img, energy, detector.absolute_gain, bg)
        accum.add(photons)
        maxphotons = np.maximum(maxphotons,photons)
    return(accum.mean,acum.std,maxphotons)
meanphotons,stdphotons,maxphotons=photonsstats(detector,bg,energy)
mask=(meanphotons>(0.1*np.mean(meanphotons)))



accum={
    'simple': accumulator(), 
    'ft3d':accumulator(), 
    'direct':accumulator(), 
    'directrad':accumulator()
}
for n,img in enumerate(detector):
    if n>=nmax: break
    photons=photonize(img, energy, detector.absolute_gain, bg)
    if args.simple: accum['simple'].add(recon.simple.corr(photons))
    if args.ft3d: accum['ft3d'].add(recon.ft.corr(photons,z))
    if args.direct: accum['direct'].add(recon.direct.corr(photons,z))
    if args.directrad: accum['directrad'].add(recon.newrad.corr(photons,z, qmax))
 
tosave=vars(args)
tosave.update({f'{k}_mean': v.mean for k,v in accum.items})
tosave.update({f'{k}_std': v.std for k,v in accum.items})
