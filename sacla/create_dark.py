#!/usr/bin/env python
from sacla import *
import argparse, os
import numpy as np
import os, shutil
from accum import *
import datetime
import scipy.signal as ss

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


def appenddata(file, key, data):
    data=np.atleast_1d(np.array(data))
    if np.array(data).dtype.kind == 'U': data=data.astype(h5py.string_dtype(encoding='ascii'))
    if key not in file.keys():
        file.create_dataset(key, compression="gzip", chunks=data.shape, data=data, maxshape=(None, *data.shape[1:]))
    else:
        file[key].resize((file[key].shape[0] + data.shape[0]), axis=0)
        file[key][-data.shape[0] :] = data


parser = argparse.ArgumentParser(description='sacla 2020 analysis dark')
parser.add_argument('inputfiles', metavar='inputfiles', type=isfile, nargs='+', help='the hdf5 inputfiles to process')
parser.add_argument('outfile', default=None, metavar='filename', type=str, help='where to save the output')
parser.add_argument('--workpath', default=None, metavar='path', type=isdir, help='the work dir (default input file dir)')
parser.add_argument('--threshold', default=5, metavar='X', type=isdir, help='if max pixel in one image is higher than X times std, ignore image')
args = parser.parse_args()

print(f'Dark run using files {args.inputfiles}')
if os.path.isfile(args.outfile):
    raise FileExistsError(f'{args.outfile} exists. will not overwrite existing file. quitting.')
outfile = h5py.File(args.outfile, 'w')
accums = {}

for f in args.inputfiles:
    if args.workpath is not None:
        workfile = os.path.join(args.workpath, os.path.basename(f))
        if os.path.isfile(workfile):
            print(f' File {workfile} exists, not copying to workdir.', flush=True)
        else:
            print(f' copying input to {workfile}', flush=True)
            shutil.copy(f, workfile)
        inputfile = saclarun(workfile, settings=Tais2020)
    else:
        inputfile = saclarun(f, settings=Tais2020)
    lowintensity=(np.array(inputfile.pulse_energy_hutch_joule)<1e-5)
    lowintensity=~ss.convolve(~lowintensity,np.ones(7, bool))[3:-3]
    inputfile=inputfile[lowintensity]
    print(f'Dark shots in {f}: {len(inputfile)}', flush=True)
    dets = [k for k in dir(inputfile) if 'detector_2d' in k]
    print(f'Detectors in {f}: {dets}', flush=True)
    
    for detname in dets:
        print(f'doing {detname} in {f}', flush=True)
        if detname not in accums.keys():
            accums[detname] = accumulator()
        det = getattr(inputfile, detname)
        m = np.max(np.array(det[:300]), axis=(1, 2))
        threshold = np.rint((m.mean() + args.threshold * m.std()) * 3.6 * det.absolute_gain)
        
        print(f' threshold: {threshold}')
        for img in det:
            img = np.array(img) * 3.6 * det.absolute_gain
            if np.any(img > threshold):
                print(f'ignored image on {detname} with max intensity {int(np.rint(np.max(img)))}>{int(threshold)} ev', flush=True)
                continue
            accums[detname].add(img)
            
print(f'saving to {args.outfile}', flush=True)

print(f'writing {list(accums.keys())}', flush=True)
for k, v in accums.items():
    appenddata(outfile, f'{k}/mean', v.mean)
    appenddata(outfile, f'{k}/std', v.std)
    appenddata(outfile, f'{k}/n', v.n)
appenddata(outfile, 'inputfiles', args.inputfiles)
appenddata(outfile, 'timestamp',int(datetime.datetime.now().strftime(f'%y%m%d%H%M%S')))

outfile.close()
print('done')
