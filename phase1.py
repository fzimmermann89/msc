#!/usr/bin/env python

import os

os.environ['OMP_NUM_THREADS'] = '14'
os.environ['OPENBLAS_NUM_THREADS'] = '14'
os.environ['MKL_NUM_THREADS'] = '14'
os.environ['VECLIB_MAXIMUM_THREADS'] = '14'
os.environ['NUMEXPR_NUM_THREADS'] = '14'
import mkl
mkl.set_num_threads(14)

from sacla import *
import argparse, os
import numpy as np
from accum import *
import os, shutil
import datetime
import h5py
import scipy.signal as ss
import collections
import fast_histogram
import signal
import sys

from h5util import *

terminated=False
def sigterm_handler(_signo, _stack_frame):
    global terminated
    terminated=True
    try:
        print('\n!!!terminated!!!',flush=True)
    except:
        pass
    
signal.signal(signal.SIGTERM, sigterm_handler)
signal.signal(signal.SIGUSR1, sigterm_handler)
signal.signal(signal.SIGINT, sigterm_handler)

def sigusr2_handler(_signo, _stack_frame):
    global args
    args.chunkend=0
    print('\n!!quitting after this file!!',flush=True)
signal.signal(signal.SIGUSR2, sigusr2_handler)


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



parser = argparse.ArgumentParser(description='sacla 2020 analysis 3')
parser.add_argument('inputfiles', metavar='inputfiles', type=isfile, nargs='+', help='the hdf5 inputfiles to process')
parser.add_argument('outfile', default=None, metavar='outputfilename', type=str, help='where to save the output')
parser.add_argument('--workpath', default=None, metavar='path', type=isdir, help='the work dir (default input file dir)')
parser.add_argument('--bgfile', default=None, dest='bgfile', type=isfile, help='background file')
parser.add_argument('--chunkstart', default=None, dest='chunkstart', type=int, help='start of chunk of inputfiles to do')
parser.add_argument('--chunkend', default=None, dest='chunkend', type=int, help='end of chunk of inputfiles to do')
parser.add_argument('--name', default='', dest='name', type=str, help='run info name')
parser.add_argument('--sample', default='', dest='sample', type=str, help='sample name')

parser.add_argument('--focus_y', default=0, dest='focus_y', type=float, help='profy in um at focus')
parser.add_argument('--kalpha', dest='kalpha', type=float, default=0, help='Kalpha energy in ev')
parser.add_argument('--pulsethres', dest='pulsethres', type=float, default=250, help='pulsethreshold in uJ')


args = parser.parse_args()
kalpha=args.kalpha

print(f'Phase1 {args.name} run using files\n {args.inputfiles}')


props = [
    'pulse_energy_hutch_joule',
    'pulse_energy_beam_joule',
    'photonEnergy',
]
motors=[
    'sampleThX',
    'sampleThY',
    'sampleThZUpper',
    'sampleThZLower',
    'sampleX',
    'sampleZ',
    'profX',
    'profZ',
    'profY',
    'octalX',
    'octalY',
    'octalZ',
    'dualX',
    'dualZ',
    'singleX',
    'singleZ',
]
attenuators=[
 'attenuator_eh_2_Al_thickness_in_meter',
 'attenuator_eh_4_Al_thickness_in_meter',
 'attenuator_eh_5_Si_uncalibrated',
 'attenuator_oh_2_Si_thickness_in_meter',
]

detectorinfo_t = collections.namedtuple('detectorinfo', ['savemeta', 'savedata', 'detectors'])
dets = {
    'dual': detectorinfo_t(True, True, ['detector_2d_2','detector_2d_3']),
    'octal': detectorinfo_t(True, True, ['detector_2d_4','detector_2d_5','detector_2d_6','detector_2d_7','detector_2d_8','detector_2d_9','detector_2d_10','detector_2d_11']),
}

histrange=(-2000, int(1e5))
histres=10
histrange=(int(histrange[0]),int(histrange[0]+histres*np.ceil(np.diff(histrange)/histres)))

def attenuator_cal(pulses):
    motorpulses = np.array((226200, 179059, 166522, 154238, 135266, 91636, 79067, 66630, 54075, 41718, 22601))
    thickness = np.array((0, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 1.5, 2.0, 2.5, 3.0)) * 1e-3
    return thickness[np.abs(motorpulses[None, :] - pulses[:, None]).argmin(axis=1)]

def calcedgeintensity(d):
    s0,s1=d.shape
    edges=(d[s0-s0//2-s0//4:0-s0//2+s0//4,:s1//4],
        d[s0-s0//2-s0//4:0-s0//2+s0//4,-s1//4:],
        d[:s0//4,s1-s1//2-s1//4:0-s1//2+s1//4],
        d[-s0//4:,s1-s1//2-s1//4:0-s1//2+s1//4])
    return np.array([np.nansum(e) for e in edges])



if args.bgfile is not None:
    print(f'using dark background from file {args.bgfile}')
    bgfile = h5py.File(args.bgfile, 'r')

accums = {}

# create output file and load old data #TODO chunking
if os.path.isfile(args.outfile):
    if not args.chunkstart is  None:
        print(f'starting at file {args.chunkstart}, loading old data')
        outfile = h5py.File(args.outfile, 'r+')
        for detoutname in outfile['detectors']:
            for tilename in outfile['detectors'][detoutname]:
                d=outfile['detectors'][detname][tilename]
                acc=accumulator(True)
                acc._n=np.array(d['n'])
                acc._mean=np.array(d['mean'])
                acc._max=np.array(d['max'])
                acc._min=np.array(d['min'])
                acc._nvar= np.square(np.array(d['std']))*np.array(d['n'])
                accums[f'detectors/{detoutname}/{tilename}']=acc

    else:
        raise FileExistsError(f'{args.outfile} exists. will not overwrite existing file. quitting.')
else:
    outfile = h5py.File(args.outfile, 'w')
print(f'saving to {args.outfile}')



# rough filtering
print('filtering...', flush=True)

try:
    sampleX = []
    pulseenergyhutch = []
    pulseenergybeam = []
    for f in args.inputfiles:
        inputfile = saclarun(f, settings=Tais2020)
        sampleX.append(np.array(inputfile.sampleX))
        pulseenergyhutch.append(np.array(inputfile.pulse_energy_hutch_joule))
        pulseenergybeam.append(np.array(inputfile.pulse_energy_beam_joule))
    beamok=np.concatenate(pulseenergybeam)>args.pulsethres*1e-6
    pulseenergythres = np.mean(np.concatenate(pulseenergyhutch)[beamok]) - 1*np.std(np.concatenate(pulseenergyhutch)[beamok])
    print(f'   energy threshold={pulseenergythres}', flush=True)
    pulseenergyok = [(eh > pulseenergythres) & (eb > args.pulsethres*1e-6) for eh, eb in zip(pulseenergyhutch, pulseenergybeam)]
    sampleXdiff = [np.abs(e[1:] - e[:-1]) for e in sampleX]
    sampleXdeltathres = np.mean(np.concatenate(sampleXdiff)) - np.std(np.concatenate(sampleXdiff))
    print(f'   delta samplex threshold={sampleXdeltathres}', flush=True)
    sampleXok = [~ss.convolve(diff < sampleXdeltathres, np.ones(7, bool))[3:-2] for diff in sampleXdiff]
    allsampleXok = np.concatenate(sampleX)[np.concatenate(sampleXok).astype(bool)]
    sampleXthreslow=np.nanpercentile(allsampleXok,1)
    sampleXthreshigh=np.nanpercentile(allsampleXok,99)
    print(f'   samplex thresholds= {sampleXthreslow}  ..  {sampleXthreshigh} ', flush=True)
    sampleXok = [np.logical_and.reduce((ok, csx>sampleXthreslow, csx<sampleXthreshigh)) for ok, csx in zip(sampleXok, sampleX)]
    ok = [np.logical_and(e1, e2) for e1, e2 in zip(sampleXok, pulseenergyok)]
    print(f'   will keep {np.sum(np.concatenate(ok))} shots', flush=True)
except Exception as e:
    print('   error in filtering')
    raise e
    
print('saving meta data', flush=True)
dictmeta = {
    'name': args.name, 
    'sample': args.sample, 
    'runs': [os.path.splitext(os.path.basename(i))[0] for i in args.inputfiles],
    'inputfiles':args.inputfiles,
    'Kalpha': float(args.kalpha), 
    'focus_y': float(args.focus_y),
    'filtering/thres_sampleX_delta':sampleXdeltathres,
    'filtering/thres_sampleX_low':sampleXthreslow,
    'filtering/thres_sampleX_high':sampleXthreshigh,
    'filtering/thres_pulseenergy':pulseenergythres,
    'filtering/shot_ok': list2array(ok),
    'interrupted':-1
}
for k, v in dictmeta.items():
    overwritedata(outfile, f'meta/{k}', np.array(v))

    print('writing settings:')
for k, v in vars(args).items():
    if v is None:
        v = 'None'
    print('  ', k, v)
    overwritedata(outfile, f'meta/arguments/{k}', np.array(v))
        
for fi, f in enumerate(args.inputfiles):
    if not args.chunkstart is None:
        if args.chunkstart>fi: continue
    if not args.chunkend is None:
        if args.chunkend<=fi: 
            print(f'reached chunkend {fi}')
            break
            
    already_added=collections.defaultdict(int) #used if interupted
    print(f'doing file {f}', flush=True)
    if np.sum(ok[fi]) == 0:
        print('   no good shots!')
        continue
    if args.workpath is not None:
        workfile = os.path.join(args.workpath, os.path.basename(f))
        if os.path.isfile(workfile):
            print(f'   File {workfile} exists, not copying to workdir.', flush=True)
        else:
            print(f'   copying input to {workfile}', flush=True)
            shutil.copy(f, workfile)
        inputfile = saclarun(workfile, settings=Tais2020)
    else:
        inputfile = saclarun(f, settings=Tais2020)

    inputfile = inputfile[ok[fi]]
    #inputfile = inputfile[:10] #TESTING

    for detoutname, detinfo in dets.items():
        print(f'   doing detector {detoutname} ', end='')
        if not (detinfo.savemeta or detinfo.savedata):
            print('skipping', flush=True)
            continue
        if detinfo.savemeta:
            print('saving meta. ', end='')
        if detinfo.savedata:
            print('saving data. ', end='')
        print()
        for tilenr,detname in enumerate(detinfo.detectors):
            try:
                det = getattr(inputfile, detname)
                print(f'     doing tile {tilenr} ')
            except KeyError:
                print(f'error: {detname} (tile {tilenr} of {detoutname}) not found in {f}')
                continue
            print('       ', end='')

            gain = det.absolute_gain
            if detinfo.savemeta:
                intensity = np.zeros(len(inputfile), np.float32)
                edgeintensity = np.zeros((len(inputfile),4), np.float32)

                hists = np.zeros((len(inputfile), int(np.diff(histrange))//histres+3), np.float32)
                if f'detectors/{detoutname}/tile{tilenr}' not in accums.keys():
                    accums[f'detectors/{detoutname}/tile{tilenr}'] = accumulator(maxmin=True)

                if f'detectors/{detoutname}/tile{tilenr}/singlephotoncount' not in accums.keys() and kalpha!=0:
                    accums[f'detectors/{detoutname}/tile{tilenr}/singlephotoncount'] = np.zeros(det[0][0].shape)
                
                if f'detectors/{detoutname}/tile{tilenr}/absfft0' not in accums.keys():
                    accums[f'detectors/{detoutname}/tile{tilenr}/absfft0'] = accumulator(maxmin=False)
                if f'detectors/{detoutname}/tile{tilenr}/absfft1' not in accums.keys():
                    accums[f'detectors/{detoutname}/tile{tilenr}/absfft1'] = accumulator(maxmin=False)
                    
            if args.bgfile is not None:
                bg = np.array(bgfile[f'{detname}/mean'])
            else:
                bg = 0
            for i in range(len(det)):
                if i%1000==0: print(i,flush=True,end=' ')
                #if i>10:break #TESTING
                if terminated:
                    outfile['meta/interrupted'][0]=-2
                    for k,v in already_added.items():
                        shrink(outfile,k,v)
                    outfile['meta/interrupted'][0]=fi
                    outfile.close()
                    print('quitting')
                    sys.exit(1)
                ev = (np.array(det[i]) * (gain * 3.6)) - bg
                if detinfo.savemeta:
                    intensity[i] = ev.sum()
                    edgeintensity[i] = calcedgeintensity(ev)
                    hists[i,:]=fast_histogram.histogram1d(np.clip(ev[np.abs(ev)>1e-3],
                                                                  histrange[0]-histres,
                                                                  histrange[1]+histres),
                                                          range=(histrange[0]-3*histres//2,histrange[1]+3*histres//2),
                                                          bins=np.shape(hists)[1])
                    accums[f'detectors/{detoutname}/tile{tilenr}'].add(ev)
                    if kalpha!=0:
                        accums[f'detectors/{detoutname}/tile{tilenr}/singlephotoncount'] += ((ev > (kalpha - 1000)) & (ev < (kalpha + 1000))).astype(float)
                    if i%10==0:    
                        accums[f'detectors/{detoutname}/tile{tilenr}/absfft0'].add(np.abs(np.fft.rfft(ev,axis=0)))
                        accums[f'detectors/{detoutname}/tile{tilenr}/absfft1'].add(np.abs(np.fft.rfft(ev,axis=1)))

                    
                if detinfo.savedata:
                    already_added[f'detectors/{detoutname}/tile{tilenr}/data']+=1
                    appenddata(outfile, f'detectors/{detoutname}/tile{tilenr}/data', np.clip(ev, - (2 ** 31), (2 ** 31) - 1).astype(np.int32)[None, ...])           
            if detinfo.savemeta:
                already_added[f'detectors/{detoutname}/tile{tilenr}/hist/run']+=1
                already_added[f'detectors/{detoutname}/tile{tilenr}/hist/shot']+=len(hists)
                already_added[f'detectors/{detoutname}/tile{tilenr}/intensity']+=len(intensity)
                already_added[f'detectors/{detoutname}/tile{tilenr}/edgeintensity']+=len(edgeintensity)

                appenddata(outfile, f'detectors/{detoutname}/tile{tilenr}/hist/shot', hists)
                appenddata(outfile, f'detectors/{detoutname}/tile{tilenr}/hist/run', np.sum(hists,axis=0)[None,:])
                appenddata(outfile, f'detectors/{detoutname}/tile{tilenr}/intensity', intensity)
                appenddata(outfile, f'detectors/{detoutname}/tile{tilenr}/edgeintensity', edgeintensity)

                histbincenters=np.concatenate(([-np.inf],np.linspace(histrange[0],histrange[1],hists.shape[1]-2),[np.inf]))
                overwritedata(outfile,f'detectors/{detoutname}/tile{tilenr}/hist/bincenters',histbincenters)
                
            if detinfo.savemeta or detinfo.savedata: 
                attr=outfile[f'detectors/{detoutname}/tile{tilenr}/'].attrs
                attr['detector_coordinate_in_micro_meter']=np.array(det.detector_coordinate_in_micro_meter)
                attr['detector_name']=np.array(det.detector_name)
                attr['detector_rotation_angle_in_degree']=np.array(det.detector_rotation_angle_in_degree)
                attr['detector_rotation_steps']=np.array((int(np.rint(det.detector_rotation_angle_in_degree/90)),(det.detector_rotation_angle_in_degree+45)%90-45))
                attr['detector_tile_position_in_pixels']=np.array((-1,1)*np.array(det.detector_coordinate_in_micro_meter)[1::-1]/np.array(det.pixel_size_in_micro_meter))
          
                
                
            print(f'{i}.',flush=True)
    # append  spectrum
    print('   saving spectrum',flush=True)
    spec = np.sum(np.array(inputfile.detector_2d_1), axis=2,dtype=np.float32)
    appenddata(outfile, f'beamspectrum', spec)
    
    print('   saving tmp accumulated data', flush=True)
    for k, v in accums.items():
        if 'singlephotoncount' in k:
            overwritedata(outfile, f'{k}',np.array(v,np.float32))
        else:
            overwritedata(outfile, f'{k}/mean', np.array(v.mean,np.float32))
            overwritedata(outfile, f'{k}/std', np.array(v.std,np.float32))
            overwritedata(outfile, f'{k}/n', np.array(v.n,np.int32))

            try:
                overwritedata(outfile, f'{k}/max', np.array(v.max,np.float32))
                overwritedata(outfile, f'{k}/min', np.array(v.min,np.float32))
            except NotImplementedError:
                pass
    
    try:  # appending point data
        for prop in props:
            appenddata(outfile, prop, np.array(getattr(inputfile, prop), dtype=np.float32))
        for motor in motors:
            appenddata(outfile, f'motors/{motor}', np.array(getattr(inputfile, motor), dtype=np.float32))
        appenddata(outfile, 'attenuator/attenuator_eh_5_Si_thickness_in_meter', np.array(attenuator_cal(inputfile.attenuator_eh_5_Si_uncalibrated), dtype=np.float32))
        appenddata(outfile, 'tag_number_list', np.array(getattr(inputfile, 'tag_number_list'), dtype=np.int64))
        for attenuator in attenuators:
            appenddata(outfile, f'attenuator/{attenuator}', np.array(getattr(inputfile, attenuator), dtype=np.float32))

        print('   added point data', flush=True)
    except Exception as e:
        print('   error in adding point data')
        print(e)

        
print('saving accumulated data', flush=True)

for k, v in accums.items():
    if 'singlephotoncount' in k:
        overwritedata(outfile, f'{k}',np.array(v,np.float32))
    else:
        overwritedata(outfile, f'{k}/mean', np.array(v.mean,np.float32))
        overwritedata(outfile, f'{k}/std', np.array(v.std,np.float32))
        overwritedata(outfile, f'{k}/n', np.array(v.n,np.int32))
        try:
            overwritedata(outfile, f'{k}/max', np.array(v.max,np.float32))
            overwritedata(outfile, f'{k}/min', np.array(v.min,np.float32))
        except NotImplementedError:
            pass

#todo
if args.bgfile is not None:
    print('saving used backgrounds')
    for detoutname, detinfo in dets.items():
        if detinfo.savemeta or detinfo.savedata:
            for tilenr,detname in enumerate(detinfo.detectors):
                bg = np.array(bgfile[f'{detname}/mean'])
                overwritedata(outfile, f'detectors/{detoutname}/tile{tilenr}/bg', bg)
    
            



outfile.close()
print('done')
