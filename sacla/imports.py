import matplotlib.pyplot as plt
%matplotlib inline 
import numpy as np
import pandas as pd
pd.options.display.max_rows = 999
import scipy as sp
import scipy.ndimage as snd
import scipy.signal as ss
import skimage.morphology as skm
import mkl_fft
import fast_histogram
import sys, os, datetime, shutil, glob, itertools, collections, pathlib
import h5py
from tqdm.notebook import tqdm
from pprint import pprint
import matplotlib.pyplot as plt
import matplotlib as mpl
%load_ext line_profiler
%load_ext memory_profiler
from sacla import *
from accum import *
import idi.reconstruction as recon
import re
from pathlib import Path


def browse(obj,depth=0):
    try:
        if type(obj)==h5py._hl.dataset.Dataset:
            print(obj.shape,obj.dtype,end='')
            if np.product(obj.shape)<10: print(np.array(obj), end='')
        else:
            for k in obj.keys():
                print('\n',depth*'   ',k,end='')
                browse(obj[k],depth+1)
            print()
    except Exception as e:
        print(e)

def zoom(cor):
        cor[cor.shape[0]//2,cor.shape[1]//2]=np.nan
        plt.matshow(cor[cor.shape[0]//2-n:cor.shape[0]//2+n,cor.shape[1]//2-n:cor.shape[1]//2+n])
    
