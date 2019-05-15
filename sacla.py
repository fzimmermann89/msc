from funchelper import *
from settings import *

import pandas as pd
class log:
    def __init__(self, filename):
        df = pd.read_csv(filename, header=[0, 1])
        fillcols = [i for i, k in enumerate(df.keys()) if not "Comment" in k]
        df.iloc[:, fillcols] = df.iloc[:, fillcols].fillna(method="ffill")
        self.dataframe = df

    def search(self, field, value):
        field = [k for k in self.dataframe.keys() if field in k]
        if len(field) != 1:
            raise KeyError
        return self.dataframe[self.dataframe[field[0]].str.contains(value, na=False)]

    def __getattr__(self, attr):
        if attr in self.__dict__.keys():
            return self.__dict__[attr]
        return getattr(self.dataframe, attr)

    def __getitem__(self, item):
        return self.dataframe[item]

    def __setitem__(self, item, value):
        self.dataframe[item] = item

    def __setattr(self, attr, value):
        if attr in self.__dict__.keys():
            self.__dict__[attr] = value
        else:
            setattr(self.dataframe, attr, value)


import h5py, numpy as np, os, re


class saclarun:
    """
    a convenience wrapper around the sacla h5 files for data analysis, similiar on lcls run
    """
    def _parsekey(self,key,partial=False):

        if key in self._shortkeys:
            return self._shortkeys[key] #shortkey

        if key in self._longkeys:
            return self._longkeys[key]  # longkey

        if key in self._longkeys.values():
            return key  # hdf5 key

        if partial:
            tmp = [vs for ks, vs in self._shortkeys.items() if key in ks]
            if len(tmp) == 1:
                return tmp[0]  # unique partial shortkey

        return None #not found


    def __init__(self, file, settings={"hide": ["camera"]}):
        """
        saclarun(path_to_sacla_h5_file)
        """

        def __load(filename):
            """
            loads the workfile to a dict
            """

            def recur(h5, path):
                ret = {}
                for key, item in h5[path].items():
                    if isinstance(item, h5py._hl.dataset.Dataset):
                        ret[key] = item[()]
                    elif isinstance(item, h5py._hl.group.Group):
                        ret[key] = recur(h5, path + key + "/")
                return ret

            with h5py.File(filename, "r") as h5:
                return recur(h5, "/")

        self._h5filename = file
        self._h5file = h5py.File(file, "r")
        self._runname = list(self._h5file.keys())[1]
        self._detectors2d = {k: sacla2d(k,self) for k in self._h5file[f"{self._runname}"].keys() if "detector" in k}
        self._tags = np.array(list(self._h5file[f"{self._runname}/{list(self._detectors2d.keys())[0]}"].keys())[1:])
        keydict = getkeys(self._h5file[f"{self._runname}/event_info"])
        # shortkeys are shortcuts to keys, longkeys are only flattend/slashes replaced by underscores
        self._shortkeys = {k.replace("/", "_"): keydict[k] for k in keydict.keys()}
        self._longkeys = {k.replace("/", "_"): k for k in keydict.values()}
        self._calibration={}

        if settings is not None:
            if 'alias' in settings: # alias replacement for shortkeys
                alias=settings['alias']
                for k, v in alias.items():
                    parsed=self._parsekey(v)
                    if parsed is None: ValueError(f'alias value {v} not found')
                    alias[k] = parsed
                self._shortkeys = {k: v for k, v in self._shortkeys.items() if v not in alias.values()} #non aliased keys
                self._shortkeys.update(alias) #aliased keys
            if "hide" in settings: #hide shortkeys by regex
                for regex in [re.compile(pattern) for pattern in settings["hide"]]:
                    for entry in list(filter(regex.match, self._shortkeys)):  # list(..) necessary because dict changes in loop
                        self._shortkeys.pop(entry)
            if "calibration" in settings: #setup dict to multiply values by calibration constant
                calibration = settings['calibration']
                for k,v in calibration.items():
                    parsed=self._parsekey(k)
                    if parsed is None: raise KeyError(f'calibration key {k} not found')
                    self._calibration[parsed] = v

        filename, fileext = os.path.splitext(file)
        self._workfilename = filename + "_work" + fileext  # might change
        # _dict contains additions to the data that will be saved to a workfile when saclarun.save() is called and is loaded on creation of saclarun
        if os.path.exists(self._workfilename):
            self._dict = __load(self._workfilename)
        else:
            self._dict = {}
        if not "singlevalues" in self._dict:  # singlevalues: scalar values
            self._dict["singlevalues"] = {}
        if not "datavalues" in self._dict:  # datavalues: values that change between tags
            self._dict["datavalues"] = {}

    @property
    def photonenergy(self):
        return np.array(self._h5file[f"{self_runname}/run_info/sacla_config/photon_energy_in_eV"]).take(0)

    @property
    def photonwavelength(self):
        return 1.239842e-6 / self.photonenergy()

    @property
    def electronenergy(self):
        return np.array(self._h5file[f"{self_runname}/run_info/sacla_config/electron_energy_in_eV"]).take(0)

    @property
    def run(self):
        """
        returns the run number
        """
        return self._runname.replace("run_", "")

    @property
    def comment(self):
        return self._h5file[f"{self._runname}/exp_info/comment"][0].decode("utf-8")

    @property
    def file(self):
        """
        returns the underlying hdf5 file
        """
        return self._h5file

    @property
    def detectorNames(self):
        """
        returns all detectors that are in the h5 file by their short name
        """
        return list(self._shortkeys.keys()) + list(self._detectors2d.keys())

    @property
    def workNames(self):
        """
        returns names of all custom data added to a run, that is not in the original h5 file
        """
        return list(self._dict["singlevalues"].keys()) + list(self._dict["datavalues"].keys()) + []

    def __len__(self):
        return len(self._tags)

    def __getattr__(self, attr):
        if attr in self._detectors2d:  # is a 2d detector
            return self._detectors2d[attr]
#             return sacla2d(attr, self)
        parsed=self._parsekey(attr)
        if parsed is not None:  # is scalar key #TODO: think about returning array or dataset
            val = np.array(self._h5file[f"{self._runname}/event_info/{parsed}"])
            if parsed in self._calibration:
                val = val * self._calibration[parsed]
            return val
        if attr in self._dict["singlevalues"]:  # is scalar work data
            return self._dict["singlevalues"][attr]
        if attr in self._dict["datavalues"]:  # is vector work data
            return self._dict["datavalues"][attr]
        raise KeyError(attr)  # not found

    def __setattr__(self, attr, val):
        if attr[0] == "_":  # for internal variables
            object.__setattr__(self, attr, val)
        else:
            if hasattr(val, "__len__") and len(val) == len(self):
                self._dict["datavalues"][attr] = val
            else:
                self._dict["singlevalues"][attr] = val

    def save(self):
        """
        saves _dict to the workfile
        """

        def recur(h5, path, dic):
            for key, item in dic.items():
                if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes, int)):
                    h5[path + str(key)] = item
                elif isinstance(item, dict):
                    recur(h5, path + str(key) + "/", item)

        with h5py.File(self._workfilename, "w") as h5:
            recur(h5, "/", self._dict)

    def __dir__(self):  # usefull for autocomplete
        return self.detectorNames + self.workNames + ["file", "comment", "detectorNames", "workNames", "run", "save", "photonwavelength", "photonenergy", "electronenergy"]

    def __getitem__(self, items):
        if isinstance(items, int):  # fast path for most used case
            if items >= len(self._tags):
                raise IndexError
            return saclarunview(items, self)
        else:  # don't want to care about slice etc handling... let's use numpy for that.
            return saclarunview(np.atleast_1d(np.arange(len(self))[items]).tolist(), self)

    def asdict(self):
        """
        returns a dict of the run. copies all point detectors into memory. usefull for numexpr
        """
        return {key: getattr(self, key) for key in self.detectorNames + self.workNames}


class saclarunview:
    """
    a helper object describing a view (in the numpy sense, meaning a selection of the elements) of a saclarun
    """

    def __init__(self, items, run):
        self._run = run
        self._items = np.atleast_1d(items)

    def __len__(self):
        return len(self._items)

    def __getitem__(self, items):
        return saclarunview(self._items[items], self._run)

    def __getattr__(self, attr):
        det = getattr(self._run, attr)
        if isinstance(det, h5py._hl.dataset.Dataset) and self._items != sorted(self._items):
            # cannot index into hdf5 dataset if indices are not sorted, resort to converting to np array
            # TODO: remove if always return array
            return np.array(det)[self._items]
        elif isinstance(det, dict) and "__partial__" in det and det["__partial__"] == True:
            ret = [det[str(i)] if str(i) in det else None for i in self._items]
            return ret[0] if len(ret) == 1 else ret
        elif hasattr(det, "__len__") and len(det) == len(self._run):
            return det[self._items]
        else:
            return det

    def __dir__(self):
        return self._run.__dir__()

    def __setattr__(self, attr, val):
        if attr[0] == "_":
            object.__setattr__(self, attr, val)
        else:
            if not hasattr(val, "__len__") or len(val) != len(self):
                raise ValueError
            dict = {str(item): val[i] for i, item in enumerate(self._items)}
            dict["__partial__"] = True
            setattr(self._run, attr, dict)

    def __iter__(self):
        for item in self._items:
            yield saclarunview(item, self._run)

    @property
    def run(self):
        """
        returns the saclarun object this view belongs to
        """
        return self._run


class sacla2d:
    """
    a helper object describing a sacla 2d detector. allows indexing without knowing the tag.
    """

    def __init__(self, detector, run):
        self._run = run
        self._detector = detector
        self._properties = self._run._h5file[f"{self._run._runname}/{self._detector}/detector_info"]

    def __getitem__(self, pos):
        if isinstance(pos, tuple):
            if not len(pos) == 3:
                raise NotImplementedError
            tags = np.atleast_1d(self._run._tags[pos[0]]).tolist()
            return h5list([(self._run._h5file[f"{self._run._runname}/{self._detector}/{tag}/detector_data"][pos[1], pos[2]]) for tag in tags])
        else:
            tags = np.atleast_1d(self._run._tags[pos]).tolist()
            return h5list([(self._run._h5file[f"{self._run._runname}/{self._detector}/{tag}/detector_data"]) for tag in tags])
        # h5list improves performance of np.array

    def __len__(self):
        return len(self._run._tags)

    def __array__(self):  # speeds up np.array(detector) converting one dataset at a time
        return np.squeeze(np.array([np.array(self._run._h5file[f"{self._run._runname}/{self._detector}/{tag}/detector_data"]) for tag in self._run._tags]))

    @property
    def shape(self):
        return (len(self), *self[0].shape)

    def __dir__(self):
        return ["shape"] + list(self._properties.keys())

    def __getattr__(self, attr):
        if attr in self._properties:
            ret = self._properties[attr]
            if ret.size == 1:
                ret = ret[0]  # unpack scalar values
            return ret
        else:
            raise AttributeError


class h5list(list):
    """
    improves the performance of np.array(h5list<h5 datasets>) compared to np.array(list<h5 datasets>) by converting each element
    """

    def __array__(self):
        return np.array([np.array(obj) for obj in self])


def getkeys(obj, identifier="", key=""):
    """
    Takes an keys() implementing nested object, such as an h5 Group and recursivly walks through it.
    Returns an dictionary with the shortest unique identifier as keys and full path as values.
    """
    ret = {}
    if hasattr(obj, "keys"):
        for k in obj.keys():
            if len(obj.keys()) > 1:  # multiple children, set the identifier to full path and recurse
                ret.update(getkeys(obj[k], key + "/" + k, key + "/" + k))
            elif len(obj.keys()) > 0:  # unique at this level, no need to extend the identifier
                ret.update(getkeys(obj[k], identifier, key + "/" + k))
            else:  # no members, dead end
                return []
        return ret
    else:  # reached end
        return {identifier[1:]: key[1:]}
    
def qsub(command, args=[], name='script', start=0, end=0):
    '''
    submits a job (array) using max resources.
    if start==end
    use $PBS_ARRAY_INDEX as argument to get index of job in array
    '''
    from subprocess import run, PIPE

    args = ' '.join(str(i) for i in args)
    pbs = (
        f'''
    #PBS -V
    #PBS -l nodes=1:ppn=14
    #PBS -l walltime=24:00:00
    #PBS -l mem=60GB
    #PBS -N {jobname}
    '''
        + (f'#PBS -J {start}-{end}' if start != end else f'export PBS_ARRAY_INDEX={start}')
        + f'''
    {command} {args}
    '''
    )
    p = run(['qsub'], stdout=PIPE, input=pbs, encoding='ascii', cwd='./logs/')
    return (p.returncode, p.stdout)
