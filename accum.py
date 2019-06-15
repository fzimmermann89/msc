import numpy as _np
import numexpr as _ne
# from memory_profiler import profile
class accumulator:
    #     _n: int = 0
    #     _mean: float = 0
    #     _nvar: float = 0
    def __init__(self):
        self._n = 0
        self._mean = None
        self._nvar = None

    def __repr__(self):
        print(type(self._mean))
        return 'accumulator[%i]' %  self._n
#     @profile
    def add(self, value, count = 1):
        if self._mean is None:
            self._mean=_np.asarray(value)
            self._nvar=_np.zeros_like(value)
            self._n = count
        else:
            delta = value - self._mean
            self._n += count
            with _np.errstate(divide='ignore', invalid='ignore'):
                self._mean = _np.add(self._mean, delta / self._n, where=count, out=self._mean)
                self._nvar = _ne.evaluate('nvar + delta * (value - mean)', local_dict={'nvar':self._nvar, 'value': value, 'delta':delta,'mean':self._mean})

    def __len__(self):
        return n

    @property
    def mean(self):
        return self._mean

    @property
    def var(self):
        return self._nvar / self._n 

    @property
    def std(self):
        return np.sqrt(self.var)
