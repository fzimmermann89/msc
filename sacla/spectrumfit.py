from scipy.stats import nbinom
from math import sqrt, ceil, factorial, exp, floor, pi
import numpy as np
from lmfit import Model
import itertools
import numexpr as ne

EPS = 1e-25


def _gaussian(x, mu, s):
    s = max(s, EPS)
    Pi = pi
    return ne.evaluate('1 / ( s* sqrt(2 * Pi)) * exp(-((x - mu) ** 2) / (2 * s ** 2))')


def _comb(x, Esignal, Escatter, Escatter2, Ebeta, psignal, pscatter, pscatter2, pbeta, sigma, M, scale, p,scattersigmamod):
    # print(f'{Esignal=}, {Escatter=}, {Ebeta=}, {psignal=}, {pscatter=}, {pbeta=}, {sigma=}, {M=}, {scale=}, {p=}')
    window = np.zeros(1 + 2 * int((Esignal * len(x)) // np.ptp(x)))
    window[len(window) // 2] = 1 - p
    window[: len(window) // 2] = p / (len(window) // 2)
    window /= window.sum()

    res = np.zeros_like(x, float)
    sigdist = nbinom(M, M / max(EPS, psignal + M))
    betadist = nbinom(M, M / max(EPS, (pbeta * psignal) + M))
    for i in range(0, 1 + ceil(np.max(x) / Esignal)):
        sigamp = max(0, sigdist.pmf(i))

        for k in range(0, ceil(np.max(x) / Ebeta)):
            betaamp = max(0, betadist.pmf(k))

            for j in range(0, 1 + floor(np.max(x) / Escatter)):
                scatamp = max(0, pscatter ** j * exp(-pscatter) / factorial(j)) if pscatter > 0 else 1

                for l in range(0, 1 + floor(np.max(x) / Escatter2)):
                    scatamp2 = max(0, pscatter2 ** l * exp(-pscatter2) / factorial(l)) if pscatter2 > 0 else 1

                    res += np.nan_to_num(
                        scatamp
                        * scatamp2
                        * sigamp
                        * betaamp
                        * _gaussian(x, Escatter * j + i * Esignal + k * Ebeta + l * Escatter2, scattersigmamod * sigma if j > 0 or l > 0 else sigma)
                    )
    return np.nan_to_num(np.convolve(np.maximum(EPS, scale * res), window, 'same'))


def _logcomb(x, Esignal, Escatter, Escatter2, Ebeta, psignal, pscatter, pscatter2, pbeta, sigma, M, scale, p,scattersigmamod):
    return np.nan_to_num(np.log10(_comb(x, Esignal, Escatter, Escatter2, Ebeta, psignal, pscatter, pscatter2, pbeta, sigma, M, scale, p,scattersigmamod)))


def spectrumfit(
    x, y, Esignal, Escatter=None, Escatter2=None, psignal=None, pscatter=None, pscatter2=None, pbetaratio=0.15, sigma=500, p=0.1, scattersigmamod=1.5
):
    """
    fit a spectrum
    first fit for all params but M
    second fit for M only at signal peak positions

    x: bins, scaled by signal energy
    y: spectrum
    psignal: initial guess for signal mean, if None estimate from y
    pscatter: initial guess for scatterin mean, if None defaults to psignal/10
    sigma: initial guess for peak sigma
    Escatter: initial guess for relative scatter energy
    """

    model1 = Model(_logcomb)  # first fit
    model2 = Model(_logcomb)  # second fit

    if psignal is None:
        psignal = np.sum(x[x > Esignal * 0.5] * y[x > Esignal * 0.5]) / np.sum(y) / Esignal
    if pscatter is None:
        pscatter = psignal / 10
    if pscatter2 is None:
        pscatter2 = psignal / 30

    params1 = model1.make_params()
    params1['Esignal'].value = 1.0 * Esignal
    params1['Esignal'].min = 0.96 * Esignal
    params1['Esignal'].max = 1.04 * Esignal
    params1['psignal'].value = psignal
    params1['psignal'].min = 0

    params1['Ebeta'].value = 1.1 * Esignal
    params1['Ebeta'].min = 1.05 * Esignal
    params1['Ebeta'].max = 1.15 * Esignal
    params1['Ebeta'].vary = True
    params1['pbeta'].value = pbetaratio
    params1['pbeta'].min = 0
    params1['pbeta'].max = 0.3
    
    params1['scattersigmamod'].value = scattersigmamod
    params1['scattersigmamod'].vary=False

    if Escatter is None:
        params1['Escatter'].vary = False
        params1['Escatter'].value = 100 * Esignal
        params1['pscatter'].value = 0
        params1['pscatter'].vary = False
    else:
        params1['Escatter'].value = Escatter
        params1['Escatter'].min = 0.95 * Escatter
        params1['Escatter'].max = 1.05 * Escatter
        params1['pscatter'].value = pscatter
        params1['pscatter'].min = 0

    if Escatter2 is None:
        params1['Escatter2'].vary = False
        params1['Escatter2'].value = 100 * Esignal
        params1['pscatter2'].value = 0
        params1['pscatter2'].vary = False
    else:
        params1['Escatter2'].value = Escatter2
        params1['Escatter2'].min = 0.95 * Escatter2
        params1['Escatter2'].max = 1.05 * Escatter2
        params1['pscatter2'].value = pscatter2
        params1['pscatter2'].min = 0

    params1['sigma'].value = sigma
    params1['scale'].value = np.trapz(y, x=x)
    params1['scale'].vary = True
    params1['M'].value = 50
    params1['M'].vary = False
    params1['p'].value = p
    params1['p'].min = 0
    params1['p'].vary = True

    # weights
    Es = [0 if E is None else E for E in (Esignal, Escatter, Escatter2)]
    rs = [[0] if E == 0 else range(ceil(1 + max(x) / E)) for E in Es]
    peaks = np.array(sorted([sum([i * E for i, E in zip(ids, Es)]) for ids in itertools.product(*rs)]))
    dist = np.abs(x[None, ...] - peaks[1:, None]).min(0)
    weights = np.maximum(0, 1 - dist / Esignal)
    w1 = np.nan_to_num(np.log10(np.maximum(y, EPS) / y[(y > np.sqrt(EPS))].min()) * weights)
    w1 /= np.max(w1)
    w1[x < -sigma / 2] = 0
    w1[w1 < EPS] = 0

    # for second fit, only look at first 3 signal peaks
    dist2 = np.abs(x[None, ...] - params1['Esignal'].value * np.arange(1, 4)[..., None]).min(0)
    if Escatter is not None:  # and first scatter peak
        dist2 = np.minimum(dist2, np.abs(x - params1['Escatter']))
    if Escatter2 is not None:  # and first scatter peak
        dist2 = np.minimum(dist2, np.abs(x - params1['Escatter2']))
    w2 = np.copy(w1)
    w2[dist2 > params1['sigma'].value] = 0
    w2[np.logical_or(x > Esignal * 3.5, x < Esignal * 0.5)] = 0

    f1 = model1.fit(
        np.log10(np.maximum(EPS, y)), params1, x=x, weights=w1, max_nfev=200
    )  # , max_nfev=400, fit_kws=dict(gtol=1e-12, xtol=1e-10, ftol=1e-12))

    params2 = f1.params.copy()
    for p in params2:
        params2[p].vary = False
    params2['M'].vary = True
    params1['M'].value = 20
    params2['psignal'].vary = True
    params2['sigma'].vary = True
    params2['scale'].vary = True
    if Escatter is not None:
        params2['pscatter'].vary = True
    if Escatter2 is not None:
        params2['pscatter2'].vary = True

    f2 = model2.fit(np.log10(np.maximum(EPS, y)), params2, x=x, weights=w2, max_nfev=400, fit_kws=dict(gtol=1e-16, xtol=1e-16, ftol=1e-16))

    return f1, f2
