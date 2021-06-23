def spectrumfit(x, y, Esignal, Escatter, psignal=None, pscatter=None, sigma=500, p=0.1):
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
    from scipy.stats import nbinom
    from math import sqrt, pi, ceil, factorial, exp
    import numpy as np
    from lmfit import Model

    EPS = 1e-20

    def gaussian(x, mu, s):
        return 1 / (max(s, EPS) * sqrt(2 * pi)) * np.exp(-((x - mu) ** 2) / (2 * max(s, EPS) ** 2))

    def comb(x, Esignal, Escatter, psignal, pscatter, sigma, M, scale, p):

        window = np.zeros(1 + 2 * int((Esignal * len(x)) // np.ptp(x)))
        window[len(window) // 2] = 1 - p
        window[: len(window) // 2] = p / (len(window) // 2)
        window /= window.sum()

        res = np.zeros_like(x, float)
        for i, j in itertools.product(range(0, ceil(np.max(x) / Esignal) + 1), range(0, ceil(np.max(x) / Escatter) + 1)):
            sigamp = max(0, nbinom.pmf(i, M, M / max(EPS, psignal + M)))
            scatamp = max(0, pscatter ** j * exp(-pscatter) / factorial(j))
            res += np.nan_to_num(scatamp * sigamp * gaussian(x, Escatter * j + i * Esignal, sigma))
        return np.nan_to_num(np.convolve(np.maximum(EPS, scale * res), window, 'same'))

    def logcomb(x, Esignal, Escatter, psignal, pscatter, sigma, M, scale, p):
        return np.nan_to_num(np.log10(comb(x, Esignal, Escatter, psignal, pscatter, sigma, M, scale, p)))

    model1 = Model(logcomb)  # first fit
    model2 = Model(logcomb)  # second fit

    if psignal is None:
        psignal = np.sum(x[x > Esignal * 0.5] * y[x > Esignal * 0.5]) / np.sum(y) / Esignal
    if pscatter is None:
        pscatter = psignal / 10

    params1 = model1.make_params()
    params1['Esignal'].value = 1.0 * Esignal
    params1['Esignal'].min = 0.95 * Esignal
    params1['Esignal'].max = 1.05 * Esignal
    params1['Escatter'].value = Escatter
    params1['Escatter'].min = 0.95 * Escatter
    params1['Escatter'].max = 1.05 * Escatter
    params1['psignal'].value = psignal
    params1['psignal'].min = 0
    params1['pscatter'].value = pscatter
    params1['pscatter'].min = 0
    params1['sigma'].value = sigma
    params1['scale'].value = np.trapz(y, x=x)
    params1['scale'].vary = True
    params1['M'].value = 10
    params1['M'].vary = False
    params1['p'].value = p
    params1['p'].vary = True

    # weights
    peaks = np.array(sorted([Esignal * n + m * Escatter for n, m in itertools.product(range(ceil(1 + max(x))), range(ceil(1 + max(x) / Escatter)))]))
    dist = np.abs(x[None, ...] - peaks[..., None]).min(0)
    weights = np.maximum(0, 1 - dist / Esignal)

    w1 = np.nan_to_num(np.log10(y / y[(y > np.sqrt(EPS))].min()) * weights)
    w1 /= np.max(w1)
    w1[x < -sigma / 2] = 0
    w1[w1 < EPS] = 0

    # for second fit, only look at first 3 signal peaks
    dist2 = np.abs(x[None, ...] - params1['Esignal'].value * np.arange(1, 4)[..., None]).min(0)
    w2 = np.copy(w1)
    w2[dist2 > 0.2 * Esignal] = 0
    w2[np.logical_or(x > Esignal * 3.5, x < Esignal * 0.5)] = 0

    f1 = model1.fit(np.log10(np.maximum(EPS, y)), params1, x=x, weights=w1)

    params2 = f1.params.copy()
    for p in params2:
        params2[p].vary = False
    params2['M'].vary = True

    f2 = model2.fit(np.log10(np.maximum(EPS, y)), params2, x=x, weights=w2)

    return f1, f2
