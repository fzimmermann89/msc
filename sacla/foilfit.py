def foilfit(image, fwhmx, fwhmy, qstep, weights=None, f=1):
    """
    fit a sinc-gauss model to image
    gauss in first dimension with width fwhmx, sinc in second dimension with fwhmy.
    fwhmx,fwhmy: width in lengthunit
    qstep: q-size of on pixel in 1/lengthunit
    weights: if None, use ~np.isnan(image)
    f: oversamplingfactor, f=1: no oversampling
    """
    import numpy as np
    
    def combination_model(xtype='g', ytype='s', os=1):
        from lmfit import Model
        import numexpr as ne

        def _gaussian(x, wid):
            return ne.evaluate('exp(-x**2 / (2*(sqrt(log(16))/wid)**2))')

        def _sinc2(x, width):
            return ne.evaluate('where(abs(x*width)<1e-10,1,2*(1-cos(x*width))/(width*x)**2)')

        xf = _gaussian if xtype == 'g' else _sinc2
        yf = _gaussian if ytype == 'g' else _sinc2

        def _comb(x, y, amp, fwhmx, fwhmy, off):
            xr = np.nanmean(xf(x, fwhmx).reshape(-1, os), 1)
            yr = np.nanmean(yf(y, fwhmy).reshape(-1, os), 1)
            r = amp * xr[:, None] * yr[None, :] + off
            return r * np.sign(fwhmx) * np.sign(fwhmy)

        return Model(_comb, independent_vars=['x', 'y'])
    
    if f<1 or not isinstance(f,int):
        raise ValueError('f should be int and at least 1')
        
    os = 1 + (f - 1) * 2
    model = combination_model(os=os)

    def r(N):
        step = 1 / os
        nl = (N) // 2 - 1
        nh = (N + 1) // 2 - 1
        return np.arange(-nl - os * step, nh + os * step, step) - (f - 1) * step

    x, y = (qstep * r(s) for s in image.shape)

    if weights is None:
        weights = (~np.isnan(image)).astype(float)

    fit = model.fit(np.nan_to_num(image), x=x, y=y, amp=np.nanmax(image), fwhmx=fwhmx, fwhmy=fwhmy, off=np.nanpercentile(image, 30), weights=weights)

    return fit, model.func(x, y, **fit.best_values)