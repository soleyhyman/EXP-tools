"""
Module to do calculations for bars, etc.
"""
import astropy.units as u
import numpy as np
import pyEXP
import warnings

class barCalcs:
    """
    Extract bar properties from EXP coefficients

    Examples
    --------
    To initalize the class, run:

    >>> barSim = barCalcs(coefsFilename='disc_coefs.h5', tUnit=u.Gyr, dt=0.01954)

    To calculate the bar angle over time, run

    >>> angles = barSim.getBarAngle(n=0, m=2)

    to get the angles in radians. To get them in degrees, set ``degrees=True``.
    the ``m`` parameter should be ``m=2`` for bar calculations, but flexibility 
    for other m modes is incorporated for other use cases. Higher order n modes
    can also be used.

    To calculate the bar pattern speed over time, run:

    >>> patternSpeed = barSim.getOmegaBar(n=0, m=2, pUnit=u.km/u.s/u.kpc)

    The above code will calculate the bar pattern speed for m=2, n=0, though 
    higher order n modes can be used as well. If tUnit is not specified, the 
    resulting array will be given as floats. If tUnit is specified but pUnit
    is set to None, the astropy units will be inverse of the specified tUnit.
    The default pUnit is defined to be km/s/kpc.

    To calculate bar strength at the order-of-magnitude level, we can use the
    approximation:

    :math:`S_{bar} = \\frac{|C_{2n}|}{|C_{00}|} \;.`

    The corresponding code is

    >>> barSim.getBarStrength(n=0)

    Note that the use of the n=0 order is not always a good choice, but this
    can be used as a quick look. For a more precise calculation, the sum of
    the first few orders is better, i.e.,

    :math: `S_{bar} = \sum_{n=0}^{4} \\frac{|C_{2n}|}{|C_{00}|} \;.`

    The corresponding code for this calculation is

    >>> barSim.getBarStrengthSummed(n=0, nmax=4)

    Higher order n modes can be used for more accuracy.
    """
    def __init__(self, coefsFilename, tUnit=None, dt=None):
        """
        Initialize barCalcs class

        Parameters
        ----------
        coefsFilename : string
            Filename of coefficients
        tUnit : astropy.units.core.PrefixUnit, optional
            Astropy time unit, by default u.Gyr
        dt : float or astropy.Quantity time quantity, optional
            Timestep resolution, by default None

        Attributes
        ----------
        coefs : pyEXP.coefs object
            Coefficient and time data from loaded coefsFilename
        tUnit : astropy.units.core.PrefixUnit or None
            Astropy time unit for the timestep, if specified
        t : ndarray or astropy.Quantity
            Timesteps corresponding to the calculated coefficients
        dt : float or astropy.Quantity
            Timestep resolution (given as astropy time quantity if 
            specified or float if not)
        mnCoefs : ndarray
            Array of coeffiencient amplitudes formatted as 
            (number of m modes, number of n modes, number of timesteps)
        num_m : int
            Number of m modes
        num_n : int
            Number of n modes
        """
        self.coefs = pyEXP.coefs.Coefs.factory(coefsFilename)
        self.tUnit = tUnit

        if self.tUnit is None:
            warnings.warn("No time unit specified, so units may not be accounted for accurately. Use at your own risk!")
            self.t = self.coefs.Times()        # get timesteps
        else:
            self.t = self.coefs.Times()*tUnit  # get timesteps in appropriate timeunit

        # Load in coefficients
        self.mnCoefs = self.coefs.getAllCoefs()    # load coefficients with shape (m,n,n_timesteps)
        self.num_m = self.mnCoefs.shape[0]         # get number of m modes used
        self.num_n = self.mnCoefs.shape[1]         # get number of n modes used

        if dt is None:
            warnings.warn("No dt specified, calculating from time array")
            self.dt = np.mean(np.diff(self.t))
        elif self.tUnit is None:
            self.dt = dt
        else:
            if type(dt) == u.quantity.Quantity:
                self.dt = dt.to(tUnit)
            else:
                self.dt = dt*tUnit

    def getBarAngle(self, n=0, m=2, degrees=False):
        """
        Calculate bar angle over time

        Parameters
        ----------
        n : int, optional
            n mode to use, by default 0
        m : int, optional
            m mode to use, by default 2
        degrees : bool, optional
            If set to true, bar angle is returned in degrees
            instead of radians, by default False

        Returns
        -------
        ndarray or astropy.units.quantity.Quantity
            Bar angle over time, returned as ndarray of angle 
            in radians (if degrees=False) or as 
            astropy.units.quantity.Quantity in degrees (if degrees=True)
        """
        sinComponent = np.imag(self.mnCoefs[m,n])
        cosComponent = np.real(self.mnCoefs[m,n])
        barAngle = (1./m)*np.arctan2(sinComponent,cosComponent)*u.rad
        if degrees==True:
            return barAngle.to(u.deg)
        else:
            return barAngle.value

    def getOmegaBar(self, n=0, m=2, pUnit=u.km/u.s/u.kpc):
        """
        Calculate bar pattern speed over time

        Parameters
        ----------
        n : int, optional
            n mode to use, by default 0, by default 0
        m : int, optional
            m mode to use, by default 0, by default 2
        pUnit : astropy.units.core.PrefixUnit or None, optional
            Astropy unit of pattern speed (frequency) or None (if
            not keeping track of units), by default u.km/u.s/u.kpc

        Returns
        -------
        ndarray
            Bar pattern speed over time, returned as ndarray of pattern
            speed (if tUnit=None) or as astropy.units.quantity.Quantity 
            in inverse of tUnit (if pUnit=False) or in pUnit
        """
        barAngle = np.unwrap(self.getBarAngle(n=n,m=m))
        dtheta_dt = np.diff(barAngle)/self.dt
        if self.tUnit is None:
            return dtheta_dt/2.
        elif pUnit is None:
            return dtheta_dt/2.
        else:
            return (dtheta_dt/2.).to(pUnit)

    def getBarStrength(self, n=0):
        """
        Calculate bar strength as ratio of C_2n/C_00

        Parameters
        ----------
        n : int, optional
            n mode of m=2 coefficient, by default 0

        Returns
        -------
        ndarray
            Array of C_2n/C_00 ratios over time
        """
        return np.abs(self.mnCoefs[2,n]/self.mnCoefs[0,0])

    def getBarStrengthSummed(self, nmin=0, nmax=4):
        """
        Calculate bar strength as sum ratio of C_2n/C_00
        from specified minimum n and maximum n

        Parameters
        ----------
        nmin : int, optional
            Minimum n mode of m=2 coefficient, by default 0
        nmax : int, optional
            Maximum n mode of m=2 coefficient, by default 4

        Returns
        -------
        ndarray
            Array of sums of C_2n/C_00 ratios over time
        """
        return np.sum([self.getBarStrength(n) for n in range(nmin, nmax+1)], axis=0)