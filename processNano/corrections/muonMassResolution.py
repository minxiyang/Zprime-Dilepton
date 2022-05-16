import numpy as np
from numpy import random


def getRes(genMass, year, bb):

    #resolution smearing and uncertainties are defined as fraction of the resolution, so we need the resolution parametrization
    a = 0.
    b = 0.
    c = 0.
    d = 0.
    e = 0.

    if year == 2016:
        if bb:
            a = 0.00701
            b = 3.32e-05
            c = -1.29e-08
            d = 2.73e-12
            e = -2.05e-16
        else:
            a = 0.0124
            b = 3.75e-05
            c = -1.52e-08
            d = 3.44e-12
            e = -2.85e-16

    elif year == 2017:
        if bb:
            a = 0.00606
            b = 3.41e-05
            c = -1.33e-08
            d = 2.39e-12
            e = -1.5e-16
        else:
            a = 0.0108
            b = 3.25e-05
            c = -1.18e-08
            d = 2.11e-12
            e = -1.35e-1

    elif year == 2018:
        if bb:
            a = 0.00608
            b = 3.42e-05
            c = -1.34e-08
            d = 2.4e-12
            e = -1.5e-16
        else:
            a = 0.0135
            b = 2.83e-05
            c = -9.71e-09
            d = 1.71e-12
            e = -1.09e-16    

    res = a + b*genMass + c*genMass**2 + d*genMass**3 + e*genMass**4
    return max(0, res)


def additionalSmearing(genMass, year, bb=True):

    res = getRes(genMass,year,bb)
    extraSmear = 0
    if not bb:
        extraSmear = res*0.567


    return random.normal(1, max(0.0001, extraSmear))


def smearingForUnc(genMass, year, bb=True):

    res = getRes(genMass, year, bb)

    extraSmearSyst = res*0.567
    if (year == 2017 or year == 2018): extraSmearSyst = res*0.42098099719583537

    return random.normal(1, max(0.0001, extraSmearSyst))


def smearMass(genMasses, year, bb=True, forUnc=True):

    if forUnc:
        result = np.array([smearingForUnc(genMass, int(year), bb) for genMass in genMasses])
    else:
        result = np.array([additionalSmearing(genMass, int(year), bb) for genMass in genMasses])

    return result
