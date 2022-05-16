import numpy as np


def calckFac(mass, region, flavor):

    if flavor == "mu":
        if region == "all":
            pars = [
                0.919027,
                5.98337e-05,
                2.56077e-08,
                -2.82876e-11,
                9.2782e-15,
                -7.77529e-19,
            ]
        elif region == "bb":
            pars = [
                0.911563,
                0.000113313,
                -2.35833e-08,
                -1.44584e-11,
                8.41748e-15,
                -8.16574e-19,
            ]
        elif region == "be":
            pars = [
                0.934502,
                2.21259e-05,
                4.14656e-08,
                -2.26011e-11,
                5.58804e-15,
                -3.92687e-19,
            ]
        # else:

        correction = pars[0]
        for i in range(1, 6):
            correction += pars[i] * mass ** i
    else:
        pars = [1.0678, -0.000120666, 3.22646e-08, -3.94886e-12]

        correction = pars[0]
        for i in range(1, 3):
            correction += pars[i] * mass ** i

    return correction


def kFac(masses, region, flavor):

    result = [calckFac(mass, region, flavor) for mass in masses]
    return np.array(result)
