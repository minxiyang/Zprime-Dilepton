from numpy import exp, sin, arctan
import numpy as np


def calcMuonRecoUncert(mass, pT1, pT2, eta1, eta2, isDimuon, year):

    if not isDimuon:
        return 1.0

    p1 = pT1 / (sin(arctan(2*exp(-1*eta1))))
    p2 = pT2 / (sin(arctan(2*exp(-1*eta2))))

    bb = True
    if (abs(eta1) > 1.2 or abs(eta2) > 1.2): 
        bb = False

    if year == 2016:
        SF1 = 1.0
        SF2 = 1.0
        if (abs(eta1) <= 1.6 and p1 > 100):
            SF1 = (0.994 - 4.08e-6 * p1)/(0.994 - 4.08e-6 * 100)
        elif (abs(eta1) > 1.6 and p1 > 200):
            SF1 = ((0.9784 - 4.73e-5 * p1)/(0.9908 - 1.26e-5 * p1)) / ((0.9784 - 4.73e-5 * 200)/(0.9908 - 1.26e-5 * 200))
        if (abs(eta2) <= 1.6 and p2 > 100):
            SF2 = (0.994 - 4.08e-6 * p2)/(0.994 - 4.08e-6 * 100)
        elif (abs(eta2) > 1.6 and p2 > 200):
            SF2 = ((0.9784 - 4.73e-5 * p2)/(0.9908 - 1.26e-5 * p2)) / ((0.9784 - 4.73e-5 * 200)/(0.9908 - 1.26e-5 * 200))

        return SF1*SF2

    elif year == 2017:

        if bb:
            return 0.99

        else:
            eff_default = 0
            eff_syst = 0

            if (mass <= 450):
                a = 13.39
                b = 6.696
                c = -4.855e+06
                d = -7.431e+06
                e = -108.8
                f = -1.138
                eff_default = a - b * exp( -( (mass - c) / d) ) + e * mass**f
			
            else:
                eff_a = 0.3148
                eff_b = 0.04447
                eff_c = 1.42
                eff_d = -5108.
                eff_e = 713.5
                eff_default = eff_a + eff_b * mass**eff_c * exp(- ((mass - eff_d ) / eff_e))

            if (mass <= 450):
                a = 1.33901e+01
                b = 6.69687e+00
                c = -4.85589e+06
                d = -7.43036e+06
                e = -1.14263e+02
                f = -1.15028e+00
                eff_syst= a - b * exp( -( (mass - c) / d) ) + e * mass**f
			
            else:
                eff_a = 3.07958e-01
                eff_b = 4.63280e-02
                eff_c = 1.35632e+00
                eff_d = -5.00475e+03
                eff_e = 7.38088e+02
                eff_syst =  eff_a + eff_b * mass**eff_c * exp(- ((mass - eff_d ) / eff_e))

            return eff_syst/eff_default


    elif year == 2018:

        if bb:
            return 0.99

        else:
            eff_default = 0
            eff_syst = 0

            if (mass <= 450):
                a = 13.39
                b = 6.696
                c = -4.855e+06
                d = -7.431e+06
                e = -108.8
                f = -1.138
                eff_default = a - b * exp( -( (mass - c) / d) ) + e * mass**f
            else:
                eff_a = 0.3148
                eff_b = 0.04447
                eff_c = 1.42
                eff_d = -5108.
                eff_e = 713.5
                eff_default = eff_a + eff_b * mass**eff_c * exp(- ((mass - eff_d ) / eff_e))

            if (mass <= 450):
                a = 1.33901e+01
                b = 6.69687e+00
                c = -4.85589e+06
                d = -7.43036e+06
                e = -1.14263e+02
                f = -1.15028e+00
                eff_syst = a - b * exp( -( (mass - c) / d) ) + e * mass**f
            else:
                eff_a = 3.07958e-01
                eff_b = 4.63280e-02
                eff_c = 1.35632e+00
                eff_d = -5.00475e+03
                eff_e = 7.38088e+02
                eff_syst =  eff_a + eff_b * mass**eff_c * exp(- ((mass - eff_d ) / eff_e))

            return eff_syst/eff_default;


def muonRecoUncert(masses, pT1, pT2, eta1, eta2, isDimuon, year):

    weights = []
    for i in range(0, masses.size):
        weights.append(calcMuonRecoUncert(masses[i], pT1[i], pT2[i], eta1[i], eta2[i], isDimuon[i], int(year))) 

    return np.array(weights)
