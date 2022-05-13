def muonScaleUncert(mass, isBB, year):

    if year == 2016:
        if (isBB): return 0.01
        else: return 0.03
    elif year == 2017:
        if (isBB): return 1. - (1.00037 -1.70128e-06*mass + 3.09916e-09*mass**2 -1.32999e-12*mass**3 +  2.99434e-16*mass**4 + -2.28065e-20*mass**5)
        else: return 1. - (1.00263 - 1.04029e-05*mass + 8.9214e-09*mass**2 -3.4176e-12*mass**3 +  6.07934e-16*mass**4  -3.73738e-20*mass**5)
    elif year == 2018:
        if (isBB): return 1. - (0.999032 + 3.36979e-06*mass -3.4122e-09*mass**2 + 1.62541e-12*mass**3  - 3.12864e-16*mass**4 + 2.18417e-20*mass**5)
        else: return 1. - (1.00051 - 2.21167e-06*mass + 2.21347e-09*mass**2 -7.72178e-13*mass**3 +  1.28101e-16*mass**4  - 8.32675e-21*mass**5)
	
    return 1.0
