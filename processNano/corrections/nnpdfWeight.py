import numpy as np


def calcNNPDFWeightDY(mass, leadingPt, region, flavor, year):

    correctionAll = 1.
    correctionBB = 1.
    correctionBE = 1.
    if flavor == "mu":
        if year == 2017:
            if mass < 120:
                correctionAll = (leadingPt<30)*0.9 + (leadingPt>30 and leadingPt<100)*((1.8245728118) + (-0.0537728412909)*leadingPt + (0.000731365981935*leadingPt)**2 + (7.16669312495e-06*leadingPt)**3 + (-1.99723894101e-07*leadingPt)**4 + (1.0112316789e-09*leadingPt)**5) + (leadingPt>100)*(1.01849023)                                                                                   
                correctionBB = (leadingPt<30)*0.9 + (leadingPt>30 and leadingPt<100)*((1.91383074609) + (-0.0596201865777)*leadingPt + (0.000811074027001)*leadingPt**2 + (7.90677720686e-06)*leadingPt**3 + (-2.21489848717e-07)*leadingPt**4 + (1.12700571973e-09)*leadingPt**5) + (leadingPt>100)*(1.00484010198) 
                correctionBE = (leadingPt<30)*0.9 + (leadingPt>30 and leadingPt<100)*((1.71913319508) + (-0.0481243962238)*leadingPt + (0.000666286154366)*leadingPt**2 + (6.45776405133e-06)*leadingPt**3 + (-1.82202504311e-07)*leadingPt**4 + (9.24567381899e-10)*leadingPt**5) + (leadingPt>100)*(1.02790393101)
            else:
                correctionAll = ((0.918129) + (6.92702e-05)*mass + (1.62175e-08)*mass**2 + (-2.47833e-11)*mass**3 + (8.75707e-15)*mass**4 + (-7.53019e-19)*mass**5)
                correctionBB = ((0.914053) + (7.91618e-05)*mass + (2.19722e-08)*mass**2 + (-3.49212e-11)*mass**3 + (1.22504e-14)*mass**4 + (-1.07347e-18)*mass**5)
                correctionBE = ((0.933214) + (3.76813e-05)*mass + (1.95612e-08)*mass**2 + (-1.2688e-11)**mass**3 + (3.69867e-15)*mass**4 + (-2.62212e-19)*mass**5)

        elif year == 2018:
            if mass < 120:
                correctionAll = (leadingPt<30)*0.9 + (leadingPt>30 and leadingPt<100)*((1.69147781688) + (-0.0473286496053)*leadingPt + (0.000661599919558)*leadingPt**2 + (6.33324308996e-06)*leadingPt**3 + (-1.80459280586e-07)*leadingPt**4 + (9.19632449685e-10)*leadingPt**5) + (leadingPt>100)*(1.02344217328);
                correctionBB = (leadingPt<30)*0.9 + (leadingPt>30 and leadingPt<100)*((1.65477513925) + (-0.0472097707001)*leadingPt + (0.000681831627146)*leadingPt**2 + (6.15645344304e-06)*leadingPt**3 + (-1.82810037593e-07)*leadingPt**4 + (9.43667804224e-10)*leadingPt**5) + (leadingPt>100)*(1.01489199674)
                correctionBE = (leadingPt<30)*0.9 + (leadingPt>30 and leadingPt<100)*((1.60977951604) + (-0.0426122819079)*leadingPt + (0.000599273084801)*leadingPt**2 + (5.88395881526e-06)*leadingPt**3 + (-1.66414436738e-07)*leadingPt**4 + (8.4690800397e-10)*leadingPt**5) + (leadingPt>100)*(1.02846360871)
            else:
                correctionAll = ((0.919027) + (5.98337e-05)*mass + (2.56077e-08)*mass**2 + (-2.82876e-11)*mass**3 + (9.2782e-15)*mass**4 + (-7.77529e-19)*mass**5)
                correctionBB = ((0.911563) + (0.000113313)*mass + (-2.35833e-08)*mass**2 + (-1.44584e-11)*mass**3 + (8.41748e-15)*mass**4 + (-8.16574e-19)*mass**5)
                correctionBE = ((0.934502) + (2.21259e-05)*mass + (4.14656e-08)*mass**2 + (-2.26011e-11)*mass**3 + (5.58804e-15)*mass**4 + (-3.92687e-19)*mass**5)

    else:
        if (year == 2017 or year == 2018):
            if (mass < 120):
                correctionBB = (leadingPt<150) * (3.596-0.2076 *leadingPt+0.005795*pow(leadingPt,2)-7.421e-05*pow(leadingPt,3)+4.447e-07*pow(leadingPt,4)-1.008e-9 *pow(leadingPt,5)) + (leadingPt>=150) * 0.969125
                correctionBE = (leadingPt<150) * (2.066-0.09495*leadingPt+0.002664*pow(leadingPt,2)-3.242e-05*pow(leadingPt,3)+1.755e-07*pow(leadingPt,4)-3.424e-10*pow(leadingPt,5)) + (leadingPt>=150) * 1.191875
            else:
                correctionBB = (mass<5000) * (0.8934 + 0.0002193 * mass - 1.961e-7*mass**2 + 8.704e-11*mass**3 - 1.551e-14*mass**4 + 1.112e-18*mass**5) + (mass >= 5000) * 1.74865
                correctionBE = (mass<5000) * (0.8989 + 0.000182  * mass - 1.839e-7*mass**2 + 1.026e-10*mass**3 - 2.361e-14*mass**4 + 1.927e-18*mass**5) + (mass >= 5000) * 1.302025

    if region == "bb":
        return correctionBB
    elif region == "be":
        return correctionBE
    else:
       return correctionAll


#not implemented for electrons since it was read from ROOT file in old framework
def calcNNPDFWeightTT(mass, region, flavor, year):

    correction = 1

    if flavor == "mu":
        if (year == 2017):
            correction = ((mass < 120) * 1. + (mass > 120 and mass < 3000)*((0.994078695151) + (2.64819793287e-05)*mass + (-3.73996461024e-08)*mass**2 + (-1.11452866827e-11)*mass**3) + (mass > 3000)*(0.436005))
        elif (year == 2018):
            correction = ((mass < 120) * 1. + (mass > 120 and mass < 3000)*((0.994078695151) + (2.64819793287e-05)*mass + (-3.73996461024e-08)*mass**2 + (-1.11452866827e-11)*mass**3) + (mass > 3000)*(0.436005))
    print (correction)
    return correction


def NNPDFWeight(masses, leadingPts, region, flavor, year, DY=True):

    result = []
    for i in range(0,len(masses)):
        if DY:
            result.append(calcNNPDFWeightDY(masses[i], leadingPts[i], region, flavor, year))
        else:
            result.append(calcNNPDFWeightTT(masses[i], region, flavor, year))
    return np.array(result)
