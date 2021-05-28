import numpy as np
import math


def find_dielectron(objs):
    objs['el_idx'] = objs.index
    idx1=objs.iloc[0].el_idx
    idx2=objs.iloc[1].el_idx
    px1_ = objs.iloc[0].pt * np.cos(objs.iloc[0].phi)
    py1_ = objs.iloc[0].pt * np.sin(objs.iloc[0].phi)
    pz1_ = objs.iloc[0].pt * np.sinh(objs.iloc[0].eta)
    e1_ = np.sqrt(px1_**2 + py1_**2 + pz1_**2 + objs.iloc[0].mass**2)
    px2_ = objs.iloc[1].pt * np.cos(objs.iloc[1].phi)
    py2_ = objs.iloc[1].pt * np.sin(objs.iloc[1].phi)
    pz2_ = objs.iloc[1].pt * np.sinh(objs.iloc[1].eta)
    e2_ = np.sqrt(px2_**2 + py2_**2 + pz2_**2 + objs.iloc[1].mass**2)
    m2 = (e1_+e2_)**2-(px1_+px2_)**2-(py1_+py2_)**2-(pz1_+pz2_)**2
    mass=math.sqrt(max(0,m2))
    return [idx1, idx2, mass]


def fill_electrons(processor, output, e1, e2, is_mc):
    e1_variable_names = [
        'e1_pt',
        'e1_eta', 'e1_phi'
    ]
    e2_variable_names = [
        'e2_pt', 
        'e2_eta', 'e2_phi'
    ]
    dielectron_variable_names = [
        'dielectron_mass',
        'dielectron_mass_res', 'dielectron_mass_res_rel',
        'dielectron_ebe_mass_res', 'dielectron_ebe_mass_res_rel',
        'dielectron_pt', 'dielectron_pt_log',
        'dielectron_eta', 'dielectron_phi',
        'dielectron_dEta', 'dielectron_dPhi',
        'dielectron_dR', 'dielectron_rap',
        'dielecron_cos_theta_cs', 'dielectron_phi_cs', 'wgt_nominal','pu_wgt'
    ]
    v_names = (
        e1_variable_names +
        e2_variable_names +
        dielectron_variable_names
    )

    # Initialize columns for electron variables
    for n in (v_names):
        output[n] = 0.0

    # Fill single electron variables
    for v in ['pt', 'eta', 'phi']:
        output[f'e1_{v}'] = e1[v]
        output[f'e2_{v}'] = e2[v]
        
            
    # Fill dielectron variables
    ee = p4_sum(e1, e2)
    for v in ['pt', 'eta', 'phi', 'mass', 'rap']:
        name = f'dielectron_{v}'
        output[name] = ee[v]
        output[name] = output[name].fillna(-999.)

    output['dielectron_pt_log'] = np.log(output.dielectron_pt[output.dielectron_pt>0])
    output.loc[output.dielectron_pt<0, 'dielectron_pt_log']=-999.

    ee_deta, ee_dphi, ee_dr = delta_r(
        e1.eta, e2.eta,
        e1.phi, e2.phi
    )
    output['dielectron_pt'] = ee.pt
    output['dielectron_eta'] = ee.eta
    output['dielectron_phi'] = ee.phi
    output['dielectron_dEta'] = ee_deta
    output['dielectron_dPhi'] = ee_dphi
    output['dielectron_dR'] = ee_dr

    output['dielectron_cos_theta_cs'],\
        output['dielectron_phi_cs'] = cs_variables(e1, e2)


