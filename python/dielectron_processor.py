import awkward
import awkward as ak
import numpy as np
# np.set_printoptions(threshold=sys.maxsize)
import pandas as pd

import coffea.processor as processor
from coffea.lookup_tools import extractor
from coffea.lumi_tools import LumiMask

from python.utils import p4_sum, delta_r, rapidity, cs_variables, bbangle
from python.timer import Timer
from python.weights import Weights

from config.parameters import parameters

from python.corrections.pu_reweight import pu_lookups, pu_evaluator
from python.corrections.l1prefiring_weights import l1pf_weights

from python.electrons import find_dielectron, fill_electrons

class DielectronProcessor(processor.ProcessorABC):
    def __init__(self, **kwargs):
        self.samp_info = kwargs.pop('samp_info', None)
        do_timer = kwargs.pop('do_timer', True)

        if self.samp_info is None:
            print("Samples info missing!")
            return

        self.do_pu = True
        self.auto_pu = True
        self.do_l1pw = False  # L1 prefiring weights
        self.year = self.samp_info.year

        self.parameters = {
            k: v[self.year] for k, v in parameters.items()}

        self.timer = Timer('global') if do_timer else None

        self._columns = self.parameters["proc_columns"]

        self.regions = self.samp_info.regions
        self.channels = self.samp_info.channels

        self.lumi_weights = self.samp_info.lumi_weights

        self.prepare_lookups()


    @property
    def accumulator(self):
        return self._accumulator

    @property
    def columns(self):
        return self._columns

    def process(self, df):

        # Initialize timer
        if self.timer:
            self.timer.update()

        # Dataset name (see definitions in config/datasets.py)
        dataset = df.metadata['dataset']
        
        is_mc = 'data' not in dataset

        # ------------------------------------------------------------#
        # Apply HLT, lumimask, genweights, PU weights
        # and L1 prefiring weights
        # ------------------------------------------------------------#

        numevents = len(df)

        # All variables that we want to save
        # will be collected into the 'output' dataframe
        output = pd.DataFrame({'run': df.run, 'event': df.event})
        output.index.name = 'entry'
        output['npv'] = df.PV.npvs
        output['met'] = df.MET.pt

        # Separate dataframe to keep track on weights
        # and their systematic variations
        weights = Weights(output)

        if is_mc:
            # For MC: Apply gen.weights, pileup weights, lumi weights,
            # L1 prefiring weights
            mask = np.ones(numevents, dtype=bool)
            genweight = df.genWeight
            weights.add_weight('genwgt', genweight)
            weights.add_weight('lumi', self.lumi_weights[dataset])
            if self.do_pu:
                pu_wgts = pu_evaluator(
                    self.pu_lookups,
                    self.parameters,
                    numevents,
                    np.array(df.Pileup.nTrueInt),
                    self.auto_pu
                )
                weights.add_weight('pu_wgt', pu_wgts, how='all')
            if self.do_l1pw:
                if self.parameters["do_l1prefiring_wgts"]:
                    if 'L1PreFiringWeight' in df.fields:
                        l1pfw = l1pf_weights(df)
                        weights.add_weight('l1prefiring_wgt', l1pfw, how='all')
                    else:
                        weights.add_weight('l1prefiring_wgt', how='dummy_vars')

        else:
            # For Data: apply Lumi mask
            lumi_info = LumiMask(self.parameters['lumimask'])
            mask = lumi_info(df.run, df.luminosityBlock)

        # Apply HLT to both Data and MC
        hlt = ak.to_pandas(df.HLT[self.parameters["el_hlt"]])
        hlt = hlt[self.parameters["el_hlt"]].sum(axis=1)

        if self.timer:
            self.timer.add_checkpoint("Applied HLT and lumimask")

        # Save raw variables before computing any corrections
        df['Electron', 'pt_raw'] = df.Electron.pt
        df['Electron', 'eta_raw'] = df.Electron.eta
        df['Electron', 'phi_raw'] = df.Electron.phi


        # for ...
        if True:  # indent reserved for loop over pT variations

            # --- conversion from awkward to pandas --- #
            el_branches = ['pt_raw','pt', 'eta', 'eta_raw', 'phi', 'phi_raw', 'mass','cutBased_HEEP','charge']
            electrons = ak.to_pandas(df.Electron[el_branches])
            if self.timer:
                    self.timer.add_checkpoint("load electron data")

            # --------------------------------------------------------#
            # Electron selection
            # --------------------------------------------------------#

            # Apply event quality flag
            flags = ak.to_pandas(df.Flag)
            flags = flags[self.parameters["event_flags"]].product(axis=1)

            # Define baseline muon selection (applied to pandas DF!)
            electrons['selection'] =  (
                (electrons.pt_raw > self.parameters["electron_pt_cut"]) &
                (abs(electrons.eta_raw) <
                 self.parameters["electron_eta_cut"]) &
                (np.min(electrons.eta_raw) < 1.442) &
                (electrons[self.parameters["electron_id"]]>0) 
            )

            # Count electrons
            nelectrons = electrons[electrons.selection].reset_index()\
                .groupby('entry')['subentry'].nunique()

            # Find opposite-sign muons
            #sum_charge = muons.loc[muons.selection, 'charge']\
            #    .groupby('entry').sum()

            # Find events with at least one good primary vertex
            #good_pv = ak.to_pandas(df.PV).npvsGood > 0

            # Define baseline event selection

            #output['two_muons'] = ((nmuons == 2) | (nmuons > 2))
            #output['two_muons'] = output['two_muons'].fillna(False)
            #print("two muons")
            #print(nmuons.values)
           
            output['event_selection'] = (mask &
                (hlt > 0) &
                #(flags > 0) &
                (nelectrons >= 2) 
                #(abs(sum_charge)<nmuons) &
                #good_pv
            )

            if self.timer:
                self.timer.add_checkpoint("Selected events and electrons")

            # --------------------------------------------------------#
            # Initialize electron variables
            # --------------------------------------------------------#

            electrons = electrons[electrons.selection & (nelectrons >= 2)]

            if self.timer:
                    self.timer.add_checkpoint("electron object selection")

            output['r'] = None
            output['s'] = dataset
            output['year'] = int(self.year)
            #print(output.shape[1])

            if electrons.shape[0] == 0:
                output = output.reindex(sorted(output.columns), axis=1)
                output = output[output.r.isin(self.regions)]
                return output

            result = electrons.groupby('entry').apply(find_dielectron)
            dielectron=pd.DataFrame(result.to_list(),columns=['idx1','idx2','mass'])
            e1=electrons.loc[dielectron.idx1.values,:]
            e2=electrons.loc[dielectron.idx2.values,:]
            e1.index = e1.index.droplevel('subentry')
            e2.index = e2.index.droplevel('subentry')
            if self.timer:
                self.timer.add_checkpoint("dielectron pair selection")

            #output['bbangle'] = bbangle(mu1, mu2)
            #print("finish")
            #output['event_selection'] = (
            #    output.event_selection & (output.bbangle>self.parameters['3dangle'])
            #)
  
            if self.timer:
                self.timer.add_checkpoint("back back angle calculation")

            output.dielectron_mass=dielectron.mass  

            # --------------------------------------------------------#
            # Select events with muons passing leading pT cut
            # and trigger matching
            # --------------------------------------------------------#

            # Events where there is at least one muon passing
            # leading muon pT cut
            
            #pass_leading_pt = (
            #    mu1.pt_raw > self.parameters["muon_leading_pt"]
            #)
            #print(pass_leading_pt)
            # update event selection with leading muon pT cut
            #output['pass_leading_pt'] = pass_leading_pt
            #output['event_selection'] = (
            #    output.event_selection  & (bbangle(mu1, mu2) < self.parameters['3dangle'] )
            #)

            #if self.timer:
            #    self.timer.add_checkpoint("Applied trigger matching")

            # --------------------------------------------------------#
            # Fill dielectron and electron variables
            # --------------------------------------------------------#
            
            fill_muons(self, output, e1, e2, is_mc)          
        
            if self.timer:
                    self.timer.add_checkpoint("all electron variables")


        # ------------------------------------------------------------#
        # Calculate other event weights
        # ------------------------------------------------------------#

        """
        if is_mc:
            do_zpt = ('dy' in dataset)
            if do_zpt:
                zpt_weight = np.ones(numevents, dtype=float)
                zpt_weight[two_muons] =\
                    self.evaluator[self.zpt_path](
                        output['dimuon_pt'][two_muons]
                    ).flatten()
                weights.add_weight('zpt_wgt', zpt_weight)

        if self.timer:
            self.timer.add_checkpoint("Computed event weights")
        """

        # ------------------------------------------------------------#
        # Fill outputs
        # ------------------------------------------------------------#

        mass = output.dielectron_mass

        #output['r'] = None
        output.loc[((output.e1_eta < 1.442) & (output.e2_eta < 1.442)), 'r'] = "bb"
        output.loc[((output.e1_eta > 1.442) | (output.e2_eta > 1.442)), 'r'] = "be"
        #output['s'] = dataset
        #output['year'] = int(self.year)

        for wgt in weights.df.columns:
            #print(wgt)
            if (wgt!='nominal'):
                continue
            output[f'wgt_{wgt}'] = weights.get_weight(wgt)

        #NNPDFFac = 0.919027 + (5.98337e-05)*mass + (2.56077e-08)*mass**2 + (-2.82876e-11)*mass**3 + (9.2782e-15)*mass**4 + (-7.77529e-19)*mass**5
        #print(NNPDFFac.head())
        #NNPDFFac_bb = 0.911563 + (0.000113313)*mass + (-2.35833e-08)*mass**2 + (-1.44584e-11)*mass*3 + (8.41748e-15)*mass**4 + (-8.16574e-19)*mass**5
        #NNPDFFac_be = 0.934502 + (2.21259e-05)*mass + (4.14656e-08)*mass**2 + (-2.26011e-11)*mass**3 + (5.58804e-15)*mass**4 + (-3.92687e-19)*mass**5
        #output.loc[((output.mu1_eta < 1.2) & (output.mu2_eta < 1.2)) ,'wgt_nominal'] = output.loc[((output.mu1_eta < 1.2) & (output.mu2_eta < 1.2)) ,'wgt_nominal'] * NNPDFFac_bb
        #output.loc[((output.mu1_eta > 1.2) | (output.mu2_eta > 1.2)) ,'wgt_nominal'] = output.loc[((output.mu1_eta > 1.2) | (output.mu2_eta > 1.2)) ,'wgt_nominal'] * NNPDFFac_be
        #print('p 5')
        #print(output['wgt_nominal'].values)        
        output = output.loc[output.event_selection, :]
        output = output.reindex(sorted(output.columns), axis=1)

        output = output[output.r.isin(self.regions)]

        if self.timer:
            self.timer.add_checkpoint("Filled outputs")
            self.timer.summary()

        return output


    def prepare_lookups(self):
        # Pile-up reweighting
        self.pu_lookups = pu_lookups(self.parameters)
        
        # --- Evaluator
        self.extractor = extractor()

        # Z-pT reweigting (disabled)
        zpt_filename = self.parameters['zpt_weights_file']
        self.extractor.add_weight_sets([f"* * {zpt_filename}"])
        if '2016' in self.year:
            self.zpt_path = 'zpt_weights/2016_value'
        else:
            self.zpt_path = 'zpt_weights/2017_value'

        # Calibration of event-by-event mass resolution
        for mode in ["Data", "MC"]:
            label = f"res_calib_{mode}_{self.year}"
            path = self.parameters['res_calib_path']
            file_path = f"{path}/{label}.root"
            self.extractor.add_weight_sets(
                [f"{label} {label} {file_path}"]
            )

        self.extractor.finalize()
        self.evaluator = self.extractor.make_evaluator()

        self.evaluator[self.zpt_path]._axes =\
            self.evaluator[self.zpt_path]._axes[0]
        return


    def postprocess(self, accumulator):
        return accumulator
