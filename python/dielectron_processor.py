import awkward
import awkward as ak
import numpy as np
# np.set_printoptions(threshold=sys.maxsize)
import pandas as pd
import coffea.processor as processor
from coffea.lookup_tools import extractor
from coffea.lumi_tools import LumiMask
from python.timer import Timer
from python.weights import Weights
from config.parameters import parameters
from python.corrections.pu_reweight import pu_lookups, pu_evaluator
from python.corrections.l1prefiring_weights import l1pf_weights
from python.electrons import find_dielectron, fill_electrons
from python.jets import prepare_jets, fill_jets
#from python.jets import jet_id, jet_puid, gen_jet_pair_mass
#from python.corrections.kFac import kFac
from python.corrections.jec import jec_factories, apply_jec

class DielectronProcessor(processor.ProcessorABC):
    def __init__(self, **kwargs):
        self.samp_info = kwargs.pop('samp_info', None)
        do_timer = kwargs.pop('do_timer', True)
        self.apply_to_output = kwargs.pop('apply_to_output', None) 
        self.do_btag_syst = kwargs.pop('do_btag_syst', None)
        self.pt_variations = kwargs.pop('pt_variations', ['nominal'])
        if self.samp_info is None:
            print("Samples info missing!")
            return

        self._accumulator = processor.defaultdict_accumulator(int)

        self.do_pu = True
        self.auto_pu = True
        self.do_l1pw = False  # L1 prefiring weights
        self.do_jecunc = False
        self.do_jerunc = False

        self.year = self.samp_info.year

        self.parameters = {
            k: v[self.year] for k, v in parameters.items()}

        self.timer = Timer('global') if do_timer else None

        self._columns = self.parameters["proc_columns"]

        self.regions = ['bb', 'be', 'ee']
        self.channels = ['ee']

        self.lumi_weights = self.samp_info.lumi_weights
        if self.do_btag_syst:
            self.btag_systs = ["jes", "lf", "hfstats1", "hfstats2",
                               "cferr1", "cferr2", "hf", "lfstats1",
                               "lfstats2"]
        else:
            self.btag_systs = []

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
        
        df['Electron', 'pt_raw'] = df.Electron.pt*(df.Electron.scEtOverPt+1.)
        df['Electron', 'eta_raw'] = df.Electron.eta + df.Electron.deltaEtaSC
        df['Electron', 'phi_raw'] = df.Electron.phi


        # for ...
        if True:  # indent reserved for loop over pT variations

            # --- conversion from awkward to pandas --- #
            el_branches = ['pt_raw', 'pt', 'scEtOverPt', 'eta', 'deltaEtaSC', 'eta_raw', 'phi', 'phi_raw', 'mass','cutBased_HEEP','charge']
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
                (abs(electrons.eta_raw) < self.parameters["electron_eta_cut"]) &
                (electrons[self.parameters["electron_id"]]>0) 
            )

            # Count electrons
            nelectrons = electrons[electrons.selection].reset_index()\
                .groupby('entry')['subentry'].nunique()
            output['event_selection'] = (mask &
                (hlt > 0) &
                (nelectrons >= 2) 
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
                #return output
                if self.apply_to_output is None:
                    return output
                else:
                    self.apply_to_output(output)
                    return self.accumulator.identity()

            result = electrons.groupby('entry').apply(find_dielectron)
            dielectron=pd.DataFrame(result.to_list(),columns=['idx1','idx2','mass'])
            e1=electrons.loc[dielectron.idx1.values,:]
            e2=electrons.loc[dielectron.idx2.values,:]
            e1.index = e1.index.droplevel('subentry')
            e2.index = e2.index.droplevel('subentry')
            if self.timer:
                self.timer.add_checkpoint("dielectron pair selection")

  
            if self.timer:
                self.timer.add_checkpoint("back back angle calculation")
            dielectron_mass=dielectron.mass

            # --------------------------------------------------------#
            # Select events with muons passing leading pT cut
            # and trigger matching
            # --------------------------------------------------------#

            # Events where there is at least one muon passing
            # leading muon pT cut
            #if self.timer:
            #    self.timer.add_checkpoint("Applied trigger matching")

            # --------------------------------------------------------#
            # Fill dielectron and electron variables
            # --------------------------------------------------------#
            
            fill_electrons(output, e1, e2, dielectron_mass, is_mc)          
        
            if self.timer:
                self.timer.add_checkpoint("all electron variables")

        # ------------------------------------------------------------#
        # Prepare jets
        # ------------------------------------------------------------#

        prepare_jets(df, is_mc)

        # ------------------------------------------------------------#
        # Apply JEC, get JEC and JER variations
        # ------------------------------------------------------------#

        jets = df.Jet

        self.do_jec = False

        # We only need to reapply JEC for 2018 data
        # (unless new versions of JEC are released)
        if ('data' in dataset) and ('2018' in self.year):
            self.do_jec = True

        apply_jec(
            df, jets, dataset, is_mc, self.year,
            self.do_jec, self.do_jecunc, self.do_jerunc,
            self.jec_factories, self.jec_factories_data
        )
        output.columns = pd.MultiIndex.from_product(
            [output.columns, ['']], names=['Variable', 'Variation']
        )

        if self.timer:
            self.timer.add_checkpoint("Jet preparation & event weights")
        
        for v_name in self.pt_variations:
            output_updated = self.jet_loop(
                v_name, is_mc, df, dataset, mask, electrons,
                e1, e2, jets, weights, numevents, output
            )
            if output_updated is not None:
                output = output_updated

        if self.timer:
            self.timer.add_checkpoint("Jet loop")
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
        output['r'] = None
        output.loc[((output.e1_eta < 1.442) & (output.e2_eta < 1.442)), 'r'] = "bb"
        output.loc[((output.e1_eta > 1.566) ^ (output.e2_eta > 1.566)), 'r'] = "be"
        output.loc[((output.e1_eta > 1.566) & (output.e2_eta > 1.566)), 'r'] = "ee"
        #output['s'] = dataset
        #output['year'] = int(self.year)

        for wgt in weights.df.columns:
            
            if (wgt=='pu_wgt_off'):
                output['pu_wgt'] = weights.get_weight(wgt)
            if (wgt!='nominal'):
                output[f'wgt_{wgt}'] = weights.get_weight(wgt)
  
                
        output = output.loc[output.event_selection, :]
        output = output.reindex(sorted(output.columns), axis=1)
        output = output[output.r.isin(self.regions)]
        output.columns=output.columns.droplevel('Variation')
        if self.timer:
            self.timer.add_checkpoint("Filled outputs")
            self.timer.summary()

        if self.apply_to_output is None:
            return output
        else:
            self.apply_to_output(output)
            return self.accumulator.identity()


    def jet_loop(self, variation, is_mc, df, dataset, mask, muons,
                 mu1, mu2, jets, weights, numevents, output):
        # weights = copy.deepcopy(weights)

        if not is_mc and variation != 'nominal':
            return

        variables = pd.DataFrame(index=output.index)

        jet_columns = [
            'pt', 'eta', 'phi', 'jetId', 'qgl', 'puId', 'mass', 'btagDeepB',
        ]
        if is_mc:
            jet_columns += ['partonFlavour', 'hadronFlavour']
        if variation == 'nominal':
            if self.do_jec:
                jet_columns += ['pt_jec', 'mass_jec']
            if is_mc and self.do_jerunc:
                jet_columns += ['pt_orig', 'mass_orig']
        '''
        # Find jets that have selected muons within dR<0.4 from them
        #matched_mu_pt = jets.matched_muons.pt_fsr
        #matched_mu_iso = jets.matched_muons.pfRelIso04_all
        #matched_mu_id = jets.matched_muons[self.parameters["muon_id"]]
        #matched_mu_pass = (
        #    (matched_mu_pt > self.parameters["muon_pt_cut"]) &
        #    (matched_mu_iso < self.parameters["muon_iso_cut"]) &
        #    matched_mu_id
        #)
        #clean = ~(ak.to_pandas(matched_mu_pass).astype(float).fillna(0.0)
        #          .groupby(level=[0, 1]).sum().astype(bool))

        # if self.timer:
        #     self.timer.add_checkpoint("Clean jets from matched muons")

        # Select particular JEC variation
        #if '_up' in variation:
        #    unc_name = 'JES_' + variation.replace('_up', '')
        #    if unc_name not in jets.fields:
        #        return
        #    jets = jets[unc_name]['up'][jet_columns]
        #elif '_down' in variation:
        #    unc_name = 'JES_' + variation.replace('_down', '')
        #    if unc_name not in jets.fields:
        #        return
        ##    jets = jets[unc_name]['down'][jet_columns]
        #else:
        '''
        jets = jets[jet_columns]
        # --- conversion from awkward to pandas --- #
        jets = ak.to_pandas(jets)
        if jets.index.nlevels == 3:
            # sometimes there are duplicates?
            jets = jets.loc[pd.IndexSlice[:, :, 0], :]
            jets.index = jets.index.droplevel('subsubentry')

        if variation == 'nominal':
            # Update pt and mass if JEC was applied
            if self.do_jec:
                jets['pt'] = jets['pt_jec']
                jets['mass'] = jets['mass_jec']

            # We use JER corrections only for systematics, so we shouldn't
            # update the kinematics. Use original values,
            # unless JEC were applied.
            if is_mc and self.do_jerunc and not self.do_jec:
                jets['pt'] = jets['pt_orig']
                jets['mass'] = jets['mass_orig']
        '''
        # ------------------------------------------------------------#
        # Apply jetID and PUID
        # ------------------------------------------------------------#
        #pass_jet_id = jet_id(jets, self.parameters, self.year)
        #pass_jet_puid = jet_puid(jets, self.parameters, self.year)

        # Jet PUID scale factors
        # if is_mc and False:  # disable for now
        #     puid_weight = puid_weights(
        #         self.evaluator, self.year, jets, pt_name,
        #         jet_puid_opt, jet_puid, numevents
        #     )
        #     weights.add_weight('puid_wgt', puid_weight)

        # ------------------------------------------------------------#
        # Select jets
        # ------------------------------------------------------------#
        #jets['clean'] = clean

        #jet_selection = (
        #    pass_jet_id & pass_jet_puid &
        #    (jets.qgl > -2) & jets.clean &
        #    (jets.pt > self.parameters["jet_pt_cut"]) &
        #    (abs(jets.eta) < self.parameters["jet_eta_cut"])
        #)

        #jets = jets[jet_selection]

        # if self.timer:
        #     self.timer.add_checkpoint("Selected jets")
        '''
        # ------------------------------------------------------------#
        # Fill jet-related variables
        # ------------------------------------------------------------#

        njets = jets.reset_index().groupby('entry')['subentry'].nunique()
        variables['njets'] = njets

        #one_jet = (njets > 0)
        #two_jets = (njets > 1)
        
        # Sort jets by pT and reset their numbering in an event
        jets = jets.sort_values(
            ['entry', 'pt'], ascending=[True, False]
        )
        jets.index = pd.MultiIndex.from_arrays(
            [
                jets.index.get_level_values(0),
                jets.groupby(level=0).cumcount()
            ],
            names=['entry', 'subentry']
        )
        # Select two jets with highest pT
        #try:

        jet1 = jets.loc[pd.IndexSlice[:, 0], :]
        jet2 = jets.loc[pd.IndexSlice[:, 1], :]
        jet1.index = jet1.index.droplevel('subentry')
        jet2.index = jet2.index.droplevel('subentry')
        Jets = [jet1, jet2]
        #else:
        #    jet1.index = jet1.index.droplevel('subentry')
        #    Jets = [jet1]
        #except Exception:
        #    return
        fill_jets(output, variables, Jets, 'el')
        if self.timer:
            self.timer.add_checkpoint("Filled jet variables")

        # --------------------------------------------------------------#
        # Fill outputs
        # --------------------------------------------------------------#

        variables.update({'wgt_nominal': weights.get_weight('nominal')})

        # All variables are affected by jet pT because of jet selections:
        # a jet may or may not be selected depending on pT variation.

        for key, val in variables.items():
            #output.loc[:, pd.IndexSlice[key, variation]] = val
            output.loc[:, key] = val
        return output

    def prepare_lookups(self):
        # Pile-up reweighting
        self.pu_lookups = pu_lookups(self.parameters)
        self.jec_factories, self.jec_factories_data = jec_factories(self.year)
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
