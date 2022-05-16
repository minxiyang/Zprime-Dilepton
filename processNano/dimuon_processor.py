import sys
sys.path.append("copperhead/")

import awkward
import awkward as ak
import numpy as np

# np.set_printoptions(threshold=sys.maxsize)
import pandas as pd
import coffea.processor as processor
from coffea.lumi_tools import LumiMask
from coffea.btag_tools import BTagScaleFactor
from processNano.timer import Timer
from processNano.weights import Weights

#correction helpers included from copperhead
from copperhead.stage1.corrections.pu_reweight import pu_lookups, pu_evaluator
from copperhead.stage1.corrections.lepton_sf import musf_lookup, musf_evaluator
from copperhead.stage1.corrections.jec import jec_factories, apply_jec
from copperhead.stage1.corrections.l1prefiring_weights import l1pf_weights
#from copperhead.stage1.corrections.lhe_weights import lhe_weights
#from copperhead.stage1.corrections.pdf_variations import add_pdf_variations
from copperhead.stage1.corrections.btag_weights import btag_weights

#high mass dilepton specific corrections
from processNano.corrections.kFac import kFac

from processNano.jets import prepare_jets, fill_jets
import copy

from processNano.muons import find_dimuon, fill_muons
from processNano.utils import bbangle

from config.parameters import parameters, muon_branches, jet_branches

from copperhead.config.jec_parameters import jec_parameters


class DimuonProcessor(processor.ProcessorABC):
    def __init__(self, **kwargs):
        self.samp_info = kwargs.pop("samp_info", None)
        do_timer = kwargs.pop("do_timer", True)
        self.pt_variations = kwargs.get("pt_variations", ["nominal"])
        self.apply_to_output = kwargs.pop("apply_to_output", None)

        self.year = self.samp_info.year
        self.parameters = {k: v[self.year] for k, v in parameters.items()}

        self.do_btag_syst = kwargs.pop("do_btag_syst", None)
        if self.do_btag_syst:
            self.btag_systs = self.parameters["btag_systs"]
        else:
            self.btag_systs = []

        if self.samp_info is None:
            print("Samples info missing!")
            return

        self.do_pu = True
        self.auto_pu = False
        self.year = self.samp_info.year
        self.do_l1pw = True  # L1 prefiring weights
        jec_pars = {k: v[self.year] for k, v in jec_parameters.items()}
        self.do_jecunc = True
        self.do_jerunc = False
        for ptvar in self.pt_variations:
            if ptvar in jec_pars["jec_variations"]:
                self.do_jecunc = True
            if ptvar in jec_pars["jer_variations"]:
                self.do_jerunc = True

        self.timer = Timer("global") if do_timer else None

        self._columns = self.parameters["proc_columns"]

        self.regions = ["bb", "be"]
        self.channels = ["mumu"]

        self.lumi_weights = self.samp_info.lumi_weights
        if self.do_btag_syst:
            self.btag_systs = [
                "jes",
                "lf",
                "hfstats1",
                "hfstats2",
                "cferr1",
                "cferr2",
                "hf",
                "lfstats1",
                "lfstats2",
            ]
        else:
            self.btag_systs = []

        self.prepare_lookups()

    def process(self, df):
        # Initialize timer
        if self.timer:
            self.timer.update()

        # Dataset name (see definitions in config/datasets.py)
        dataset = df.metadata["dataset"]

        is_mc = "data" not in dataset

        # ------------------------------------------------------------#
        # Apply HLT, lumimask, genweights, PU weights
        # and L1 prefiring weights
        # ------------------------------------------------------------#

        numevents = len(df)

        # All variables that we want to save
        # will be collected into the 'output' dataframe
        output = pd.DataFrame(
            {"run": df.run, "event": df.event, "luminosityBlock": df.luminosityBlock}
        )
        output.index.name = "entry"
        output["npv"] = df.PV.npvs
        output["met"] = df.MET.pt

        # Separate dataframe to keep track on weights
        # and their systematic variations
        weights = Weights(output)

        if is_mc:
            # For MC: Apply gen.weights, pileup weights, lumi weights,
            # L1 prefiring weights
            mask = np.ones(numevents, dtype=bool)
            genweight = df.genWeight
            weights.add_weight("genwgt", genweight)
            weights.add_weight("lumi", self.lumi_weights[dataset])
            if self.do_pu:
                pu_wgts = pu_evaluator(
                    self.pu_lookups,
                    self.parameters,
                    numevents,
                    np.array(df.Pileup.nTrueInt),
                    self.auto_pu,
                )
                weights.add_weight("pu_wgt", pu_wgts, how="all")
            if self.do_l1pw:
                if self.parameters["do_l1prefiring_wgts"]:
                    if "L1PreFiringWeight" in df.fields:
                        l1pfw = l1pf_weights(df)
                        weights.add_weight("l1prefiring_wgt", l1pfw, how="all")
                    else:
                        weights.add_weight("l1prefiring_wgt", how="dummy_vars")

        else:
            # For Data: apply Lumi mask
            lumi_info = LumiMask(self.parameters["lumimask_Pre-UL_mu"])
            mask = lumi_info(df.run, df.luminosityBlock)
        # Apply HLT to both Data and MC

        hlt = ak.to_pandas(df.HLT[self.parameters["mu_hlt"]])
        hlt = hlt[self.parameters["mu_hlt"]].sum(axis=1)

        if self.timer:
            self.timer.add_checkpoint("Applied HLT and lumimask")

        # ------------------------------------------------------------#
        # Update muon kinematics with Rochester correction,
        # FSR recovery and GeoFit correction
        # Raw pT and eta are stored to be used in event selection
        # ------------------------------------------------------------#

        # Save raw variables before computing any corrections
        df["Muon", "pt_raw"] = df.Muon.pt
        df["Muon", "eta_raw"] = df.Muon.eta
        df["Muon", "phi_raw"] = df.Muon.phi
        if is_mc:
            df["Muon", "pt_gen"] = df.Muon.matched_gen.pt
            df["Muon", "eta_gen"] = df.Muon.matched_gen.eta
            df["Muon", "phi_gen"] = df.Muon.matched_gen.phi
            df["Muon", "idx"] = df.Muon.genPartIdx

        # for ...
        if True:  # indent reserved for loop over muon pT variations
            # According to HIG-19-006, these variations have negligible
            # effect on significance, but it's better to have them
            # implemented in the future

            # --- conversion from awkward to pandas --- #
            muon_branches_local = copy.copy(muon_branches)
            if is_mc:
                muon_branches_local += [
                    "genPartFlav",
                    "genPartIdx",
                    "pt_gen",
                    "eta_gen",
                    "phi_gen",
                    "idx",
                ]
            muons = ak.to_pandas(df.Muon[muon_branches_local])
            if self.timer:
                self.timer.add_checkpoint("load muon data")
            muons = muons.dropna()
            muons = muons.loc[:, ~muons.columns.duplicated()]
            # --------------------------------------------------------#
            # Select muons that pass pT, eta, isolation cuts,
            # muon ID and quality flags
            # Select events with 2 OS muons, no electrons,
            # passing quality cuts and at least one good PV
            # --------------------------------------------------------#

            # Apply event quality flag
            output["r"] = None
            output["dataset"] = dataset
            output["year"] = int(self.year)
            if dataset == "dyInclusive50":
                muons = muons[muons.genPartFlav == 15]
            flags = ak.to_pandas(df.Flag)
            flags = flags[self.parameters["event_flags"]].product(axis=1)
            muons["pass_flags"] = True
            if self.parameters["muon_flags"]:
                muons["pass_flags"] = muons[self.parameters["muon_flags"]].product(
                    axis=1
                )
            # Define baseline muon selection (applied to pandas DF!)
            muons["selection"] = (
                (muons.pt_raw > self.parameters["muon_pt_cut"])
                & (abs(muons.eta_raw) < self.parameters["muon_eta_cut"])
                & (muons.tkRelIso < self.parameters["muon_iso_cut"])
                & (muons[self.parameters["muon_id"]] > 0)
                & (muons.dxy < self.parameters["muon_dxy"])
                & (
                    (muons.ptErr.values / muons.pt.values)
                    < self.parameters["muon_ptErr/pt"]
                )
                & muons.pass_flags
            )

            # Count muons
            nmuons = (
                muons[muons.selection]
                .reset_index()
                .groupby("entry")["subentry"]
                .nunique()
            )
            # Find opposite-sign muons
            sum_charge = muons.loc[muons.selection, "charge"].groupby("entry").sum()

            # Find events with at least one good primary vertex
            good_pv = ak.to_pandas(df.PV).npvsGood > 0

            # Define baseline event selection

            output["two_muons"] = (nmuons == 2) | (nmuons > 2)
            output["two_muons"] = output["two_muons"].fillna(False)
            output["event_selection"] = (
                mask
                & (hlt > 0)
                & (flags > 0)
                & (nmuons >= 2)
                & (abs(sum_charge) < nmuons)
                & good_pv
            )
            if self.timer:
                self.timer.add_checkpoint("Selected events and muons")

            # --------------------------------------------------------#
            # Initialize muon variables
            # --------------------------------------------------------#

            # Find pT-leading and subleading muons
            muons = muons[muons.selection & (nmuons >= 2) & (abs(sum_charge) < nmuons)]

            if self.timer:
                self.timer.add_checkpoint("muon object selection")
            if muons.shape[0] == 0:
                output = output.reindex(sorted(output.columns), axis=1)
                output = output[output.r.isin(self.regions)]

                # return output
                if self.apply_to_output is None:
                    return output
                else:
                    self.apply_to_output(output)
                    return self.accumulator.identity()

            result = muons.groupby("entry").apply(find_dimuon, is_mc=is_mc)
            if is_mc:
                dimuon = pd.DataFrame(
                    result.to_list(), columns=["idx1", "idx2", "mass", "mass_gen"]
                )
            else:
                dimuon = pd.DataFrame(
                    result.to_list(), columns=["idx1", "idx2", "mass"]
                )
            mu1 = muons.loc[dimuon.idx1.values, :]
            mu2 = muons.loc[dimuon.idx2.values, :]
            mu1.index = mu1.index.droplevel("subentry")
            mu2.index = mu2.index.droplevel("subentry")

            if self.timer:
                self.timer.add_checkpoint("dimuon pair selection")

            output["bbangle"] = bbangle(mu1, mu2)

            output["event_selection"] = output.event_selection & (
                output.bbangle > self.parameters["3dangle"]
            )

            if self.timer:
                self.timer.add_checkpoint("back back angle calculation")

            # --------------------------------------------------------#
            # Select events with muons passing leading pT cut
            # and trigger matching
            # --------------------------------------------------------#

            # Events where there is at least one muon passing
            # leading muon pT cut

            # if self.timer:
            #    self.timer.add_checkpoint("Applied trigger matching")

            # --------------------------------------------------------#
            # Fill dimuon and muon variables
            # --------------------------------------------------------#
            fill_muons(self, output, mu1, mu2, is_mc, self.year, weights)

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
        if ("data" in dataset) and ("2018" in self.year):
            self.do_jec = False

        apply_jec(
            df,
            jets,
            dataset,
            is_mc,
            self.year,
            self.do_jec,
            self.do_jecunc,
            self.do_jerunc,
            self.jec_factories,
            self.jec_factories_data,
        )
        output.columns = pd.MultiIndex.from_product(
            [output.columns, [""]], names=["Variable", "Variation"]
        )

        if self.timer:
            self.timer.add_checkpoint("Jet preparation & event weights")

        for v_name in self.pt_variations:
            output_updated = self.jet_loop(
                v_name,
                is_mc,
                df,
                dataset,
                mask,
                muons,
                mu1,
                mu2,
                jets,
                jet_branches,
                weights,
                numevents,
                output,
            )
            if output_updated is not None:
                output = output_updated

        if self.timer:
            self.timer.add_checkpoint("Jet loop")

        if is_mc:
            """
            do_zpt = ('dy' in dataset)
            if do_zpt:
                zpt_weight = np.ones(numevents, dtype=float)
                zpt_weight[two_muons] =\
                    self.evaluator[self.zpt_path](
                        output['dimuon_pt'][two_muons]
                    ).flatten()
                weights.add_weight('zpt_wgt', zpt_weight)
            """

            do_musf = True
            if do_musf:
                muID, muIso, muTrig = musf_evaluator(
                    self.musf_lookup, self.year, numevents, mu1, mu2
                )
                weights.add_weight("muID", muID, how="all")
                weights.add_weight("muIso", muIso, how="all")
                weights.add_weight("muTrig", muTrig, how="all")
            else:
                weights.add_weight("muID", how="dummy_all")
                weights.add_weight("muIso", how="dummy_all")
                weights.add_weight("muTrig", how="dummy_all")

        if self.timer:
            self.timer.add_checkpoint("Computed event weights")

        # ------------------------------------------------------------#
        # Fill outputs
        # ------------------------------------------------------------#

        output.loc[
            ((abs(output.mu1_eta) < 1.2) & (abs(output.mu2_eta) < 1.2)), "r"
        ] = "bb"
        output.loc[
            ((abs(output.mu1_eta) > 1.2) | (abs(output.mu2_eta) > 1.2)), "r"
        ] = "be"

        output["year"] = int(self.year)

        for wgt in weights.df.columns:

            if wgt != "nominal":
                continue
            output[f"wgt_{wgt}"] = weights.get_weight(wgt)
        if is_mc and "dy" in dataset:
            mass_bb = output[output["r"] == "bb"].dimuon_mass_gen.to_numpy()
            mass_be = output[output["r"] == "be"].dimuon_mass_gen.to_numpy()
            output.loc[
                ((abs(output.mu1_eta) < 1.2) & (abs(output.mu2_eta) < 1.2)), "wgt_nominal"
            ] = (
                output.loc[
                    ((abs(output.mu1_eta) < 1.2) & (abs(output.mu2_eta) < 1.2)), "wgt_nominal"
                ]
                * kFac(mass_bb, "bb", "mu")
            ).values
            output.loc[
                ((abs(output.mu1_eta) > 1.2) | (abs(output.mu2_eta) > 1.2)), "wgt_nominal"
            ] = (
                output.loc[
                    ((abs(output.mu1_eta) > 1.2) | (abs(output.mu2_eta) > 1.2)), "wgt_nominal"
                ]
                * kFac(mass_be, "be", "mu")
            ).values
        output = output.loc[output.event_selection, :]
        output = output.reindex(sorted(output.columns), axis=1)
        output = output[output.r.isin(self.regions)]
        output.columns = output.columns.droplevel("Variation")
        if self.timer:
            self.timer.add_checkpoint("Filled outputs")
            self.timer.summary()

        if self.apply_to_output is None:
            return output
        else:
            self.apply_to_output(output)
            return self.accumulator.identity()

    def jet_loop(
        self,
        variation,
        is_mc,
        df,
        dataset,
        mask,
        muons,
        mu1,
        mu2,
        jets,
        jet_branches,
        weights,
        numevents,
        output,
    ):

        if not is_mc and variation != "nominal":
            return

        variables = pd.DataFrame(index=output.index)
        jet_branches_local = copy.copy(jet_branches)

        if is_mc:
            jets["pt_gen"] = jets.matched_gen.pt
            jets["eta_gen"] = jets.matched_gen.eta
            jets["phi_gen"] = jets.matched_gen.phi

            jet_branches_local += [
                "partonFlavour",
                "hadronFlavour",
                "pt_gen",
                "eta_gen",
                "phi_gen",
                "sf",
            ]
            btag_sf = BTagScaleFactor(
                "data/b-tagging/DeepCSV_102XSF_WP_V1.csv", "tight"
            )
            sf = btag_sf("central", jets.hadronFlavour, np.abs(jets.eta), jets.pt)
            jets["sf"] = sf

        if variation == "nominal":
            if self.do_jec:
                jet_branches_local += ["pt_jec", "mass_jec"]
            if is_mc and self.do_jerunc:
                jet_branches_local += ["pt_orig", "mass_orig"]

        # Find jets that have selected muons within dR<0.4 from them
        matched_mu_pt = jets.matched_muons.pt
        matched_mu_iso = jets.matched_muons.pfRelIso04_all
        matched_mu_id = jets.matched_muons[self.parameters["muon_id"]]
        matched_mu_pass = (
            (matched_mu_pt > self.parameters["muon_pt_cut"]) &
            (matched_mu_iso < self.parameters["muon_iso_cut"]) &
            matched_mu_id
        )
        clean = ~(ak.to_pandas(matched_mu_pass).astype(float).fillna(0.0)
                  .groupby(level=[0, 1]).sum().astype(bool))

        if self.timer:
            self.timer.add_checkpoint("Clean jets from matched muons")

        # Select particular JEC variation
        if "_up" in variation:
            unc_name = "JES_" + variation.replace("_up", "")
            if unc_name not in jets.fields:
                return
            jets = jets[unc_name]["up"][jet_branches_local]
        elif "_down" in variation:
            unc_name = "JES_" + variation.replace("_down", "")
            if unc_name not in jets.fields:
                return
            jets = jets[unc_name]["down"][jet_branches_local]
        else:
            jets = jets[jet_branches_local]

        # --- conversion from awkward to pandas --- #
        jets = ak.to_pandas(jets)

        jets = jets.dropna()

        if jets.index.nlevels == 3:
            # sometimes there are duplicates?
            jets = jets.loc[pd.IndexSlice[:, :, 0], :]
            jets.index = jets.index.droplevel("subsubentry")

        if variation == "nominal":
            # Update pt and mass if JEC was applied
            if self.do_jec:
                jets["pt"] = jets["pt_jec"]
                jets["mass"] = jets["mass_jec"]

            # We use JER corrections only for systematics, so we shouldn't
            # update the kinematics. Use original values,
            # unless JEC were applied.
            if is_mc and self.do_jerunc and not self.do_jec:
                jets["pt"] = jets["pt_orig"]
                jets["mass"] = jets["mass_orig"]

        # ------------------------------------------------------------#
        # Apply jetID
        # ------------------------------------------------------------#
        # Sort jets by pT and reset their numbering in an event
        # jets = jets.sort_values(["entry", "pt"], ascending=[True, False])
        jets.index = pd.MultiIndex.from_arrays(
            [jets.index.get_level_values(0), jets.groupby(level=0).cumcount()],
            names=["entry", "subentry"],
        )
        # Select two jets with highest pT

        jets["selection"] = 0
        jets.loc[
            (
                (jets.pt > 30.0)
                & (abs(jets.eta) < 2.4)
                & (jets.btagDeepB > parameters["UL_btag_medium"][self.year])
                & (jets.jetId >= 2)
            ),
            "selection",
        ] = 1
        nBjets = jets.loc[:, "selection"].groupby("entry").sum()
        variables["njets"] = nBjets
        jets = jets.sort_values(["entry", "pt"], ascending=[True, False])
        jet1 = jets.groupby("entry").nth(0)
        jet2 = jets.groupby("entry").nth(1)
        Jets = [jet1, jet2]
        fill_jets(output, variables, Jets, is_mc=is_mc)
        if self.timer:
            self.timer.add_checkpoint("Filled jet variables") 

        # ------------------------------------------------------------#
        # Calculate btag SF
        # ------------------------------------------------------------#
        # --- Btag weights --- #
        if is_mc:
            bjet_sel_mask = output.event_selection

            btag_wgt, btag_syst = btag_weights(
                self, self.btag_lookup, self.btag_systs, jets, weights, bjet_sel_mask
            )
            weights.add_weight("btag_wgt", btag_wgt)

            # --- Btag weights variations --- #
            for name, bs in btag_syst.items():
                weights.add_weight(f"btag_wgt_{name}", bs, how="only_vars")

            if self.timer:
                self.timer.add_checkpoint(
                    "Applied B-tag weights"
                )

        # --------------------------------------------------------------#
        # Fill outputs
        # --------------------------------------------------------------#

        variables.update({"wgt_nominal": weights.get_weight("nominal")})

        # All variables are affected by jet pT because of jet selections:
        # a jet may or may not be selected depending on pT variation.

        for key, val in variables.items():
            output.loc[:, key] = val

        del df
        del muons
        del jets
        del mu1
        del mu2
        return output

    def prepare_lookups(self):
        self.jec_factories, self.jec_factories_data = jec_factories(self.year)
        # Muon scale factors
        self.musf_lookup = musf_lookup(self.parameters)
        # Pile-up reweighting
        self.pu_lookups = pu_lookups(self.parameters)
        # Btag weights
        self.btag_lookup = BTagScaleFactor(
            self.parameters["btag_sf_csv"],
            BTagScaleFactor.RESHAPE,
            "iterativefit,iterativefit,iterativefit",
        )

        # --- Evaluator
        #self.extractor = extractor()
        # PU ID weights
        #puid_filename = self.parameters["puid_sf_file"]
        #self.extractor.add_weight_sets([f"* * {puid_filename}"])

        #self.extractor.finalize()
        #self.evaluator = self.extractor.make_evaluator()

        return

    @property
    def accumulator(self):
        return processor.defaultdict_accumulator(int)

    @property
    def muoncolumns(self):
        return muon_branches

    @property
    def jetcolumns(self):
        return jet_branches


    def postprocess(self, accumulator):
        return accumulator
