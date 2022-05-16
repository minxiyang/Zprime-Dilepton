class Variable(object):
    def __init__(self, name_, caption_, nbins_, xmin_, xmax_, ymin_, ymax_, binning_ = [], norm_to_bin_width_=False, xminPlot_=None, xmaxPlot_=None):
        self.name = name_
        self.caption = caption_
        self.nbins = nbins_
        self.xmin = xmin_
        self.xmax = xmax_
        self.ymin = ymin_
        self.ymax = ymax_
        if xminPlot_ is not None:
            self.xminPlot = xminPlot_
        else:
            self.xminPlot = xmin_
        if xmaxPlot_ is not None:
            self.xmaxPlot = xmaxPlot_
        else:
            self.xmaxPlot = xmax_
        self.binning = binning_
        self.norm_to_bin_width = norm_to_bin_width_


variables = []

massBinningMuMu = (
        [j for j in range(120, 150, 5)]
        + [j for j in range(150, 200, 10)]
        + [j for j in range(200, 600, 20)]
        + [j for j in range(600, 900, 30)]
        + [j for j in range(900, 1250, 50)]
        + [j for j in range(1250, 1610, 60)]
        + [j for j in range(1610, 1890, 70)]
        + [j for j in range(1890, 3970, 80)]
        + [j for j in range(3970, 6070, 100)]
        + [6070]
    )

variables.append(Variable("dimuon_mass", r"$m_{\mu\mu}$ [GeV]", len(massBinningMuMu)-1, 200, 4900, 1e-5, 1e5, binning_ = massBinningMuMu, norm_to_bin_width_=True))
variables.append(Variable("dimuon_mass_resUnc", r"$m_{\mu\mu}$ [GeV] (res. unc.)", len(massBinningMuMu)-1, 200, 4900, 1e-5, 1e5, binning_ = massBinningMuMu, norm_to_bin_width_=True))
variables.append(Variable("dimuon_mass_scaleUncUp", r"$m_{\mu\mu}$ [GeV] (scale unc. up)", len(massBinningMuMu)-1, 200, 4900, 1e-5, 1e5, binning_ = massBinningMuMu, norm_to_bin_width_=True))
variables.append(Variable("dimuon_mass_scaleUncDown", r"$m_{\mu\mu}$ [GeV] (scale unc. down)", len(massBinningMuMu)-1, 200, 4900, 1e-5, 1e5, binning_ = massBinningMuMu, norm_to_bin_width_=True))
variables.append(Variable("dimuon_mass_gen", r"generated $m_{\mu\mu}$ [GeV]", len(massBinningMuMu)-1, 200, 4900, 1e-5, 1e5, binning_ = massBinningMuMu, norm_to_bin_width_=True))

variables_lookup = {}
for v in variables:
    variables_lookup[v.name] = v
