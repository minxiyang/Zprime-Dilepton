import subprocess



backgrounds_muons_2018 = [

        ('WZ', '/WZ_TuneCP5_13TeV-pythia8/RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15-v3/MINIAODSIM'),
        ('WZ3LNu', '/WZTo3LNu_TuneCP5_13TeV-powheg-pythia8/RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15_ext1-v2/MINIAODSIM'),
        ('WZ2L2Q', '/WZTo2L2Q_13TeV_amcatnloFXFX_madspin_pythia8/RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15-v1/MINIAODSIM'),

            ('ZZ', '/ZZ_TuneCP5_13TeV-pythia8/RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15-v2/MINIAODSIM'),
            ('ZZ2L2Nu', '/ZZTo2L2Nu_TuneCP5_13TeV_powheg_pythia8/RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15_ext1-v2/MINIAODSIM'),
            ('ZZ2L2Nu_ext', '/ZZTo2L2Nu_TuneCP5_13TeV_powheg_pythia8/RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15_ext2-v2/MINIAODSIM'),
            ('ZZ2L2Q', '/ZZTo2L2Q_13TeV_amcatnloFXFX_madspin_pythia8/RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15-v1/MINIAODSIM'),
            ('ZZ4L', '/ZZTo4L_TuneCP5_13TeV_powheg_pythia8/RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15_ext1-v2/MINIAODSIM'),
            ('ZZ4L_ext', '/ZZTo4L_TuneCP5_13TeV_powheg_pythia8/RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15_ext2-v2/MINIAODSIM'),
                    ('WWinclusive', '/WW_TuneCP5_13TeV-pythia8/RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15-v2/MINIAODSIM'),
        ('WW200to600', '/WWTo2L2Nu_MLL_200To600_NNPDF31_13TeV-powheg/RunIIFall17MiniAODv2-PU2017_12Apr2018_94X_mc2017_realistic_v14-v1/MINIAODSIM'),
        ('WW600to1200', '/WWTo2L2Nu_MLL_600To1200_v3_NNPDF31_13TeV-powheg/RunIIFall17MiniAODv2-PU2017_12Apr2018_94X_mc2017_realistic_v14-v1/MINIAODSIM'),
        ('WW1200to2500', '/WWTo2L2Nu_MLL_1200To2500_NNPDF31_13TeV-powheg/RunIIFall17MiniAODv2-PU2017_12Apr2018_94X_mc2017_realistic_v14-v1/MINIAODSIM'),
        ('WW2500', '/WWTo2L2Nu_MLL_2500ToInf_NNPDF31_13TeV-powheg/RunIIFall17MiniAODv2-PU2017_12Apr2018_94X_mc2017_realistic_v14-v1/MINIAODSIM'),

        ('dyInclusive50', '/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15-v1/MINIAODSIM'),

            ('Wjets', '/WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8/RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15-v2/MINIAODSIM'),

        ('ttbar_lep', '/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15-v1/MINIAODSIM'),
        ('ttbar_lep_500to800_ext', '/TTToLL_MLL_500To800_41to65_NNPDF31_13TeV-powheg/RunIIFall17MiniAODv2-PU2017_12Apr2018_94X_mc2017_realistic_v14-v1/MINIAODSIM'),
        ('ttbar_lep_500to800', '/TTToLL_MLL_500To800_0to20_NNPDF31_13TeV-powheg/RunIIFall17MiniAODv2-PU2017_12Apr2018_94X_mc2017_realistic_v14-v1/MINIAODSIM'),
        ('ttbar_lep_800to1200', '/TTToLL_MLL_800To1200_NNPDF31_13TeV-powheg/RunIIFall17MiniAODv2-PU2017_12Apr2018_94X_mc2017_realistic_v14-v1/MINIAODSIM'),
        ('ttbar_lep_1200to1800', '/TTToLL_MLL_1200To1800_NNPDF31_13TeV-powheg/RunIIFall17MiniAODv2-PU2017_12Apr2018_94X_mc2017_realistic_v14-v1/MINIAODSIM'),
        ('ttbar_lep_1800toInf','/TTToLL_MLL_1800ToInf_NNPDF31_13TeV-powheg/RunIIFall17MiniAODv2-PU2017_12Apr2018_94X_mc2017_realistic_v14-v1/MINIAODSIM'),


            ('Wantitop', '/ST_tW_antitop_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8/RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15_ext1-v3/MINIAODSIM'),
        ('tW', '/ST_tW_top_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8/RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15_ext1-v3/MINIAODSIM'),

]


dy_electrons_2018=[
        ('dy120to200', '/ZToEE_NNPDF31_TuneCP5_13TeV-powheg_M_120_200/RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15-v1/MINIAODSIM'),
            ('dy200to400', '/ZToEE_NNPDF31_TuneCP5_13TeV-powheg_M_200_400/RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15-v1/MINIAODSIM'),
        ('dy400to800', '/ZToEE_NNPDF31_TuneCP5_13TeV-powheg_M_400_800/RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15-v1/MINIAODSIM'),
            ('dy800to1400', '/ZToEE_NNPDF31_TuneCP5_13TeV-powheg_M_800_1400/RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15-v1/MINIAODSIM'),
        ('dy1400to2300', '/ZToEE_NNPDF31_TuneCP5_13TeV-powheg_M_1400_2300/RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15-v1/MINIAODSIM'),
    ('dy2300to3500', '/ZToEE_NNPDF31_TuneCP5_13TeV-powheg_M_2300_3500/RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15-v1/MINIAODSIM'),
        ('dy3500to4500', '/ZToEE_NNPDF31_TuneCP5_13TeV-powheg_M_3500_4500/RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15-v1/MINIAODSIM'),
    ('dy4500to6000', '/ZToEE_NNPDF31_TuneCP5_13TeV-powheg_M_4500_6000/RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15-v1/MINIAODSIM'),
    ('dy6000toInf', '/ZToEE_NNPDF31_TuneCP5_13TeV-powheg_M_6000_Inf/RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15-v1/MINIAODSIM'),]

keys=[]
#for tup in dy_electrons_2018:

#    keys.append(tup[0])
#print(keys)
dataset_list={}
for tup in dy_electrons_2018:

    dataset = tup[1]
    dataset = dataset.split('MiniAOD')[0]+'*/NANOAOD*'
    print(tup[0]) 
    cmd = 'dasgoclient --query==" dataset= %s" '% dataset
    #print(cmd)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,stderr=subprocess.PIPE, shell=True)
    result = proc.stdout.readlines()
    result = [r.rstrip().decode("utf-8") for r in result]
    #result = [r for r in result if 'NanoAODv6' in r]
    print(result)
    dataset_list[tup[0]]=result


print (dataset_list)
with open('dy_electrons_2018datalist.txt', 'a') as output:
    output.write(str(dataset_list))
