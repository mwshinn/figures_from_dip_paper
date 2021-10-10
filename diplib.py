"""Library for the dip project"""


import sqlite3
import numpy as np
import scipy
import paranoid as pns
import pandas
import scipy.signal
import design
import statsmodels.api as sm
import sklearn.decomposition
import seaborn as sns
import ddm
import scipy.signal
from itertools import groupby
import matplotlib.pyplot as plt
import h5py
import pickle

#import matplotlib.pyplot as plt
#import seaborn as sns

CM_DIR = '/home/max/Research_data/leelab/color-match/'
#CM_DIR = '/home/max/Research_data/leelab/color-match/_build'
USE_CACHE = True

# Memoization to speed things up
if USE_CACHE:
    import joblib
    #memoize = joblib.Memory("/home/max/Tmp/colormatch/cachedir2").cache
    memoize = joblib.Memory("/home/max/Tmp/colormatchv4/cachedir").cache
else:
    memoize = lambda x : x


MonkeyType = pns.Set(["Q", "P"])

#################### Display ####################
def get_color(coh=70, ps=None):
    if coh == 50:
        return (.5, .5, .5)
    ind = 0 if ps == 800 else 4 if ps == 400 else 1
    desat = 1 if coh in [70, 63] else .3 if coh in [53, 52] else .6
    pal = ['#1b9e77','#d95f02','#7570b3']
    pal = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33']
    return sns.color_palette(pal, desat=desat)[ind]

def kern_color(kern):
    #pal = sns.color_palette("Set1")
    pal = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628'] 
    if kern in ['E', 'EC', 'sample', 'samp']:
        return pal[2]
    elif kern in ['SI', 'SO', 'SC', 'SIC', 'SOC', 'saccade', 'sac']:
        return pal[3]
    elif kern in ['PH', 'PL', 'presample', 'ps']:
        return pal[6]
    elif kern == 'R1':
        return (.5, .5, 0)
    elif kern == 'R2':
        return (0, .5, .5)

def pc_colors(pc_num):
    #colors = [sns.color_palette("deep")[i] for i in [0, 5, 7, 8]]
    #colors = [plt.cm.gist_earth(i) for i in [.85, .7, .55, .4]]
    colors = [(250/255, 207/255, 224/255), (224/255, 205/255, 255/255), (254/255, 255/255, 162/255), (189/255, 232/255, 239/255)]
    return colors[pc_num]

#################### Loading data ####################

#db = sqlite3.connect(CM_DIR+"colormatch.sqlite")
db = sqlite3.connect(CM_DIR+"colormatch.sqlite")
db.isolation_level = None
dbc = db.cursor()

@pns.accepts(monkey=MonkeyType)
@pns.returns(pandas.DataFrame)
@memoize
def spikes_df(monkey):
    alignstr_pre = "(spike - coalesce(presamp_start, sample_start))" # Presample onset
    alignstr_samp = "(spike - cm_behavior.sample_start)" # Sample onset
    alignstr_sac = "(spike - (cm_behavior.resp_time + coalesce(presamp_start, sample_start)))" # Saccade onset
    
    if monkey == "Q":
        cohs = "(30, 40, 47, 50, 53, 60, 70)"
    if monkey == "P":
        cohs = "(37, 43, 48, 50, 52, 57, 63)"
    
    idstr = "(100*sessions.id + spikestab.cell_id)"
    tridstr = "(100*sessions.id + spikestab.cell_id)*10000 + cm_behavior.trial"
    dbc.execute("""select {alignstr_pre} as spiketime_pre,
                          {alignstr_samp} as spiketime_samp,
                          {alignstr_sac} as spiketime_sac,
                          cm_behavior.resp_time as saccadetime,
                          {idstr} as cellid,
                          sessions.id as session_id,
                          {tridstr} as trialid,
                          cm_behavior.samp_perc_fclr as coh,
                          cm_behavior.presamp_spec_time as ps,
                          cm_behavior.correct as correct,
                          cm_behavior.high_rew as high_rew,
                          cm_behavior.choice_rew as choice_rew,
                          not cm_behavior.choice_rf as choice_in_rf -- We flip this since the coding in the sqlite database coding is unintuitive
                   from spikestab, cm_behavior, neurontab, sessions
                   where sessions.id = spikestab.session_id 
                         and neurontab.session_id = spikestab.session_id and neurontab.cell_id = spikestab.cell_id 
                         and cm_behavior.session_id = spikestab.session_id and cm_behavior.trial = spikestab.trial 
                         and neurontab.valid == 1 and cm_behavior.cm_sac_off is not null
                         and sessions.monkey == '{monkey}'
                         and cm_behavior.samp_perc_fclr in {cohs}
                """.format(alignstr_pre=alignstr_pre, alignstr_samp=alignstr_samp, alignstr_sac=alignstr_sac, idstr=idstr, tridstr=tridstr, cohs=cohs, monkey=monkey))
    
    column_names = [description[0] for description in dbc.description]
    alltrials = pandas.DataFrame(dbc.fetchall(), columns=column_names)
    alltrials['coh'] = np.abs(alltrials['coh']-50)+50
    return alltrials

@pns.accepts(monkey=MonkeyType)
@pns.returns(pandas.DataFrame)
@memoize
def behavior_df(monkey):
    if monkey == "Q":
        cohs = "(30, 40, 47, 50, 53, 60, 70)"
    if monkey == "P":
        cohs = "(37, 43, 48, 50, 52, 57, 63)"

    # There is no trial_id field because our trial_id depends on the cell, whereas these are per-session not per-cell
    dbc.execute("""select cm_behavior.resp_time as saccadetime,
                          cm_behavior.samp_perc_fclr as coh,
                          cm_behavior.presamp_spec_time as ps,
                          cm_behavior.correct as correct,
                          cm_behavior.high_rew as high_rew,
                          cm_behavior.choice_rew as choice_rew,
                          not cm_behavior.choice_rf as choice_in_rf, -- We flip this since the coding in the sqlite database coding is unintuitive
                          cm_behavior.targ_in_radius as target_radius,
                          sessions.name as session_name,
                          sessions.id as session_id,
                          cm_behavior.trial as trial_id
                   from cm_behavior, sessions
                   where     cm_behavior.cm_sac_off is not null
                         and sessions.monkey == '{monkey}'
                         and cm_behavior.session_id == sessions.id
                         and cm_behavior.samp_perc_fclr in {cohs}
                """.format(cohs=cohs, monkey=monkey))
    
    column_names = [description[0] for description in dbc.description]
    alltrials = pandas.DataFrame(dbc.fetchall(), columns=column_names)
    alltrials['coh'] = np.abs(alltrials['coh']-50)+50
    return alltrials

@pns.accepts(session=pns.Number, monkey=MonkeyType)
@pns.returns(pandas.DataFrame)
@memoize
def eyetracking_df(monkey, session):
    alignstr_pre = "(eyetab.time - coalesce(presamp_start, sample_start))" # Presample onset
    alignstr_samp = "(eyetab.time - cm_behavior.sample_start)" # Sample onset
    alignstr_sac = "(eyetab.time - (cm_behavior.resp_time + coalesce(presamp_start, sample_start)))" # Saccade onset
    
    if monkey == "Q":
        cohs = "(30, 40, 47, 50, 53, 60, 70)"
    if monkey == "P":
        cohs = "(37, 43, 48, 50, 52, 57, 63)"
    
    idstr = "100*sessions.id"
    tridstr = "1000000*sessions.id + cm_behavior.trial"
    dbc.execute("""select {alignstr_pre} as time_pre,
                          {alignstr_samp} as time_samp,
                          {alignstr_sac} as time_sac,
                          {tridstr} as trialid,
                          xpos,
                          ypos,
                          cm_behavior.resp_time as saccadetime,
                          cm_behavior.samp_perc_fclr as coh,
                          cm_behavior.presamp_spec_time as ps,
                          cm_behavior.correct as correct,
                          cm_behavior.high_rew as high_rew,
                          not cm_behavior.choice_rf as choice_in_rf -- We flip this since the coding in the sqlite database coding is unintuitive
                   from cm_behavior, sessions, eyetab
                   where sessions.id = eyetab.session_id 
                         and sessions.id == {session}
                         and cm_behavior.session_id = eyetab.session_id
                         and cm_behavior.cm_sac_off is not null
                         and cm_behavior.trial = eyetab.trial
                         and sessions.monkey == '{monkey}'
                         and cm_behavior.samp_perc_fclr in {cohs}
                """.format(alignstr_pre=alignstr_pre, alignstr_samp=alignstr_samp, alignstr_sac=alignstr_sac, tridstr=tridstr, monkey=monkey, session=session, cohs=cohs))
    
    column_names = [description[0] for description in dbc.description]
    alltrials = pandas.DataFrame(dbc.fetchall(), columns=column_names)
    return alltrials

@pns.accepts(monkey=MonkeyType)
@pns.returns(pandas.DataFrame)
@memoize
def cells_df(monkey):
    idstr = "(100*sessions.id + neurontab.cell_id)"
    dbc.execute("""select {idstr} as cell_id,
                          chamber_ap, chamber_ml,
                          penangle, penrad,
                          depth,
                          sac, dsel,
                          channel, unit,
                          name
                   from neurontab, sessions
                   where sessions.id = neurontab.session_id 
                         and neurontab.valid == 1
                         and sessions.monkey == '{monkey}'
                """.format(idstr=idstr, monkey=monkey))
    
    column_names = [description[0] for description in dbc.description]
    alltrials = pandas.DataFrame(dbc.fetchall(), columns=column_names)
    return alltrials

@pns.accepts(monkey=MonkeyType)
@pns.returns(pandas.DataFrame)
@memoize
def lfp_df(monkey):
    if monkey == "Q":
        cohs = "(30, 40, 47, 50, 53, 60, 70)"
    if monkey == "P":
        cohs = "(37, 43, 48, 50, 52, 57, 63)"

    # There is no trial_id field because our trial_id depends on the cell, whereas these are per-session not per-cell
    dbc.execute("""select cm_behavior.resp_time as saccadetime,
                          coalesce(presamp_start, sample_start) as presample_start,
                          sample_start,
                          cm_behavior.samp_perc_fclr as coh,
                          cm_behavior.presamp_spec_time as ps,
                          cm_behavior.correct as correct,
                          cm_behavior.high_rew as high_rew,
                          cm_behavior.choice_rew as choice_rew,
                          lfptab.session_id as session_id,
                          lfptab.trial as trial,
                          lfptab.channel as channel,
                          time_start,
                          time_stop,
                          name as session_name,
                          hdf5_path
                   from cm_behavior, sessions, lfptab
                   where     cm_behavior.cm_sac_off is not null
                         and sessions.monkey == '{monkey}'
                         and lfptab.session_id == sessions.id
                         and lfptab.trial == cm_behavior.trial
                         and cm_behavior.session_id == sessions.id
                         and cm_behavior.samp_perc_fclr in {cohs}
                """.format(cohs=cohs, monkey=monkey))
    
    column_names = [description[0] for description in dbc.description]
    alltrials = pandas.DataFrame(dbc.fetchall(), columns=column_names)
    alltrials['coh'] = np.abs(alltrials['coh']-50)+50
    return alltrials

@pns.accepts(monkey=MonkeyType)
@pns.returns(pandas.DataFrame)
@memoize
def ms_spikes_df(monkey):
    alignstr_target = "(spike - target_start)" # Presample onset
    alignstr_saccade = "(spike - ms_fix_off - resp_time)" # Saccade
    alignstr_memstart = "(spike - ms_mem_start)" # Memory period start
    idstr = "(100*sessions.id + spikestab.cell_id)"
    tridstr = "(100*sessions.id + spikestab.cell_id)*10000 + ms_behavior.trial"
    dbc.execute("""select {alignstr_target} as spiketime_target,
                          {alignstr_saccade} as spiketime_saccade,
                          {alignstr_memstart} as spiketime_memory,
                          resp_time as rt,
                          ms_angle as angle,
                          {idstr} as cellid,
                          {tridstr} as trialid,
                          name
                   from spikestab, ms_behavior, neurontab, sessions
                   where sessions.id = spikestab.session_id 
                         and neurontab.session_id = spikestab.session_id and neurontab.cell_id = spikestab.cell_id 
                         and ms_behavior.session_id = spikestab.session_id and ms_behavior.trial = spikestab.trial 
                         and neurontab.valid == 1 and ms_behavior.tr_good == 1
                         and sessions.monkey == '{monkey}'
                """.format(alignstr_target=alignstr_target, alignstr_saccade=alignstr_saccade, alignstr_memstart=alignstr_memstart, idstr=idstr, tridstr=tridstr, monkey=monkey))
    
    column_names = [description[0] for description in dbc.description]
    alltrials = pandas.DataFrame(dbc.fetchall(), columns=column_names)
    return alltrials

@memoize
@pns.accepts(MonkeyType)
@pns.returns(pns.List(pns.NDArray(t=pns.Number, d=1)))
def get_waveforms(monkey):
    def select_unique(df, key, value):
        res = df.query('%s == %s' % (key, value))
        assert len(res) == 1
        return res.iloc[0]

    cells = get_cell_ids(monkey)
    df = cells_df(monkey)
    mean_waveforms = []
    for cell in cells:
        cell_row = select_unique(df, "cell_id", cell)
        waveform = np.loadtxt(CM_DIR+"/waveforms/%s.swf" % cell_row['name'], delimiter=',')
        waveform = waveform[waveform[:,1]==cell_row['channel'],:]
        waveform = waveform[waveform[:,2]==cell_row['unit'],:]
        # The waveform shifted for this cell half way through the
        # task, so we only use the first 30000 trials.
        if cell == 1101:
            waveform = waveform[0:30000,:]
        mean_waveform = np.mean(waveform[:,6:], axis=0)
        mean_waveforms.append(mean_waveform)
    return mean_waveforms


@memoize
@pns.accepts(monkey=MonkeyType)
def get_cell_ids(monkey):
    spikes = spikes_df(monkey)
    ids = list(sorted(set(spikes['cellid'])))
    if monkey == "P":
        return ids[6:]
    return ids

@pns.accepts(monkey=MonkeyType)
def get_session_ids(monkey):
    cellids = get_cell_ids(monkey)
    return list(sorted(set([cid//100 for cid in cellids])))

#################### Regression model ####################

@memoize
@pns.accepts(MonkeyType, pns.Natural1, model=pns.Set(["default"]))
def _get_regression_model(monkey, cell, model="default"):
    cell_spikes = spikes_df(monkey).query('cellid == %i and saccadetime < 2450' % cell) 
    
    dm = design.DesignMatrix(binsize=25, mintime=-500, maxtime=2500)
    dm.add_regressor(design.RegressorPoint("sample", bins_after=12))
    dm.add_regressor(design.RegressorPointScaled("samplecoh", 12))
    dm.add_regressor(design.RegressorPointScaled("saccade-inrf", bins_before=8, bins_after=0))
    dm.add_regressor(design.RegressorPoint("saccade", bins_before=8, bins_after=0))
    dm.add_regressor(design.RegressorConstant("presample", 1))
    dm.add_regressor(design.RegressorConstantScaled("presample-hr", 1))
    trials = list(sorted(set(cell_spikes['trialid'])))
    mb = design.MatrixBuilder(dm)
    for tr in trials:
        trial_spikes = cell_spikes.query('trialid == %i' % tr)
        spiketimes = trial_spikes['spiketime_pre']
        ps = trial_spikes.iloc[0]['ps']
        coh = trial_spikes.iloc[0]['coh']
        inrf = trial_spikes.iloc[0]['choice_in_rf']
        corr = trial_spikes.iloc[0]['correct']
        hr_in_rf = (corr + inrf + trial_spikes.iloc[0]['high_rew']) % 2 == 1 # Determined through a truth table
        saccadetime = trial_spikes.iloc[0]['saccadetime']
        mb.add_spiketrain({"sample_time": ps if ps != 0 else None, 
                             "samplecoh_time": ps if ps != 0 else None, 
                             "samplecoh_val": np.abs(coh-50)/50,
                             "saccade-inrf_time": saccadetime,
                             "saccade-inrf_val": 1 if inrf else -1,
                             "saccade_time": saccadetime,
                             "presample-hr_val" : 1 if hr_in_rf else -1
                            },
                            spiketimes, trial_start=0, trial_end=saccadetime)
    
    res = sm.OLS(mb.get_y(), mb.get_x()).fit()
    
    total_sum_of_squares = np.sum(np.square(mb.get_y()))
    resid_sum_of_squares = np.sum(np.square(mb.get_y() - res.predict()))
    explained_variance_ratio = 1-(resid_sum_of_squares/total_sum_of_squares)
    res.explained_variance_ratio = explained_variance_ratio
    return {"dm": dm,
            # "res": res,
            # "Y_full": mb.get_y(),
            # "pred": res.predict(),
            "params": res.params,
            "bse": res.bse,
            "ess": res.ess,
            "centered_tss": res.centered_tss,
            "uncentered_tss": res.uncentered_tss,
            "fvalue": res.fvalue,
            "likelihood": res.llf,
            "mse_model": res.mse_model,
            "mse_resid": res.mse_resid,
            "mse_total": res.mse_total,
            "resid": res.resid,
            "condition_number": res.condition_number,
            "rsquared": res.rsquared,
            "explained_variance_ratio": explained_variance_ratio}

@memoize
@pns.accepts(MonkeyType, pns.Natural1, model=pns.Set(["default"]))
def _get_regression_cpds(monkey, cell, model="default"):
    cell_spikes = spikes_df(monkey).query('cellid == %i and saccadetime < 2450' % cell) 
    
    dm = design.DesignMatrix(binsize=25, mintime=-500, maxtime=2500)
    dm.add_regressor(design.RegressorPoint("sample", bins_after=16))
    dm.add_regressor(design.RegressorPointScaled("samplecoh", 16))
    dm.add_regressor(design.RegressorPointScaled("saccade-inrf", bins_before=8, bins_after=0))
    dm.add_regressor(design.RegressorPoint("saccade", bins_before=8, bins_after=0))
    dm.add_regressor(design.RegressorConstant("presample", 1))
    dm.add_regressor(design.RegressorConstantScaled("presample-hr", 1))
    sets = [[], ['samplecoh'], ['samplecoh', 'sample'], ['saccade-inrf'], ['saccade', 'saccade-inrf'], ['presample-hr'], ['presample-hr', 'presample']]
    cpds = []
    for s in sets:
        dm_reduced = dm.skip(s)
        trials = list(sorted(set(cell_spikes['trialid'])))
        mb = design.MatrixBuilder(dm_reduced)
        for tr in trials:
            trial_spikes = cell_spikes.query('trialid == %i' % tr)
            spiketimes = trial_spikes['spiketime_pre']
            ps = trial_spikes.iloc[0]['ps']
            coh = trial_spikes.iloc[0]['coh']
            inrf = trial_spikes.iloc[0]['choice_in_rf']
            corr = trial_spikes.iloc[0]['correct']
            hr_in_rf = (corr + inrf + trial_spikes.iloc[0]['high_rew']) % 2 == 1 # Determined through a truth table
            saccadetime = trial_spikes.iloc[0]['saccadetime']
            mb.add_spiketrain({"sample_time": ps if ps != 0 else None, 
                               "samplecoh_time": ps if ps != 0 else None, 
                               "samplecoh_val": np.abs(coh-50)/50,
                               "saccade-inrf_time": saccadetime,
                               "saccade-inrf_val": 1 if inrf else -1,
                               "saccade_time": saccadetime,
                               "presample-hr_val" : 1 if hr_in_rf else -1
                                },
                                spiketimes, trial_start=0, trial_end=saccadetime)

        res = sm.OLS(mb.get_y(), mb.get_x()).fit()

        total_sum_of_squares = np.sum(np.square(mb.get_y()))
        resid_sum_of_squares = np.sum(np.square(mb.get_y() - res.predict()))
        if len(s) == 0:
            baseline_rss = resid_sum_of_squares
        cpds.append((s, {"rss": resid_sum_of_squares, "tss": total_sum_of_squares}))
    for cpd in cpds:
        cpd[1]['cpd'] = (cpd[1]['rss']-baseline_rss)/cpd[1]['rss']
    return cpds


@pns.accepts(monkey=MonkeyType, model=pns.Set(["default"]))
def get_regression_models(monkey, model="default"):
    cells = get_cell_ids(monkey)
    regs = dict()
    for cell in cells:
        regs[cell] = _get_regression_model(monkey, cell, model)
        regs[cell]['res'] = None
        regs[cell]['Y_full'] = None
        regs[cell]['pred'] = None
        print("Cleaned")
        print("Finished regression for cell", cell)
    return regs

@pns.accepts(monkey=MonkeyType, model=pns.Set(["default"]))
def get_regression_cpds(monkey, model="default"):
    cells = get_cell_ids(monkey)
    regs = dict()
    for cell in cells:
        regs[cell] = _get_regression_cpds(monkey, cell, model)
        print("Finished CPDs for cell", cell)
    return regs

#################### Regression model (saccade control) ####################

@memoize
@pns.accepts(MonkeyType, pns.Natural1, model=pns.Set(["default"]))
def _get_saccade_control_regression_model(monkey, cell, model="default"):
    cell_spikes = spikes_df(monkey).query('cellid == %i and saccadetime < 2300' % cell)

    dm = design.DesignMatrix(binsize=25, mintime=-500, maxtime=2500)
    dm.add_regressor(design.RegressorPoint("sample", bins_after=16))
    dm.add_regressor(design.RegressorPointScaled("samplecoh", 16))
    dm.add_regressor(design.RegressorPoint("saccade-inrf", bins_before=24, bins_after=8))
    dm.add_regressor(design.RegressorPoint("saccade", bins_before=24, bins_after=8))
    dm.add_regressor(design.RegressorPointScaled("saccadecoh", bins_before=30, bins_after=8))
    dm.add_regressor(design.RegressorPointScaled("saccadecoh-inrf", bins_before=30, bins_after=8))
    dm.add_regressor(design.RegressorConstant("presample", 1))
    dm.add_regressor(design.RegressorConstantScaled("presample-hr", 1))
    trials = list(sorted(set(cell_spikes['trialid'])))
    X_full = dm.empty_matrix()
    Y_full = np.asarray([])
    for tr in trials:
        trial_spikes = cell_spikes.query('trialid == %i' % tr)
        spiketimes = trial_spikes['spiketime_pre']
        binned_spikes = dm.bin_spikes(list(spiketimes[spiketimes<2300]))
        ps = trial_spikes.iloc[0]['ps']
        coh = trial_spikes.iloc[0]['coh']
        inrf = trial_spikes.iloc[0]['choice_in_rf']
        corr = trial_spikes.iloc[0]['correct']
        hr_in_rf = (corr + inrf + trial_spikes.iloc[0]['high_rew']) % 2 == 1 # Determined through a truth table
        saccadetime = trial_spikes.iloc[0]['saccadetime']
        X = dm.build_matrix({"sample_time": ps if ps != 0 else None, 
                             "samplecoh_time": ps if ps != 0 else None, 
                             "samplecoh_val": np.abs(coh-50)/50,
                             "saccade-inrf_time": saccadetime if inrf else None,
                             "saccade_time": saccadetime,
                             "saccadecoh-inrf_time": saccadetime if inrf else None,
                             "saccadecoh_time": saccadetime,
                             "saccadecoh-inrf_val": np.abs(coh-50)/50,
                             "saccadecoh_val": np.abs(coh-50)/50,
                             "presample-hr_val" : 1 if hr_in_rf else 0,
           },
                            trial_end=saccadetime)
        X_full = np.concatenate([X_full, X], axis=0)
        Y_full = np.concatenate([Y_full, binned_spikes])
    
    res = sm.OLS(Y_full, X_full).fit()
    
    total_sum_of_squares = np.sum(np.square(Y_full))
    resid_sum_of_squares = np.sum(np.square(Y_full - res.predict()))
    explained_variance_ratio = 1-(resid_sum_of_squares/total_sum_of_squares)
    res.explained_variance_ratio = explained_variance_ratio
    return {"dm": dm,
            "params": res.params,
            "bse": res.bse,
            "ess": res.ess,
            "centered_tss": res.centered_tss,
            "uncentered_tss": res.uncentered_tss,
            "fvalue": res.fvalue,
            "likelihood": res.llf,
            "mse_model": res.mse_model,
            "mse_resid": res.mse_resid,
            "mse_total": res.mse_total,
            "resid": res.resid,
            "condition_number": res.condition_number,
            "rsquared": res.rsquared,
            "explained_variance_ratio": explained_variance_ratio}

@pns.accepts(monkey=MonkeyType, model=pns.Set(["default"]))
def get_saccade_control_regression_models(monkey, model="default"):
    cells = get_cell_ids(monkey)
    regs = dict()
    for cell in cells:
        regs[cell] = _get_saccade_control_regression_model(monkey, cell, model)
        print("Finished regression for cell", cell)
    return regs

#################### Cell properties ####################

@pns.accepts(monkey=MonkeyType, regressor=pns.Set(["sample", "sample-nocoh", "presample"]), noncentered=pns.Boolean)
def get_pcs_noncentered(monkey, regressor="sample", noncentered=True):
    if noncentered:
        pca = sklearn.decomposition.TruncatedSVD(n_components=4)
    else:
        raise NotImplementedError("We're not doing this anymore, remember?")
        pca = sklearn.decomposition.PCA(n_components=10)
    cells = get_cell_ids(monkey)
    regs = get_regression_models(monkey)
    example_cell = cells[0]
    dm = regs[example_cell]['dm']
    if regressor == "sample":
        times = dm.get_regressor_from_output("samplecoh", regs[example_cell]['params'])[0]
        inds = (times >= 0) & (times <= 300)
        ds = np.asarray([dm.get_regressor_from_output("samplecoh", regs[cellid]['params'])[1][inds] for cellid in cells])
    elif regressor == "sample-nocoh":
        times = dm.get_regressor_from_output("sample", regs[example_cell]['params'])[0]
        inds = (times >= 0) & (times <= 300)
        ds = np.asarray([dm.get_regressor_from_output("sample", regs[cellid]['params'])[1][inds] for cellid in cells])
    elif regressor == "presample":
        times = dm.get_regressor_from_output("presample", regs[example_cell]['params'])[0]
        inds = (times >= 0) & (times <= 300)
        ds = np.asarray([dm.get_regressor_from_output("presample", regs[cellid]['params'])[1][inds] for cellid in cells])
        ds = (ds - np.mean(ds[:,0:4], axis=1, keepdims=True))
    trans = pca.fit_transform(ds)
    return (pca.singular_values_,  pca.components_.T, trans, times[inds])

@pns.accepts(monkey=MonkeyType, start_time=pns.Natural0, end_time=pns.Natural0)
def get_dip_score(monkey, start_time=125, end_time=175):
    regs = get_regression_models(monkey)
    cells = get_cell_ids(monkey)
    example_cell = cells[0]
    # Rather than selecting times as indices for the dip function, do
    # it in a more general way that will work if the timestep is
    # changed.
    times = regs[example_cell]['dm'].get_regressor_from_output("samplecoh", regs[example_cell]['params'])[0]
    keys_region = (times >= 0) & (times <= 300)
    keys_dip = (times >= start_time) & (times <= end_time)
    dip_score = lambda x : (-np.mean(x[keys_dip]) + np.mean(x[keys_region]))/np.std(x[keys_region])
    return [dip_score(regs[c]["dm"].get_regressor_from_output("samplecoh", regs[c]["params"])[1]) for c in cells]

@pns.accepts(monkey=MonkeyType, start_time=pns.Natural0, end_time=pns.Natural0)
def get_presample_dip_score(monkey, start_time=100, end_time=150):
    regs = get_regression_models(monkey)
    cells = get_cell_ids(monkey)
    example_cell = cells[0]
    times = regs[example_cell]['dm'].get_regressor_from_output("presample", regs[example_cell]['params'])[0]
    # Rather than selecting times as indices for the dip function, do
    # it in a more general way that will work if the timestep is
    # changed.
    keys_region = (times >= 0) & (times <= 300)
    keys_dip = (times >= start_time) & (times <= end_time)
    dip_score = lambda x : (-np.mean(x[keys_dip]) + np.mean(x[keys_region]))/np.std(x[keys_region])
    # Find the average between the high reward coefficients and the low reward coefficients
    reg_ts = lambda name,c : regs[c]["dm"].get_regressor_from_output(name, regs[c]["params"])[1]
    return [dip_score(reg_ts("presample",c)) for c in cells]

@pns.accepts(monkey=MonkeyType, period=pns.Set(["fixation", "iti"]))
def get_cell_fr(monkey, period="fixation"):
    if period == "fixation":
        time_window = (-1100, -800)
    elif period == "iti":
        time_window = (-2000, -1500)
    cells = get_cell_ids(monkey)
    frs = []
    for cell in cells:
        activity = get_cell_conditional_activity(monkey=monkey, cellid=cell, smooth=0, time_range=time_window, align="presample")[1]
        frs.append(np.mean(activity))
    return frs

@pns.accepts(monkey=MonkeyType)
def get_vm_index(monkey):
    # VM index computed as in
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2804409/ but only
    # for trials in the RF.  We reverse it because they use a
    # different direction here than they use in their other papers.
    # We do NOT use
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2657052/ because it
    # makes no sense.
    ms_spikes = ms_spikes_df(monkey)
    cells = get_cell_ids(monkey)
    vm_indices = []
    for cell in cells:
        cell_spikes = ms_spikes.query(f"cellid == {cell}")
        n_trials = len(set(cell_spikes['trialid']))
        angles = list(sorted(set(cell_spikes['angle'])))
        rf = None # Angle of the RF
        rf_fr = -np.inf # Firing rate of the RF
        # Detect the RF as the highest FR during the memory period
        for angle in angles:
            angle_spikes = cell_spikes.query(f"angle == {angle} and 0 <= spiketime_memory and spiketime_memory <= 1000")
            n_trials_angle = len(set(angle_spikes['trialid']))
            rate_angle = len(angle_spikes)/n_trials_angle if n_trials_angle > 0 else 0
            if rate_angle == 0:
                print(f"Warning, FR is 0 for angle {angle} in cell {cell}")
            if rate_angle > rf_fr:
                rf_fr = rate_angle
                rf = angle
        # Find the quantities from the "Data Analysis" section of
        # Ray, ..., Schall (2009, J Neurophysiology)
        angle_spikes = cell_spikes.query(f"angle == {angle}")
        n_trials_angle = len(set(angle_spikes['trialid']))
        V = len(angle_spikes.query("50 <= spiketime_target and spiketime_target <= 200"))
        M = len(angle_spikes.query("-100 <= spiketime_saccade and spiketime_saccade <= 50"))
        vm_indices.append((V-M)/(V+M) if V+M > 0 else 0)
    return vm_indices
        
@pns.accepts(monkey=MonkeyType)
def get_dsi_index(monkey):
    # DSI index (sic) computed as the maximum selectivity from target
    # onset, saccade, and delay.  It is defined as in
    # https://onlinelibrary.wiley.com/doi/full/10.1111/j.0953-816X.2003.03130.x
    # under "Data analysis"
    #
    # Really, DSI would make MUCH more sense as mutual information...
    ms_spikes = ms_spikes_df(monkey)
    cells = get_cell_ids(monkey)
    dsis = []
    for cell in cells:
        print("cell", cell)
        cell_spikes = ms_spikes.query(f"cellid == {cell}")
        n_trials = len(set(cell_spikes['trialid']))
        angles = list(sorted(set(cell_spikes['angle'])))
        targets = []
        saccades = []
        memories = []
        # If we didn't do the memory saccade for this neuron, set DSI to 0.
        if len(angles) == 0:
            dsis.append(0)
            print("Warning, no memory saccade, DSI set to 0")
            continue
        for angle in angles:
            angle_spikes = cell_spikes.query(f"angle == {angle}")
            n_trials_angle = len(set(angle_spikes['trialid']))
            target = len(angle_spikes.query("50 <= spiketime_target and spiketime_target <= 200"))/150/n_trials_angle*1000
            saccade = len(angle_spikes.query("-100 <= spiketime_saccade and spiketime_saccade <= 50"))/150/n_trials_angle*1000
            memory = len(angle_spikes.query(f"0 <= spiketime_memory and spiketime_memory <= 1000"))/1000/n_trials_angle*1000
            targets.append(target)
            saccades.append(saccade)
            memories.append(memory)
        dsi_target = (len(targets) - np.sum(targets)/np.max(targets))/(len(targets)-1)
        dsi_saccade = (len(saccades) - np.sum(saccades)/np.max(saccades))/(len(saccades)-1)
        dsi_memory = (len(memories) - np.sum(memories)/np.max(memories))/(len(memories)-1)
        dsis.append(dsi_memory)
    return dsis

@pns.accepts(monkey=MonkeyType)
def get_peak_to_peak(monkey):
    # TODO I don't know the units here
    waveforms = get_waveforms(monkey)
    extrema = []
    for wf in waveforms:
        # I played around with this a while back and found it works
        # reasonably well.  I don't necessarily remember why it works
        # though.
        proto_minima = scipy.signal.find_peaks(-wf, width=3)[0]
        proto_peaks = scipy.signal.find_peaks(wf, width=3)[0]
        if len(proto_minima) == 1:
            minimum = proto_minima[0]
        elif len(proto_minima) == 2:
            minimum = proto_minima[0] if abs(proto_minima[0]-10) < abs(proto_minima[1]-10) else proto_minima[1]
        else:
            raise ValueError("Minimum error")
        if len(proto_peaks) == 1:
            peak = proto_peaks[0]
        elif len(proto_peaks) > 1 and max(proto_peaks) > minimum:
            peak = proto_peaks[next(i for i in range(0, len(proto_peaks)) if proto_peaks[i] > minimum)]
        else:
            raise ValueError("Peak error")
        extrema.append((minimum, peak))
    return list(map(lambda x : np.abs(x[1]-x[0]), extrema))
        

#################### Activity/RT traces ####################

@memoize
@pns.accepts(coh=pns.Maybe(pns.Natural0), ps=pns.Maybe(pns.Natural0), choice_in_rf=pns.Maybe(pns.Boolean), hr_choice=pns.Maybe(pns.Boolean),
             correct=pns.Maybe(pns.Boolean), hr_in_rf=pns.Maybe(pns.Boolean), hr_correct=pns.Maybe(pns.Boolean),
             align=pns.Set(["sample", "presample", "saccade"]),
             time_range=pns.Tuple(pns.Integer, pns.Integer), cellid=pns.Maybe(pns.Natural1), monkey=MonkeyType)
def _get_cell_conditional_activity(monkey=None, coh=None, ps=None, choice_in_rf=None, hr_in_rf=None, correct=None, hr_correct=None, hr_choice=None, align="sample", time_range=(-1000,1000), cellid=None):
    cells = get_cell_ids(monkey)
    spikes = spikes_df(monkey)
    example_cell = cells[0]
    rt_select = (time_range[0]-500, time_range[1]+500)
    if align == "presample":
        align_key = "spiketime_pre"
    elif align == "sample":
        align_key = "spiketime_samp"
    elif align == "saccade":
        align_key = "spiketime_sac"
    # Stepwise build the sample
    if cellid is not None:
        spikes = spikes.query(f"cellid == {cellid}")
    if coh is not None:
        spikes = spikes.query(f"coh == {coh}")
    if ps is not None:
        spikes = spikes.query(f"ps == {ps}")
    if correct is not None:
        spikes = spikes.query(f"correct == {correct}")
    if hr_correct is not None:
        spikes = spikes.query(f"high_rew == {hr_correct}")
    if hr_choice is not None:
        spikes = spikes.query(f"choice_rew == {hr_choice}")
    if hr_in_rf is not None:
        spikes = spikes.query(f"(choice_rew == choice_in_rf) == {hr_in_rf}")
    if choice_in_rf is not None:
        spikes = spikes.query(f"choice_in_rf == {choice_in_rf}")
    if ps is not None or align == "sample": # 50% coherence trials don't have a valid presample duration
        spikes = spikes[spikes['coh']!=50]
    spikes = spikes.query(f"{align_key} >= {rt_select[0]} and {align_key} <= {rt_select[1]}")
    spikes['aligned_rt'] = spikes[align_key]
    return spikes


@pns.accepts(time_range=pns.Tuple(pns.Integer, pns.Integer),
             smooth=pns.Natural0, zscore=pns.Or(pns.Boolean, pns.String))
def get_cell_conditional_activity(time_range=(-1000,1000), smooth=5, zscore=False, **kwargs):
    TIMESTEP = 20
    rt_select = (time_range[0]-100, time_range[1]+100)
    spikes = _get_cell_conditional_activity(time_range=rt_select, **kwargs)
    ntrials = len(set(spikes['trialid']))
    times_bin = np.arange(rt_select[0], rt_select[1]+.1, TIMESTEP)
    times = times_bin[0:-1] + TIMESTEP/2
    ts = np.histogram(spikes['aligned_rt'], bins=times_bin, density=False)[0]/ntrials*(1000/TIMESTEP)
    inds = (time_range[0] <= times) & (times <= time_range[1]) # Cut off the ends
    if zscore is not False:
        if 'cellid' in kwargs.keys():
            mu,sigma = get_zscore_coefs(monkey=kwargs['monkey'], cellid=kwargs['cellid'], method=zscore)
        else:
            raise NotImplementedError("Z-score for population activity")
        ts = (ts - mu)/sigma
    if smooth != 0:
        ts = scipy.signal.savgol_filter(ts, smooth, 1)
    return (times[inds]/1000, ts[inds])

@pns.accepts(time_range=pns.Tuple(pns.Integer, pns.Integer),
             timebin=pns.Natural1, zscore=pns.Or(pns.Boolean, pns.String))
def get_cell_conditional_activity_by_trial(time_range=(-1000,1000), timebin=20, zscore=False, **kwargs):
    TIMESTEP = timebin
    spikes = _get_cell_conditional_activity(time_range=time_range, **kwargs)
    times_bin = np.arange(time_range[0], time_range[1]+.1, TIMESTEP)
    times = times_bin[0:-1] + TIMESTEP/2
    trials = list(sorted(set(spikes['trialid'])))
    tss = [np.histogram(spikes.query(f'trialid == {trid}')['aligned_rt'], bins=times_bin, density=False)[0] for trid in trials]
    tss = np.asarray(tss).squeeze()
    if zscore is not False:
        if 'cellid' in kwargs.keys():
            mu,sigma = get_zscore_coefs(monkey=kwargs['monkey'], cellid=kwargs['cellid'], method=zscore)
        else:
            raise NotImplementedError("Z-score for population activity")
        tss = (tss - mu)/sigma
    return (times/1000, tss)

@pns.accepts(rtbin=pns.Or(pns.Tuple(pns.Range(0,1), pns.Range(0,1)), pns.Tuple(pns.Integer, pns.Integer)), time_range=pns.Tuple(pns.Integer, pns.Integer),
             smooth=pns.Natural0, zscore=pns.Or(pns.Boolean, pns.String))
def get_cell_conditional_activity_by_rtbin(rtbin=(0, .5), time_range=(-1000,1000), smooth=5, zscore=False, **kwargs):
    TIMESTEP = 20
    rt_select = (time_range[0]-100, time_range[1]+100)
    spikes = _get_cell_conditional_activity(time_range=rt_select, **kwargs)
    rts = np.asarray([g[0] for g in groupby(spikes['spiketime_samp'] - spikes['spiketime_sac'])])
    if isinstance(rtbin[0], int) and isinstance(rtbin[1], int): # If integers, use these as the rt bins
        rtsplit = rtbin
        print("Is int")
    elif rts != []: # If floats between 0 and 1, use as quantiles.  Guard against quantile function throwing an error if rts is empty.
        rtsplit = np.quantile(rts, rtbin)
    else: # Doesn't matter here since rts empty, dummy value
        rtsplit = (-1e9, 1e9)
    spikes = spikes.query(f"spiketime_samp-spiketime_sac >= {rtsplit[0]} and spiketime_samp-spiketime_sac <= {rtsplit[1]}")
    ntrials = len(set(spikes['trialid']))
    times_bin = np.arange(rt_select[0], rt_select[1]+.1, TIMESTEP)
    times = times_bin[0:-1] + TIMESTEP/2
    ts = np.histogram(spikes['aligned_rt'], bins=times_bin, density=False)[0]/ntrials*(1000/TIMESTEP)
    inds = (time_range[0] <= times) & (times <= time_range[1]) # Cut off the ends
    if zscore is not False:
        if 'cellid' in kwargs.keys():
            mu,sigma = get_zscore_coefs(monkey=kwargs['monkey'], cellid=kwargs['cellid'], method=zscore)
        else:
            raise NotImplementedError("Z-score for population activity")
        ts = (ts - mu)/sigma
    if smooth != 0:
        ts = scipy.signal.savgol_filter(ts, smooth, 1)
    return (times[inds]/1000, ts[inds])

@memoize
@pns.accepts(coh=pns.Maybe(pns.Natural0), ps=pns.Maybe(pns.Natural0), hr_choice=pns.Maybe(pns.Boolean),
             correct=pns.Maybe(pns.Boolean), hr_correct=pns.Maybe(pns.Boolean),
             align=pns.Set(["sample", "presample"]), time_range=pns.Tuple(pns.Integer, pns.Integer),
             monkey=MonkeyType)
def _get_rt_conditional_activity(monkey, coh=None, ps=None, correct=None, hr_correct=None, hr_choice=None, align="sample", time_range=(0, 2000)):
    spikes = spikes_df(monkey)
    # get the first entry for each spike, which is sufficient for the rt
    spikes = spikes[np.append([True], (np.asarray(spikes['saccadetime'][0:-1]) != np.asarray(spikes['saccadetime'][1:])))]
    # Stepwise build the sample
    if coh is not None:
        spikes = spikes.query(f"coh == {coh}")
    if ps is not None:
        spikes = spikes.query(f"ps == {ps}")
    if correct is not None:
        spikes = spikes.query(f"correct == {correct}")
    if hr_correct is not None:
        spikes = spikes.query(f"high_rew == {hr_correct}")
    if hr_choice is not None:
        spikes = spikes.query(f"choice_rew == {hr_choice}")
    if ps is not None or align == "sample": # 50% coherence trials don't have a valid presample duration
        spikes = spikes[spikes['coh']!=50]
    ps_dur = spikes['spiketime_pre'] - spikes['spiketime_samp']
    if align == "presample":
        rts = spikes['saccadetime']
    elif align == "sample":
        rts = spikes['saccadetime'] - ps_dur
    return rts

@pns.accepts(monkey=MonkeyType, time_range=pns.Tuple(pns.Integer, pns.Integer),
             smooth=pns.Natural0, resample=pns.Maybe(pns.Integer), binsize=pns.Natural1)
def get_rt_conditional_activity(monkey, smooth=5, time_range=(0, 2000), resample=False, binsize=20, **kwargs):
    rts = _get_rt_conditional_activity(monkey, time_range=time_range, **kwargs)
    if resample is not None: # For bootstrapping
        rng = np.random.RandomState(resample)
        rts = rts.iloc[rng.choice(len(rts), len(rts), replace=True)]
    ntrials = len(rts)
    times_bin = np.arange(-1000, 3000+binsize/100, binsize)
    times = times_bin[0:-1] + binsize/2
    ts = np.histogram(rts, bins=times_bin, density=False)[0]/ntrials*(1000/binsize)
    if smooth != 0:
        ts = scipy.signal.savgol_filter(ts, smooth, 1)
    # Cut off the ends
    inds = (time_range[0] <= times) & (times <= time_range[1])
    return (times[inds]/1000, ts[inds])


#################### Microsaccades ####################

@pns.accepts(time=pns.NDArray(d=1, t=pns.Number), xpos=pns.NDArray(d=1, t=pns.Integer), ypos=pns.NDArray(d=1, t=pns.Integer),
             lmbda=pns.Number, dt=pns.Natural1, min_sac_len=pns.Natural1)
@pns.returns(pns.Tuple(pns.List(pns.Number), pns.List(pns.Positive0)))
@pns.requires("time.shape == xpos.shape")
@pns.requires("time.shape == ypos.shape")
@pns.ensures("all(t in time for t in return[0])")
def detect_microsaccades(time, xpos, ypos, lmbda=6, dt=2, min_sac_len=3):
    xvel = (xpos[:-4] + xpos[1:-3] - xpos[3:-1] - xpos[4:])/(6*dt)
    yvel = (ypos[:-4] + ypos[1:-3] - ypos[3:-1] - ypos[4:])/(6*dt)
    speed = np.sqrt(xvel**2 + yvel**2)
    # Am I supposed to take the square root of the next line?  Engbert
    # and Kliegl (2003) is not clear.  I am making decisions here
    # based on the interpretation in the GazeParser software package.
    x_sigma = np.sqrt(np.median(xvel**2) - np.median(xvel)**2)
    y_sigma = np.sqrt(np.median(yvel**2) - np.median(yvel)**2)
    if x_sigma == 0 or y_sigma == 0:
        print("Warning: velocity was zero")
        return ([], [])
    # Find all indices where the velocity passes some threshold
    inds = np.where(((xvel/(x_sigma*lmbda))**2 + (yvel/(y_sigma*lmbda))**2) > 1)[0]
    microsaccades = [] # All detected microsaccades
    i_start = 0 # Keep track of start of current putitive microsaccade
    for i in range(1, len(inds)):
        # If the two detected indices are neighbors, this is the same
        # putitive microsaccade as the previous index
        if inds[i]-inds[i-1] <= 1:
            continue
        # Number of indices comprising the current microsaccade
        msac_len = (i - 1) - i_start
        if msac_len > min_sac_len: # Microsaccade found
            microsaccades.append((2+inds[i_start], 2+inds[i-1]))
        i_start = i
    # I should probably check if there was half of a saccade at the
    # end of the timeseries but I don't.
    times = [time[int(np.mean(ms))] for ms in microsaccades]
    distances = [np.sqrt((xpos[ms_start]-xpos[ms_stop])**2 + (ypos[ms_start]-xpos[ms_stop])**2) for ms_start,ms_stop in microsaccades]
    return (times, distances)



@memoize
@pns.accepts(monkey=MonkeyType)
def get_microsaccades_df(monkey):
    sessions = get_session_ids(monkey)
    all_microsaccades = []
    for session in sessions:
        print("Starting session", session)
        edf = eyetracking_df(monkey, session)
        trials = list(sorted(set(edf['trialid'])))
        for tr in trials:
            edf_filt = edf.query('trialid == %i' % tr)
            time_pre = np.asarray(edf_filt['time_pre'])
            time_samp = np.asarray(edf_filt['time_samp'])
            time_sac = np.asarray(edf_filt['time_sac'])
            ps_actual = time_pre[0] - time_samp[0]
            _xpos = np.asarray(edf_filt['xpos'])
            _ypos = np.asarray(edf_filt['ypos'])
            coh = edf_filt.iloc[0]['coh']
            ps = edf_filt.iloc[0]['ps']
            correct = edf_filt.iloc[0]['correct']
            saccadetime = edf_filt.iloc[0]['saccadetime']
            high_rew = edf_filt.iloc[0]['high_rew']
            microsaccades, distances = detect_microsaccades(time_pre, _xpos, _ypos)
            all_microsaccades.extend([(session, tr, ms, ms-ps_actual, coh, ps, correct, saccadetime, high_rew, dist) for ms,dist in zip(microsaccades, distances)])
    
    msdf = pandas.DataFrame(all_microsaccades, columns=["session", "trialid", "microsaccadetime_pre", "microsaccadetime_samp", "coh", "ps", "correct", "saccadetime", "high_rew", "saccade_distance"])
    return msdf

@memoize
@pns.accepts(coh=pns.Maybe(pns.Natural0), ps=pns.Maybe(pns.Natural0), hr_choice=pns.Maybe(pns.Boolean),
             correct=pns.Maybe(pns.Boolean), hr_correct=pns.Maybe(pns.Boolean), max_dist=pns.Maybe(pns.Positive),
             monkey=MonkeyType)
def _get_microsaccade_conditional_activity(monkey, coh=None, ps=None, correct=None, hr_correct=None, hr_choice=None, max_dist=None):
    ms_times = get_microsaccades_df(monkey)
    # Stepwise build the sample
    if coh is not None:
        ms_times = ms_times.query(f"coh == {coh}")
    if ps is not None:
        ms_times = ms_times.query(f"ps == {ps}")
    if correct is not None:
        ms_times = ms_times.query(f"correct == {correct}")
    if hr_correct is not None:
        ms_times = ms_times.query(f"high_rew == {hr_correct}")
    if hr_choice is not None:
        ms_times = ms_times.query(f"choice_rew == {hr_choice}")
    if max_dist is not None:
        ms_times = ms_times.query(f"saccade_distance <= {max_dist}")
    return ms_times

@pns.accepts(binsize=pns.Natural1, time_range=pns.Tuple(pns.Integer, pns.Integer), monkey=MonkeyType,
             smooth=pns.Natural0, resample=pns.Maybe(pns.Integer), align=pns.Set(["sample", "presample"]))
def get_microsaccade_conditional_activity(monkey, smooth=5, binsize=20, resample=None, align="sample", time_range=(0, 2000), **kwargs):
    ms_times = _get_microsaccade_conditional_activity(monkey, **kwargs)
    if 'ps' in kwargs.keys() or align == "sample": # 50% coherence trials don't have a valid presample duration
        ms_times = ms_times[ms_times['coh']!=50]
    if align == "presample":
        rts = ms_times['microsaccadetime_pre']
    elif align == "sample":
        rts = ms_times['microsaccadetime_samp']
    if resample is not None: # For bootstrapping
        rng = np.random.RandomState(resample)
        rts = rts.iloc[rng.choice(len(rts), len(rts), replace=True)]
    ntrials = len(set(ms_times['trialid']))
    times_bin = np.arange(-1000, 3000, binsize)
    times = times_bin[0:-1] + binsize/2
    ts = np.histogram(rts, bins=times_bin, density=False)[0]/ntrials*(1000/binsize)
    if smooth != 0:
        ts = scipy.signal.savgol_filter(ts, smooth, 1)
    # Cut off the ends
    inds = (time_range[0] <= times) & (times <= time_range[1])
    return (times[inds]/1000, ts[inds])

#################### DDM ####################

def get_ddm_model():
    import cleanmodels2 as cm2
    import ddm
    return ddm.Model(name='m1_collapse_leaky_leaktarget_none_lapse', drift=cm2.DriftUrgencyGated(snr=ddm.Fitted(10.115540997901837, minval=0.5, maxval=40), noise=ddm.Fitted(1.0711420216601528, minval=0.01, maxval=4), t1=0, t1slope=0, cohexp=1, maxcoh=70, leak=ddm.Fitted(13.49673532315082, minval=0.01, maxval=40), leaktarget=ddm.Fitted(0.06930768967085771, minval=0, maxval=0.9)), noise=cm2.NoiseUrgency(noise=ddm.Fitted(1.0711420216601528, minval=0.01, maxval=4), t1=0, t1slope=0), bound=cm2.BoundCollapsingExponentialDelay(B=1, tau=ddm.Fitted(0.8786700045333606, minval=0.1, maxval=10), t1=0), IC=cm2.ICPoint(x0=ddm.Fitted(0.06930768967085771, minval=0, maxval=0.9)), overlay=ddm.OverlayNonDecision(nondectime=ddm.Fitted(0.234419916513916, minval=0.1, maxval=0.3)), dx=0.02, dt=0.02, T_dur=3.0)

def get_ddm_model_dv():
    import cleanmodels2 as cm2
    import ddm
    return ddm.Model(name='m1_collapse_leaky_leaktarget_none_lapse', drift=cm2.DriftUrgencyGated(snr=ddm.Fitted(10.115540997901837, minval=0.5, maxval=40), noise=ddm.Fitted(.50711420216601528, minval=0.01, maxval=4), t1=0, t1slope=0, cohexp=1, maxcoh=70, leak=ddm.Fitted(1.49673532315082, minval=0.01, maxval=40), leaktarget=ddm.Fitted(0.4930768967085771, minval=0, maxval=0.9)), noise=cm2.NoiseUrgency(noise=ddm.Fitted(.50711420216601528, minval=0.01, maxval=4), t1=0, t1slope=0), bound=cm2.BoundCollapsingExponentialDelay(B=1, tau=ddm.Fitted(0.8786700045333606, minval=0.1, maxval=10), t1=0), IC=cm2.ICPoint(x0=ddm.Fitted(0.06930768967085771, minval=0, maxval=0.9)), overlay=ddm.OverlayNonDecision(nondectime=ddm.Fitted(0.204419916513916, minval=0.1, maxval=0.3)), dx=0.02, dt=0.02, T_dur=3.0)

@memoize
def get_ddm_conditional_rts(corr=True, coh=None, ps=None, highreward=None, time_range=(-200, 400), align="sample"):
    m = get_ddm_model()
    m.dt = .005
    m.dx = .005
    _coh = coh if coh is not None else [53, 60, 70]
    _ps = ps if ps is not None else [0, 400, 800]
    _highreward = highreward if highreward is not None else [0, 1]
    s = ddm.solve_partial_conditions(m, conditions={"coherence": _coh, "presample": _ps, "highreward": _highreward})
    pdf = s.pdf_corr() if corr == True else s.pdf_err()
    times = m.t_domain()
    if align == "sample":
        if ps is not None:
            times -= ps/1000
        else:
            raise ValueError("Cannot align to sample with multiple presample durations")
    inds = (time_range[0]/1000 <= times) & (times <= time_range[1]/1000)
    return times[inds], pdf[inds]
    

def get_ddm_mean_activity(coh, ps, highreward, time_range, align="sample"):
    m = get_ddm_model_dv()
    # This technically doesn't take into consideration RTs which have
    # already passed the bound... it should be an okay approximation
    # for the short timescales it is used for though.
    nondec = m.get_dependence("overlay").nondectime
    m.get_dependence("overlay").nondectime = 0
    conds = {"coherence": coh, "presample": ps, "highreward": highreward}
    s = m.solve(conditions=conds, return_evolution=True)
    scaling = 1 - (s.corr + s.err)
    ts = np.mean(s.pdf_evolution()*np.reshape(m.x_domain(conditions=conds), (-1, 1)), axis=0) * scaling + np.cumsum(s.corr) - np.cumsum(s.err)
    t_domain = s.model.t_domain()
    inds = (t_domain+nondec >= (time_range[0]+ps)/1000) & (t_domain+nondec <= (time_range[1]+ps)/1000)
    if align == "sample":
        return t_domain[inds]+nondec-ps/1000, ts[inds]

def get_ddm_conditional_activity(**kwargs):
    by_trial = get_ddm_conditional_activity_by_trial(**kwargs)
    timestep = by_trial[0][1] - by_trial[0][0]
    return by_trial[0], np.mean(by_trial[1], axis=0)/timestep

@memoize
def get_ddm_conditional_activity_by_trial(coh, ps, highreward, trials=100, seed=0, time_range=(0, 1000), align="sample"):
    m = get_ddm_model_dv() 
    ndtime = m._overlay.nondectime
    m._overlay = ddm.OverlayNone()
    trajs = []
    for i in range(0, trials):
        # _coh = coh if coh is not None else np.random.choice([53, 60, 70])
        # _ps = ps if ps is not None else np.random.choice([0, 400, 800])
        # _highreward = highreward if highreward is not None else np.random.choice([0, 1])
        traj = m.simulate_trial(conditions={"coherence": coh, "presample": ps, "highreward": highreward}, cutoff=False, seed=seed+i*1000)
        trajs.append(traj)
    t_domain = m.t_domain()+ndtime
    if align == "sample":
        t_domain -= ps/1000
    inds = (time_range[0]/1000 <= t_domain) & (t_domain <= time_range[1]/1000)
    timestep = t_domain[1] - t_domain[0]
    return t_domain[inds], np.asarray(trajs)[:,inds]*timestep
    
    

############################# Analysis #############################

def bootstrap_trialwise_ci(monkey, N=1000, seed=0, func="data", smooth=0, **kwargs):
    if func == "trial":
        T, _trials = get_cell_conditional_activity_by_trial(monkey=monkey, **kwargs)/timestep
        timestep = T[1]-T[0]
        trials = np.asarray(_trials)/timestep
    elif func == "data":
        _trials = []
        for c in get_cell_ids(monkey):
            try:
                T, _activity = get_cell_conditional_activity(monkey=monkey, cellid=c, smooth=0, **kwargs)
            except FloatingPointError:
                print("ERROR accessing cell", c)
                continue
            _trials.append(_activity)
        trials = np.asarray(_trials)
    elif func == "ddm":
        T, _trials = get_ddm_conditional_activity_by_trial(**kwargs, seed=seed)
        timestep = T[1]-T[0]
        trials = np.asarray(_trials)/timestep
    ntrials = len(_trials)
    if smooth != 0:
        for i in range(0,ntrials):
            trials[i,:] = scipy.signal.savgol_filter(trials[i,:], smooth, 1)
    print("bootstrap trials", ntrials)
    print("Trials shape", trials.shape)
    samps = []
    rng = np.random.RandomState(seed)
    for i in range(0, N):
        inds = rng.choice(list(range(0, len(trials))), len(trials))
        traj = np.mean(trials[inds,:], axis=0)
        samps.append(traj)
    #import matplotlib.pyplot as plt
    #plt.plot(np.asarray(samps).T)
    #plt.show()
    return np.quantile(samps, [.025, .975], axis=0)

def bootstrap_significance(monkey, params1, params2, N=1000, seed=0, func="data"):
    if func == "trial":
        T1, _trials1 = get_cell_conditional_activity_by_trial(monkey=monkey, **params1)
        T2, _trials2 = get_cell_conditional_activity_by_trial(monkey=monkey, **params2)
    elif func == "data":
        _trials1 = []
        _trials2 = []
        for c in get_cell_ids(monkey):
            try:
                T1, _activity1 = get_cell_conditional_activity(monkey=monkey, cellid=c, smooth=0, **params1)
                T2, _activity2 = get_cell_conditional_activity(monkey=monkey, cellid=c, smooth=0, **params2)
            except FloatingPointError:
                print("ERROR accessing cell", c)
                continue
            _trials1.append(_activity1)
            _trials2.append(_activity2)
        trials1 = np.asarray(_trials1)
        trials2 = np.asarray(_trials2)
    elif func == "ddm":
        T1, _trials1 = get_ddm_conditional_activity_by_trial(**params1)
        T2, _trials2 = get_ddm_conditional_activity_by_trial(**params2)
    assert len(T1) == len(T2), "Different timesteps"
    timestep = T1[1]-T1[0]
    samps1 = []
    samps2 = []
    rng = np.random.RandomState(seed)
    for i in range(0, N):
        inds = rng.choice(list(range(0, len(trials1))), len(trials1))
        traj = np.mean(trials1[inds,:], axis=0)
        samps1.append(traj)
    for i in range(0, N):
        inds = rng.choice(list(range(0, len(trials2))), len(trials2))
        traj = np.mean(trials2[inds,:], axis=0)
        samps2.append(traj)
    samp_diff = np.asarray(samps1) - np.asarray(samps2)
    #quants = np.quantile(samp_diff, [.025, .975], axis=0)
    #return (quants[0,:]*quants[1,:] > 0).astype(int)
    return np.mean(samp_diff<0, axis=0)

def bootstrap_rts_ci(N=1000, seed=0, smooth=0, **kwargs):
    samps = [get_rt_conditional_activity(resample=i+seed*10000, smooth=smooth, **kwargs)[1] for i in range(0, N)]
    return np.quantile(samps, [.025, .975], axis=0)

def bootstrap_rts_significance(params1, params2, N=1000, seed=0):
    samps1 = [get_rt_conditional_activity(resample=i+seed*10000, smooth=0, **params1)[1] for i in range(0, N)]
    samps2 = [get_rt_conditional_activity(resample=i+seed*10000, smooth=0, **params2)[1] for i in range(0, N)]
    samps1 = np.asarray(samps1)
    samps2 = np.asarray(samps2)
    samps_diff = samps1 - samps2
    return np.mean(samps_diff<=0, axis=0)

@memoize
def bootstrap_microsaccade_ci(N=1000, seed=0, smooth=0, **kwargs):
    samps = [get_microsaccade_conditional_activity(resample=i+seed*10000, smooth=smooth, **kwargs)[1] for i in range(0, N)]
    return np.quantile(samps, [.025, .975], axis=0)

def bootstrap_microsaccade_significance(params1, params2, N=1000, seed=0):
    samps1 = [get_microsaccade_conditional_activity(resample=i+seed*10000, **params1)[1] for i in range(0, N)]
    samps2 = [get_microsaccade_conditional_activity(resample=i+seed*10000, **params2)[1] for i in range(0, N)]
    samps1 = np.asarray(samps1)
    samps2 = np.asarray(samps2)
    samps_diff = samps1 - samps2
    return np.mean(samps_diff<=0, axis=0)

#################### Convenience ####################

@memoize
@pns.accepts(monkey=MonkeyType)
def get_mean_conditional_activity(monkey, **kwargs):
    traces = []
    for c in get_cell_ids(monkey):
        try:
            T, trace = get_cell_conditional_activity(monkey=monkey, cellid=c, **kwargs)
            traces.append(trace)
        except FloatingPointError:
            print("ERROR: Invalid cell", c, "(excluding from mean)")
    return (T, np.mean(traces, axis=0))

@pns.accepts(monkey=MonkeyType, resample=pns.Maybe(pns.Natural0))
def get_mean_conditional_activity_by_rtbin(monkey, resample=None, **kwargs):
    traces = []
    cellids = get_cell_ids(monkey)
    if resample is not None:
        rng = np.random.RandomState(resample)
        cellids = [cellids[i] for i in rng.choice(len(cellids), len(cellids), replace=True)]
    for c in cellids:
        try:
            T, trace = get_cell_conditional_activity_by_rtbin(monkey=monkey, cellid=c, **kwargs)
            traces.append(trace)
        except FloatingPointError:
            print("ERROR: Invalid cell", c, "(excluding from mean)")
    return (T, np.mean(traces, axis=0))

def get_zscore_coefs(monkey, cellid, method):
    if method == "inrf":
        activity = get_cell_conditional_activity(monkey=monkey, align="presample", time_range=(0, 800), ps=800, smooth=0, hr_in_rf=True, cellid=cellid)[1]
    elif method == "all":
        activity = get_cell_conditional_activity(monkey=monkey, align="presample", time_range=(0, 800), ps=800, smooth=0, cellid=cellid)[1]
    elif method == "inrf_subtract":
        activity = get_cell_conditional_activity(monkey=monkey, align="presample", time_range=(0, 800), ps=800, smooth=0, hr_in_rf=True, cellid=cellid)[1]
        return (np.mean(activity), 1)
    elif method == "outrf_subtract":
        activity = get_cell_conditional_activity(monkey=monkey, align="presample", time_range=(0, 800), ps=800, smooth=0, hr_in_rf=False, cellid=cellid)[1]
    elif method == "bothrf_subtract":
        activity = get_cell_conditional_activity(monkey=monkey, align="presample", time_range=(0, 800), ps=800, smooth=0, cellid=cellid)[1]
        return (np.mean(activity), 1)
    return (np.mean(activity), np.std(activity))


#################### Plotting ####################

def plot_significance(canvas, axname, points, dx, height=.8):
    from canvas import Point
    segments = []
    p = points
    if len(points) == 0: return # If there are no significant points
    current_segment = p[0]
    i = -1 # If there is only one significant point
    for i in range(0, len(points)-1):
        if p[i+1] - p[i] < dx*1.001:
            continue
        else:
            segments.append((current_segment-dx/2, p[i]+dx/2))
            current_segment = p[i+1]
    segments.append((current_segment-dx/2, p[i+1]+dx/2))
    for seg in segments:
        start = Point(seg[0], 0, axname) >> Point(0, height, "axis_"+axname)
        stop = Point(seg[1], 0, axname) >> Point(0, height, "axis_"+axname)
        canvas.add_line(start, stop, c='k', lw=3)


def make_gridlegend(c, pos, shorten=False, zero=True, forrnn=False):
    from canvas import Vector, Width
    cohs = {0: 50, 1: 53, 2: 60, 3: 70}
    pss = {0: 0, 1: 400, 2: 800}
    psslabel = {0: 0, 1: 400, 2: 800} if not forrnn else  {0: 0, 1: 200, 2: 400}
    coh_label = {50: "Zero", 53: "Low", 60: "Middle", 70: "High"}
    spacing = Vector(.3, .15, "absolute")
    linelen = Width(.15, "absolute")
    toptextoffset = Vector(0, -.05, "absolute")
    titleoffset = Vector(0, .15, "absolute")
    for ypos in [0, 1, 2, 3]:
        if ypos == 0 and zero is False: continue
        for xpos in [0, 1, 2]:
            coh = cohs[ypos]
            ps,pslabel = pss[xpos], psslabel[xpos]
            basepos = pos + spacing.width()*xpos + spacing.height()*ypos
            if coh == 50 and ps != 0:
                continue
            print(basepos)
            if ypos != 0:
                c.add_line(basepos, basepos+linelen, c=get_color(coh, ps), lw=5)
            else: # Zero coh
                c.add_line(basepos, basepos+linelen+spacing.width()*2, c=get_color(coh, ps), lw=5)
            if ypos == 3:
                c.add_text(f"{pslabel}", basepos+spacing.height()+linelen/2+toptextoffset, horizontalalignment="center", verticalalignment="bottom")
                if xpos == 2:
                    c.add_text("Coherence" if not shorten else "Coh.", basepos+spacing, horizontalalignment="left", weight="bold")
                if xpos == 1:
                    c.add_text("Presample (ms)" if not forrnn else "Presample (steps)", basepos+spacing.height()+linelen/2+titleoffset, horizontalalignment="center", weight="bold")
        c.add_text(coh_label[coh], basepos+spacing.width(), horizontalalignment="left")

def scientific_notation(n, sigfigs=0, round_direction="auto"):
    """Format number `n` in scientific notation
    
    Utility function to transform the number `n` into the string
    "$10^k$" for some k.  
    
    We implement this manually instead of using
    ticker._formatSciNotation from matplotlib in order to gain more
    control over sigfigs and rounding.
    """
    import re
    import math
    numstr = '%1.10e' % n # Example: '-3.1415926536e-02'
    if numstr[0] == '-': # Remove the sign if there is a negative sign
        significand_sign = '-'
        numstr = numstr[1:]
    else:
        significand_sign = '+'
    # Separate parts before and after decimal point and the exponent (and its sign)
    wholenum, part2 = numstr.split('.', 2)
    decimal, exponent_plus_sign = part2.split('e')
    exponent_sign = exponent_plus_sign[0]
    exponent = exponent_plus_sign[1:]
    # Make sure all four parts are as we expect
    assert len(wholenum) == 1, "Invalid decimal format"
    assert exponent_sign in ['+', '-'], "Invalid exponent sign"
    assert str(int(wholenum)) == wholenum, "Invalid whole number"
    #assert str(int(exponent)) == exponent, f"Invalid exponent: {exponent}"
    #assert str(int(decimal)) == decimal, f"Invalid decimal: {decimal}"
    if n*100 > 1:
        return "%.2f" % ((math.ceil(n*100) if round_direction == "up" else math.floor(n*100) if round_direction == "down" else round(n*100))/100)
    if sigfigs == 0: # Only display 10^k
        print(significand_sign+wholenum+"."+decimal)
        if round_direction == "up":
            final_exponent = int(exponent_sign+exponent) + int(math.ceil(float(significand_sign+wholenum+"."+decimal)/10))
        elif round_direction == "down":
            final_exponent = int(exponent_sign+exponent) + int(math.floor(float(significand_sign+wholenum+"."+decimal)/10))
        elif round_direction == "auto":
            final_exponent = int(exponent_sign+exponent) + int(round(float(significand_sign+wholenum+"."+decimal)/10))
        return ('-' if significand_sign == '-' else '') +"10^{"+str(final_exponent)+"}"
    if sigfigs > 0:
        pass # TODO
        
    # f = matplotlib.ticker.ScalarFormatter(useOffset=False, useMathText=True)
    # g = lambda x,pos : "${}$".format(f._formatSciNotation())
    # fmt = matplotlib.ticker.FuncFormatter(g)
    # return fmt(n)

@pns.accepts(coh=pns.Maybe(pns.Natural0), ps=pns.Maybe(pns.Natural0), hr_choice=pns.Maybe(pns.Boolean),
             correct=pns.Maybe(pns.Boolean), hr_correct=pns.Maybe(pns.Boolean), smooth=pns.Natural0,
             session_id=pns.Maybe(pns.Natural0), align=pns.Set(["sample", "presample", "saccade"]), zscore=pns.Boolean,
             channel=pns.Maybe(pns.Natural0), trial=pns.Maybe(pns.Natural0),
             time_range=pns.Tuple(pns.Integer, pns.Integer), session_name=pns.Maybe(pns.String), monkey=MonkeyType)
def get_lfp_conditional_activity(monkey=None, coh=None, ps=None, correct=None, hr_correct=None, hr_choice=None, align="sample", time_range=(-1000,1000), session_name=None, session_id=None, trial=None, zscore=True, channel=None, smooth=0):
    cells = get_cell_ids(monkey)
    lfp = lfp_df(monkey)
    example_cell = cells[0]
    rt_select = time_range
    hdf_file = h5py.File(CM_DIR+"colormatch-lfp.hdf5")
    # Stepwise build the sample
    if session_name is not None:
        lfp = lfp.query(f"session_name == '{session_name}'")
    if trial is not None:
        lfp = lfp.query(f"trial == {trial}")
    if session_id is not None:
        lfp = lfp.query(f"session_id == {session_id}")
    if coh is not None:
        lfp = lfp.query(f"coh == {coh}")
    if ps is not None:
        lfp = lfp.query(f"ps == {ps}")
    if correct is not None:
        lfp = lfp.query(f"correct == {correct}")
    if hr_correct is not None:
        lfp = lfp.query(f"high_rew == {hr_correct}")
    if channel is not None:
        lfp = lfp.query(f"channel == {channel}")
    if hr_choice is not None:
        lfp = lfp.query(f"choice_rew == {hr_choice}")
    if ps is not None or align == "sample": # 50% coherence trials don't have a valid presample duration
        lfp = lfp[lfp['coh']!=50]
    lfps = []
    for _,trial in lfp.iterrows():
        if align == "presample":
            alignment_offset = trial['presample_start']
        elif align == "sample":
            alignment_offset = trial['sample_start']
        elif align == "saccade":
            alignment_offset = trial['saccadetime'] + trial['presample_start']
        lfp_vals = hdf_file[trial['hdf5_path']]
        assert len(lfp_vals) > 0
        times = np.asarray(list(range(trial['time_start'], -1000+len(lfp_vals))))
        assert len(times) == len(lfp_vals)
        if np.isnan(trial['saccadetime']):
            print("Trial saccadetime was nan, skipping")
            continue
        inds = (times-alignment_offset >= time_range[0]) & (times-alignment_offset <= time_range[1])
        this_lfp = lfp_vals[inds]
        if smooth != 0:
            this_lfp = scipy.signal.savgol_filter(this_lfp, smooth, 3)
        if zscore:
            lfps.append((this_lfp-np.mean(lfp_vals[0:1000]))/np.std(lfp_vals[0:1000]))
        else:
            lfps.append(this_lfp)
    assert all((len(e)==len(lfps[0])) for e in lfps)
    return lfps


def psave(filename, data, overwrite=True):
    if overwrite == False:
        import os.path
        assert os.path.isfile(filename) == False, "File already exists, can't overwrite"
    with open(filename, "wb") as f:
        pickle.dump(data, f)

def pload(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def plock(filename, data=()):
    try:
        with open(filename, "xb") as f:
            pickle.dump(data, f)
            return True
    except FileExistsError:
        return False
