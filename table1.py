from canvas import *
import diplib
import seaborn as sns
import pandas
import scipy

MONKEY = "Q"

fn = "all_cpds.pkl"
if diplib.plock(fn):
    cpds_Q = diplib.get_regression_cpds("Q")
    cpds_P = diplib.get_regression_cpds("P")
    diplib.psave(fn, (cpds_Q, cpds_P))
else:
    cpds_Q, cpds_P = diplib.pload(fn)

list_to_text = {"": "Full model",
                "presample-hr_presample": "Presample kernel (P+PH)",
                "samplecoh_sample": "Sample kernel (E+EC)",
                "saccade_saccade-inrf": "Saccade kernel (S+SI)",
                'samplecoh': "Evidence kernel (EC)",
                "saccade-inrf": "RF-dependent saccade kernel (SI)",
                "presample-hr": "Reward-dependent presample kernel (PH)",
}

list_to_count = {"": 304,
                'samplecoh': 288,
                "samplecoh_sample": 272,
                "saccade-inrf": 288,
                "saccade_saccade-inrf": 272,
                "presample-hr": 184,
                "presample-hr_presample": 64}

    
for MONKEY,all_cpds in [("Q", cpds_Q), ("P", cpds_P)]:
    mods = diplib.get_regression_models(MONKEY)
    dm = mods[list(mods.keys())[0]]['dm']
    metric = 'samplecoh'
    vals = {}
    meanvals = {'': np.nan}
    for regressor in dm.regressor_names():
        vals[regressor] = []
        for c in mods.keys():
            adj = 0
            # if regressor == "presample":
            #     times = dm.get_regressor_from_output("presample", mods[c]['params'])[0]
            #     inds = (times >= 0) & (times <= 300)
            #     ds = dm.get_regressor_from_output("presample", mods[c]['params'])[1][inds]
            #     adj = np.mean(ds)
            signeg = dm.get_regressor_from_output(regressor, mods[c]['bse']*1.96+(mods[c]['params']-adj))[1] < 0
            sigpos = -dm.get_regressor_from_output(regressor, mods[c]['bse']*1.96+(mods[c]['params']-adj))[1] > 0
            vals[regressor].append((sum(sigpos) + sum(signeg))/len(sigpos))
        meanvals[regressor] = np.mean(vals[regressor])

    meanvals['samplecoh_sample'] = meanvals['sample']
    meanvals['presample-hr_presample'] = meanvals['presample']
    meanvals['saccade_saccade-inrf'] = meanvals['saccade']

    sample_sizes = {cellid: len(set(diplib.spikes_df(MONKEY).query('cellid == %i and saccadetime < 2300' % cellid).trialid)) for cellid in all_cpds.keys()}
    _df = []
    for cell,e in all_cpds.items():
        for model_name_list, stats in e:
            model_name = "_".join(model_name_list)
            _df.append((model_name, cell, stats['rss'], stats['tss'], stats['cpd']))
    df = pandas.DataFrame(_df, columns=["model_name", "cell", "rss", "tss", "cpd"])
    df['R2'] = df.apply(lambda x : 1-(x['rss']/x['tss']), axis=1)
    df['full_name'] = df.apply(lambda x : list_to_text[x['model_name']], axis=1)
    df['sample_size'] = len(mods[list(mods.keys())[0]]['Y_full'])
    df['n_params'] = df.apply(lambda x : list_to_count[x['model_name']], axis=1)
    df['n_params_full'] = list_to_count[""]
    df = df.merge(df.query('model_name == ""')[['rss', 'cell']], on='cell', suffixes=("", "_full"))
    df['df_n'] = df['n_params_full']-df['n_params']
    df['df_d'] = df['sample_size']-df['n_params_full']
    df['F_crit_05'] = df.apply(lambda x : scipy.stats.f.ppf(.05, dfn=x['df_n'], dfd=x['df_d']), axis=1)
    df['F'] = ((df['rss']-df['rss_full'])/(df['df_n'])) / (df['rss_full']/df['df_d'])
    df['fractimessig'] = df.apply(lambda x : meanvals[x['model_name']], axis=1)
    #df.to_pickle("all_cpds_Q.pandas.pkl")
    df = df.query("cell > 104")
    df['F_sig'] = (df['F'] > df['F_crit_05']).astype(int)

    aggs = [('Minimum', 'min'), ('25%', lambda x : np.quantile(x, .25)), ('Median', 'median'), ('75%', lambda x : np.quantile(x, .75)), ('Maximum', 'max')]
    table = df.query('model_name != ""').groupby(['full_name'])['cpd'].agg(aggs).reindex(list(list_to_text.values())[1:]).reset_index()
    table['metric'] = 'CPD'
    table_r2 = df.query('model_name == ""').groupby(['full_name'])['R2'].agg(aggs).reset_index()
    table_r2['metric'] = 'R2'
    table_F = df.query('model_name != ""').groupby(['full_name'])['F_sig', 'fractimessig'].mean().reset_index()

    full_table = table_r2.append(table).merge(table_F[['F_sig', 'fractimessig', 'full_name']], on='full_name', how='outer').set_index(['metric', 'full_name'])
    full_table.to_csv(f"regression_table_{MONKEY}.csv")
    print(full_table)

