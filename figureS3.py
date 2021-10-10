from canvas import *
import diplib
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import pandas
import scipy

def lighten_color(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


stats = {"Q": {}, "P": {}}
coefs = {}
for MONKEY in ["Q", "P"]:
    spikes = diplib.spikes_df(MONKEY)
    sigs_sac = []
    sigs_ps = []
    sigs_samp = []
    ispos_sac = []
    ispos_ps = []
    ispos_samp = []
    tmp_coefs_sac = []
    tmp_coefs_ps = []
    tmp_coefs_samp = []
    tmp_coefs_di = []
    tmp_coef_sac = None
    tmp_coef_ps = None
    tmp_coef_samp = None
    dis = diplib.get_dip_score(MONKEY)
    CELLS = diplib.get_cell_ids(MONKEY)
    nonsig = {"coh": False, "choice_in_rf": False, "hr_in_rf": False, "Intercept": False}
    # Saccade modulation
    for i,cellid in enumerate(CELLS):
        print(cellid)
        spikes['hr_in_rf'] = (spikes['choice_in_rf'] + spikes['correct'] + spikes['high_rew']) % 2 # Determined through a truth table
        
        spike_counts_sac = spikes.query(f"cellid == {cellid} and -50 < spiketime_sac and spiketime_sac < 50").groupby(['trialid', 'coh', 'ps', 'choice_in_rf', 'hr_in_rf'])['trialid'].count().rename('spike_count').reset_index()
        try:
            m = sm.ols("spike_count ~ coh + choice_in_rf + hr_in_rf", data=spike_counts_sac).fit()
            sigs_sac.append(m.pvalues<.05)
            ispos_sac.append(m.tvalues>0)
            tmp_coef_sac = m.params
        except FloatingPointError:
            print("Excepted sac")
            sigs_sac.append(nonsig)
            ispos_sac.append(nonsig)
            
        spike_counts_ps = spikes.query(f"cellid == {cellid} and 100 < spiketime_pre and spiketime_pre < 200").groupby(['trialid', 'coh', 'ps', 'choice_in_rf', 'hr_in_rf'])['trialid'].count().rename('spike_count').reset_index()
        try:
            m = sm.ols("spike_count ~ coh + choice_in_rf + hr_in_rf", data=spike_counts_ps).fit()
            sigs_ps.append(m.pvalues<.05)
            ispos_ps.append(m.tvalues>0)
            tmp_coef_ps = m.params
        except (ValueError, FloatingPointError):
            print("Excepted ps")
            sigs_ps.append(nonsig)
            ispos_ps.append(nonsig)
            
        spike_counts_samp = spikes.query(f"cellid == {cellid} and 100 < spiketime_samp and spiketime_samp < 200").groupby(['trialid', 'coh', 'ps', 'choice_in_rf', 'hr_in_rf'])['trialid'].count().rename('spike_count').reset_index()
        try:
            m = sm.ols("spike_count ~ coh + choice_in_rf + hr_in_rf", data=spike_counts_samp).fit()
            sigs_samp.append(m.pvalues<.05)
            ispos_samp.append(m.tvalues>0)
            tmp_coef_samp = m.params
        except FloatingPointError:
            print("Excepted samp")
            sigs_samp.append(nonsig)
            ispos_samp.append(nonsig)
        if all(e is not None for e in [tmp_coef_ps, tmp_coef_sac, tmp_coef_samp]):
            tmp_coefs_samp.append(tmp_coef_samp)
            tmp_coefs_ps.append(tmp_coef_ps)
            tmp_coefs_sac.append(tmp_coef_sac)
            tmp_coefs_di.append(dis[i])
            
    coefs[MONKEY] = {}
    coefs[MONKEY]['samp'] = tmp_coefs_samp
    coefs[MONKEY]['ps'] = tmp_coefs_ps
    coefs[MONKEY]['sac'] = tmp_coefs_sac
    coefs[MONKEY]['dipindex'] = tmp_coefs_di
    for name,ispos,sigs in [("samp", ispos_samp, sigs_samp),
                            ("ps", ispos_ps, sigs_ps),
                            ("sac", ispos_sac, sigs_sac)]:
        stats[MONKEY][name] = {}
        df_samp = pandas.concat([pandas.DataFrame(map(dict, ispos)).add_prefix("ispos"), pandas.DataFrame(map(dict, sigs)).add_prefix("sig")], axis=1)
        print(df_samp)
        for v in m.params.keys():
            stats[MONKEY][name][v] = (df_samp.query(f"sig{v} == True")[f'ispos{v}'].sum(), df_samp[f'sig{v}'].sum(), len(df_samp))

print("Q Coh vs choice_in_rf", scipy.stats.spearmanr([tc['coh'] for tc in coefs['Q']], [tc['choice_in_rf'] for tc in coefs['Q']]))
print("P Coh vs choice_in_rf", scipy.stats.spearmanr([tc['coh'] for tc in coefs['P']], [tc['choice_in_rf'] for tc in coefs['P']]))

for m in ["Q", "P"]:
    for typ in ['coh', 'choice_in_rf', 'hr_in_rf']:
        for per in ['samp', 'ps', 'sac']:
            print(f"{m} {typ} ({per}) vs coh (samp)", scipy.stats.spearmanr([tc[typ] for tc in coefs[m][per]], [tc['coh'] for tc in coefs[m]['samp']]))
            print(f"{m} {typ} ({per}) vs dip index", scipy.stats.spearmanr([tc[typ] for tc in coefs[m][per]], coefs[m]['dipindex']))

for typ in ['coh', 'choice_in_rf', 'hr_in_rf']:
    for per in ['samp', 'ps', 'sac']:
        print(f"Both {typ} ({per}) vs coh (samp)", scipy.stats.spearmanr([tc[typ] for m in ["Q", "P"] for tc in coefs[m][per]], [tc['coh'] for m in ["Q", "P"] for tc in coefs[m]['samp']]))
        print(f"Both {typ} ({per}) vs dip index", scipy.stats.spearmanr([tc[typ] for m in ["Q", "P"] for tc in coefs[m][per]], [e for m in ["Q", "P"] for e in coefs[m]['dipindex']]))

c = Canvas(6, 3, "in", fontsize=8)
c.add_axis("motor", Point(.5, .5, "in"), Point(2.9, 2.9, "in"))
c.add_axis("choice", Point(3.5, .5, "in"), Point(5.9, 2.9, "in"))
ax = c.ax("motor")
# ax.scatter([e for m in ["Q", "P"] for e in coefs[m]['dipindex']], [tc["choice_in_rf"] for m in ["Q", "P"] for tc in coefs[m]["sac"]], c='k')
# ax.set_xlabel("Dip index")
ax.scatter([e['coh'] for m in ["Q", "P"] for e in coefs[m]['samp']], [tc["choice_in_rf"] for m in ["Q", "P"] for tc in coefs[m]["sac"]], c='k')
# ax.axvline(-1.96)
# ax.axvline(1.96)
ax.set_xlabel("Dip selectivity")
ax.set_ylabel("Choice selectivity")
sns.despine(ax=ax)
ax = c.ax("choice")
# ax.scatter([e for m in ["Q", "P"] for e in coefs[m]['dipindex']], [tc["hr_in_rf"] for m in ["Q", "P"] for tc in coefs[m]["ps"]], c='k')
ax.scatter([e['coh'] for m in ["Q", "P"] for e in coefs[m]['samp']], [tc["hr_in_rf"] for m in ["Q", "P"] for tc in coefs[m]["ps"]], c='k')
# ax.axvline(-1.96)
# ax.axvline(1.96)
# ax.set_xlabel("Dip index")
ax.set_xlabel("Dip selectivity")
ax.set_ylabel("Reward selectivity")
sns.despine(ax=ax)
scipy.stats.spearmanr([e for m in ["Q", "P"] for e in coefs[m]['dipindex']], [tc["choice_in_rf"] for m in ["Q", "P"] for tc in coefs[m]["sac"]])
scipy.stats.spearmanr([e for m in ["Q", "P"] for e in coefs[m]['dipindex']], [tc["hr_in_rf"] for m in ["Q", "P"] for tc in coefs[m]["ps"]])
c.save("figureR2.png", dpi=450)

c = Canvas(6.9, 2, "in")
c.add_axis("mQ", Point(.7, .5, "in"), Point(2.5, 1.5, "in"))
c.add_axis("mP", Point(3.3, .5, "in"), Point(5.3, 1.5, "in"))
ids = ["hr_in_rf", "coh", "choice_in_rf"]
ids_to_names = {"Intercept": "Intercept", "coh": "Coherence", "choice_in_rf": "Choice", "hr_in_rf": "Reward"}
names = [ids_to_names[i] for i in ids]
taskconds = ["ps", "samp", "sac"]
taskconds_to_name = {"ps": "Presample", "samp": "Sample", "sac": "Saccade"}
c.add_legend(Point(5.8, 1.35, "in"), [(taskconds_to_name[i], {"c": diplib.kern_color(i), "linewidth":6}) for i in taskconds], sym_width=Width(1, "Msize"))
for MONKEY in ["Q", "P"]:
    
    bars_x = []
    bars_height = []
    bars_bottom = []
    bars_color = []
    for i,name in enumerate(taskconds):
        for j,v in enumerate(ids):
            x = j*5 + i
            snv = stats[MONKEY][name][v]
            # Positive coefficients
            bars_x.append(x)
            bars_bottom.append(0)
            bars_height.append(snv[0]/snv[2])
            bars_color.append(lighten_color(diplib.kern_color(name), 1.1))
            # Negative coefficients
            bars_x.append(x)
            bars_bottom.append(bars_height[-1])
            bars_height.append((snv[1]-snv[0])/snv[2])
            bars_color.append(lighten_color(diplib.kern_color(name), .6))
            
    ax = c.ax("m"+MONKEY)
    ax.bar(x=bars_x, height=bars_height, bottom=bars_bottom, color=bars_color)
    for x,bottom in zip(bars_x, bars_bottom):
        if bottom == 0: continue
        ax.plot([x-.4, x+.4], [bottom, bottom], c='k', solid_capstyle='butt', linewidth=1.5)
        
    ax.set_yticks([0, .5, 1])
    ax.set_xticks(np.asarray([0, 1, 2])*5+(len(names)-1)/2)
    ax.set_xticklabels(names)
    ax.set_title(f"Monkey {1 if MONKEY=='Q' else 2}")
    sns.despine(ax=ax)

c.ax("mQ").set_ylabel("Fraction neurons significant")

c.save("figureS3.pdf")
