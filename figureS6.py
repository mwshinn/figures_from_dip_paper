# Show that the dip is not associated with the saccade

from canvas import Canvas, Vector, Point, Width, Height
import diplib
import scipy
import scipy.stats
import numpy as np
from itertools import groupby
import seaborn as sns


def detect_local_minimum(T, ts, start, tol=.5, backwards=False):
    i = next(i for i in range(0, len(T)) if T[i] > start)
    inc = -1 if backwards else 1
    i_min = -1
    minval = np.inf
    while minval >= ts[i] - tol:
        if ts[i] < minval:
            i_min = i
            minval = ts[i]
        i += inc
    return T[i_min]

colors = sns.color_palette("Reds_d", 5)

c = Canvas(6.9, 8, fontsize=8)


LEFT1 = .5
RIGHT1 = 1.5
LEFT2 = 2.1
RIGHT2 = 3.1


diagram_width = Vector(RIGHT2-LEFT1, .6, "absolute")
diagram_center = Point(.5, 0, "figure") >> Point(0, 7.5, "absolute")
c.add_axis("diagram",  diagram_center-diagram_width/2, diagram_center+diagram_width/2)

for MONKEY in ["Q", "P"]:
    if MONKEY == "P":
        c.add_unit("monkeyp", Vector(1, 1, "absolute"), Point(3.6, 0, "absolute"))
        c.set_default_unit("monkeyp")
    else:
        c.set_default_unit("absolute")
    
    c.add_text(f"Monkey { {'Q': 1, 'P': 2}[MONKEY]}", Point(LEFT1+(RIGHT2-LEFT1)/2, 6.9), weight="bold", horizontalalignment="center")
    
    HC = 63 if MONKEY == "P" else 70
    
    
    c.add_grid(["posweights"+MONKEY, "negweights"+MONKEY], 2, Point(LEFT1, 4.8), Point(LEFT1+(RIGHT1-LEFT1)*.66, 6.4), spacing=Vector(0, .1))
    
    c.add_grid(["posweightssac"+MONKEY, "negweightssac"+MONKEY], 2, Point(RIGHT2-(RIGHT2-LEFT2)*1.33, 4.8), Point(RIGHT2, 6.4), spacing=Vector(0, .1))
    
    
    #c.add_grid(["diagram", "posweightssac", "posweights", "negweights"], 2, Point(.5, 4.5), Point(3.6, 7.6), size=Vector(1, 1))
    
    c.add_grid(["march_samp"+MONKEY, "march_sac"+MONKEY, "plot_samp"+MONKEY, "plot_sac"+MONKEY], 2, Point(LEFT1, 1.1), Point(RIGHT2, 4), size=Vector(1, 1))
    
    
    # Legend
    if MONKEY == "Q":
        LCENTER = Point(.5, 0, "figure") >> Point(0, .5, "absolute")
        LSTEP = Vector(.50, 0)
        LSIZE = Vector(.3, 0)
        c.add_text("RT bin", LCENTER+Height(.1), horizontalalignment="center", verticalalignment="center")
        for i in range(-2, 3):
            center = LCENTER+i*LSTEP
            c.add_line(center - LSIZE/2, center + LSIZE/2, linewidth=4, color=colors[i+2])
            bincenter = 50 + i*20
            c.add_text(f"{bincenter-10}-{bincenter+10}%", center - Height(.1), horizontalalignment="center", verticalalignment="center")
    
    #################### Regression models ####################
    
    CELLS = diplib.get_cell_ids(monkey=MONKEY)
    regs = diplib.get_saccade_control_regression_models(monkey=MONKEY)
    
    
    if MONKEY == "Q":
        ax = c.ax("diagram")
        T,traj = diplib.get_cell_conditional_activity(monkey=MONKEY, ps=800, coh=HC, hr_in_rf=True, align="presample", time_range=(-1000, 2000))
        example_cell = CELLS[0]
        dm = regs[example_cell]['dm']
        T_samplecoh = dm.get_regressor_from_output("samplecoh", regs[example_cell]['params'])[0]
        T_saccade = dm.get_regressor_from_output("saccade-inrf", regs[example_cell]['params'])[0]
        T_presample = dm.get_regressor_from_output("presample-hr", regs[example_cell]['params'])[0]
        
        SACTIME = 1100
        PS = 800
        ax.plot(T*1000, traj, c='k')
        ax.plot(T_samplecoh+PS, len(T_samplecoh)*[0], c=diplib.kern_color('EC'), linewidth=4)
        ax.plot(T_saccade+SACTIME, len(T_saccade)*[5], c=diplib.kern_color('SC'), linewidth=4)
        ax.plot(T_presample, len(T_presample)*[10], c=diplib.kern_color('PH'), linewidth=4)
        ax.axvline(SACTIME, c=diplib.kern_color('SC'), linestyle='--')
        ax.axvline(PS, c=diplib.kern_color('EC'), linestyle='--')
        ax.axvline(0, c=diplib.kern_color('PH'), linestyle='--')
        ax.set_xlim(-200, 1600)
        ax.set_xlabel("Time from presample (ms)")
        c.add_text("Sample", Point(PS, 0, "diagram") >> Point(0, 1.1, "axis_diagram"), color=diplib.kern_color('EC'))
        c.add_text("Presample", Point(0, 0, "diagram") >> Point(0, 1.1, "axis_diagram"), color=diplib.kern_color('PH'))
        c.add_text("Saccade", Point(SACTIME, 0, "diagram") >> Point(0, 1.1, "axis_diagram"), color=diplib.kern_color('SC'))
        sns.despine(ax=ax)
    
    
    
    
    
    
    
    
    
    
    T_dip_overlay, dip_overlay = diplib.get_cell_conditional_activity(monkey=MONKEY, ps=800, coh=HC, hr_in_rf=True, align="sample", time_range=(0, 400))
    dip_overlay = dip_overlay/np.mean(dip_overlay)
    
    times = regs[CELLS[0]]['dm'].get_regressor_from_output("samplecoh", regs[CELLS[0]]['params'])[0]
    sig_neg_coef = times * 0
    sig_pos_coef = times * 0
    n_cell_regs = 0
    for cellid in CELLS:
        if cellid not in regs.keys(): continue
        dm = regs[cellid]['dm']
        estimate = dm.get_regressor_from_output("samplecoh", regs[cellid]['params'])[1]
        sem = dm.get_regressor_from_output("samplecoh", regs[cellid]['bse'])[1]
        sig_neg_coef += (estimate+1.64*sem < 0).astype(int)
        sig_pos_coef += (estimate-1.64*sem > 0).astype(int)
        n_cell_regs += 1
    
    
    ax = c.ax("posweights"+MONKEY)
    ax.bar(times, sig_pos_coef/n_cell_regs*100, 25, color=diplib.kern_color('EC'))
    ax.set_ylabel("% positive")
    #ax.set_xticks([0, 100, 200, 300, 400])
    #ax.set_xticklabels([0, "", 200, "", 400])
    ax.set_xticks([])
    if MONKEY == "Q":
        ax.set_ylim(0, 40)
        ax.set_yticks([10, 20, 30])
    elif MONKEY == "P":
        ax.set_ylim(0, 80)
        ax.set_yticks([20, 40, 60])
    #ax.set_xlabel("Time after sample (ms)")
    ax.set_title("% cells with significant\nevidence* coefficient")
    ax.set_xlim(0, 400)
    sns.despine(ax=ax)
    #ax.plot(T_dip_overlay*1000, dip_overlay/2-.3, alpha=.2, c='k')
    
    ax = c.ax("negweights"+MONKEY)
    ax.bar(times, sig_neg_coef/n_cell_regs*100, 25, color=diplib.kern_color('EC'))
    #ax.set_title("Negative\ncoherence $\\times$ sample\nregression coefficients")
    ax.set_xticks([0, 100, 200, 300, 400])
    ax.set_xticklabels([0, "", 200, "", 400])
    if MONKEY == "Q":
        ax.set_ylim(0, 40)
        ax.set_yticks([10, 20, 30])
    elif MONKEY == "P":
        ax.set_ylim(0, 80)
        ax.set_yticks([20, 40, 60])
    ax.set_xlim(0, 400)
    ax.set_xlabel("Time from sample (ms)")
    ax.set_ylabel("% negative")
    ax.invert_yaxis()
    sns.despine(ax=ax)
    
    
    
    
    
    
    
    
    
    
    
    times = regs[CELLS[0]]['dm'].get_regressor_from_output("saccadecoh-inrf", regs[CELLS[0]]['params'])[0]
    sig_neg_coef = times * 0
    sig_pos_coef = times * 0
    n_cell_regs = 0
    for cellid in CELLS:
        if cellid not in regs.keys(): continue
        dm = regs[cellid]['dm']
        estimate = dm.get_regressor_from_output("saccadecoh-inrf", regs[cellid]['params'])[1]
        sem = dm.get_regressor_from_output("saccadecoh-inrf", regs[cellid]['bse'])[1]
        sig_neg_coef += (estimate+1.64*sem < 0).astype(int)
        sig_pos_coef += (estimate-1.64*sem > 0).astype(int)
        n_cell_regs += 1
    
    
    ax = c.ax("posweightssac"+MONKEY)
    ax.bar(times, sig_pos_coef/n_cell_regs*100, 25, color=diplib.kern_color('SC'))
    ax.set_ylabel("% positive")
    #ax.set_xticks([0, 100, 200, 300, 400])
    #ax.set_xticklabels([0, "", 200, "", 400])
    ax.set_xticks([])
    if MONKEY == "Q":
        ax.set_ylim(0, 40)
        ax.set_yticks([10, 20, 30])
    elif MONKEY == "P":
        ax.set_ylim(0, 80)
        ax.set_yticks([20, 40, 60])
    #ax.set_xlabel("Time after sample (ms)")
    ax.set_title("% cells with significant\nSIC* kernel")
    ax.set_xlim(-800, 0)
    #ax.plot(T_dip_overlay*1000, dip_overlay/2-.3, alpha=.2, c='k')
    sns.despine(ax=ax)
    
    ax = c.ax("negweightssac"+MONKEY)
    ax.bar(times, sig_neg_coef/n_cell_regs*100, 25, color=diplib.kern_color('SC'))
    #ax.set_title("Negative\ncoherence $\\times$ sample\nregression coefficients")
    #ax.set_xticks([0, 100, 200, 300, 400])
    #ax.set_xticklabels([0, "", 200, "", 400])
    if MONKEY == "Q":
        ax.set_ylim(0, 40)
        ax.set_yticks([10, 20, 30])
    elif MONKEY == "P":
        ax.set_ylim(0, 80)
        ax.set_yticks([20, 40, 60])
    ax.set_xlim(-800, 0)
    ax.set_xlabel("Time from saccade (ms)")
    ax.set_ylabel("% negative")
    ax.invert_yaxis()
    sns.despine(ax=ax, top=False)
    ax.set_xticks([-800, -700, -600, -500, -400, -300, -200, -100, 0])
    ax.set_xticklabels([-800, "", -600, "", -400, "", -200, "", 0])
    
    
    
    #################### Timebin analysis traces ####################
    
    
    
    params = {"coh": HC, "ps": 800, "choice_in_rf": False, "hr_in_rf": True, "monkey": MONKEY}
    
    rt_bins = [(i*.2, (i+1)*.2) for i in range(0, 5)]
    
    rts_per_bin = {rt_bin : [] for rt_bin in rt_bins}
    
    CELLS = diplib.get_cell_ids(MONKEY)
    
    for cell in CELLS:
        spikes = diplib._get_cell_conditional_activity(time_range=(-1000, 1000), cellid=cell, **params)
        rts = np.asarray([g[0] for g in groupby(spikes['spiketime_samp'] - spikes['spiketime_sac'])])
        if len(rts) == 0:
            continue
        for rt_bin in rt_bins:
            quantiles = np.quantile(rts, rt_bin)
            rts_per_bin[rt_bin].extend(list(rts[(rts>=quantiles[0])&(rts<=quantiles[1])]))
    
    mean_rt_per_bin = [np.mean(rts_per_bin[rt_bin])/1000 for rt_bin in rt_bins]
    
    ax = c.ax("march_samp"+MONKEY)
    ax.set_title("Sample-aligned\nactivity by RT bin")
    for i,rt_bin in enumerate(rt_bins):
        T, x = diplib.get_mean_conditional_activity_by_rtbin(rtbin=rt_bin, **params, zscore="inrf_subtract")
        ax.plot(T, x, c=colors[i])
    
    ax.set_xlim(-.15, .35)
    ax.set_xticks([0, .2])
    ax.set_xticklabels([0, 200])
    ax.axvline(0, c='k', linestyle='--')
    #c.add_text("Sample\nonset", Point(.01, 0, "march_samp"+MONKEY) >> Point(0, .2, "axis_march_samp"+MONKEY), horizontalalignment="left")
    ax.set_xlabel("Time from sample (ms)")
    ax.set_ylabel("FEF activity")
    #plt.legend(rt_bins)
    sns.despine(ax=ax)
    if MONKEY == "Q":
        ax.set_ylim(-9, 9)
    if MONKEY == "P":
        ax.set_ylim(-8, 28)
    
    ax = c.ax("march_sac"+MONKEY)
    ax.set_title("Saccade-aligned\nactivity by RT bin")
    for i,rt_bin in enumerate(rt_bins):
        T, x = diplib.get_mean_conditional_activity_by_rtbin(rtbin=rt_bin, **params, zscore="inrf_subtract", align="saccade")
        ax.plot(T, x, c=colors[i])
    
    ax.set_xlim(-.35, .05)
    ax.set_xticks([-.2, 0])
    ax.set_xticklabels([-200, 0])
    ax.axvline(0, c='k', linestyle='--')
    #c.add_text("Saccade\nonset", Point(-.01, 0, "march_sac"+MONKEY) >> Point(0, .2, "axis_march_sac"+MONKEY), horizontalalignment="right")
    ax.set_xlabel("Time from saccade (ms)")
    ax.set_ylabel("FEF activity")
    if MONKEY == "Q":
        ax.set_ylim(-9, 9)
    if MONKEY == "P":
        ax.set_ylim(-8, 28)
    #plt.legend(rt_bins)
    sns.despine(ax=ax)
    
    #################### Timebin analysis: Compare sample vs saccade dip minima locations ####################
    
    @diplib.memoize
    def samp_sac_local_minima(MONKEY):
        samp_local_minima = {rt_bin : [] for rt_bin in rt_bins}
        ax = c.ax("march_samp"+MONKEY)
        for i in range(0, 15):
            for rt_bin in rt_bins:
                T, x = diplib.get_mean_conditional_activity_by_rtbin(rtbin=rt_bin, **params, zscore="inrf_subtract", resample=i)
                samp_local_minima[rt_bin].append(detect_local_minimum(T, x, .12))
        
        sac_local_minima = {rt_bin : [] for rt_bin in rt_bins}
        ax = c.ax("march_sac"+MONKEY)
        for i in range(0, 15):
            for rt_bin in rt_bins:
                T, x = diplib.get_mean_conditional_activity_by_rtbin(rtbin=rt_bin, **params, zscore="inrf_subtract", align="saccade", resample=i)
                sac_local_minima[rt_bin].append(detect_local_minimum(T, x, -.12, backwards=True))
        return samp_local_minima, sac_local_minima
    
    samp_local_minima, sac_local_minima = samp_sac_local_minima(MONKEY)
    
    
    ax = c.ax("plot_samp"+MONKEY)
    ax.cla()
    q = lambda q,rt_bin : np.quantile(samp_local_minima[rt_bin], q)
    ax.scatter(mean_rt_per_bin, [q(.5, rt_bin) for rt_bin in rt_bins], c=colors)
    ax.errorbar(mean_rt_per_bin, [q(.5, rt_bin) for rt_bin in rt_bins], yerr=np.asarray([(q(.5, rt_bin)-q(.25, rt_bin), q(.75, rt_bin)-q(.5, rt_bin))  for rt_bin in rt_bins]).T, linestyle=' ', color='k')
    ax.set_xlabel("Mean RT (ms)")
    ax.set_ylabel("Dip minimum time (ms)")
    ax.set_ylim(.12, .23)
    if MONKEY == "Q":
        ax.set_xticks([.25, .35])
        ax.set_xticklabels([250, 350])
    else:
        ax.set_xticks([.4, .6])
        ax.set_xticklabels([400, 600])
    ax.set_yticks([.15, .2])
    ax.set_yticklabels([150, 200])
    
    ax.set_title("Sample-aligned dip\nlocation by RT bin")
    # I confirmed in the source code that the p-value for kendall tau in scipy is the Mann-Kendall test
    flatdata = np.asarray([(b[0], v1) for b,v in samp_local_minima.items() for v1 in v])
    r,p = scipy.stats.kendalltau(flatdata[:,0], flatdata[:,1])
    pform = f"={p:.4f}" if p > .0001 else "$<"+diplib.scientific_notation(p)+"$"
    c.add_text(f"tau={r:.2f}\np{pform}", Point(.55, .7, "axis_plot_samp"+MONKEY), horizontalalignment="left")
    sns.despine(ax=ax)
    
    ax = c.ax("plot_sac"+MONKEY)
    if MONKEY == "Q":
        ax.cla()
        q = lambda q,rt_bin : np.quantile(sac_local_minima[rt_bin], q)
        ax.scatter(mean_rt_per_bin, [q(.5, rt_bin) for rt_bin in rt_bins], c=colors)
        ax.errorbar(mean_rt_per_bin, [q(.5, rt_bin) for rt_bin in rt_bins], yerr=np.asarray([(q(.5, rt_bin)-q(.25, rt_bin), q(.75, rt_bin)-q(.5, rt_bin))  for rt_bin in rt_bins]).T, linestyle=' ', color='k')
        ax.set_xlabel("Mean RT (ms)")
        ax.set_ylabel("Dip minimum time (ms)")
        ax.set_title("Saccade-aligned dip\nlocation by RT bin")
        ax.set_xticks([.25, .35])
        ax.set_xticklabels([250, 350])
        ax.set_yticks([-.2, -.15])
        ax.set_yticklabels([-200, -150])
        flatdata = np.asarray([(b[0], v1) for b,v in sac_local_minima.items() for v1 in v])
        r,p = scipy.stats.kendalltau(flatdata[:,0], flatdata[:,1])
        pform = f"={p:.4f}" if p > .0001 else "$<"+diplib.scientific_notation(p)+"$"
        c.add_text(f"tau={r:.2f}\np{pform}", Point(.6, .7, "axis_plot_sac"+MONKEY), horizontalalignment="left")
        sns.despine(ax=ax)
    else:
        ax.axis("off")

c.add_figure_labels([("a", "diagram"), ("b", "posweightsQ", Width(-.2)), ("c", "posweightssacQ"), ("d", "march_sampQ"), ("e", "march_sacQ"), ("f", "plot_sampQ"), ("g", "plot_sacQ"), ("h", "posweightsP", Width(-.2)), ("i", "posweightssacP"), ("j", "march_sampP"), ("k", "march_sacP"), ("l", "plot_sampP")])
    

c.save("figureS6.pdf")
