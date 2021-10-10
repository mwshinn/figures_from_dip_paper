import diplib
import numpy as np
import matplotlib.pyplot as plt
import scipy
import seaborn as sns
from canvas import Canvas, Point, Vector

for MONKEY,FILENAME in [("Q", "figure5.pdf"), ("P", "figureS7.pdf")]:
    c = Canvas(6.9, 3.7, fontsize=8)
    # c.use_latex(preamble=r'\usepackage{color}\usepackage[scaled]{helvet}\usepackage[helvet]{sfmath}')
    
    # c.add_axis("diagram", Point(.8, 2.1, "absolute"), Point(2.2, 2.7, "absolute"))

    c.add_grid(["population", "placeholder", "kernels", "ph2", "ph3", "pcs", "loadinghist", "dip_vs_psdip"], 2, Point(.5, .5, "absolute"), Point(6.7, 3.4, "absolute"), size=Vector(1.0, 1.0, "absolute"))
    c.ax("ph2").axis("off")
    c.ax("ph3").axis("off")

    # c.add_axis("population", Point(.5, 6.1, "absolute"), Point(1.5, 7.1, "absolute"))
    
    # c.add_grid(["placeholder", "kernels", "pcs", "loadinghist"], 2, Point(.5, 2.5, "absolute"), Point(3.1, 5.4, "absolute"), size=Vector(1.0, 1.0, "absolute"))
    # c.add_axis("dip_vs_psdip", Point(1.5, .5, "absolute"), Point(2.5, 1.5, "absolute"))
    
    c.add_figure_labels([("a", "population"), ("b", "placeholder"), ("c", "kernels"), ("d", "pcs"), ("e", "loadinghist"), ("f", "dip_vs_psdip")])
    
    regs = diplib.get_regression_models(MONKEY)
    cells = diplib.get_cell_ids(MONKEY)
    
    # Rather than selecting times as indices for the dip function, do
    # it in a more general way that will work if the timestep is
    # changed.
    
    
    if MONKEY == "Q":
        HC,MC,LC = 70, 60, 53
    elif MONKEY == "P":
        HC,MC,LC = 63, 57, 52

    # Population activity
    ax = c.ax("population")
    SMOOTH = 3
    ax.plot(*diplib.get_rt_conditional_activity(monkey=MONKEY, coh=50, smooth=SMOOTH, time_range=(-200, 500), align="presample"), color=diplib.get_color(ps=800, coh=50))
    for coh in [LC, MC, HC]:
        for ps in [0, 400, 800]:
            ax.plot(*diplib.get_mean_conditional_activity(monkey=MONKEY, coh=coh, ps=ps, smooth=SMOOTH, time_range=(-200, 500), align="presample"), color=diplib.get_color(ps=ps, coh=coh))
    if MONKEY == "Q":
        ax.set_ylim(19, 27)
        ax.set_yticks([20, 25])
    else:
        ax.set_ylim(24, 36)
        ax.set_yticks([25, 30, 35])
    ax.set_xlim(0, .300)
    ax.axvline(0, linestyle='--', color='k')
    sns.despine(right=False, top=False, ax=ax)
    ax.set_xticks([0, .1, .2, .3])
    ax.set_xticklabels(["0", "100", "200", "300"])
    ax.set_xlabel("Time from presample (ms)")
    ax.set_ylabel("Normalized FEF activity")
    c.add_text("Stimulus dip", Point(.5, 1.08, "axis_population"))

    diplib.make_gridlegend(c, Point(-.2, -1.3, "axis_population")+Vector(0, 0, "absolute"), shorten=True)

    # Find the minimum from the population activity of sample and presample dips
    kern_times = regs[cells[0]]["dm"].get_regressor_from_output("samplecoh", regs[cells[0]]["params"])[0]
    kerns = [regs[c]["dm"].get_regressor_from_output("samplecoh", regs[c]["params"])[1] for c in cells]
    
    _pstimes = regs[cells[0]]['dm'].get_regressor_from_output("presample", regs[cells[0]]['params'])[0]
    _pskern_inds = (_pstimes >= 0) & (_pstimes <= 300)
    pstimes = _pstimes[_pskern_inds]
    pskerns = [regs[c]["dm"].get_regressor_from_output("presample", regs[c]["params"])[1][_pskern_inds] for c in cells]

    argmintwo_close = lambda x : np.abs(np.subtract(*sorted(list(range(0, len(x))), key=lambda y : x[y])[0:2])) < 3
    # getmin = lambda x : np.median(np.argsort(x)[0:3])
    getmin = lambda x : np.argsort(x)[0]
    kerns_adj = [(k) for k,psk in zip(kerns,pskerns)]
    pskerns_adj = [(psk-np.mean(psk[0:4])) for k,psk in zip(kerns,pskerns)]
    meankern = np.mean(kerns_adj, axis=0)
    semkern = np.std(kerns_adj, axis=0)/np.sqrt(len(kerns_adj))
    meanpskern = np.mean(pskerns_adj, axis=0)
    sempskern = np.std(pskerns_adj, axis=0)/np.sqrt(len(pskerns_adj))
    
    ax = c.ax("kernels")
    ax.errorbar(kern_times, meankern/np.std(meankern), yerr=semkern*1.64/np.std(meankern), c=diplib.kern_color('EC'))
    # for k in kerns_adj:
    #     ax.plot(kern_times, (k)/np.std(k), c=diplib.kern_color('EC'), alpha=.1, linewidth=1)
    
    ax.errorbar(pstimes, meanpskern/np.std(meanpskern), yerr=sempskern*1.64/np.std(meanpskern), c=diplib.kern_color('PH'))
    # for k in pskerns_adj:
    #     ax.plot(pstimes, (k)/np.std(k), c=diplib.kern_color('PH'), alpha=.1, linewidth=1)
    
    ax.set_yticklabels(map(lambda x : "%g" % x, ax.get_yticks())) # Fix negative signs which don't show up for some latex-related reason
    
    ax.set_xlabel("Time from (pre)sample (ms)")
    ax.set_ylabel("Normalized kernel")
    ax.set_xlim(0, 300)
    ax.set_xticks([0, 100, 200, 300])
    sns.despine(ax=ax)
    c.add_text("Mean kernels", Point(.5, 1.08, "axis_kernels"))


    c.add_legend(Point(.9, .4, "axis_kernels"), [("Evidence kernel", {"c": diplib.kern_color('EC')}), ("Stimulus kernel", {"c": diplib.kern_color('PH')})], line_spacing=Vector(0, 1.2, "Msize"), sym_width=Vector(1, 0, "Msize"), padding_sep=Vector(1, 0, "Msize"))
    
    #plt.plot(kern_times, meankern)
    #plt.plot(pstimes, meanpskern)
    #plt.show()
    
    # Bootstrap a CI
    median_diffs = []
    for i in range(0, 10000):
        choices = np.random.choice(len(kerns), len(kerns))
        bs_kerns = np.asarray(kerns)[choices]
        bs_pskerns = np.asarray(pskerns_adj)[choices]
        bs_mediankern = np.median(bs_kerns, axis=0)
        bs_medianpskern = np.median(bs_pskerns, axis=0)
        median_diffs.append(kern_times[np.argmin(bs_mediankern)]-pstimes[np.argmin(bs_medianpskern)])
    
    
    ci95 = (np.quantile(median_diffs, .025), np.quantile(median_diffs, .975))
    
    # Make a dip index
    i_inner = meankern.argsort()[0:3]
    i_inner_ps = meanpskern.argsort()[0:3]
    # dipindex = [(-np.mean(k[i_inner]) + np.mean(k[0:12]))/np.std(k[0:12]) for k in kerns]
    # psdipindex = [(-np.mean(k[i_inner_ps]) + np.mean(k[0:12]))/np.std(k[0:12]) for k in pskerns]
    dipindex = diplib.get_dip_score(MONKEY)
    psdipindex = diplib.get_presample_dip_score(MONKEY)
    
    #plt.hist(dipindex, alpha=.5)
    #plt.hist(psdipindex, alpha=.5)
    #plt.show()

    c.ax("placeholder").axis("off")
    c.add_grid(["posweights", "negweights"], 2, Point(0, 0, "axis_placeholder"), Point(1, 1, "axis_placeholder"), spacing=Vector(0, .1, "absolute"))
    times = regs[cells[0]]['dm'].get_regressor_from_output("presample", regs[cells[0]]['params'])[0]
    i_times = (times > 0) & (times < 300)
    times = times[i_times]
    sig_neg_coef = times * 0
    sig_pos_coef = times * 0
    n_cell_regs = 0
    for cellid in cells:
        if cellid not in regs.keys(): continue
        dm = regs[cellid]['dm']
        estimate = dm.get_regressor_from_output("presample", regs[cellid]['params'])[1][i_times]
        sem = dm.get_regressor_from_output("presample", regs[cellid]['bse'])[1][i_times]
        sig_neg_coef += ((estimate-np.mean(estimate[0:4]))+1.64*sem < 0).astype(int)
        sig_pos_coef += ((estimate-np.mean(estimate[0:4]))-1.64*sem > 0).astype(int)
        n_cell_regs += 1


    ax = c.ax("posweights")
    ax.bar(times, sig_pos_coef/n_cell_regs*100, 25, color=diplib.kern_color('PH'))
    ax.set_ylabel("% positive")
    ax.set_xticks([])
    if MONKEY == "Q":
        ax.set_ylim(0, 70)
        ax.set_yticks([20, 40, 60])
    elif MONKEY == "P":
        ax.set_ylim(0, 75)
        ax.set_yticks([20, 40, 60])
    c.add_text("% cells with significant\nstimulus kernel", Point(.5, 1.1, "axis_posweights"))
    ax.set_xlim(0, 300)
    sns.despine(ax=ax)

    ax = c.ax("negweights")
    ax.bar(times, sig_neg_coef/n_cell_regs*100, 25, color=diplib.kern_color('PH'))
    #ax.set_title("Negative\ncoherence $\\times$ sample\nregression coefficients")
    ax.set_xticks([0, 100, 200, 300])
    if MONKEY == "Q":
        ax.set_ylim(0, 70)
        ax.set_yticks([20, 40, 60])
    elif MONKEY == "P":
        ax.set_ylim(0, 75)
        ax.set_yticks([20, 40, 60])
    ax.set_xlim(0, 300)
    ax.set_xlabel("Time from presample (ms)")
    ax.set_ylabel("% negative")
    sns.despine(top=False, ax=ax)
    ax.invert_yaxis()
    sns.despine(ax=ax, top=False)
    print("="*100, "pos neg at 100-125", sig_pos_coef[4]/n_cell_regs, sig_neg_coef[4]/n_cell_regs)
    
    
    
    ax = c.ax("dip_vs_psdip")
    ax.scatter(dipindex, psdipindex, c='k', s=8)
    #ax.axvline(0, linestyle='--', linewidth=.5)
    #ax.axhline(0, linestyle='--', linewidth=.5)
    scipy.stats.spearmanr(psdipindex, dipindex)
    scipy.stats.pearsonr(psdipindex, dipindex)
    ax.set_xlabel("Evidence dip index")
    ax.set_ylabel("Stimulus dip index")
    ax.set_yticklabels(map(lambda x : "%g" % x, ax.get_yticks())) # Fix negative signs which don't show up for some latex-related reason
    sns.despine(ax=ax)
    
    rval, pval = scipy.stats.spearmanr(psdipindex, dipindex)
    
    if pval > .0001:
        c.add_text(f"$r_s={np.abs(rval):.3f}$\n$p = {round(pval, 4)}$", Point(.8, .2, "axis_dip_vs_psdip"), bbox=dict(boxstyle="round", ec=(0.0, 0.0, 0.0, 0.3), fc=(.95, .95, .95, .7)))
    else:
        c.add_text(f"$r_s={np.abs(rval):.4f}$\n$p < {diplib.scientific_notation(pval, round_direction='up')}$", Point(.8, .2, "axis_dip_vs_psdip"), bbox=dict(boxstyle="round", ec=(0.0, 0.0, 0.0, 0.3), fc=(.95, .95, .95, .7)))
    
    

    # # Show there is a difference in dip time and presample dip time
    # psmintimes = [pstimes[np.argmin(psk)] for psk in pskerns_adj if argmintwo_close(psk)]
    # mintimes = [kern_times[np.argmin(k)] for k in kerns_adj if argmintwo_close(k)]

    assert len(kerns_adj) == len(pskerns_adj)
    mean_kerns = np.mean(kerns_adj, axis=0)
    mean_pskerns = np.mean(pskerns_adj, axis=0)
    real_difference = getmin(mean_kerns) - getmin(mean_pskerns)
    null_differences = []
    for i in range(0, 10000):
        inds = np.random.choice(len(kerns_adj), len(kerns_adj), replace=True)
        mean_kerns = np.mean(np.asarray(kerns_adj)[inds,:], axis=0)
        mean_pskerns = np.mean(np.asarray(pskerns_adj)[inds,:], axis=0)
        null_differences.append(getmin(mean_kerns) - getmin(mean_pskerns))
    pval = np.sum(real_difference <= 0 for d in null_differences)/len(null_differences)
    print("="*100, "Permuation pval:", MONKEY, pval, "real difference", real_difference, getmin(mean_kerns), getmin(mean_pskerns))
    

    
    # ax = c.ax("neuron_timing_diffs")
    # ax.hist(np.asarray(mintimes)-psmintimes, color='k')
    # ax.set_xlabel("Difference in PS-dip and dip time")
    # ax.set_ylabel("\\# Cells")
    # ax.axvline(np.median(mintimes)-np.median(psmintimes), linestyle='--', color='k')
    # ax.set_xlim(-250, 250)
    # ax.set_xticklabels(map(lambda x : "%g" % x, ax.get_xticks())) # Fix negative signs which don't show up for some latex-related reason
    
    
    
    # # Do a permutation test
    # real_difference = np.median(mintimes) - np.median(psmintimes)
    # null_differences = []
    # alltimes = mintimes + psmintimes
    # for i in range(0, 10000):
    #     np.random.shuffle(alltimes)
    #     groupA = alltimes[0:len(mintimes)]
    #     groupB = alltimes[len(mintimes):]
    #     null_differences.append(np.median(groupA) - np.median(groupB))

    # print(real_difference, null_differences, mintimes)
    # pval = np.sum(real_difference <= np.abs(d) for d in null_differences)/len(null_differences)
    # print("="*100, "Permuation pval:", MONKEY, pval)
    # #c.add_text(f"p={pval}", Point(.9, .8, "axis_neuron_timing_diffs"))
    
    # plt.figure()
    # plt.hist(null_differences, bins=50)
    # plt.axvline(real_difference)
    # plt.show()
    
    
    
    
    
    
    
    # Check to see if the new dip index correlates with the old one
    
    PCNUM = 1 if MONKEY == "P" else 1
    
    pcs = diplib.get_pcs_noncentered(monkey=MONKEY)
    pc_scores = pcs[2][:,PCNUM]
    
    # ax = c.ax("pc1_vs_dip")
    # if MONKEY == "P":
    #     ax.scatter(dipindex, -1*pc_scores, c='k', s=8)
    #     rval, pval = scipy.stats.spearmanr(pc_scores, dipindex)
    # else:
    #     ax.scatter(dipindex, pc_scores, c='k', s=8)
    #     rval, pval = scipy.stats.spearmanr(pc_scores, dipindex)
    
    # ax.set_xlabel("Evidence dip index")
    # ax.set_ylabel(f"Evidence kernel SV{PCNUM+1} loading")
    
    # ax.set_yticklabels(map(lambda x : "%g" % x, ax.get_yticks())) # Fix negative signs which don't show up for some latex-related reason
    # sns.despine(ax=ax)
    
    # c.add_text(f"$r_s={np.abs(rval):.4f}$\n$p < {diplib.scientific_notation(pval, round_direction='up')}$", Point(.3, .85, "axis_pc1_vs_dip"))
    
    
    
    #################### Plot PCs ####################

    ax = c.ax("pcs")
    pcs = diplib.get_pcs_noncentered(monkey=MONKEY, regressor="presample") # I know this is weird, historical reasons
    times = pcs[3]

    if MONKEY == "Q":
        PCNUM = 1
    elif MONKEY == "P":
        PCNUM = 1

    for i in range(0, 2):
        ax.plot(times, pcs[1][:,i], c='k', lw=(2.75 if i!=PCNUM else 4), alpha=1, solid_capstyle="round")
        ax.plot(times, pcs[1][:,i], c=diplib.pc_colors(i), lw=(2 if i!=PCNUM else 3), alpha=1, solid_capstyle="round")
        #ax.plot(times, ((-1 if i==0 else 1)*pcs[1]*pcs[0])[:,i], c=diplib.pc_colors(i), lw=(1.5 if i!=PCNUM else 3), alpha=(.4 if i!=PCNUM else 1))

    print("="*100, "monkey", MONKEY, "corr", np.corrcoef(meanpskern, pcs[1][:,PCNUM])[0,1])
    ax.set_xlabel("Time from sample (ms)")
    ax.set_ylabel("Singular vector weight")
    ax.set_title("Stimulus kernel\nsingular vectors")
    #c.add_legend(Point(2.3, 3.6, "absolute"), [("Var explained: %i%%" % (100*pcs[0][i]) , {"lw": 5, "color": colors[i]}) for i in range(0, 4)])
    ax.set_xlim(0, 300)
    ax.set_xticks([0, 100, 200, 300])
    ax.axhline(0, c='gray', linestyle='--', zorder=-10)
    sns.despine(ax=ax)

    ax = c.ax("loadinghist")
    pc_weights = pcs[2][:,PCNUM]
    if MONKEY == "P":
        h = ax.hist(pc_weights, bins=np.arange(-.4, .4001, .03333), color=diplib.pc_colors(PCNUM))
        ax.set_xlim(-.4, .4)
    else:
        h = ax.hist(pc_weights, bins=np.arange(-1.2, 1.2001, .1), color=diplib.pc_colors(PCNUM))
        ax.set_xlim(-1.2, 1.2)
    ax.axvline(0, c='gray', linestyle='--')
    ax.set_ylabel("# cells")
    ax.set_xlabel(f"Stimulus kernel SV{PCNUM+1} factor score")
    ax.set_title(f"Cell factor scores for\nstimulus kernel SV2")
    ax.plot(np.repeat(h[1], 2), [0]+list(np.repeat(h[0], 2))+[0], c='k', linewidth=.5)
    sns.despine(ax=ax)
    print("N in hist > 0:", sum(pc_weights>0), "/", len(pc_weights))
    
    # Diagram
    # ax = c.ax("diagram")
    # CELLS = diplib.get_cell_ids(MONKEY)
    # example_cell = 3401 if MONKEY == "Q" else 6101
    # regmod = diplib.get_regression_models(monkey=MONKEY)[example_cell]
    # ax.plot(*regmod['dm'].get_regressor_from_output("samplecoh", regmod['params']), color='k')
    # #ax.plot([125, 175], 2*[-1], c="r", linewidth=4)
    # #ax.plot([0, 300], 2*[-1.2], c="g", linewidth=4)
    # ax.set_xlim(0, 350)
    # if MONKEY == "Q":
    #     ax.set_ylim(-1, .4)
    # elif MONKEY == "P":
    #     ax.set_ylim(-3, 2.5)
    # c.add_line(Point(min(kern_times[i_inner]), 0, "diagram") >> Point(0, .1, "axis_diagram"), Point(max(kern_times[i_inner]), 0, "diagram") >> Point(0, .1, "axis_diagram"), c=diplib.kern_color('R1'), linewidth=4, solid_capstyle="butt")
    # c.add_line(Point(0, 0, "diagram") >> Point(0, .04, "axis_diagram"), Point(300, 0, "diagram") >> Point(0, .04, "axis_diagram"), c=diplib.kern_color('R2'), linewidth=4, solid_capstyle="butt")
    # c.add_text("$I_1$", Vector(.05, 0, "cm") + Point(max(kern_times[i_inner]), 0, "diagram") >> Point(0, .1, "axis_diagram"), color=diplib.kern_color('R1'), horizontalalignment="left", verticalalignment="bottom")
    # c.add_text("$I_2$", Point(300, 0, "diagram") >> Point(0, .04, "axis_diagram"), color=diplib.kern_color('R2'), horizontalalignment="left", verticalalignment="bottom")
    # c.add_text("$\\frac{mean(\\textcolor[rgb]{"+",".join(map(str, diplib.kern_color('R2')))+"}{I_2})-mean(\\textcolor[rgb]{"+",".join(map(str,diplib.kern_color('R1')))+"}{I_1})}{stdev(\\textcolor[rgb]{"+",".join(map(str,diplib.kern_color('R2')))+"}{I_2})}$", Point(1.15, .4, "axis_diagram"), size=8)
    # c.add_text("Evidence or onset\nkernel", Point(.32, .8, "axis_diagram"))
    # ax.set_yticklabels(map(lambda x : "%g" % x, ax.get_yticks())) # Fix negative signs which don't show up for some latex-related reason
    # ax.set_xlabel("Time (ms)")
    # #ax.set_ylabel("Kernel value")
    # ax.set_yticks([])
    # ax.set_title("Dip index schematic")
    # sns.despine(ax=ax)
    
    
    
    
    c.save(FILENAME)



