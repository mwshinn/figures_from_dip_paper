from canvas import *
import diplib
import seaborn as sns

USE50 = True


regname = "sample"
#regname = "samplecoh"

c = Canvas(6.9, 4, fontsize=8)


LEFT1 = .6
RIGHT1 = 1.6
LEFT2 = 2.2
RIGHT2 = 3.2

for MONKEY in ["Q", "P"]:
    if MONKEY == "Q":
        HC,MC,LC = 70, 60, 53
    elif MONKEY == "P":
        HC,MC,LC = 63, 57, 52

    if MONKEY == "Q":
        c.set_default_unit("absolute")
    elif MONKEY == "P":
        c.add_unit("monkeyp", Vector(1, 1, "absolute"), Point(3.5, 0, "absolute"))
        c.set_default_unit("monkeyp")
    
    c.add_grid(["cell1kern"+MONKEY, "cell2kern"+MONKEY], 1, Point(LEFT1, 2.0), Point(RIGHT2, 3.0+(RIGHT1-LEFT1)), size=Vector(RIGHT1-LEFT1, RIGHT1-LEFT1))
    
    c.add_axis("means"+MONKEY, Point(LEFT1, .6), Point(RIGHT1, 1.6))
    c.add_grid(["posweights"+MONKEY, "negweights"+MONKEY], 2, Point(LEFT2, .6), Point(RIGHT2, 1.6), spacing=Vector(0, .1))
    
    # c.add_axis("varplot"+MONKEY, Point(LEFT1, .6), Point(RIGHT1, 1.6))
    # c.add_axis("pcs"+MONKEY, Point(LEFT2, .6), Point(RIGHT2, 1.6))
    
    c.add_text(f"Monkey { {'Q': 1, 'P':2}[MONKEY]}", Point(LEFT1+(RIGHT2-LEFT1)/2, 3.8), weight="bold", horizontalalignment="center")
    
    CELLS = diplib.get_cell_ids(monkey=MONKEY)
    regs = diplib.get_regression_models(monkey=MONKEY)
    
    
    
    #################### Example cells ####################
    
    example_cells = [3401, 4001] if MONKEY == "Q" else [6101, 7901] # 7901
    for i,cellid in enumerate(example_cells):
        ax = c.ax(f"cell{i+1}kern"+MONKEY)
        dm = regs[cellid]['dm']
        T,params = dm.get_regressor_from_output(regname, regs[cellid]['params'])
        bse = dm.get_regressor_from_output(regname, regs[cellid]['bse'])[1]
        ax.errorbar(T, params, yerr=bse, c='k')
        ax.set_xlim(0, 300)
        ax.set_clip_on(False)
        ax.set_xticks([0, 100, 200, 300])
        ax.set_ylabel("Sample kernel")
        ax.set_xlabel("Time from sample (ms)")
        ax.set_title(f"Cell {cellid}")
        ax.axhline(0, c='gray', linestyle='--')
        sns.despine(ax=ax)

    
    # c.add_legend(Point(.05, .25, "axis_cell2act"), [("800ms, high coh", {"color":diplib.get_color(ps=800,coh=HC), "linewidth": 5}),
    #                                              ("800ms, low coh", {"color":diplib.get_color(ps=800,coh=LC), "linewidth": 5}),
    #                                              ("400ms, high coh", {"color":diplib.get_color(coh=HC, ps=400), "linewidth": 5}),
    #                                              ("400ms, low coh", {"color":diplib.get_color(coh=LC, ps=400), "linewidth": 5})])
    
    #################### Significant regression coefficients ####################
    
    
    
    
    T_dip_overlay, dip_overlay = diplib.get_cell_conditional_activity(monkey=MONKEY, ps=800, coh=HC, hr_in_rf=True, align="sample", time_range=(0, 400))
    dip_overlay = dip_overlay/np.mean(dip_overlay)
    
    times = regs[CELLS[0]]['dm'].get_regressor_from_output(regname, regs[CELLS[0]]['params'])[0]
    kernels = []
    sig_neg_coef = times * 0
    sig_pos_coef = times * 0
    n_cell_regs = 0
    for cellid in CELLS:
        if cellid not in regs.keys(): continue
        dm = regs[cellid]['dm']
        estimate = dm.get_regressor_from_output(regname, regs[cellid]['params'])[1]
        kernels.append(estimate)
        sem = dm.get_regressor_from_output(regname, regs[cellid]['bse'])[1]
        sig_neg_coef += (estimate+1.64*sem < 0).astype(int)
        sig_pos_coef += (estimate-1.64*sem > 0).astype(int)
        n_cell_regs += 1
    
    
    ax = c.ax("posweights"+MONKEY)
    ax.bar(times, sig_pos_coef/n_cell_regs*100, 25, color='k')
    ax.set_ylabel("% positive")
    #ax.set_xticks([0, 100, 200, 300, 400])
    #ax.set_xticklabels([0, "", 200, "", 400])
    ax.set_xticks([])
    if MONKEY == "Q":
        ax.set_ylim(0, 45)
        ax.set_yticks([15, 30])
    elif MONKEY == "P":
        ax.set_ylim(0, 60)
        ax.set_yticks([20, 40])
    #ax.set_xlabel("Time after sample (ms)")
    ax.set_title("% cells with significant\n sample kernel")
    ax.set_xlim(0, 300)
    #ax.plot(T_dip_overlay*1000, dip_overlay/2-.3, alpha=.2, c='k')
    sns.despine(ax=ax)
    
    ax = c.ax("negweights"+MONKEY)
    ax.bar(times, sig_neg_coef/n_cell_regs*100, 25, color='k')
    #ax.set_title("Negative\ncoherence $\\times$ sample\nregression coefficients")
    ax.set_xticks([0, 100, 200, 300])
    if MONKEY == "Q":
        ax.set_ylim(0, 45)
        ax.set_yticks([15, 30])
    elif MONKEY == "P":
        ax.set_ylim(0, 60)
        ax.set_yticks([20, 40])
    ax.set_xlim(0, 300)
    ax.set_xlabel("Time from sample (ms)")
    ax.set_ylabel("% negative")
    ax.invert_yaxis()
    sns.despine(ax=ax, top=False)
    
    ax = c.ax("means"+MONKEY)
    km = np.mean(kernels, axis=0)
    ksem = np.std(kernels, axis=0)/np.sqrt(len(kernels))
    ax.errorbar(times, km, yerr=ksem*1.64, c='k')
    ax.axhline(0, c='gray', linestyle='--')
    ax.set_xlim(0, 300)
    ax.set_title("Mean evidence kernel")
    ax.set_xlabel("Time from sample (ms)")
    ax.set_ylabel("Mean kernel")
    sns.despine(ax=ax)
    
    #################### Plot PCs ####################
    
    # ax = c.ax("pcs"+MONKEY)
    # pcs = diplib.get_pcs_noncentered(monkey=MONKEY, regressor=("sample" if regname == "samplecoh" else "sample-nocoh")) # I know this is weird, historical reasons
    
    # if MONKEY == "Q":
    #     PCNUM = 0
    # elif MONKEY == "P":
    #     PCNUM = 0
    
    # for i in range(0, 4):
    #     ax.plot(times, ((1 if i==0 else 1)*pcs[1]*pcs[0])[:,i], c='k', lw=(1.5 if i!=PCNUM else 4), alpha=1)
    #     ax.plot(times, ((1 if i==0 else 1)*pcs[1]*pcs[0])[:,i], c=diplib.pc_colors(i), lw=(1 if i!=PCNUM else 3), alpha=1)
    #     #ax.plot(times, (pcs[1]*pcs[0])[:,i], c=diplib.kern_color(i), lw=(1.5 if i!=PCNUM else 3), alpha=(.4 if i!=PCNUM else 1))
    # ax.set_xlabel("Time from sample (ms)")
    # ax.set_ylabel("SV weight")
    # ax.set_title("E kernel SV components")
    # #c.add_legend(Point(2.3, 3.6), [("Var explained: %i%%" % (100*pcs[0][i]) , {"lw": 5, "color": colors[i]}) for i in range(0, 4)])
    # ax.set_xlim(0, 400)
    # ax.set_xticks([0, 100, 200, 300, 400])
    # ax.set_xticklabels([0, "", 200, "", 400])
    # ax.axhline(0, c='gray', linestyle='--')
    # sns.despine(ax=ax)
    
    # ax = c.ax("loadinghist"+MONKEY)
    # pc_weights = pcs[2][:,PCNUM]
    # ax.hist(pc_weights, bins=np.arange(-2.5, 2.5001, .25), color='k')
    # ax.set_xlim(-2.5, 2.5)
    # ax.axvline(0, c='gray', linestyle='--')
    # ax.set_ylabel("# cells")
    # ax.set_xlabel(f"E kernel SV{PCNUM+1} loading")
    # ax.set_title(f"Loading on SV{PCNUM+1} by cell")
    # sns.despine(ax=ax)
    
    # ax = c.ax("varplot"+MONKEY)
    # bars = ax.bar(range(1, 1+len(pcs[0])), pcs[0]*100, color='k')
    # for i in range(0, 4):
    #     bars[i].set_color(diplib.pc_colors(i))
    # ax.set_title("E kernel SV\nexplained variance")
    # ax.set_xlabel("SV #")
    # ax.set_xticks([1, 2, 3, 4])
    # ax.set_ylabel("Explanatory power")
    # sns.despine(ax=ax)


c.add_figure_labels([("a", "cell1kernQ"), ("b", "meansQ"), ("c", "posweightsQ"), ("d", "cell1kernP"), ("e", "meansP"), ("f", "posweightsP")])

c.save(f"figureS5.pdf")

    
