from canvas import Canvas, Vector, Point, Width, Height
import diplib
import scipy
import numpy as np
import seaborn as sns

c = Canvas(6.9, 8.5, fontsize=8, fontsize_ticks=7)
for MONKEY in ["Q", "P"]:
    if MONKEY == "P":
        c.add_unit("monkeyp", Vector(1, 1, "absolute"), Point(3.4, 0, "absolute"))
        c.set_default_unit("monkeyp")
    else:
        c.set_default_unit("absolute")
    
    
    c.add_grid(list(map(lambda x : x+MONKEY, ["pc", "pc_pre", "pspc", "pspc_pre", "fr", "fr_pre", "dsi", "dsi_pre", "vmindex", "vmindex_pre", "peakpeak", "peakpeak_pre"])), 6, Point(.55, .5), Point(2.9, 8), size=Vector(.9, .9))
    c.add_text(f"Monkey { {'Q': 1, 'P': 2}[MONKEY]}", (Point(0, 0, "pc"+MONKEY) | Point(1, 0, "pc_pre"+MONKEY)) >> Point(0, 8.4),  weight="bold", horizontalalignment="center")

    
    
    #################### Compare dip score to other stuff ####################
    
    pcs = diplib.get_pcs_noncentered(MONKEY)
    pspcs = diplib.get_pcs_noncentered(MONKEY, regressor="presample")
    # all_cpds = diplib.get_regression_cpds(MONKEY)
    # The np.nan here is a hack: it should throw an error in the
    # unlikely case that samplecoh isn't the second element of the
    # list
    # cpds_evidence = [all_cpds[c][1][1]['cpd'] if all_cpds[c][1][0] == ['samplecoh'] else np.nan for c in all_cpds.keys()]
    
    items = [("pc", pcs[2][:,0], "Evidence kernel\nSV1 factor score"),
             ("pspc", pspcs[2][:,1], "Stimulus kernel\nSV2 factor score"),
             #("cpd", cpds_evidence, "Evidence kernel CPD"),
             ("fr", diplib.get_cell_fr(MONKEY, 'iti'), "Firing rate"),
             ("dsi", diplib.get_dsi_index(MONKEY), "DSI"),
             ("vmindex", diplib.get_vm_index(MONKEY), "VM index"),
             ("peakpeak", diplib.get_peak_to_peak(MONKEY), "Peak-to-peak width")]
    
    for item in items:
        for pre in [True, False]:
            if pre:
                dip_scores = diplib.get_presample_dip_score(MONKEY)
                name = item[0]+"_pre"
            else:
                dip_scores = diplib.get_dip_score(MONKEY)
                name = item[0]
            ax = c.ax(name+MONKEY)
            if item == items[0]:
                ax.set_title("Stimulus dip" if pre else "Evidence dip")
            ax.plot(dip_scores, item[1], markersize=2, c='k', linestyle="none", marker='o')
            ax.plot(dip_scores, np.poly1d(np.polyfit(dip_scores, item[1], 1))(dip_scores), c="k")
            r,p = scipy.stats.spearmanr(dip_scores, item[1])
            #ax.set_title(f"r={r:.2f}, p={p:.4f}")
            ptext = f"$p={p:.3f}$" if p >= .0001 else "p<$10^{-3}$"
            c.add_text(f"$r_s={r:.2f}$\n{ptext}", Point(.9, .7, "axis_"+name+MONKEY), horizontalalignment="left", size=7,
                       bbox=dict(boxstyle="round", ec=(0.0, 0.0, 0.0, 0.3), fc=(.95, .95, .95, .7)))
            if item[0] == "cpd" and MONKEY == "Q":
                ax.set_yticks([0, .001, .002])
            if item == items[-1]:
                ax.set_xlabel("Stimulus dip index" if pre else "Evidence dip index")
            else:
                ax.set_xticklabels(['', ''])
            if not pre:
                ax.set_ylabel(item[2])
            else:
                ax.set_yticklabels([])
            sns.despine(ax=ax)
    


c.save("figureS8.pdf")
