from canvas import *
import diplib
import seaborn as sns
import matplotlib.pyplot as plt
import sys


USE50 = True
USE53 = True
USE60 = True
DRAFT = False
c = Canvas(6.9, 7.0, fontsize=8)

for MONKEY in ["Q", "P"]:
    if MONKEY == "Q":
        HC,MC,LC = 70, 60, 53
        offset = 3.4
    elif MONKEY == "P":
        HC,MC,LC = 63, 57, 52
        offset = 0
    c.add_axis("all_rts"+MONKEY, Point(.6, 2.2+offset, "absolute"), Point(6.5, 3.2+offset, "absolute"))
    c.add_grid(["data"+MONKEY, ("ddm"+MONKEY if MONKEY == "P" else None)], 1, Point(3, .5+offset, "absolute"), Point(6.5, 1.7+offset, "absolute"), size=Vector(1.2, 1.2, "absolute"))
    
    rts = diplib.spikes_df(MONKEY)
    if MONKEY == "Q":
        diplib.make_gridlegend(c, Point(0, .2, "axis_data"+MONKEY)-Vector(2.3, 0, "absolute"))
    
    ##################### All RTs ####################
    
    SMOOTH = 5
    ax = c.ax("all_rts"+MONKEY)
    for coh in [LC, MC, HC]:
        for ps in [0, 400, 800]:
            ax.plot(*diplib.get_rt_conditional_activity(monkey=MONKEY, coh=coh, ps=ps, smooth=SMOOTH, time_range=(-200, 1400), align="presample"), color=diplib.get_color(coh=coh, ps=ps))
    
    ax.set_title("RT distribution (presample-aligned)")
    ax.set_xticklabels(["", "0", "", "400", "", "800"])
    ax.set_ylabel("RT histogram")
    if MONKEY == "Q":
        ax.set_yticks([0, 2, 4, 6, 8])
        ax.set_ylim(0, 8)
    elif MONKEY == "P":
        ax.set_yticks([0, .5, 1])
        ax.set_ylim(0, 1.4)
    
    ax.set_xlim(-.2, 1.4225)
    ax.xaxis.set_minor_locator(plt.matplotlib.ticker.AutoMinorLocator(2))
    
    loffset = Width(.03, "absolute")
    ax.axvline(0, color='k', linestyle="-")
    c.add_text("Presample\nstart", -loffset+Point(-.005, 0, "all_rts"+MONKEY) >> Point(0, 1, "axis_all_rts"+MONKEY), color='k', horizontalalignment="right", verticalalignment="top")
    ax.axvline(0, color=diplib.get_color(ps=0, coh=HC), linestyle="--")
    c.add_text("Sample\nstart\n(0 ms PS)", loffset+Point(.005, 0, "all_rts"+MONKEY) >> Point(0, 1, "axis_all_rts"+MONKEY), color=diplib.get_color(ps=0, coh=HC), horizontalalignment="left", verticalalignment="top")
    ax.axvline(.4, color=diplib.get_color(ps=400, coh=HC), linestyle="--")
    c.add_text("Sample\nstart\n(400 ms PS)", loffset+Point(.4, 0, "all_rts"+MONKEY) >> Point(0, 1, "axis_all_rts"+MONKEY), color=diplib.get_color(ps=400, coh=HC), horizontalalignment="left", verticalalignment="top")
    ax.axvline(.8, color=diplib.get_color(ps=800, coh=HC), linestyle="--")
    c.add_text("Sample\nstart\n(800 ms PS)", loffset+Point(.8, 0, "all_rts"+MONKEY) >> Point(0, 1, "axis_all_rts"+MONKEY), color=diplib.get_color(ps=800, coh=HC), horizontalalignment="left", verticalalignment="top")
    
    ax.axvspan(.75, 1.08, color=diplib.get_color(ps=800, coh=HC), alpha=.1, zorder=-1)
    c.add_text("Time from presample (ms)", Point(1, -.07, "axis_all_rts"+MONKEY), horizontalalignment="right", verticalalignment="top")
    
    sns.despine(ax=ax)
    
    
    #################### RTs (800ms) ####################
    print("Starting data")
    ax = c.ax("data"+MONKEY)
    
    SMOOTH = 5
    T, activity70 = diplib.get_rt_conditional_activity(monkey=MONKEY, coh=HC, ps=800, smooth=SMOOTH, time_range=(-200, 500), align="sample")
    ax.plot(T, activity70, c=diplib.get_color(ps=800, coh=HC))
    if not DRAFT:
        bounds70 = diplib.bootstrap_rts_ci(monkey=MONKEY, N=1000, coh=HC, ps=800, time_range=(-200, 500), seed=1, smooth=SMOOTH)
        ax.fill_between(T, bounds70[0,:], bounds70[1,:], color=diplib.get_color(ps=800, coh=HC), alpha=.4)
    if USE60:
        activity60 = diplib.get_rt_conditional_activity(monkey=MONKEY, coh=MC, ps=800, smooth=SMOOTH, time_range=(-200, 500), align="sample")[1]
        ax.plot(T, activity60, c=diplib.get_color(ps=800, coh=MC))
        if not DRAFT:
            bounds60 = diplib.bootstrap_rts_ci(monkey=MONKEY, N=1000, coh=MC, ps=800, smooth=SMOOTH, time_range=(-200, 500), seed=1)
            ax.fill_between(T, bounds60[0,:], bounds60[1,:], color=diplib.get_color(ps=800, coh=MC), alpha=.4)
    
    if USE50:
        activity50 = diplib.get_rt_conditional_activity(monkey=MONKEY, coh=50, smooth=SMOOTH, time_range=(600, 1300), align="presample")[1]
        ax.plot(T, activity50, c=diplib.get_color(ps=800, coh=50))
        if not DRAFT:
            bounds50 = diplib.bootstrap_rts_ci(monkey=MONKEY, N=1000, coh=50, time_range=(600, 1300), smooth=SMOOTH, align="presample", seed=1)
            ax.fill_between(T, bounds50[0,:], bounds50[1,:], color=diplib.get_color(ps=800, coh=50), alpha=.4)
        #sigs = diplib.bootstrap_rts_significance(params1=dict(monkey=MONKEY, coh=50, time_range=(600, 1300), align="presample"), params2=dict(monkey=MONKEY, coh=HC, ps=800, align="sample", time_range=(-200, 500)), N=500, seed=1)
    
    if USE53:
        activity53 = diplib.get_rt_conditional_activity(monkey=MONKEY, coh=LC, ps=800, smooth=SMOOTH, time_range=(-200, 500), align="sample")[1]
        ax.plot(T, activity53, c=diplib.get_color(ps=800, coh=LC))
        if not DRAFT:
            bounds53 = diplib.bootstrap_rts_ci(monkey=MONKEY, N=1000, coh=LC, ps=800, smooth=SMOOTH, time_range=(-200, 500), seed=1)
            ax.fill_between(T, bounds53[0,:], bounds53[1,:], color=diplib.get_color(ps=800, coh=LC), alpha=.4)
            sigs = diplib.bootstrap_rts_significance(params1=dict(monkey=MONKEY, coh=LC, ps=800, time_range=(-200, 500)), params2=dict(monkey=MONKEY, coh=HC, ps=800, time_range=(-200, 500)), N=500, seed=1)
    
    
    ax.set_xlim(-.05, .28)
    ax.set_xticks([0, .1, .2])
    ax.set_xticklabels([0, 100, 200])
    if MONKEY == "Q":
        ax.set_ylim(0, 1.3)
        ax.set_yticks([0, 1])
    else:
        ax.set_ylim(0, .12)
        ax.set_yticks([0, .05, .1])
    
    ax.set_xlabel("Time from sample (ms)")
    if not DRAFT:
        diplib.plot_significance(c, "data"+MONKEY, T[(sigs<.01) & (T > -.05) & (T < .28)], dx=T[1]-T[0])
    ax.axvline(0, color=diplib.get_color(ps=800, coh=HC), linestyle="--")
    sns.despine(right=False, top=False, ax=ax)
    ax.set_ylabel("RT histogram")
    #ax.axvspan(-.05, .28, color=diplib.get_color(ps=800, coh=HC), alpha=.1)
    if MONKEY == "Q":
        c.add_arrow(Point(.06, .06, ("data"+MONKEY, "axis_data"+MONKEY)), Point(.13, .12, "data"+MONKEY))
    else:
        c.add_arrow(Point(.06, .06, ("data"+MONKEY, "axis_data"+MONKEY)), Point(.13, .02, "data"+MONKEY))
    
    c.add_text("Dip", Point(.03, .06, ("data"+MONKEY, "axis_data"+MONKEY)))
    
    
    #################### DDM (800ms) ####################
    if MONKEY == "P":
        print("Starting DDM")
        
        ax = c.ax("ddm"+MONKEY)
        N_trials = 500
        T, activity70 = diplib.get_ddm_conditional_rts(coh=HC, ps=800, time_range=(-200, 500))
        ax.plot(T, activity70, c=diplib.get_color(ps=800, coh=HC))
        if USE50:
            activity50 = diplib.get_ddm_conditional_rts(coh=50, ps=800, time_range=(-200, 500))[1]
            ax.plot(T, activity50, c=diplib.get_color(ps=800, coh=50))
        
        if USE53:
            activity53 = diplib.get_ddm_conditional_rts(coh=LC, ps=800, time_range=(-200, 500))[1]
            ax.plot(T, activity53, c=diplib.get_color(ps=800, coh=LC))
        
        if USE60:
            activity60 = diplib.get_ddm_conditional_rts(coh=MC, ps=800, time_range=(-200, 500))[1]
            ax.plot(T, activity60, c=diplib.get_color(ps=800, coh=MC))
        
        ax.set_ylim(0, 1.3)
        ax.set_xlim(-.05, .28)
        ax.set_xticks([0, .1, .2])
        ax.set_xticklabels([0, 100, 200])
        ax.set_yticks([0, 1])
        ax.set_xlabel("Time from sample (ms)")
        sns.despine(right=False, top=False, ax=ax)
        
        c.add_text("GDDM RT prediction", Point(.5, 1.1, "axis_ddmP"), weight="bold", size=9)
        
        c.add_arrow(Point(.2, .25, "axis_ddm"+MONKEY), Point(.8, .55, "axis_ddm"+MONKEY))
        c.add_text("Monotonic\nincrease", Point(.2, .25, "axis_ddm"+MONKEY)+Vector(0, 0.06, "inches"), rotation=27, horizontalalignment="left", verticalalignment="bottom")
        
    
    
    #################### Connecting lines ####################
    
    c.add_poly([Point(.75, 0, "all_rts"+MONKEY) >> Point(0, 0, "axis_all_rts"+MONKEY),
                Point(0, 1, "axis_data"+MONKEY),
                Point(1, 1, "axis_data"+MONKEY),
                Point(1.08, 0, "all_rts"+MONKEY) >> Point(0, 0, "axis_all_rts"+MONKEY)],
               facecolor=diplib.get_color(ps=800, coh=HC), alpha=.1, fill=True, edgecolor='k')


c.add_text("Monkey 1", Point(0, 1, "axis_all_rtsQ") + Vector(-1.4, .45, "cm"), weight="bold", size=9, ha="left")
c.add_text("Monkey 2", Point(0, 1, "axis_all_rtsP") + Vector(-1.4, .45, "cm"), weight="bold", size=9, ha="left")

c.add_figure_labels([("a", "all_rtsQ", Vector(-.5, -.3, "cm")),
                     ("b", "dataQ", Vector(0, -.3, "cm")),
                     ("c", "all_rtsP", Vector(-.5, -.3, "cm")),
                     ("d", "dataP", Vector(0, -.3, "cm")),
                     ("e", "ddmP", Vector(0, -.3, "cm"))])

c.save(f"figure2.pdf")
