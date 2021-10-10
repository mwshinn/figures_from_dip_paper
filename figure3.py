from canvas import *
import diplib
import matplotlib
import seaborn as sns
import sys


MONKEY = "Q"
USE50 = True
USE53 = True
USE60 = True
SIG50 = True
DRAFT = False
alpha = .2



conds = (dict(), "")
zscore = ("bothrf_subtract", "bothrf_subtract")

rts = diplib.spikes_df(MONKEY)


if DRAFT:
    USE50 = USE53 = USE60 = False

c = Canvas(6.9, 8.1, fontsize=8)
offset = 3.55
c.add_axis("all_fefQ", Point(.6, 3.3+offset, "absolute"), Point(6.5, 4.3+offset, "absolute"))
c.add_grid(["data0Q", "data400Q", "dataQ"], 1, Point(.7, 1.9+offset, "absolute"), Point(5.2, 2.9+offset, "absolute"), size=Vector(1.1, 1.1, "absolute"))
offset = 0.4
c.add_axis("all_fefP", Point(.6, 3.3+offset, "absolute"), Point(6.5, 4.3+offset, "absolute"))
c.add_grid(["data0P", "data400P", "dataP"], 1, Point(.7, 1.9+offset, "absolute"), Point(5.2, 2.9+offset, "absolute"), size=Vector(1.1, 1.1, "absolute"))
c.add_grid(["ddm0", "ddm400", "ddm"], 1, Point(.7, .4, "absolute"), Point(5.2, 1.5, "absolute"), size=Vector(1.1, 1.1, "absolute"))



c.add_figure_labels([("a", "all_fefQ", Vector(-.5, -.3, "cm")),
                     ("b", "data0Q", Vector(0, -.3, "cm")),
                     ("c", "data400Q", Vector(0, -.3, "cm")),
                     ("d", "dataQ", Vector(0, -.3, "cm")),
                     ("e", "all_fefP", Vector(-.5, -.3, "cm")),
                     ("f", "data0P", Vector(0, -.3, "cm")),
                     ("g", "data400P", Vector(0, -.3, "cm")),
                     ("h", "dataP", Vector(0, -.3, "cm")),
                     ("i", "ddm0", Vector(0, -.3, "cm")),
                     ("j", "ddm400", Vector(0, -.3, "cm")),
                     ("k", "ddm", Vector(0, -.3, "cm"))])

diplib.make_gridlegend(c, Point(1.15, .2, "axis_dataQ")+Vector(0, 0, "absolute"))


c.add_text("Monkey 1", Point(0, 1, "axis_all_fefQ") + Vector(-1.4, .45, "cm"), weight="bold", size=9, ha="left")
c.add_text("Monkey 2", Point(0, 1, "axis_all_fefP") + Vector(-1.4, .45, "cm"), weight="bold", size=9, ha="left")
c.add_text("GDDM DV prediction", Point(0, 1, "axis_ddm0") + Vector(-1.4, .45, "cm")-Vector(.1, 0, "absolute"), weight="bold", size=9, ha="left")

#################### All FEF ####################

for MONKEY in ["Q", "P"]:
    if MONKEY == "Q":
        HC,MC,LC = 70, 60, 53
    elif MONKEY == "P":
        HC,MC,LC = 63, 57, 52
    SMOOTH = 3
    ax = c.ax("all_fef"+MONKEY)
    for coh in [LC, MC, HC]:
        for ps in [0, 400, 800]:
            ax.plot(*diplib.get_mean_conditional_activity(monkey=MONKEY, coh=coh, ps=ps, smooth=SMOOTH, time_range=(-200, 1400), align="presample", zscore=zscore[0], **conds[0]), color=diplib.get_color(coh=coh, ps=ps))
    
    ax.set_title("FEF activity (presample-aligned)")
    ax.set_xticklabels(["", "0", "", "400", "", "800"])
    ax.set_ylabel("FEF activity")
    if MONKEY == "Q":
        ax.set_ylim(-10, 20)
    else:
        ax.set_ylim(-10, 30)
    
    ax.set_xlim(-.2, 1.4225)
    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
    sns.despine(ax=ax)
    
    loffset = Width(.03, "absolute")
    ax.axvline(0, color='k', linestyle="-")
    c.add_text("Presample\nstart", -loffset+Point(-.005, 0, "all_fef"+MONKEY) >> Point(0, 1, "axis_all_fef"+MONKEY), color='k', horizontalalignment="right", verticalalignment="top")
    ax.axvline(0, color=diplib.get_color(ps=0, coh=HC), linestyle="--")
    c.add_text("Sample start\n(0 ms PS)", loffset+Point(0, 0, "all_fef"+MONKEY) >> Point(0, 1, "axis_all_fef"+MONKEY), color=diplib.get_color(ps=0, coh=HC), horizontalalignment="left", verticalalignment="top")
    ax.axvline(.4, color=diplib.get_color(ps=400, coh=HC), linestyle="--")
    c.add_text("Sample start\n(400 ms PS)", loffset+Point(.4, 0, "all_fef"+MONKEY) >> Point(0, 1, "axis_all_fef"+MONKEY), color=diplib.get_color(ps=400, coh=HC), horizontalalignment="left", verticalalignment="top")
    ax.axvline(.8, color=diplib.get_color(ps=800, coh=HC), linestyle="--")
    c.add_text("Sample start\n(800 ms PS)", loffset+Point(.8, 0, "all_fef"+MONKEY) >> Point(0, 1, "axis_all_fef"+MONKEY), color=diplib.get_color(ps=800, coh=HC), horizontalalignment="left", verticalalignment="top")
    
    ax.axvspan(-.05, .28, color=diplib.get_color(ps=0, coh=HC), alpha=.1, zorder=-1)
    ax.axvspan(.35, .68, color=diplib.get_color(ps=400, coh=HC), alpha=.1, zorder=-1)
    ax.axvspan(.75, 1.08, color=diplib.get_color(ps=800, coh=HC), alpha=.1, zorder=-1)
    
    #################### Data (800ms) ####################
    print("Starting data"+MONKEY)
    SMOOTH = 3
    ax = c.ax("data"+MONKEY)
    T, activity70 = diplib.get_mean_conditional_activity(monkey=MONKEY, coh=HC, ps=800, **conds[0], smooth=SMOOTH, time_range=(-200, 400), zscore=zscore[0])
    ax.plot(T, activity70, c=diplib.get_color(ps=800, coh=HC))
    if not DRAFT:
        bounds70 = diplib.bootstrap_trialwise_ci(monkey=MONKEY, N=1000, coh=HC, ps=800, **conds[0], smooth=SMOOTH, time_range=(-200, 400), zscore=zscore[0])
        ax.fill_between(T, bounds70[0,:], bounds70[1,:], color=diplib.get_color(ps=800, coh=HC), alpha=alpha, zorder=-1)
    
    if USE50:
        activity50 = diplib.get_mean_conditional_activity(monkey=MONKEY, coh=50, smooth=SMOOTH, **conds[0], time_range=(600, 1200), zscore=zscore[1], align="presample")[1]
        ax.plot(T, activity50, c=diplib.get_color(coh=50))
        if not DRAFT:
            bounds50 = diplib.bootstrap_trialwise_ci(monkey=MONKEY, N=1000, coh=50, smooth=SMOOTH, **conds[0], time_range=(600, 1200), zscore=zscore[1], align="presample")
            ax.fill_between(T, bounds50[0,:], bounds50[1,:], color=diplib.get_color(coh=50), alpha=alpha, zorder=-1)
    
    if USE53:
        activity53 = diplib.get_mean_conditional_activity(monkey=MONKEY, coh=LC, ps=800, **conds[0], smooth=SMOOTH, time_range=(-200, 400), zscore=zscore[0])[1]
        ax.plot(T, activity53, c=diplib.get_color(ps=800, coh=LC))
        if not DRAFT:
            bounds53 = diplib.bootstrap_trialwise_ci(monkey=MONKEY, N=1000, coh=LC, ps=800, **conds[0], smooth=SMOOTH, time_range=(-200, 400), zscore=zscore[0])
            ax.fill_between(T, bounds53[0,:], bounds53[1,:], color=diplib.get_color(ps=800, coh=LC), alpha=alpha, zorder=-1)
    
    if USE60:
        activity60 = diplib.get_mean_conditional_activity(monkey=MONKEY, coh=MC, ps=800, **conds[0], smooth=SMOOTH, time_range=(-200, 400), zscore=zscore[0])[1]
        ax.plot(T, activity60, c=diplib.get_color(ps=800, coh=MC))
        if not DRAFT:
            bounds60 = diplib.bootstrap_trialwise_ci(monkey=MONKEY, N=1000, coh=MC, ps=800, **conds[0], smooth=SMOOTH, time_range=(-200, 400), zscore=zscore[0])
            ax.fill_between(T, bounds60[0,:], bounds60[1,:], color=diplib.get_color(ps=800, coh=MC), alpha=alpha, zorder=-1)
    
    if not DRAFT:
        if SIG50:
            sigs = diplib.bootstrap_significance(MONKEY, dict(coh=50, time_range=(600, 1200), **conds[0], zscore=zscore[1], align="presample"), dict(coh=HC, ps=800, time_range=(-200, 400), **conds[0], zscore=zscore[0]), N=10000)
        else:
            sigs = diplib.bootstrap_significance(MONKEY, dict(coh=LC, ps=800, **conds[0], zscore=zscore[0], time_range=(-200, 400)), dict(coh=HC, ps=800, time_range=(-200, 400), **conds[0], zscore=zscore[0]), N=10000)
    
    
    
    sns.despine(right=False, top=False, ax=ax)
    ax.set_xlim(-.05, .28)
    ax.set_xticks([0, .1, .2])
    ax.set_xticklabels(["0", "100", "200"])
    if not DRAFT:
        diplib.plot_significance(c, "data"+MONKEY, T[(sigs<.05) & (T>-.05) & (T<.28)], dx=T[1]-T[0])
    
    ax.axvline(0, color=diplib.get_color(ps=800, coh=HC), linestyle="--")
    
    ax.set_xlabel("Time from sample (ms)")
    if MONKEY == "Q":
        ax.set_ylim(-2, 12)
    else:
        ax.set_ylim(3, 17)
    
    if MONKEY == "Q":
        c.add_arrow(Point(.06, .1, ("data"+MONKEY, "axis_data"+MONKEY)), Point(.14, 0, "data"+MONKEY))
    else:
        c.add_arrow(Point(.06, .1, ("data"+MONKEY, "axis_data"+MONKEY)), Point(.14, 6, "data"+MONKEY))
    
    c.add_text("Dip", Point(.03, .1, ("data"+MONKEY, "axis_data"+MONKEY)))
    
    
    #################### Data (400 ms) ####################
    print("Starting data (400ms)")
    ax = c.ax("data400"+MONKEY)
    T, activity70 = diplib.get_mean_conditional_activity(monkey=MONKEY, coh=HC, ps=400, **conds[0], smooth=SMOOTH, zscore=zscore[0], time_range=(-200, 400))
    ax.plot(T, activity70, c=diplib.get_color(ps=400, coh=HC))
    if not DRAFT:
        bounds70 = diplib.bootstrap_trialwise_ci(monkey=MONKEY, N=1000, coh=HC, ps=400, **conds[0], smooth=SMOOTH, zscore=zscore[0], time_range=(-200, 400))
        ax.fill_between(T, bounds70[0,:], bounds70[1,:], color=diplib.get_color(ps=400, coh=HC), alpha=alpha, zorder=-1)
    
    if USE50:
        activity50 = diplib.get_mean_conditional_activity(monkey=MONKEY, coh=50, **conds[0], smooth=SMOOTH, zscore=zscore[1], time_range=(200, 800), align="presample")[1]
        ax.plot(T, activity50, c=diplib.get_color(coh=50))
        if not DRAFT:
            bounds50 = diplib.bootstrap_trialwise_ci(monkey=MONKEY, N=1000, coh=50, **conds[0], smooth=SMOOTH, zscore=zscore[1], time_range=(200, 800), align="presample")
            ax.fill_between(T, bounds50[0,:], bounds50[1,:], color=diplib.get_color(coh=50), alpha=alpha, zorder=-1)
    
    if USE53:
        activity53 = diplib.get_mean_conditional_activity(monkey=MONKEY, coh=LC, ps=400, **conds[0], smooth=SMOOTH, zscore=zscore[0], time_range=(-200, 400))[1]
        ax.plot(T, activity53, c=diplib.get_color(ps=400, coh=LC))
        if not DRAFT:
            bounds53 = diplib.bootstrap_trialwise_ci(monkey=MONKEY, N=1000, coh=LC, ps=400, **conds[0], smooth=SMOOTH, zscore=zscore[0], time_range=(-200, 400))
            ax.fill_between(T, bounds53[0,:], bounds53[1,:], color=diplib.get_color(ps=400, coh=LC), alpha=alpha, zorder=-1)
    
    if USE60:
        activity60 = diplib.get_mean_conditional_activity(monkey=MONKEY, coh=MC, ps=400, **conds[0], smooth=SMOOTH, zscore=zscore[0], time_range=(-200, 400))[1]
        ax.plot(T, activity60, c=diplib.get_color(ps=400, coh=MC))
        if not DRAFT:
            bounds60 = diplib.bootstrap_trialwise_ci(monkey=MONKEY, N=1000, coh=MC, ps=400, **conds[0], smooth=SMOOTH, zscore=zscore[0], time_range=(-200, 400))
            ax.fill_between(T, bounds60[0,:], bounds60[1,:], color=diplib.get_color(ps=400, coh=MC), alpha=alpha, zorder=-1)
    
    if not DRAFT:
        if SIG50:
            sigs = diplib.bootstrap_significance(MONKEY, dict(coh=50, align="presample", **conds[0], zscore=zscore[1], time_range=(200, 800)), dict(coh=HC, ps=400, **conds[0], zscore=zscore[0], time_range=(-200, 400)), N=10000)
        else:
            sigs = diplib.bootstrap_significance(MONKEY, dict(coh=LC, ps=400, **conds[0], zscore=zscore[0], time_range=(-200, 400)), dict(coh=HC, ps=400, zscore=zscore[0], time_range=(-200, 400), **conds[0]), N=10000)
    
    sns.despine(right=False, top=False, ax=ax)
    ax.set_xlim(-.05, .28)
    ax.set_xticks([0, .1, .2])
    ax.set_xticklabels(["0", "100", "200"])
    if not DRAFT:
        diplib.plot_significance(c, "data400"+MONKEY, T[(sigs<.05) & (T>-.05) & (T<.28)], dx=T[1]-T[0])
    
    ax.axvline(0, color=diplib.get_color(ps=400, coh=HC), linestyle="--")
    #ax.axvspan(-.05, .28, color=diplib.get_color(ps=400, coh=HC), alpha=.1)
    ax.set_xlabel("Time from sample (ms)")
    ax.set_ylim(-7, 7)
    if MONKEY == "Q":
        c.add_arrow(Point(.06, .1, ("data400"+MONKEY, "axis_data400"+MONKEY)), Point(.15, -3, "data400"+MONKEY))
    else:
        c.add_arrow(Point(.06, .1, ("data400"+MONKEY, "axis_data400"+MONKEY)), Point(.15, -1.5, "data400"+MONKEY))
    
    c.add_text("Dip", Point(.03, .1, ("data400"+MONKEY, "axis_data400"+MONKEY)))
    
    #################### Data (0 ms) ####################
    print("Starting data (0ms)")
    ax = c.ax("data0"+MONKEY)
    T, activity70 = diplib.get_mean_conditional_activity(monkey=MONKEY, coh=HC, ps=0, **conds[0], smooth=SMOOTH, zscore=zscore[0], time_range=(-200, 400))
    ax.plot(T, activity70, c=diplib.get_color(ps=0, coh=HC))
    if not DRAFT:
        bounds70 = diplib.bootstrap_trialwise_ci(monkey=MONKEY, N=1000, coh=HC, ps=0, **conds[0], smooth=SMOOTH, zscore=zscore[0], time_range=(-200, 400))
        ax.fill_between(T, bounds70[0,:], bounds70[1,:], color=diplib.get_color(ps=0, coh=HC), alpha=alpha, zorder=-1)
    
    if USE50:
        activity50 = diplib.get_mean_conditional_activity(monkey=MONKEY, coh=50, **conds[0], smooth=SMOOTH, zscore=zscore[1], time_range=(-200, 400), align="presample")[1]
        ax.plot(T, activity50, c=diplib.get_color(coh=50))
        if not DRAFT:
            bounds50 = diplib.bootstrap_trialwise_ci(monkey=MONKEY, N=1000, coh=50, **conds[0], smooth=SMOOTH, zscore=zscore[1], time_range=(-200, 400), align="presample")
            ax.fill_between(T, bounds50[0,:], bounds50[1,:], color=diplib.get_color(coh=50), alpha=alpha, zorder=-1)
    
    if USE53:
        activity53 = diplib.get_mean_conditional_activity(monkey=MONKEY, coh=LC, ps=0, **conds[0], smooth=SMOOTH, zscore=zscore[0], time_range=(-200, 400))[1]
        ax.plot(T, activity53, c=diplib.get_color(ps=0, coh=LC))
        if not DRAFT:
            bounds53 = diplib.bootstrap_trialwise_ci(monkey=MONKEY, N=1000, coh=LC, ps=0, **conds[0], smooth=SMOOTH, zscore=zscore[0], time_range=(-200, 400))
            ax.fill_between(T, bounds53[0,:], bounds53[1,:], color=diplib.get_color(ps=0, coh=LC), alpha=alpha, zorder=-1)
    
    if USE60:
        activity60 = diplib.get_mean_conditional_activity(monkey=MONKEY, coh=MC, ps=0, **conds[0], smooth=SMOOTH, zscore=zscore[0], time_range=(-200, 400))[1]
        ax.plot(T, activity60, c=diplib.get_color(ps=0, coh=MC))
        if not DRAFT:
            bounds60 = diplib.bootstrap_trialwise_ci(monkey=MONKEY, N=1000, coh=MC, ps=0, **conds[0], smooth=SMOOTH, zscore=zscore[0], time_range=(-200, 400))
            ax.fill_between(T, bounds60[0,:], bounds60[1,:], color=diplib.get_color(ps=0, coh=MC), alpha=alpha, zorder=-1)
    
    if not DRAFT:
        if SIG50:
            sigs = diplib.bootstrap_significance(MONKEY, dict(coh=50, align="presample", **conds[0], zscore=zscore[1], time_range=(-200, 400)), dict(coh=HC, ps=0, **conds[0], zscore=zscore[0], time_range=(-200, 400)), N=10000)
        else:
            sigs = diplib.bootstrap_significance(MONKEY, dict(coh=LC, ps=0, **conds[0], zscore=zscore[0], time_range=(-200, 400)), dict(coh=HC, ps=0, zscore=zscore[0], time_range=(-200, 400), **conds[0]), N=10000)
    
    sns.despine(right=False, top=False, ax=ax)
    ax.set_xlim(-.05, .28)
    ax.set_xticks([0, .1, .2])
    ax.set_xticklabels(["0", "100", "200"])
    if not DRAFT:
        diplib.plot_significance(c, "data0"+MONKEY, T[(sigs<.05) & (T>-.05) & (T<.28)], dx=T[1]-T[0])
    
    ax.axvline(0, color=diplib.get_color(ps=0, coh=HC), linestyle="--")
    #ax.axvspan(-.05, .28, color=diplib.get_color(ps=0, coh=HC), alpha=.1)
    ax.set_ylabel("FEF activity")
    ax.set_xlabel("Time from sample (ms)")
    if MONKEY == "Q":
        ax.set_ylim(-7, 7)
    else:
        ax.set_ylim(-12, 2)
        
    c.add_poly([Point(-.05, 0, "all_fef"+MONKEY) >> Point(0, 0, "axis_all_fef"+MONKEY),
                Point(0, 1, "axis_data0"+MONKEY),
                Point(1, 1, "axis_data0"+MONKEY),
                Point(.28, 0, "all_fef"+MONKEY) >> Point(0, 0, "axis_all_fef"+MONKEY)],
               facecolor=diplib.get_color(ps=0, coh=HC), alpha=.1, fill=True, edgecolor='k')
    
    
    c.add_poly([Point(.35, 0, "all_fef"+MONKEY) >> Point(0, 0, "axis_all_fef"+MONKEY),
                Point(0, 1, "axis_data400"+MONKEY),
                Point(1, 1, "axis_data400"+MONKEY),
                Point(.68, 0, "all_fef"+MONKEY) >> Point(0, 0, "axis_all_fef"+MONKEY)],
               facecolor=diplib.get_color(ps=400, coh=HC), alpha=.1, fill=True, edgecolor='k')
    
    c.add_poly([Point(.75, 0, "all_fef"+MONKEY) >> Point(0, 0, "axis_all_fef"+MONKEY),
                Point(0, 1, "axis_data"+MONKEY),
                Point(1, 1, "axis_data"+MONKEY),
                Point(1.08, 0, "all_fef"+MONKEY) >> Point(0, 0, "axis_all_fef"+MONKEY)],
               facecolor=diplib.get_color(ps=800, coh=HC), alpha=.1, fill=True, edgecolor='k')


#################### DDM (800ms) ####################

print("Starting ddm")
ax = c.ax("ddm")
N_trials = 50
T, activity70 = diplib.get_ddm_mean_activity(coh=HC, ps=800, highreward=1, time_range=(-200, 400))
ax.plot(T, activity70, c=diplib.get_color(ps=800, coh=HC))
if USE50:
    T, activity50 = diplib.get_ddm_mean_activity(coh=50, ps=800, highreward=1, time_range=(-200, 400))
    ax.plot(T, activity50, c=diplib.get_color(ps=800, coh=50))

if USE53:
    T, activity53 = diplib.get_ddm_mean_activity(coh=LC, ps=800, highreward=1, time_range=(-200, 400))
    ax.plot(T, activity53, c=diplib.get_color(ps=800, coh=LC))

if USE60:
    T, activity60 = diplib.get_ddm_mean_activity(coh=MC, ps=800, highreward=1, time_range=(-200, 400))
    ax.plot(T, activity60, c=diplib.get_color(ps=800, coh=MC))


ax.set_xlabel("Time from sample (ms)")
ax.axvline(0, color=diplib.get_color(ps=800, coh=HC), linestyle="--")
sns.despine(right=False, top=False, ax=ax)
ax.set_yticks([0, .5, 1])
ax.set_ylim(0, .8)
ax.set_xlim(-.05, .28)
ax.set_xticks([0, .1, .2])
ax.set_xticklabels(["0", "100", "200"])

c.add_arrow(Point(.2, .50, "axis_ddm"), Point(.8, .75, "axis_ddm"))
c.add_text("Monotonic\nincrease", Point(.2, .53, "axis_ddm")+Vector(0, 0.03, "inches"), rotation=22, horizontalalignment="left", verticalalignment="bottom")

#################### DDM (400ms) ####################

print("Starting ddm")
ax = c.ax("ddm400")
N_trials = 50
T, activity70 = diplib.get_ddm_mean_activity(coh=HC, ps=400, highreward=1, time_range=(-200, 400))
ax.plot(T, activity70, c=diplib.get_color(ps=400, coh=HC))
if USE50:
    T, activity50 = diplib.get_ddm_mean_activity(coh=50, ps=400, highreward=1, time_range=(-200, 400))
    ax.plot(T, activity50, c=diplib.get_color(ps=400, coh=50))

if USE53:
    T, activity53 = diplib.get_ddm_mean_activity(coh=LC, ps=400, highreward=1, time_range=(-200, 400))
    ax.plot(T, activity53, c=diplib.get_color(ps=400, coh=LC))

if USE60:
    T, activity60 = diplib.get_ddm_mean_activity(coh=MC, ps=400, highreward=1, time_range=(-200, 400))
    ax.plot(T, activity60, c=diplib.get_color(ps=400, coh=MC))


ax.set_yticks([0, .5, 1])
ax.set_xlim(-.05, .28)
ax.set_xticks([0, .1, .2])
ax.set_xticklabels(["0", "100", "200"])
ax.set_xlabel("Time from sample (ms)")
ax.axvline(0, color=diplib.get_color(ps=400, coh=HC), linestyle="--")
sns.despine(right=False, top=False, ax=ax)
ax.set_ylim(0, .8)

c.add_arrow(Point(.2, .2, "axis_ddm400"), Point(.8, .35, "axis_ddm400"))
c.add_text("Monotonic\nincrease", Point(.2, .2, "axis_ddm400")+Vector(0, 0.03, "inches"), rotation=15, horizontalalignment="left", verticalalignment="bottom")


#################### DDM (0ms) ####################

print("Starting ddm")
ax = c.ax("ddm0")
N_trials = 50
T, activity70 = diplib.get_ddm_mean_activity(coh=HC, ps=0, highreward=1, time_range=(-200, 400))
ax.plot(T, activity70, c=diplib.get_color(ps=0, coh=HC))
if USE50:
    T, activity50 = diplib.get_ddm_mean_activity(coh=50, ps=0, highreward=1, time_range=(-200, 400))
    ax.plot(T, activity50, c=diplib.get_color(ps=0, coh=50))

if USE53:
    T, activity53 = diplib.get_ddm_mean_activity(coh=LC, ps=0, highreward=1, time_range=(-200, 400))
    ax.plot(T, activity53, c=diplib.get_color(ps=0, coh=LC))

if USE60:
    T, activity60 = diplib.get_ddm_mean_activity(coh=MC, ps=0, highreward=1, time_range=(-200, 400))
    ax.plot(T, activity60, c=diplib.get_color(ps=0, coh=MC))


ax.set_yticks([0, .5, 1])
ax.set_xlim(-.05, .28)
ax.set_xticks([0, .1, .2])
ax.set_xticklabels(["0", "100", "200"])
ax.set_ylabel("Decision variable")
ax.set_xlabel("Time from sample (ms)")
ax.axvline(0, color=diplib.get_color(ps=0, coh=HC), linestyle="--")
sns.despine(right=False, top=False, ax=ax)
ax.axvspan(-.05, T[0], color='gray', alpha=.25)
ax.set_ylim(0, .8)


#################### Connecting lines ####################


c.add_text("Time from presample (ms)", Point(1, -.07, "axis_all_fef"+MONKEY), horizontalalignment="right", verticalalignment="top")

c.save(f"figure3.pdf")


