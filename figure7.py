import paranoid.ignore
import diplib
import numpy as np
import matplotlib.pyplot as plt
import scipy
import seaborn as sns

from canvas import Canvas, Height, Width, Vector, Point
c = Canvas(6.9, 7.5, fontsize=8)


c.add_grid(["micros0Q", "micros400Q", "micros800Q"], 1, Point(.55, 4.2, "absolute"), Point(4.75, 5.4, "absolute"), size=Vector(1.2, 1.2, "absolute"))
c.add_axis("all_microsQ", Point(.6, 5.7, "absolute"), Point(6.5, 7.1, "absolute"))

c.add_grid(["micros0P", "micros400P", "micros800P"], 1, Point(.55, .5, "absolute"), Point(4.75, 1.7, "absolute"), size=Vector(1.2, 1.2, "absolute"))
c.add_axis("all_microsP", Point(.6, 2.2, "absolute"), Point(6.5, 3.4, "absolute"))



USE50 = True
USE53 = True
USE60 = True

for MONKEY in ["Q", "P"]:
    if MONKEY == "Q":
        HC,MC,LC = 70, 60, 53
    elif MONKEY == "P":
        HC,MC,LC = 63, 57, 52
    
    # ##################### All Micros ####################
    
    SMOOTH = 3
    ax = c.ax("all_micros"+MONKEY)
    for coh in [LC, MC, HC]:
        for ps in [0, 400, 800]:
            ax.plot(*diplib.get_microsaccade_conditional_activity(monkey=MONKEY, coh=coh, ps=ps, smooth=SMOOTH, time_range=(-200, 1400), align="presample", max_dist=200), color=diplib.get_color(coh=coh, ps=ps))
    
    ax.axvline(0, color=diplib.get_color(ps=0, coh=HC), linestyle="--")
    ax.axvline(.4, color=diplib.get_color(ps=400, coh=HC), linestyle="--")
    ax.axvline(.8, color=diplib.get_color(ps=800, coh=HC), linestyle="--")
    ax.set_ylabel("Microsaccade rate (hz)")
    
    ax.set_xticklabels(["", "0", "", "400", "", "800"])
    ax.set_yticks([0, 2, 4])
    ax.set_xlim(-.2, 1.4225)
    ax.set_ylim(0, 5.25)
    ax.set_title("Microsaccade rate (presample-aligned)")
    ax.xaxis.set_minor_locator(plt.matplotlib.ticker.AutoMinorLocator(2))
    
    loffset = Width(.03, "absolute")
    ax.axvline(0, color='k', linestyle="-")
    c.add_text("Presample\nstart", -loffset+Point(-.005, 0, "all_micros"+MONKEY) >> Point(0, 1, "axis_all_micros"+MONKEY), color='k', horizontalalignment="right", verticalalignment="top")
    ax.axvline(0, color=diplib.get_color(ps=.005, coh=HC), linestyle="--")
    c.add_text("Sample\nstart\n(0 ms PS)", loffset+Point(.005, 0, "all_micros"+MONKEY) >> Point(0, 1, "axis_all_micros"+MONKEY), color=diplib.get_color(ps=0, coh=HC), horizontalalignment="left", verticalalignment="top")
    ax.axvline(.4, color=diplib.get_color(ps=400, coh=HC), linestyle="--")
    c.add_text("Sample\nstart\n(400 ms PS)", loffset+Point(.4, 0, "all_micros"+MONKEY) >> Point(0, 1, "axis_all_micros"+MONKEY), color=diplib.get_color(ps=400, coh=HC), horizontalalignment="left", verticalalignment="top")
    ax.axvline(.8, color=diplib.get_color(ps=800, coh=HC), linestyle="--")
    c.add_text("Sample\nstart\n(800 ms PS)", loffset+Point(.8, 0, "all_micros"+MONKEY) >> Point(0, 1, "axis_all_micros"+MONKEY), color=diplib.get_color(ps=800, coh=HC), horizontalalignment="left", verticalalignment="top")
    
    ax.axvspan(.75, 1.08, color=diplib.get_color(ps=800, coh=HC), alpha=.1, zorder=-1)
    ax.axvspan(.35, .68, color=diplib.get_color(ps=400, coh=HC), alpha=.1, zorder=-1)
    ax.axvspan(-.05, .28, color=diplib.get_color(ps=0, coh=HC), alpha=.1, zorder=-1)
    
    c.add_text("Time from presample (ms)", Point(1, -.07, "axis_all_micros"+MONKEY), horizontalalignment="right", verticalalignment="top")
    sns.despine(ax=ax)
    
    #################### Microsaccades (800ms presamp) ####################
    print("Starting micros800")
    ax = c.ax("micros800"+MONKEY)
    T, activity70 = diplib.get_microsaccade_conditional_activity(monkey=MONKEY, coh=HC, ps=800, smooth=SMOOTH, time_range=(-200, 500), align="sample", max_dist=200)
    bounds70 = diplib.bootstrap_microsaccade_ci(monkey=MONKEY, N=500, coh=HC, ps=800, time_range=(-200, 500), smooth=SMOOTH, seed=1, align="sample", max_dist=200)
    ax.plot(T, activity70, c=diplib.get_color(ps=800, coh=HC))
    ax.fill_between(T, bounds70[0,:], bounds70[1,:], color=diplib.get_color(ps=800, coh=HC), alpha=.4)
    if USE60:
        activity60 = diplib.get_microsaccade_conditional_activity(monkey=MONKEY, coh=MC, ps=800, smooth=SMOOTH, time_range=(-200, 500), align="sample", max_dist=200)[1]
        bounds60 = diplib.bootstrap_microsaccade_ci(monkey=MONKEY, N=500, coh=MC, ps=800, time_range=(-200, 500), smooth=SMOOTH, align="sample", seed=1, max_dist=200)
        ax.plot(T, activity60, c=diplib.get_color(ps=800, coh=MC))
        ax.fill_between(T, bounds60[0,:], bounds60[1,:], color=diplib.get_color(ps=800, coh=MC), alpha=.4)
    if USE50:
        activity50 = diplib.get_microsaccade_conditional_activity(monkey=MONKEY, coh=50, smooth=SMOOTH, time_range=(600, 1300), align="presample", max_dist=200)[1]
        bounds50 = diplib.bootstrap_microsaccade_ci(monkey=MONKEY, N=500, coh=50, smooth=SMOOTH, time_range=(600, 1300), align="presample", seed=1, max_dist=200)
        ax.plot(T, activity50, c=diplib.get_color(ps=800, coh=50))
        ax.fill_between(T, bounds50[0,:], bounds50[1,:], color=diplib.get_color(ps=800, coh=50), alpha=.4)
    if USE53:
        activity53 = diplib.get_microsaccade_conditional_activity(monkey=MONKEY, coh=LC, ps=800, smooth=SMOOTH, time_range=(-200, 500), align="sample", max_dist=200)[1]
        bounds53 = diplib.bootstrap_microsaccade_ci(monkey=MONKEY, N=500, coh=LC, ps=800, smooth=SMOOTH, time_range=(-200, 500), align="sample", seed=1, max_dist=200)
        ax.plot(T, activity53, c=diplib.get_color(ps=800, coh=LC))
        ax.fill_between(T, bounds53[0,:], bounds53[1,:], color=diplib.get_color(ps=800, coh=LC), alpha=.4)
    
    if USE50:
        sigs = diplib.bootstrap_microsaccade_significance(params1=dict(monkey=MONKEY, coh=50, time_range=(600, 1300), align="presample", smooth=SMOOTH, max_dist=200), params2=dict(monkey=MONKEY, coh=HC, ps=800, time_range=(-200, 500), smooth=SMOOTH, align="sample", max_dist=200), N=500, seed=1)
    else:
        sigs = diplib.bootstrap_microsaccade_significance(params1=dict(monkey=MONKEY, coh=LC, ps=800, smooth=SMOOTH, time_range=(-200, 500), align="sample", max_dist=200), params2=dict(monkey=MONKEY, coh=HC, ps=800, time_range=(-200, 500), smooth=SMOOTH, align="sample", max_dist=200), N=500, seed=1)
    
    if MONKEY == "Q":
        ax.set_ylim(0, 3.2)
    else:
        ax.set_ylim(0, 1.6)
    ax.set_xlim(-.05, .28)
    ax.set_xticks([0, .1, .2])
    ax.set_xticklabels(["0", "100", "200"])
    ax.set_xlabel("Time from sample (ms)")
    diplib.plot_significance(c, "micros800"+MONKEY, T[(sigs<.01) & (T > -.05) & (T < .28)], dx=T[1]-T[0])
    ax.axvline(0, color=diplib.get_color(ps=800, coh=HC), linestyle="--")
    #ax.axvspan(-.05, .28, color=diplib.get_color(ps=800, coh=HC), alpha=.1)
    sns.despine(right=False, top=False, ax=ax)
    ax.xaxis.set_minor_locator(plt.matplotlib.ticker.AutoMinorLocator(2))
    
    if MONKEY == "Q":
        c.add_arrow(Point(.06, .06, ("micros800"+MONKEY, "axis_micros800"+MONKEY)), Point(.13, .4, "micros800"+MONKEY))
        c.add_text("Dip", Point(.03, .06, ("micros800"+MONKEY, "axis_micros800"+MONKEY)))
    else:
        c.add_arrow(Point(.06, .8, ("micros800"+MONKEY, "axis_micros800"+MONKEY)), Point(.13, .6, "micros800"+MONKEY))
        c.add_text("Dip", Point(.06, .8, ("micros800"+MONKEY, "axis_micros800"+MONKEY)), ha="right")
    
    #################### Microsaccades (400ms presamp) ####################
    print("Starting micros400")
    ax = c.ax("micros400"+MONKEY)
    T, activity70 = diplib.get_microsaccade_conditional_activity(monkey=MONKEY, coh=HC, ps=400, smooth=SMOOTH, time_range=(-200, 500), align="sample", max_dist=200)
    bounds70 = diplib.bootstrap_microsaccade_ci(monkey=MONKEY, N=500, coh=HC, ps=400, smooth=SMOOTH, time_range=(-200, 500), seed=1, align="sample", max_dist=200)
    ax.plot(T, activity70, c=diplib.get_color(ps=400, coh=HC))
    ax.fill_between(T, bounds70[0,:], bounds70[1,:], color=diplib.get_color(ps=400, coh=HC), alpha=.4)
    if USE60:
        activity60 = diplib.get_microsaccade_conditional_activity(monkey=MONKEY, coh=MC, ps=400, smooth=SMOOTH, time_range=(-200, 500), align="sample", max_dist=200)[1]
        bounds60 = diplib.bootstrap_microsaccade_ci(monkey=MONKEY, N=500, coh=MC, ps=400, time_range=(-200, 500), smooth=SMOOTH, align="sample", seed=1, max_dist=200)
        ax.plot(T, activity60, c=diplib.get_color(ps=400, coh=MC))
        ax.fill_between(T, bounds60[0,:], bounds60[1,:], color=diplib.get_color(ps=400, coh=MC), alpha=.4)
    if USE50:
        activity50 = diplib.get_microsaccade_conditional_activity(monkey=MONKEY, coh=50, smooth=SMOOTH, time_range=(200, 900), align="presample", max_dist=200)[1]
        bounds50 = diplib.bootstrap_microsaccade_ci(monkey=MONKEY, N=500, coh=50, time_range=(200, 900), smooth=SMOOTH, align="presample", seed=1, max_dist=200)
        ax.plot(T, activity50, c=diplib.get_color(ps=400, coh=50))
        ax.fill_between(T, bounds50[0,:], bounds50[1,:], color=diplib.get_color(ps=400, coh=50), alpha=.4)
    if USE53:
        activity53 = diplib.get_microsaccade_conditional_activity(monkey=MONKEY, coh=LC, ps=400, smooth=SMOOTH, time_range=(-200, 500), align="sample", max_dist=200)[1]
        bounds53 = diplib.bootstrap_microsaccade_ci(monkey=MONKEY, N=500, coh=LC, ps=400, time_range=(-200, 500), smooth=SMOOTH, align="sample", seed=1, max_dist=200)
        ax.plot(T, activity53, c=diplib.get_color(ps=400, coh=LC))
        ax.fill_between(T, bounds53[0,:], bounds53[1,:], color=diplib.get_color(ps=400, coh=LC), alpha=.4)
    
    if USE50:
        sigs = diplib.bootstrap_microsaccade_significance(params1=dict(monkey=MONKEY, coh=50, smooth=SMOOTH, time_range=(200, 900), align="presample", max_dist=200), params2=dict(monkey=MONKEY, coh=HC, smooth=SMOOTH, ps=400, time_range=(-200, 500), align="sample", max_dist=200), N=500, seed=2)
    else:
        sigs = diplib.bootstrap_microsaccade_significance(params1=dict(monkey=MONKEY, coh=LC, ps=400, smooth=SMOOTH, time_range=(-200, 500), align="sample", max_dist=200), params2=dict(monkey=MONKEY, coh=HC, ps=400, smooth=SMOOTH, time_range=(-200, 500), align="sample", max_dist=200), N=500, seed=1)
    
    if MONKEY == "Q":
        ax.set_ylim(0, 3.2)
    else:
        ax.set_ylim(0, 1.6)
    ax.set_xlim(-.05, .28)
    ax.set_xticks([0, .1, .2])
    ax.set_xticklabels(["0", "100", "200"])
    
    ax.axvline(0, color=diplib.get_color(ps=400, coh=HC), linestyle="--")
    #ax.axvspan(-.05, .28, color=diplib.get_color(ps=400, coh=HC), alpha=.1)
    ax.set_xlabel("Time from sample (ms)")
    diplib.plot_significance(c, "micros400"+MONKEY, T[(sigs<.01) & (T > -.05) & (T < .28)], dx=T[1]-T[0])
    sns.despine(right=False, top=False, ax=ax)
    
    if MONKEY == "Q":
        c.add_arrow(Point(.06, .06, ("micros400"+MONKEY, "axis_micros400"+MONKEY)), Point(.16, .4, "micros400"+MONKEY))
        c.add_text("Dip", Point(.03, .06, ("micros400"+MONKEY, "axis_micros400"+MONKEY)))
    
    #################### Microsaccades (0ms presamp) ####################
    print("Starting micros0")
    ax = c.ax("micros0"+MONKEY)
    T, activity70 = diplib.get_microsaccade_conditional_activity(monkey=MONKEY, coh=HC, ps=0, smooth=SMOOTH, time_range=(-200, 500), align="sample", max_dist=200)
    bounds70 = diplib.bootstrap_microsaccade_ci(monkey=MONKEY, N=500, coh=HC, ps=0, smooth=SMOOTH, time_range=(-200, 500), seed=1, align="sample", max_dist=200)
    ax.plot(T, activity70, c=diplib.get_color(ps=0, coh=HC))
    ax.fill_between(T, bounds70[0,:], bounds70[1,:], color=diplib.get_color(ps=0, coh=HC), alpha=.4)
    if USE60:
        activity60 = diplib.get_microsaccade_conditional_activity(monkey=MONKEY, coh=MC, ps=0, smooth=SMOOTH, time_range=(-200, 500), align="sample", max_dist=200)[1]
        bounds60 = diplib.bootstrap_microsaccade_ci(monkey=MONKEY, N=500, coh=MC, ps=0, time_range=(-200, 500), smooth=SMOOTH, align="sample", seed=1, max_dist=200)
        ax.plot(T, activity60, c=diplib.get_color(ps=0, coh=MC))
        ax.fill_between(T, bounds60[0,:], bounds60[1,:], color=diplib.get_color(ps=0, coh=MC), alpha=.4)
    if USE50:
        activity50 = diplib.get_microsaccade_conditional_activity(monkey=MONKEY, coh=50, smooth=SMOOTH, time_range=(-200, 500), align="presample", max_dist=200)[1]
        bounds50 = diplib.bootstrap_microsaccade_ci(monkey=MONKEY, N=500, coh=50, time_range=(-200, 500), smooth=SMOOTH, align="presample", seed=1, max_dist=200)
        ax.plot(T, activity50, c=diplib.get_color(ps=0, coh=50))
        ax.fill_between(T, bounds50[0,:], bounds50[1,:], color=diplib.get_color(ps=0, coh=50), alpha=.4)
    if USE53:
        activity53 = diplib.get_microsaccade_conditional_activity(monkey=MONKEY, coh=LC, ps=0, smooth=SMOOTH, time_range=(-200, 500), align="sample", max_dist=200)[1]
        bounds53 = diplib.bootstrap_microsaccade_ci(monkey=MONKEY, N=500, coh=LC, ps=0, time_range=(-200, 500), smooth=SMOOTH, align="sample", seed=1, max_dist=200)
        ax.plot(T, activity53, c=diplib.get_color(ps=0, coh=LC))
        ax.fill_between(T, bounds53[0,:], bounds53[1,:], color=diplib.get_color(ps=0, coh=LC), alpha=.4)
    if USE50:
        sigs = diplib.bootstrap_microsaccade_significance(params1=dict(monkey=MONKEY, coh=50, smooth=SMOOTH, time_range=(-200, 500), align="presample", max_dist=200), params2=dict(monkey=MONKEY, coh=HC, ps=0, smooth=SMOOTH, time_range=(-200, 500), align="sample", max_dist=200), N=500, seed=1)
    else:
        sigs = diplib.bootstrap_microsaccade_significance(params1=dict(monkey=MONKEY, coh=LC, ps=0, smooth=SMOOTH, time_range=(-200, 500), align="sample", max_dist=200), params2=dict(monkey=MONKEY, coh=HC, ps=0, smooth=SMOOTH, time_range=(-200, 500), align="sample", max_dist=200), N=500, seed=1)
    
    ax.set_ylim(0, 3.2)
    ax.set_xlim(-.05, .28)
    ax.set_xticks([0, .1, .2])
    ax.set_xticklabels(["0", "100", "200"])
    
    ax.axvline(0, color=diplib.get_color(ps=0, coh=HC), linestyle="--")
    #ax.axvspan(-.05, .28, color=diplib.get_color(ps=0, coh=HC), alpha=.1)
    ax.set_xlabel("Time from sample (ms)")
    ax.set_ylabel("Microsaccade rate (hz)")
    diplib.plot_significance(c, "micros0"+MONKEY, T[(sigs<.01) & (T > -.05) & (T < .28)], dx=T[1]-T[0])
    sns.despine(right=False, top=False, ax=ax)
    
    #################### Connectors ####################
    
    c.add_poly([Point(.75, 0, "all_micros"+MONKEY) >> Point(0, 0, "axis_all_micros"+MONKEY),
                Point(0, 1, "axis_micros800"+MONKEY),
                Point(1, 1, "axis_micros800"+MONKEY),
                Point(1.08, 0, "all_micros"+MONKEY) >> Point(0, 0, "axis_all_micros"+MONKEY)],
            facecolor=diplib.get_color(ps=800, coh=HC), alpha=.1, fill=True, edgecolor='k')
    
    c.add_poly([Point(.35, 0, "all_micros"+MONKEY) >> Point(0, 0, "axis_all_micros"+MONKEY),
                Point(0, 1, "axis_micros400"+MONKEY),
                Point(1, 1, "axis_micros400"+MONKEY),
                Point(.68, 0, "all_micros"+MONKEY) >> Point(0, 0, "axis_all_micros"+MONKEY)],
            facecolor=diplib.get_color(ps=400, coh=HC), alpha=.1, fill=True, edgecolor='k')
    
    c.add_poly([Point(-.05, 0, "all_micros"+MONKEY) >> Point(0, 0, "axis_all_micros"+MONKEY),
                Point(0, 1, "axis_micros0"+MONKEY),
                Point(1, 1, "axis_micros0"+MONKEY),
                Point(.28, 0, "all_micros"+MONKEY) >> Point(0, 0, "axis_all_micros"+MONKEY)],
            facecolor=diplib.get_color(ps=0, coh=HC), alpha=.1, fill=True, edgecolor='k')



diplib.make_gridlegend(c, Point(1, 0, "axis_micros800Q")+Vector(.4, 0, "absolute"))
c.add_figure_labels([("a", "all_microsQ"), ("b", "micros0Q"), ("c", "micros400Q"), ("d", "micros800Q"), ("e", "all_microsP"), ("f", "micros0P"), ("g", "micros400P"), ("h", "micros800P")])

c.add_text("Monkey 1", Point(0, 1, "axis_all_microsQ") + Vector(-.6, .7, "cm"), weight="bold", size=9)
c.add_text("Monkey 2", Point(0, 1, "axis_all_microsP") + Vector(-.6, .7, "cm"), weight="bold", size=9)

c.save(f"figure7.pdf")

