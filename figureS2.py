# Smoothing grid figure for the reviewers
import diplib
from canvas import Canvas, Vector, Point
import seaborn as sns

c = Canvas(6.9, 3.6, "in", fontsize=7)
bins = [5, 10, 20, 50]
smooths = [0, 3, 5, 7]
c.add_grid([f"mQb{b}s{s}" for b in bins for s in smooths], 4, Point(.4, .5, "in"), Point(3.1, 3.3, "in"), size=Vector(.55, .55, "in"), unitname="Qgrid")
c.add_grid([f"mPb{b}s{s}" for b in bins for s in smooths], 4, Point(3.9, .5, "in"), Point(6.6, 3.3, "in"), size=Vector(.55, .55, "in"), unitname="Pgrid")

for MONKEY in ["Q", "P"]:
    if MONKEY == "Q":
        HC,MC,LC = 70, 60, 53
    elif MONKEY == "P":
        HC,MC,LC = 63, 57, 52
    
    for b in bins:
        for s in smooths:
            print(b,s)
            axname = f"m{MONKEY}b{b}s{s}"
            ax = c.ax(axname)
            for coh in [HC, MC, LC]:
                T, hist = diplib.get_rt_conditional_activity(MONKEY, smooth=s, time_range=(-400, 400), align="sample", ps=800, coh=coh, binsize=b)
                ax.plot(T*1000, hist, c=diplib.get_color(ps=800, coh=coh))
            ax.plot(T*1000, diplib.get_rt_conditional_activity(MONKEY, smooth=s, time_range=(400, 1200), align="presample", coh=50, binsize=b)[1], c=diplib.get_color(coh=50))
            filtertext = f"width = {s}" if s > 0 else "no filter"
            c.add_text(f"{b} ms bins\n{filtertext}", Point(.05, .5, "axis_"+axname), ha="left", va="bottom", size=6)
            ax.set_xlim(-150, 250)
            if MONKEY == "Q":
                ax.set_ylim(0, 2.5)
            elif MONKEY == "P":
                ax.set_ylim(0, .25)
            if b != bins[-1]:
                ax.set_xticks([])
            if s != smooths[0]:
                ax.set_yticks([])
            sns.despine(ax=ax, bottom=(b!=bins[-1]), left=(s!=smooths[0]))
            if s == 5 and b == 20:
                sns.despine(ax=ax, right=False, top=False)

c.add_text("Monkey 1", Point(.5, 1.07, "Qgrid"), weight="bold")
c.add_text("Monkey 2", Point(.5, 1.07, "Pgrid"), weight="bold")
c.add_text("Time from sample (ms)", Point(.5, 0, "Qgrid")-Vector(0, .35, "in"))
c.add_text("Time from sample (ms)", Point(.5, 0, "Pgrid")-Vector(0, .35, "in"))
c.add_text("RT density", Point(-.12, .5, "Qgrid"), rotation=90)
c.add_text("RT density", Point(-.12, .5, "Pgrid"), rotation=90)

c.save("figureS2.pdf")
