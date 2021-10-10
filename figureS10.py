import diplib
from canvas import Canvas, Point, Vector
import seaborn as sns

c = Canvas(6.9, 4.3, "in")
c.add_grid(["0Q", "400Q", "800Q", "0P", "400P", "800P"], 2, Point(.5, .5, "in"), Point(5.2, 3.9, "in"), size=Vector(1.2, 1.4, "in"))

for MONKEY in ["Q", "P"]:
    cohs = [53, 60, 70] if MONKEY == "Q" else [52, 57, 63]
    for ps in [0, 400, 800]:
        axname = f"{ps}{MONKEY}"
        ax = c.ax(axname)
        for hr_in_rf in [True, False]:
            for coh in cohs:
                linestyle = '-' if hr_in_rf else '--'
                T, traj = diplib.get_mean_conditional_activity(MONKEY, ps=ps, coh=coh, hr_in_rf=hr_in_rf, time_range=(-200, 400)) 
                ax.plot(1000*T, traj, linestyle=linestyle, color=diplib.get_color(coh=coh, ps=ps))
            _, zerocoh = diplib.get_mean_conditional_activity(MONKEY, coh=50, hr_in_rf=hr_in_rf, align="presample", time_range=(ps-200, ps+400))
            ax.plot(1000*T, zerocoh, linestyle=linestyle, color=diplib.get_color(coh=50, ps=ps))
        ax.set_xlim(-150, 350)
        if MONKEY == "Q":
            ax.set_ylim(15, 53)
        else:
            ax.set_ylim(20, 75)
        ax.set_xlabel("Time from sample (ms)")
        if ps == 0:
            ax.set_ylabel("FEF activity")
        sns.despine(ax=ax)

diplib.make_gridlegend(c, Point(5.5, 2.6, "in"), shorten=True)
c.add_legend(Point(5.5, 2.4, "in"), [("Large reward in RF", dict(linestyle="-", color="grey")),
                                     ("Small reward in RF", dict(linestyle="--", color="grey"))],
             line_spacing=Vector(0, 1.5, "Msize")
             )

c.add_text("Monkey 1", Point(0, 1, "axis_0Q") + Vector(-.9, .3, "cm"), weight="bold", size=9, ha="left")
c.add_text("Monkey 2", Point(0, 1, "axis_0P") + Vector(-.9, .3, "cm"), weight="bold", size=9, ha="left")

c.save("figureS10.pdf")
c.show()

