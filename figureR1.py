import diplib
import matplotlib.pyplot as plt
from canvas import Canvas, Point, Vector
import seaborn as sns

c = Canvas(6, 3, "in", fontsize=6)

c.add_axis("fplus", Point(.4, .4, "in"), Point(2.8, 2.8, "in"))
c.add_axis("fminus", Point(3.4, .4, "in"), Point(5.8, 2.8, "in"))

monkey = "Q"
for coh in ([53, 60, 70] if monkey == "Q" else [52, 57, 63]):
    for ps in [0, 400, 800]:
        inrf = diplib.get_mean_conditional_activity(monkey=monkey, ps=ps, coh=coh, align="sample", time_range=(-600, 600), choice_in_rf=True)
        outrf = diplib.get_mean_conditional_activity(monkey=monkey, ps=ps, coh=coh, align="sample", time_range=(-600, 600), choice_in_rf=False)
        c.ax("fplus").plot(inrf[0], inrf[1]+outrf[1], c=diplib.get_color(coh=coh, ps=ps))
        c.ax("fminus").plot(inrf[0], inrf[1]-outrf[1], c=diplib.get_color(coh=coh, ps=ps))

# inrf = diplib.get_mean_conditional_activity(monkey=monkey, coh=50, align="presample", time_range=(800+-600, 800+600), choice_in_rf=True)
# outrf = diplib.get_mean_conditional_activity(monkey=monkey, coh=50, align="presample", time_range=(800+-600, 800+600), choice_in_rf=False)
# c.ax("fplus").plot(inrf[0]-.8, inrf[1]+outrf[1], c=diplib.get_color(coh=50))
# c.ax("fminus").plot(inrf[0]-.8, inrf[1]-outrf[1], c=diplib.get_color(coh=50))
# inrf = diplib.get_mean_conditional_activity(monkey=monkey, coh=50, align="presample", time_range=(-600, 600), choice_in_rf=True)
# outrf = diplib.get_mean_conditional_activity(monkey=monkey, coh=50, align="presample", time_range=(-600, 600), choice_in_rf=False)
# c.ax("fplus").plot(inrf[0], inrf[1]+outrf[1], c=diplib.get_color(coh=50))
# c.ax("fminus").plot(inrf[0], inrf[1]-outrf[1], c=diplib.get_color(coh=50))

c.ax("fplus").set_title("Choice in-RF plus choice out-RF")
c.ax("fminus").set_title("Choice in-RF minus choice out-RF")
for axname in ["fplus", "fminus"]:
    ax = c.ax(axname)
    ax.set_xlabel("Time from sample (s)")
    ax.set_ylabel("FEF activity")
    sns.despine(ax=ax)

diplib.make_gridlegend(c, Point(0.6, 1.8, "in"), zero=False)
c.save("figureR1.pdf")

