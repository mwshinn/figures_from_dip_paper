from canvas import Canvas, Vector, Point
import seaborn as sns
import diplib
import scipy.stats

c = Canvas(6.9, 2.8, "inches", fontsize=8)

MONKEY = "Q"

c.add_image("static/task-alt3.pdf", Point(0, 1, "figure")+Vector(.3, 0, "inches"), unitname="taskfig", width=Vector(3.61, 0, "inches"), height=Vector(0, 2.78, "inches"), horizontalalignment="left", verticalalignment="top")

c.add_arrow(Point(.9, 1.80, "inches"), Point(.8, 1.45, "inches"), linewidth=1, arrowstyle="->,head_width=2,head_length=4")

c.add_axis("psycho", Point(4.5, .4, "inches"), Point(5.3, 1.2, "inches"))
c.add_axis("chrono", Point(5.9, .4, "inches"), Point(6.7, 1.2, "inches"))
c.add_axis("times", Point(4.5, 2, "inches"), Point(6.7, 1, ("inches", "taskfig"))-Vector(0, .3, "inches"))

#diplib.make_gridlegend(c, Point(1.5, 1.3, "inches"), zero=False, shorten=True)
# c.add_legend(Point(5.6, 2.2, "inches"),
#              [(f"{ps} ms presample", {"c": diplib.get_color(ps=ps), "linewidth": 2}) for ps in [0, 400, 800]],
#              line_spacing=Vector(0, 1.3, "Msize"), sym_width=Vector(1.4, 0, "Msize"), padding_sep=Vector(.8, 0, "Msize"))

c.add_figure_labels([("a", "taskfig", Vector(0, -.3, "inches")), ("b", "times"), ("c", "psycho"), ("d", "chrono")])

from ddm import Model, Fitted
from ddm.models import *
from dipmodels import *
m_gate = Model(name='',
      drift=DriftUrgencyGatedDip(snr=8.88568084926296, noise=1.0548720586722578, t1=0.36901214878189553, t1slope=1.8788206931608151, maxcoh=70, leak=7.168279198532247, leaktarget=0, leaktargramp=0.2727682274028674, dipstart=Fitted(-0.21938989703271497, minval=-0.4, maxval=0), dipstop=Fitted(0.006437664287952751, minval=0, maxval=0.5), diptype=3, dipparam=0, nd2=0),
      noise=NoiseUrgencyDip(noise=1.0548720586722578, t1=0.36901214878189553, t1slope=1.8788206931608151, dipstart=Fitted(-0.21938989703271497, minval=-0.4, maxval=0), dipstop=Fitted(0.006437664287952751, minval=0, maxval=0.5), diptype=3, nd2=0),
      bound=BoundDip(B=1, dipstart=Fitted(-0.21938989703271497, minval=-0.4, maxval=0), dipstop=Fitted(0.006437664287952751, minval=0, maxval=0.5), diptype=3),
      IC=ICPoint(x0=0),
      overlay=OverlayChain(overlays=[OverlayNonDecision(nondectime=Fitted(0.22246039919157512, minval=0, maxval=0.3)),
                                     OverlayDipRatio(detect=Fitted(6.271333742309737, minval=2, maxval=50), diptype=3),
                                     #OverlayPoissonMixture(pmixturecoef=0.07855503975842383, rate=1.1033286141789573)
      ]),
      #dx=0.002,
      #dt=0.002,
      dx=.005,dt=.005,
      T_dur=6.0)


def plot_psychometric(ax, ps):
    ax = c.ax(ax)
    ps_rts = rts.query(f'ps == {ps} and spiketime_sac < spiketime_samp')
    p_hrs = []
    std_hrs = []
    cohs = [50, LC, MC, HC]
    for coh in cohs:
        if coh == 50:
            coh_rts = rts.query(f'coh == {coh}')
        else:
            coh_rts = ps_rts.query(f'coh == {coh} and ps == {ps}')
        p_hr = np.mean(coh_rts['correct'])
        std_hr = scipy.stats.sem(coh_rts['correct'])
        p_hrs.append(p_hr)
        std_hrs.append(std_hr)
    xs = (np.asarray(cohs)-50)/50
    print(p_hrs, xs)
    ax.errorbar(xs, p_hrs, yerr=std_hrs, c=diplib.get_color(ps=ps), markersize=8, marker='.', clip_on=False)
    ax.set_xticks([0, .5])
    ax.set_yticks([0, .5, 1])
    ax.set_ylim(0, 1)
    ax.set_xlabel("Coherence")
    ax.set_ylabel("P(correct)")
    sns.despine(ax=ax)


def plot_model_psychometric(ax, ps, resolution=13):
    ax = c.ax(ax)
    p_model_hrs = []
    cohs = np.linspace(50, HC, resolution)
    for coh in cohs:
        coh_query = abs(coh-50)+50
        p_model_corr = ddm.solve_partial_conditions(m_gate, conditions={"coherence": coh_query, "highreward": [0, 1], "presample": ps}).prob_correct()
        p_model_hrs.append(p_model_corr)
    xs = (np.asarray(cohs)-50)/50
    ax.plot(xs, p_model_hrs, c=diplib.get_color(ps=ps), lw=1.4)

def plot_chronometric(ax, rts):
    ax = c.ax(ax)
    for ps in [0, 400, 800]:
        ps_rts = rts.query(f'ps == {ps} and spiketime_sac < spiketime_samp')
        cohs = [50, LC, MC, HC]
        c_rts = []
        mean_rts = []
        std_rts = []
        for coh in cohs:
            if coh == 50:
                coh_rts = rts.query(f'spiketime_sac < spiketime_samp and coh == 50')
            else:
                coh_rts = ps_rts.query(f'coh == {coh} and ps == {ps} and correct == 1')
            c_rts.append(coh/50-1)
            mean_rts.append(np.mean(coh_rts['saccadetime']))
            std_rts.append(scipy.stats.sem(coh_rts['saccadetime']))
        ax.errorbar(c_rts, mean_rts, yerr=std_rts, c=diplib.get_color(ps=ps), markersize=8, marker='.', clip_on=False)
    ax.set_ylabel("Mean RT (ms)")
    ax.set_xlabel("Coherence")
    ax.set_xticks([0, .5])
    sns.despine(ax=ax)

def plot_chronometric_model(ax, resolution=13):
    ax = c.ax(ax)
    for ps in [0, 400, 800]:
        cohs = np.linspace(50, HC, int(resolution//2))
        c_rts = []
        mean_rts = []
        for coh in cohs:
            s = ddm.solve_partial_conditions(m_gate, conditions={"coherence": coh, "highreward": [0, 1], "presample": ps})
            c_rts.append(coh/50-1)
            mean_rts.append(s.mean_decision_time()*1000)
        ax.plot(c_rts, mean_rts, c=diplib.get_color(ps=ps), lw=1.4)



HC,MC,LC = 70, 60, 53
spikes = diplib.spikes_df(MONKEY)
# get the first entry for each spike, which is sufficient for the rt
rts = spikes[np.append([True], (np.asarray(spikes['saccadetime'][0:-1]) != np.asarray(spikes['saccadetime'][1:])))]
plot_chronometric("chrono", rts)
# plot_chronometric_model("chrono", resolution=3)
plot_psychometric("psycho", 0)
plot_psychometric("psycho", 400)
plot_psychometric("psycho", 800)
# plot_model_psychometric("psycho", 0, resolution=3)
# plot_model_psychometric("psycho", 400, resolution=3)
# plot_model_psychometric("psycho", 800, resolution=3)

#c.ax("psycho").set_xlabel("")
#c.ax("psycho").set_xticklabels([])




#################### Make time plot ####################

ax = c.ax("times")

END = 1500
START = -480
ax.plot([START, 0], [.2, .2], c=diplib.get_color(coh=50), lw=1, solid_capstyle="butt")
ax.plot([START, 0], [.5, .5], c=diplib.get_color(coh=50), lw=1, solid_capstyle="butt")
ax.plot([START, 0], [.8, .8], c=diplib.get_color(coh=50), lw=1, solid_capstyle="butt")
ax.plot([0, END], [.2, .2], c=diplib.get_color(ps=0), lw=10, solid_capstyle="butt")
ax.plot([0, 400], [.5, .5], c=diplib.get_color(coh=50), lw=10, solid_capstyle="butt")
ax.plot([400, END], [.5, .5], c=diplib.get_color(ps=400), lw=10, solid_capstyle="butt")
ax.plot([0, 800], [.8, .8], c=diplib.get_color(coh=50), lw=10, solid_capstyle="butt")
ax.plot([800, END], [.8, .8], c=diplib.get_color(ps=800), lw=10, solid_capstyle="butt")

#ax.plot([START, 0], [1.1, 1.1], c=diplib.get_color(coh=50), lw=1, solid_capstyle="butt")
#ax.plot([0, END], [1.1, 1.1], c=diplib.get_color(coh=50), lw=10, solid_capstyle="butt")



ax.set_xlim(-400, 1400)
ax.set_ylim(0, 1)

c.add_text("Presample\nduration (ms)", Point(0, .5, ("axis_times", "times"))-Vector(.45, 0, "in"), verticalalignment="center", rotation=90)
c.add_text("0", Point(START, .2, "times"), horizontalalignment="right", color=diplib.get_color(ps=0))
c.add_text("400", Point(START, .5, "times"), horizontalalignment="right", color=diplib.get_color(ps=400))
c.add_text("800", Point(START, .8, "times"), horizontalalignment="right", color=diplib.get_color(ps=800))
#c.add_text("$\infty$ ms", Point(START, 1.1, "times"), horizontalalignment="right", color=diplib.get_color(coh=50))

#c.add_text("Zero coherence", Point(400, .8, "times"), color="w")
#c.add_text("Sample", Point(1000, .8, "times"), color="w")
c.add_text("Presample", Point(400, 1.1, "times"), color="k")
c.add_text("Sample", Point(1100, 1.1, "times"), color="k")
c.add_text("Fixation", Point(-200, 1.1, "times"), color="k")


ax.set_yticks([])
sns.despine(left=True, ax=ax)
ax.set_xticks([0, 400, 800, 1200])
#ax.set_xticklabels(["0", "400 ms", "800 ms"])
ax.set_xlabel("Time from presample onset (ms)")
#c.add_text("Time:", Point(0, -.335, "axis_times"), verticalalignment="bottom", horizontalalignment="left")
#c.add_text("Task sequence", Point(.5, 1.1, "axis_times"))

c.save("figure1.pdf")
c.save("figure1.png", dpi=600)
