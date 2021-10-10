from canvas import *
import diplib
import paranoid.ignore
import seaborn as sns

DRAFT = False

from ddm import Model, Fitted
from ddm.models import *
from dipmodels import *
def get_model(monkey):
    if monkey == "Q":
        return Model(name='',
      drift=DriftUrgencyGatedDip(snr=Fitted(9.055489816606826, minval=0.5, maxval=20), noise=Fitted(1.0210007162574837, minval=0.2, maxval=2), t1=Fitted(0.3655364577704902, minval=0, maxval=1), t1slope=Fitted(2.0110724435351504, minval=0, maxval=3), maxcoh=70, leak=Fitted(6.915762729607658, minval=-10, maxval=30), leaktarget=Fitted(0.08164321017639141, minval=-0.5, maxval=0.5), leaktargramp=Fitted(0.14801117597537627, minval=0, maxval=3), dipstart=Fitted(-0.21934331140214572, minval=-0.25, maxval=0), dipstop=Fitted(0.00907722935591773, minval=0, maxval=0.1), diptype=3, dipparam=0, nd2=0),
      noise=NoiseUrgencyDip(noise=Fitted(1.0210007162574837, minval=0.2, maxval=2), t1=Fitted(0.3655364577704902, minval=0, maxval=1), t1slope=Fitted(2.0110724435351504, minval=0, maxval=3), dipstart=Fitted(-0.21934331140214572, minval=-0.25, maxval=0), dipstop=Fitted(0.00907722935591773, minval=0, maxval=0.1), diptype=3, nd2=0),
      bound=BoundDip(B=1, dipstart=Fitted(-0.21934331140214572, minval=-0.25, maxval=0), dipstop=Fitted(0.00907722935591773, minval=0, maxval=0.1), diptype=3),
      IC=ICPoint(x0=Fitted(0.08164321017639141, minval=-0.5, maxval=0.5)),
      overlay=OverlayChain(overlays=[OverlayNonDecision(nondectime=Fitted(0.22032108588143898, minval=0, maxval=0.3)), OverlayDipRatio(detect=Fitted(4.451814358885121, minval=2, maxval=50), diptype=3), OverlayPoissonMixture(pmixturecoef=Fitted(0.053169489731494414, minval=0, maxval=0.1), rate=Fitted(1.1880292212319197, minval=0.1, maxval=2))]),
      dx=0.005,
      dt=0.005,
      T_dur=6.0)
    elif monkey == "P":
        return Model(name='',
      drift=DriftUrgencyGatedDip(snr=Fitted(4.982349578945338, minval=0.5, maxval=20), noise=Fitted(0.9493607207603991, minval=0.2, maxval=2), t1=Fitted(0.24989636700560622, minval=0, maxval=1), t1slope=Fitted(1.3263731969806372, minval=0, maxval=3), maxcoh=63, leak=Fitted(7.692866409205677, minval=-10, maxval=30), leaktarget=Fitted(0.018005919237749864, minval=-0.5, maxval=0.5), leaktargramp=Fitted(0.20753823987831005, minval=0, maxval=3), dipstart=Fitted(-0.24439145528487205, minval=-0.25, maxval=0), dipstop=Fitted(0.014273725045643261, minval=0, maxval=0.1), diptype=3, dipparam=0, nd2=0),
      noise=NoiseUrgencyDip(noise=Fitted(0.9493607207603991, minval=0.2, maxval=2), t1=Fitted(0.24989636700560622, minval=0, maxval=1), t1slope=Fitted(1.3263731969806372, minval=0, maxval=3), dipstart=Fitted(-0.24439145528487205, minval=-0.25, maxval=0), dipstop=Fitted(0.014273725045643261, minval=0, maxval=0.1), diptype=3, nd2=0),
      bound=BoundDip(B=1, dipstart=Fitted(-0.24439145528487205, minval=-0.25, maxval=0), dipstop=Fitted(0.014273725045643261, minval=0, maxval=0.1), diptype=3),
      IC=ICPoint(x0=Fitted(0.018005919237749864, minval=-0.5, maxval=0.5)),
      overlay=OverlayChain(overlays=[OverlayNonDecision(nondectime=Fitted(0.23301300378556872, minval=0, maxval=0.3)), OverlayDipRatio(detect=Fitted(4.8221145313327245, minval=2, maxval=50), diptype=3), OverlayPoissonMixture(pmixturecoef=Fitted(0.031182198519672435, minval=0, maxval=0.1), rate=Fitted(0.42210305081388555, minval=0.1, maxval=2))]),
      dx=0.005,
      dt=0.005,
      T_dur=6.0)

c = Canvas(6.9, 3.0, fontsize=8)
c.add_axis("psychoQ", Point(.6, 1.9, "in"), Point(1.4, 2.7, "in"))
c.add_axis("psychoP", Point(1, 0.0, "axis_psychoQ")+Vector(.3, 0, "in"), Point(1, 1, "axis_psychoQ")+Vector(1.2, 0, "in"))

c.add_axis("chronoP", Point(4.5, 1.9, "in"), Point(5.3, 2.7, "in"))
c.add_axis("chronoQ", Point(0, 0.0, "axis_chronoP")-Vector(1.1, 0, "in"), Point(0, 1, "axis_chronoP")-Vector(.3, 0, "in"))

c.add_axis("psychotimeQ", Point(0, .5, ("axis_psychoQ", "absolute")), Point(1, 1.2, ("axis_psychoP", "in")))
c.add_axis("psychotimeP", Point(0, .5, ("axis_chronoQ", "absolute")), Point(1, 1.2, ("axis_chronoP", "in")))
# c.add_axis("ddm_rt", Point(.6, .5, "absolute"), Point(6.5, 1.2, "absolute"))

# m_gate_Q = get_model("Q")
# ax = c.ax("ddm_rt")
# for coh in [57, 60, 70]:
#     for ps in [0, 400, 800]:
#         ax.plot(m_gate_Q.t_domain()*1000, ddm.solve_partial_conditions(m_gate_Q, conditions={"coherence": coh, "presample": ps, "highreward": [0,1]}).pdf_corr(), c=diplib.get_color(ps=ps, coh=coh))
#         ax.set_xlim(0, 1500)

# ax.plot(m_gate_Q.t_domain()*1000, ddm.solve_partial_conditions(m_gate_Q, conditions={"coherence": 50, "presample": 0, "highreward": [0,1]}).pdf_corr(), c=diplib.get_color(ps=0, coh=50))
# ax.set_xlabel("Presample-aligned RT (ms)")
# ax.set_ylabel("Responses")
# sns.despine(ax=ax)

# loffset = Width(.03, "absolute")
# #ax.axvline(-.005, color='k', linestyle="--")
# #c.add_text("Presample\nstart", -loffset+Point(-.005, 0, "ddm_rt") >> Point(0, 1, "axis_ddm_rt"), color='k', horizontalalignment="right", verticalalignment="top")
# ax.axvline(5, color=diplib.get_color(ps=0), linestyle="--")
# c.add_text("Sample\nstart\n(0 ms PS)", loffset+Point(10, 0, "ddm_rt") >> Point(0, 1, "axis_ddm_rt"), color=diplib.get_color(ps=0), horizontalalignment="left", verticalalignment="top")
# ax.axvline(400, color=diplib.get_color(ps=400), linestyle="--")
# c.add_text("Sample\nstart\n(400 ms PS)", loffset+Point(400, 0, "ddm_rt") >> Point(0, 1, "axis_ddm_rt"), color=diplib.get_color(ps=400), horizontalalignment="left", verticalalignment="top")
# ax.axvline(800, color=diplib.get_color(ps=800), linestyle="--")
# c.add_text("Sample\nstart\n(800 ms PS)", loffset+Point(800, 0, "ddm_rt") >> Point(0, 1, "axis_ddm_rt"), color=diplib.get_color(ps=800), horizontalalignment="left", verticalalignment="top")

#################### Timed Psychometric ####################

proportion_ci = lambda a,b,z=1.96 : z*np.sqrt(a/(a+b+1e-10)*b/(a+b+1e-10)/(a+b+1e-10))

for animal in ["Q", "P"]:
    ax = c.ax("psychotime"+animal)
    ax.cla()
    for ps in [0, 400, 800]:
        df = diplib.behavior_df(animal)
        hist_corr = np.histogram(df.query(f'correct==True and ps=={ps} and saccadetime>={ps}')['saccadetime'].dropna(), bins=np.linspace(0, 2000, 41))[0]
        hist_err = np.histogram(df.query(f'correct==False and ps=={ps} and saccadetime>={ps}')['saccadetime'].dropna(), bins=np.linspace(0, 2000, 41))[0]
        ratio = hist_corr/(hist_corr+hist_err+.0001)
        ci_bound = proportion_ci(hist_corr, hist_err)
        ci = np.asarray((np.minimum(ci_bound, ratio), np.minimum(ci_bound, 1-ratio)))
        valid = (hist_corr > 0) & (hist_err > 0)
        
        ax.errorbar(np.linspace(25, 1975, 40)[valid], ratio[valid], yerr=ci[:,valid], c=diplib.get_color(ps=ps), elinewidth=1)
        ax.axvline(ps, c=diplib.get_color(ps=ps), linewidth=2, linestyle='--', clip_on=False)
    sns.despine(ax=ax, left=True)
    ax.set_xlim(0, 2000)
    ax.set_xlabel("Time from presample (ms)")
    ax.set_ylabel("P(correct)")
    ax.set_title(f"Monkey {1 if animal=='Q' else 2}")




def plot_psychometric(ax, ps, rts):
    ax = c.ax(ax)
    ps_rts = rts.query(f'ps == {ps} and spiketime_sac < spiketime_samp')
    p_hrs = []
    errorbars = []
    cohs = [50, LC, MC, HC]
    for coh in cohs:
        if coh == 50:
            p_hr = .5
            errorbars.append(0)
        else:
            coh_rts = ps_rts.query(f'coh == {coh} and ps == {ps}')
            p_hr = np.mean(coh_rts['correct'])
            errorbars.append(proportion_ci(sum(coh_rts['correct']), sum(1-coh_rts['correct'])))
        p_hrs.append(p_hr)
    xs = (np.asarray(cohs)-50)/50
    print(p_hrs, xs, errorbars)
    ax.plot(xs, p_hrs, c=diplib.get_color(ps=ps), markersize=10, marker='.', linestyle="none")
    ax.errorbar(xs, p_hrs, yerr=np.asarray(errorbars), linestyle='none', elinewidth=1, color=diplib.get_color(ps=ps))
    ax.set_xticks([0, .5])
    ax.set_yticks([0, .5, 1])
    sns.despine(ax=ax)

def plot_model_psychometric(ax, ps, resolution=13):
    m_gate = get_model(MONKEY)
    ax = c.ax(ax)
    p_model_hrs = []
    cohs = np.linspace(50, HC, resolution)
    for coh in cohs:
        coh_query = abs(coh-50)+50
        p_model_corr = ddm.solve_partial_conditions(m_gate, conditions={"coherence": coh_query, "highreward": [0,1], "presample": ps}).prob_correct()
        p_model_hrs.append(p_model_corr)
    xs = (np.asarray(cohs)-50)/50
    ax.plot(xs, p_model_hrs, c=diplib.get_color(ps=ps), lw=2)
    sns.despine(ax=ax)

def plot_chronometric(ax, rts):
    ax = c.ax(ax)
    for ps in [0, 400, 800]:
        ps_rts = rts.query(f'ps == {ps} and spiketime_sac < spiketime_samp')
        cohs = [50, LC, MC, HC]
        c_rts = []
        mean_rts = []
        for coh in cohs:
            if coh == 50:
                coh_rts = rts.query(f'spiketime_sac < spiketime_samp and coh == 50')
            else:
                coh_rts = ps_rts.query(f'coh == {coh} and ps == {ps} and correct == 1')
            c_rts.append(coh/50-1)
            mean_rts.append(np.mean(coh_rts['saccadetime']))
        ax.plot(c_rts, mean_rts, c=diplib.get_color(ps=ps), markersize=10, marker='.', linestyle='none')
    sns.despine(ax=ax)

def plot_chronometric_model(ax, resolution=13):
    m_gate = get_model(MONKEY)
    ax = c.ax(ax)
    for ps in [0, 400, 800]:
        cohs = np.linspace(50, HC, int(resolution//2))
        c_rts = []
        mean_rts = []
        for coh in cohs:
            s = ddm.solve_partial_conditions(m_gate, conditions={"coherence": coh, "highreward": [0, 1], "presample": ps})
            c_rts.append(coh/50-1)
            mean_rts.append(s.mean_decision_time()*1000)
        ax.plot(c_rts, mean_rts, c=diplib.get_color(ps=ps), lw=2)
    ax.set_xticks([0, .5])
    #ax.set_xlim(-.35, .35)
    ax.set_ylabel("Presample-aligned\nRT (ms)")
    sns.despine(ax=ax)



for MONKEY in ["Q", "P"]:
    if MONKEY == "Q":
        HC,MC,LC = 70, 60, 53
    elif MONKEY == "P":
        HC,MC,LC = 63, 57, 52
    
    spikes = diplib.spikes_df(MONKEY)
    # get the first entry for each spike, which is sufficient for the rt
    rts = spikes[np.append([True], (np.asarray(spikes['saccadetime'][0:-1]) != np.asarray(spikes['saccadetime'][1:])))]
    
    plot_chronometric("chrono"+MONKEY, rts)
    plot_psychometric("psycho"+MONKEY, 0, rts)
    plot_psychometric("psycho"+MONKEY, 400, rts)
    plot_psychometric("psycho"+MONKEY, 800, rts)
    if not DRAFT:
        plot_chronometric_model("chrono"+MONKEY)
        plot_model_psychometric("psycho"+MONKEY, 0)
        plot_model_psychometric("psycho"+MONKEY, 400)
        plot_model_psychometric("psycho"+MONKEY, 800)


ax = c.ax("psychoQ")
ax.set_ylabel("P(correct)")
ax.set_xticks([0, .5])
ax.set_yticks([.5, 1])
#ax.set_xlim(-.55, .55)
#ax.set_xticklabels([])
ax.set_title("Monkey 1")
ax.set_xlabel("Coherence")

ax = c.ax("psychoP")
#ax.set_ylabel("P(high reward)")
ax.set_xlabel("Coherence")
ax.set_xticks([0, .5])
ax.set_yticks([.5, 1])
#ax.set_xlim(-.35, .35)
ax.set_title("Monkey 2")
ax.set_yticklabels([])

ax = c.ax("chronoQ")
ax.set_xlabel("Coherence")
ax.set_title("Monkey 1")
ax.set_ylim(300, 1700)
#ax.set_xticklabels([])

ax = c.ax("chronoP")
ax.set_xlabel("RT (ms)")
ax.set_xlabel("Coherence")
ax.set_ylabel("")
ax.set_yticklabels([])
ax.set_title("Monkey 2")
ax.set_ylim(300, 1700)

#diplib.make_gridlegend(c, Point(1, 0, "axis_chronoP")+Vector(.2, .05, "absolute"), shorten=True)
c.add_legend(Point(1, .5, "axis_chronoP")+Vector(.4, .05, "absolute"),
             [("0 ms", {"color": diplib.get_color(ps=0), "linewidth":4}),
              ("400 ms", {"color": diplib.get_color(ps=400), "linewidth":4}),
              ("800 ms", {"color": diplib.get_color(ps=800), "linewidth":4}),
             ],
             line_spacing=Vector(0, 1.5, "Msize"), padding_sep=Vector(.7, 0, "Msize"), sym_width=Vector(1.5, 0, "Msize"))
c.add_text("Presample", Point(1, .5, "axis_chronoP")+Vector(.6, .20, "absolute"), weight="bold")
c.add_figure_labels([("a", "psychoQ"), ("b", "chronoQ"), ("c", "psychotimeQ")])

c.save("figureS1.pdf")

