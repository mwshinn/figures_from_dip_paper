from ddm import *
from ddm.models import *
import ddm
from dipmodels import *
m1_gate = Model(name='',
      drift=DriftUrgencyGatedDip(snr=Fitted(9.055489816606826, minval=0.5, maxval=20), noise=Fitted(1.0210007162574837, minval=0.2, maxval=2), t1=Fitted(0.3655364577704902, minval=0, maxval=1), t1slope=Fitted(2.0110724435351504, minval=0, maxval=3), maxcoh=70, leak=Fitted(6.915762729607658, minval=-10, maxval=30), leaktarget=Fitted(0.08164321017639141, minval=-0.5, maxval=0.5), leaktargramp=Fitted(0.14801117597537627, minval=0, maxval=3), dipstart=Fitted(-0.21934331140214572, minval=-0.25, maxval=0), dipstop=Fitted(0.00907722935591773, minval=0, maxval=0.1), diptype=3, dipparam=0, nd2=0),
      noise=NoiseUrgencyDip(noise=Fitted(1.0210007162574837, minval=0.2, maxval=2), t1=Fitted(0.3655364577704902, minval=0, maxval=1), t1slope=Fitted(2.0110724435351504, minval=0, maxval=3), dipstart=Fitted(-0.21934331140214572, minval=-0.25, maxval=0), dipstop=Fitted(0.00907722935591773, minval=0, maxval=0.1), diptype=3, nd2=0),
      bound=BoundDip(B=1, dipstart=Fitted(-0.21934331140214572, minval=-0.25, maxval=0), dipstop=Fitted(0.00907722935591773, minval=0, maxval=0.1), diptype=3),
      IC=ICPoint(x0=Fitted(0.08164321017639141, minval=-0.5, maxval=0.5)),
      overlay=OverlayChain(overlays=[OverlayNonDecision(nondectime=Fitted(0.22032108588143898, minval=0, maxval=0.3)), OverlayDipRatio(detect=Fitted(4.451814358885121, minval=2, maxval=50), diptype=3), OverlayPoissonMixture(pmixturecoef=Fitted(0.053169489731494414, minval=0, maxval=0.1), rate=Fitted(1.1880292212319197, minval=0.1, maxval=2))]),
      dx=0.005,
      dt=0.005,
      T_dur=3.0)

m2_gate = Model(name='',
      drift=DriftUrgencyGatedDip(snr=Fitted(4.982349578945338, minval=0.5, maxval=20), noise=Fitted(0.9493607207603991, minval=0.2, maxval=2), t1=Fitted(0.24989636700560622, minval=0, maxval=1), t1slope=Fitted(1.3263731969806372, minval=0, maxval=3), maxcoh=63, leak=Fitted(7.692866409205677, minval=-10, maxval=30), leaktarget=Fitted(0.018005919237749864, minval=-0.5, maxval=0.5), leaktargramp=Fitted(0.20753823987831005, minval=0, maxval=3), dipstart=Fitted(-0.24439145528487205, minval=-0.25, maxval=0), dipstop=Fitted(0.014273725045643261, minval=0, maxval=0.1), diptype=3, dipparam=0, nd2=0),
      noise=NoiseUrgencyDip(noise=Fitted(0.9493607207603991, minval=0.2, maxval=2), t1=Fitted(0.24989636700560622, minval=0, maxval=1), t1slope=Fitted(1.3263731969806372, minval=0, maxval=3), dipstart=Fitted(-0.24439145528487205, minval=-0.25, maxval=0), dipstop=Fitted(0.014273725045643261, minval=0, maxval=0.1), diptype=3, nd2=0),
      bound=BoundDip(B=1, dipstart=Fitted(-0.24439145528487205, minval=-0.25, maxval=0), dipstop=Fitted(0.014273725045643261, minval=0, maxval=0.1), diptype=3),
      IC=ICPoint(x0=Fitted(0.018005919237749864, minval=-0.5, maxval=0.5)),
      overlay=OverlayChain(overlays=[OverlayNonDecision(nondectime=Fitted(0.23301300378556872, minval=0, maxval=0.3)), OverlayDipRatio(detect=Fitted(4.8221145313327245, minval=2, maxval=50), diptype=3), OverlayPoissonMixture(pmixturecoef=Fitted(0.031182198519672435, minval=0, maxval=0.1), rate=Fitted(0.42210305081388555, minval=0.1, maxval=2))]),
      dx=0.005,
      dt=0.005,
      T_dur=3.0)

m1_reset = Model(name='',
      drift=DriftUrgencyGatedDip(snr=Fitted(8.540384994452571, minval=0.5, maxval=20), noise=Fitted(0.939630320448272, minval=0.2, maxval=2), t1=Fitted(0.376926843805476, minval=0, maxval=1), t1slope=Fitted(2.0550531206940854, minval=0, maxval=3), maxcoh=70, leak=Fitted(5.303231630655627, minval=-10, maxval=30), leaktarget=Fitted(0.07750000597007649, minval=-0.5, maxval=0.5), leaktargramp=Fitted(0.235498611069553, minval=0, maxval=3), dipstart=Fitted(-0.16006925975791975, minval=-0.25, maxval=0), dipstop=Fitted(0.002948614642891985, minval=0, maxval=0.1), diptype=2, dipparam=Fitted(7.237159473710119, minval=0, maxval=50), nd2=0),
      noise=NoiseUrgencyDip(noise=Fitted(0.939630320448272, minval=0.2, maxval=2), t1=Fitted(0.376926843805476, minval=0, maxval=1), t1slope=Fitted(2.0550531206940854, minval=0, maxval=3), dipstart=Fitted(-0.16006925975791975, minval=-0.25, maxval=0), dipstop=Fitted(0.002948614642891985, minval=0, maxval=0.1), diptype=2, nd2=0),
      bound=BoundDip(B=1, dipstart=Fitted(-0.16006925975791975, minval=-0.25, maxval=0), dipstop=Fitted(0.002948614642891985, minval=0, maxval=0.1), diptype=2),
      IC=ICPoint(x0=Fitted(0.07750000597007649, minval=-0.5, maxval=0.5)),
      overlay=OverlayChain(overlays=[OverlayNonDecision(nondectime=Fitted(0.21101574057013914, minval=0, maxval=0.3)), OverlayDipRatio(detect=Fitted(36.69568994991643, minval=2, maxval=50), diptype=2), OverlayPoissonMixture(pmixturecoef=Fitted(0.07870117978421114, minval=0, maxval=0.1), rate=Fitted(1.0578807371570935, minval=0.1, maxval=2))]),
      dx=0.005,
      dt=0.005,
      T_dur=3.0)

m2_reset = Model(name='',
      drift=DriftUrgencyGatedDip(snr=Fitted(5.558893164532837, minval=0.5, maxval=20), noise=Fitted(1.1266769704065898, minval=0.2, maxval=2), t1=Fitted(0.2051644625489287, minval=0, maxval=1), t1slope=Fitted(1.3327003007885718, minval=0, maxval=3), maxcoh=63, leak=Fitted(12.481735596979597, minval=-10, maxval=30), leaktarget=Fitted(0.015907537602473848, minval=-0.5, maxval=0.5), leaktargramp=Fitted(0.1596782056231758, minval=0, maxval=3), dipstart=Fitted(-0.22654774186806043, minval=-0.25, maxval=0), dipstop=Fitted(0.0039057285122713475, minval=0, maxval=0.1), diptype=2, dipparam=Fitted(8.805973950181391, minval=0, maxval=50), nd2=0),
      noise=NoiseUrgencyDip(noise=Fitted(1.1266769704065898, minval=0.2, maxval=2), t1=Fitted(0.2051644625489287, minval=0, maxval=1), t1slope=Fitted(1.3327003007885718, minval=0, maxval=3), dipstart=Fitted(-0.22654774186806043, minval=-0.25, maxval=0), dipstop=Fitted(0.0039057285122713475, minval=0, maxval=0.1), diptype=2, nd2=0),
      bound=BoundDip(B=1, dipstart=Fitted(-0.22654774186806043, minval=-0.25, maxval=0), dipstop=Fitted(0.0039057285122713475, minval=0, maxval=0.1), diptype=2),
      IC=ICPoint(x0=Fitted(0.015907537602473848, minval=-0.5, maxval=0.5)),
      overlay=OverlayChain(overlays=[OverlayNonDecision(nondectime=Fitted(0.23378004824747164, minval=0, maxval=0.3)), OverlayDipRatio(detect=Fitted(2.1985943043271923, minval=2, maxval=50), diptype=2), OverlayPoissonMixture(pmixturecoef=Fitted(0.03404382751452179, minval=0, maxval=0.1), rate=Fitted(0.4317755989703897, minval=0.1, maxval=2))]),
      dx=0.005,
      dt=0.005,
      T_dur=3.0)

m1_pep = Model(name='',
      drift=DriftUrgencyGatedDip(snr=Fitted(8.832972019445862, minval=0.5, maxval=20), noise=Fitted(1.0078561963371921, minval=0.2, maxval=2), t1=Fitted(0.40100276591512546, minval=0, maxval=1), t1slope=Fitted(2.0989210837866445, minval=0, maxval=3), maxcoh=70, leak=Fitted(6.429409284269061, minval=-10, maxval=30), leaktarget=Fitted(0.07250151889268981, minval=-0.5, maxval=0.5), leaktargramp=Fitted(0.18332467537867733, minval=0, maxval=3), dipstart=Fitted(-0.12522642238252274, minval=-0.25, maxval=0), dipstop=Fitted(0.028692116593979435, minval=0, maxval=0.1), diptype=1, dipparam=0, nd2=0),
      noise=NoiseUrgencyDip(noise=Fitted(1.0078561963371921, minval=0.2, maxval=2), t1=Fitted(0.40100276591512546, minval=0, maxval=1), t1slope=Fitted(2.0989210837866445, minval=0, maxval=3), dipstart=Fitted(-0.12522642238252274, minval=-0.25, maxval=0), dipstop=Fitted(0.028692116593979435, minval=0, maxval=0.1), diptype=1, nd2=0),
      bound=BoundDip(B=1, dipstart=Fitted(-0.12522642238252274, minval=-0.25, maxval=0), dipstop=Fitted(0.028692116593979435, minval=0, maxval=0.1), diptype=1),
      IC=ICPoint(x0=Fitted(0.07250151889268981, minval=-0.5, maxval=0.5)),
      overlay=OverlayChain(overlays=[OverlayNonDecision(nondectime=Fitted(0.19756445939296025, minval=0, maxval=0.3)), OverlayDipRatio(detect=Fitted(9.216110069665165, minval=2, maxval=50), diptype=1), OverlayPoissonMixture(pmixturecoef=Fitted(0.06051383398198901, minval=0, maxval=0.1), rate=Fitted(1.0357620299786299, minval=0.1, maxval=2))]),
      dx=0.005,
      dt=0.005,
      T_dur=3.0)

m2_pep = Model(name='',
      drift=DriftUrgencyGatedDip(snr=Fitted(4.505873697665654, minval=0.5, maxval=20), noise=Fitted(0.8646978315917069, minval=0.2, maxval=2), t1=Fitted(0.3055286858049485, minval=0, maxval=1), t1slope=Fitted(1.3181380911773535, minval=0, maxval=3), maxcoh=63, leak=Fitted(5.61039827833859, minval=-10, maxval=30), leaktarget=Fitted(0.017027948732443905, minval=-0.5, maxval=0.5), leaktargramp=Fitted(0.2497449262789112, minval=0, maxval=3), dipstart=Fitted(-0.14716771202266943, minval=-0.25, maxval=0), dipstop=Fitted(0.00891457419940691, minval=0, maxval=0.1), diptype=1, dipparam=0, nd2=0),
      noise=NoiseUrgencyDip(noise=Fitted(0.8646978315917069, minval=0.2, maxval=2), t1=Fitted(0.3055286858049485, minval=0, maxval=1), t1slope=Fitted(1.3181380911773535, minval=0, maxval=3), dipstart=Fitted(-0.14716771202266943, minval=-0.25, maxval=0), dipstop=Fitted(0.00891457419940691, minval=0, maxval=0.1), diptype=1, nd2=0),
      bound=BoundDip(B=1, dipstart=Fitted(-0.14716771202266943, minval=-0.25, maxval=0), dipstop=Fitted(0.00891457419940691, minval=0, maxval=0.1), diptype=1),
      IC=ICPoint(x0=Fitted(0.017027948732443905, minval=-0.5, maxval=0.5)),
      overlay=OverlayChain(overlays=[OverlayNonDecision(nondectime=Fitted(0.2099092842459185, minval=0, maxval=0.3)), OverlayDipRatio(detect=Fitted(11.526459113448741, minval=2, maxval=50), diptype=1), OverlayPoissonMixture(pmixturecoef=Fitted(0.03848253954263658, minval=0, maxval=0.1), rate=Fitted(0.3484094548380183, minval=0.1, maxval=2))]),
      dx=0.005,
      dt=0.005,
      T_dur=3.0)

_df = []
for m,mname,MONKEY in [(m1_gate, "motor", "Q"),(m2_gate, "motor", "P"),
                       (m1_pep, "pause", "Q"),(m2_pep, "pause", "P"),
                       (m1_reset, "reset", "Q"),(m2_reset, "reset", "P"),]:
    pss = [0, 400, 800]
    cohs = [53, 60, 70] if MONKEY == "Q" else [52, 57, 63]
    highrews = [0, 1]
    dipstart = m._bounddep.dipstart
    dipstop = m._bounddep.dipstop
    for coh in cohs:
        for ps in [0, 400, 800]:
            for rew in highrews:
                pre = m.solve(conditions={"coherence": coh, "presample": ps, "highreward": rew})
                pre_pcorr = pre.prob_correct()
                pre_rt = pre.mean_decision_time()
                pre_rt_all = np.sum((pre.corr+pre.err)*pre.model.t_domain()) / (pre.prob_correct()+pre.prob_error())
                m._bounddep.dipstart = .1
                m._bounddep.dipstop = .1
                m._driftdep.dipstart = .1
                m._driftdep.dipstop = .1
                m._noisedep.dipstart = .1
                m._noisedep.dipstop = .1
                post = m.solve(conditions={"coherence": coh, "presample": ps, "highreward": rew})
                post_pcorr = post.prob_correct()
                post_rt = post.mean_decision_time()
                post_rt_all = np.sum((post.corr+post.err)*post.model.t_domain()) / (post.prob_correct()+post.prob_error())
                m._bounddep.dipstart = dipstart
                m._bounddep.dipstop = dipstop
                m._driftdep.dipstart = dipstart
                m._driftdep.dipstop = dipstop
                m._noisedep.dipstart = dipstart
                m._noisedep.dipstop = dipstop
                _df.append([mname, MONKEY, coh, ps, rew, pre_pcorr, post_pcorr, pre_rt, post_rt, pre_rt_all, post_rt_all])
                print(_df[-1])
    pre = ddm.solve_partial_conditions(m, conditions={"coherence": cohs, "presample": pss, "highreward": highrews})
    pre_pcorr = pre.prob_correct()
    pre_rt = pre.mean_decision_time()
    pre_rt_all = np.sum((pre.corr+pre.err)*pre.model.t_domain()) / (pre.prob_correct()+pre.prob_error())
    m._bounddep.dipstart = .1
    m._bounddep.dipstop = .1
    m._driftdep.dipstart = .1
    m._driftdep.dipstop = .1
    m._noisedep.dipstart = .1
    m._noisedep.dipstop = .1
    post = ddm.solve_partial_conditions(m, conditions={"coherence": cohs, "presample": pss, "highreward": highrews})
    post_pcorr = post.prob_correct()
    post_rt = post.mean_decision_time()
    post_rt_all = np.sum((post.corr+post.err)*post.model.t_domain()) / (post.prob_correct()+post.prob_error())
    m._bounddep.dipstart = dipstart
    m._bounddep.dipstop = dipstop
    m._driftdep.dipstart = dipstart
    m._driftdep.dipstop = dipstop
    m._noisedep.dipstart = dipstart
    m._noisedep.dipstop = dipstop
    _df.append([mname, MONKEY, -1, -1, -1, pre_pcorr, post_pcorr, pre_rt, post_rt, pre_rt_all, post_rt_all])
    print(_df[-1])

import pandas
import diplib
df = pandas.DataFrame(_df, columns=["model", "monkey", "coherence", "presample", "high_reward", "with", "without", "with_rt", "without_rt", "with_rtall", "without_rtall"])
df["improvement"] = df["with"] - df["without"]
df["improvement_rt"] = df["with_rt"] - df["without_rt"]
df["improvement_rtall"] = df["with_rtall"] - df["without_rtall"]
df['color'] = df.apply(lambda row : diplib.get_color(row['coherence'], row['presample']), axis=1)
df['improvement_rt'] *= 1000
df['improvement_rtall'] *= 1000
df.to_pickle("model-performance.pandas.pkl")
