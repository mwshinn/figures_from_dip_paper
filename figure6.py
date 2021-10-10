import paranoid.ignore
import diplib
import numpy as np
import matplotlib.pyplot as plt
import scipy
import seaborn as sns
import pandas
import sys

from canvas import Canvas, Height, Width, Vector, Point

MONKEY = sys.argv[1]

USE50 = True
USE53 = True
USE60 = True

if MONKEY == "Q":
    HC,MC,LC = 70, 60, 53
elif MONKEY == "P":
    HC,MC,LC = 63, 57, 52

c = Canvas(6.9, 7.1, fontsize=8)

#c.add_axis("model", Point(.2, .2, "figure"), Point(.5, .9, "figure"))
#c.add_axis("data", Point(.65, .2, "figure"), Point(.95, .9, "figure"))
c.add_grid(["model_pep_rt", "model_leak_rt", "model_gate_rt", "data_rt"], 1, Point(.55, 4.1, "absolute"), Point(6.7, 5.2, "absolute"), size=Vector(1.1, 1.1, "absolute"))
c.add_grid(["model_pep_sat", "model_leak_sat", "model_gate_sat", "ghost"], 1, Point(.55, 2.3, "absolute"), Point(6.7, 3.4, "absolute"), size=Vector(1.1, 1.1, "absolute"))
c.add_grid(["model_pep", "model_leak", "model_gate", "data"], 1, Point(.55, .5, "absolute"), Point(6.7, 1.6, "absolute"), size=Vector(1.1, 1.1, "absolute"))

c.ax("ghost").axis("off")

c.add_axis("compare", Point(0, 1, "axis_data_rt") + Vector(.1, .65, "in"), Point(1, 1, "axis_data_rt") + Vector(-.1, 1.5, "in"))


for name,num,fullname in [("pep", "1", "Pause model"), ("leak", "2", "Reset model"), ("gate", "3", "Motor suppression model")]:
    print(name)
    c.add_image(f"static/diagram{num}.png", Point(0, 1, "axis_model_"+name+"_rt") + Vector(-.1, .3, "absolute"), width=Width(1.4, "absolute"), horizontalalignment="left", verticalalignment="bottom")
    c.add_text(fullname, Point(.5, 1, "axis_model_"+name+"_rt") + Vector(0, 1.65, "absolute"), horizontalalignment="center")
    c.add_text(fullname, Point(.5, 1.1, "axis_model_"+name+"_rt"))
    c.add_text(fullname, Point(.5, 1.1, "axis_model_"+name+"_sat"))
    c.add_text(fullname, Point(.5, 1.1, "axis_model_"+name))

#################### Data ####################

ax = c.ax("data")
T, mean_inrf = diplib.get_mean_conditional_activity(monkey=MONKEY, ps=800, coh=HC, hr_in_rf=True, hr_choice=True, correct=True, time_range=(-600, 600))
T, mean_outrf = diplib.get_mean_conditional_activity(monkey=MONKEY, ps=800, coh=HC, hr_in_rf=True, hr_choice=False, correct=True, time_range=(-600, 600))
ax.plot(T, mean_inrf, linestyle='-', c=(.8, .8, .8), lw=4)
ax.plot(T, mean_outrf, linestyle='-', c='r', lw=1.5)

# bounds_inrf = diplib.bootstrap_trialwise_ci(N=1000, monkey=MONKEY, ps=800, coh=HC, hr_in_rf=True, hr_choice=True, correct=True, time_range=(-600, 600))
# bounds_outrf = diplib.bootstrap_trialwise_ci(N=1000, monkey=MONKEY, ps=800, coh=HC, hr_in_rf=True, hr_choice=False, correct=True, time_range=(-600, 600))

# ax.fill_between(T, bounds_inrf[0,:], bounds_inrf[1,:], color='r', alpha=.2, zorder=-1)
# ax.fill_between(T, bounds_outrf[0,:], bounds_outrf[1,:], color=(.8, .8, .8), alpha=.5, zorder=-1)


if MONKEY == "Q":
    ax.set_ylim(23, 38)
elif MONKEY == "P":
    ax.set_ylim(38, 68)

ax.set_ylabel("Firing rate (hz)")
ax.set_xlabel("Time (ms) from sample")
ax.set_xticks([-.200, 0, .200])
ax.set_xlim(-.400, .400)
ax.set_xticklabels(["-200", "0", "200"])
sns.despine(right=False, top=False, ax=ax)

if MONKEY == "Q":
    c.add_text("Rebound", Point(.6, .90, "axis_data"), horizontalalignment="right")
    c.add_arrow(Point(.62, .90, "axis_data"), Point(.23, 31, "data"))
else:
    c.add_text("Rebound", Point(.5, .70, "axis_data"), horizontalalignment="right")
    c.add_arrow(Point(.5, .70, "axis_data"), Point(.18, 55, "data"))

if MONKEY == "Q":
    c.add_text("Dip", Point(-.05, .2, ("data", "axis_data")), horizontalalignment="right")
    c.add_arrow(Point(-.05, .2, ("data", "axis_data")), Point(.130, 29, "data"))
else:
    c.add_text("Dip", Point(.03, .1, ("data", "axis_data")), horizontalalignment="right")
    c.add_arrow(Point(.03, .1, ("data", "axis_data")), Point(.170, 45.5, "data"))



c.add_text("FEF activity", Point(.5, 1.1, "axis_data"), horizontalalignment="center")
c.add_text("RT distribution", Point(.5, 1.1, "axis_data_rt"), horizontalalignment="center")

#################### Fits ####################

try:
    df = pandas.read_pickle("model-performance.pandas.pkl")
except:
    raise IOError("Please run figure6-runmodels.py to generate the dataframe")

df['color'] = df.apply(lambda row : diplib.get_color(coh=row['coherence'], ps=row['presample']), axis=1)
models = ["pause", "reset", "motor"]
axis_names = {"pause": "pep", "reset": "leak", "motor": "gate"}
lims = {"Q": ((-.01, .15), (-5, 45)), "P": ((-.01, .08), (-5, 25))}
for model in models:
    ax = c.ax("model_"+axis_names[model]+"_sat")
    ax.scatter(x='improvement', y='improvement_rt', color='color', marker='o', data=df.query(f'high_reward == 1 and monkey == {MONKEY!r} and model == {model!r}'), edgecolor='none', s=40)
    ax.scatter(x='improvement', y='improvement_rt', color='color', marker='o', data=df.query(f'high_reward == 0 and monkey == {MONKEY!r} and model == {model!r}'), facecolor='none', linewidth=2, s=30)
    # One point, only for the mean
    ax.scatter(x='improvement', y='improvement_rt', c='k', s=35, marker='+', data=df.query(f'coherence == -1 and monkey == {MONKEY!r} and model == {model!r}'))
    # Make the plot look nice
    ax.set_xlabel("$\Delta$ accuracy")
    if model == models[0]:
        ax.set_ylabel("$\Delta$ RT (ms)")
    ax.set_xlim(lims[MONKEY][0])
    ax.set_ylim(lims[MONKEY][1])
    sns.despine(ax=ax)

diplib.make_gridlegend(c, Point(-.1, .35, "axis_ghost"), shorten=True, zero=False)

c.add_legend(Point(-.05, .3, "axis_ghost"), [("Mean", dict(markersize=np.sqrt(35), marker='+', c='k', linestyle='None')),
                                            ("Large reward", dict(markersize=np.sqrt(40), markeredgecolor='none', c='gray', linestyle='None', marker='o')),
                                            ("Small reward", dict(markersize=np.sqrt(30), markerfacecolor='none', c='gray', linestyle='None', marker='o', markeredgewidth=2))],
             line_spacing=Vector(0, 1.4, "Msize"), sym_width=Vector(1, 0, "Msize"))

#################### Model ####################

import ddm

class DriftCM(ddm.Drift):
    name = "Color match drift"
    required_parameters = ['drift', 'leak', 'bias', 'leakdip', 'pep']
    required_conditions = ['coh', 'ps']
    def get_drift(self, t, x, conditions, **kwargs):
        ps_time = conditions['ps']/1000
        dip_time = .1
        drift_rate = 0 if t < ps_time + dip_time/2 else self.drift * conditions['coh']
        if ps_time < t and t < ps_time + dip_time and self.leakdip:
            return drift_rate - x * self.leakdip
        elif ps_time < t and t < ps_time + dip_time and self.pep:
            return 0
        elif t < ps_time:
            return drift_rate - (x - self.bias) * self.leak
        else:
            return drift_rate - (x - self.bias) * self.leak

class NoisePEP(ddm.Noise):
    name = "constant noise with optional PEP"
    required_parameters = ["noise"]
    required_conditions = ["ps"]
    def get_noise(self, t, conditions, **kwargs):
        ps_time = conditions['ps']/1000
        dip_time = .1
        if ps_time < t and t < ps_time + dip_time:
            return 0
        else:
            return self.noise


class ICPoint(ddm.InitialCondition):
    """Initial condition: a dirac delta function in the center of the domain."""
    name = "point_source"
    required_parameters = ["x0"]
    def get_IC(self, x, dx, conditions={}):
        start = np.round(self.x0/dx)
        shift_i = int(start + (len(x)-1)/2)
        assert shift_i >= 0 and shift_i < len(x), "Invalid initial conditions"
        pdf = np.zeros(len(x))
        pdf[shift_i] = 1. # Initial condition at x=self.x0.
        return pdf

from diplib import memoize
@memoize
def get_traj(m, conditions, seed):
    return  m.simulate_trial(conditions=conditions, cutoff=False, seed=seed, rk4=False)

def traj_choice(traj, bound, t_domain):
    assert len(traj) == len(bound) == len(t_domain)
    inds = np.nonzero(np.abs(traj)>bound)
    if len(inds) == 0:
        return 0
    if t_domain[inds[0][0]] < 0: # Ensure response after the cue
        return 0
    if traj[inds[0][0]] > 0: # Correct
        return 1
    if traj[inds[0][0]] < 0: # Error
        return -1
    

conds = {"coherence": 70, "presample": 800, "highreward": 1}
def plot_model(m, ax, loops=125, seed=0, corr=True):
    if corr:
        conds['highreward'] = 1
        coef = 1
    else:
        conds['highreward'] = 0
        coef = -1
    bound = [m.get_dependence("bound").get_bound(t-m.get_dependence("overlay").nondectime, conditions=conds)**(1/2) for t in m.t_domain()]
    rng = np.random.RandomState(seed)
    t_nd = m.get_dependence('overlay').nondectime
    trials = []
    for i in range(0, loops):
        print(i)
        while True:# while (max(traj) > 1 and corr==True) or (max(traj) < 1 and corr==False): # Only correct trials
            traj = get_traj(m, conditions=conds, seed=rng.randint(1e8))
            if traj_choice(traj, bound, m.t_domain()-t_nd) != 1:
                print("Trying againn")
                continue 
            trials.append(scipy.signal.savgol_filter(traj, 3, 1))
            break
    ts = np.mean(trials, axis=0)
    #ts = scipy.signal.savgol_filter(ts, 15, 1)
    ts = scipy.ndimage.filters.gaussian_filter(ts, 4)
    ax.plot(m.t_domain()*1000-800, coef*ts/bound, c='k', linestyle=('--' if corr==False else '-'))
    #plt.plot(np.mean(trials, axis=0), linestyle='--', linewidth=5, c='k')
    #plt.show()

def plot_model2(m, ax, corr=True):
    if corr:
        conds['highreward'] = 1
        coef = 1
    else:
        conds['highreward'] = 0
        coef = -1
    #bound = 1
    #bound = [m.get_dependence("bound").get_bound(t-m.get_dependence("overlay").nondectime, conditions=conds) for t in m.t_domain()]
    bound = [m.get_dependence("bound").get_bound(t, conditions=conds) for t in m.t_domain()]
    t_nd = m.get_dependence('overlay').nondectime
    sol = m.solve_numerical(conditions=conds, method="implicit", return_evolution=True)
    #plt.imshow(np.log(sol.pdf_evolution()+.00001)); plt.show()
    ts = np.mean(sol.pdf_evolution()*np.reshape(m.x_domain(conditions=conds), (-1, 1)), axis=0)/(.0000001+np.mean(sol.pdf_evolution(), axis=0))
    #ts = scipy.signal.savgol_filter(ts, 15, 1)
    #ts = scipy.ndimage.filters.gaussian_filter(ts, 4)
    if corr:
        ts = np.maximum(ts, 0)
    else:
        ts = np.minimum(ts, 0)
    ax.plot((t_nd+m.t_domain())*1000-800, scipy.convolve(coef*ts/bound, scipy.stats.norm(0,1).pdf(np.linspace(-3, 3, 50))/np.sum(scipy.stats.norm(0,1).pdf(np.linspace(-3, 3, 100))), 'same'), c=('r' if corr==False else (.8, .8, .8)), linestyle=('-' if corr==False else '-'), lw=(4 if corr else 1.5))
    #plt.plot(np.mean(trials, axis=0), linestyle='--', linewidth=5, c='k')
    #plt.show()

from ddm import Model, Fitted
from ddm.models import *
from dipmodels import *
if MONKEY == "Q":
    m_gate = Model(name='',
        drift=DriftUrgencyGatedDip(snr=Fitted(9.055489816606826, minval=0.5, maxval=20), noise=Fitted(1.0210007162574837, minval=0.2, maxval=2), t1=Fitted(0.3655364577704902, minval=0, maxval=1), t1slope=Fitted(2.0110724435351504, minval=0, maxval=3), maxcoh=70, leak=Fitted(6.915762729607658, minval=-10, maxval=30), leaktarget=Fitted(0.08164321017639141, minval=-0.5, maxval=0.5), leaktargramp=Fitted(0.14801117597537627, minval=0, maxval=3), dipstart=Fitted(-0.21934331140214572, minval=-0.25, maxval=0), dipstop=Fitted(0.00907722935591773, minval=0, maxval=0.1), diptype=3, dipparam=0, nd2=0),
        noise=NoiseUrgencyDip(noise=Fitted(1.0210007162574837, minval=0.2, maxval=2), t1=Fitted(0.3655364577704902, minval=0, maxval=1), t1slope=Fitted(2.0110724435351504, minval=0, maxval=3), dipstart=Fitted(-0.21934331140214572, minval=-0.25, maxval=0), dipstop=Fitted(0.00907722935591773, minval=0, maxval=0.1), diptype=3, nd2=0),
        bound=BoundDip(B=1, dipstart=Fitted(-0.21934331140214572, minval=-0.25, maxval=0), dipstop=Fitted(0.00907722935591773, minval=0, maxval=0.1), diptype=3),
        IC=ICPoint(x0=Fitted(0.08164321017639141, minval=-0.5, maxval=0.5)),
        overlay=OverlayChain(overlays=[OverlayNonDecision(nondectime=Fitted(0.22032108588143898, minval=0, maxval=0.3)), OverlayDipRatio(detect=Fitted(4.451814358885121, minval=2, maxval=50), diptype=3), OverlayPoissonMixture(pmixturecoef=Fitted(0.053169489731494414, minval=0, maxval=0.1), rate=Fitted(1.1880292212319197, minval=0.1, maxval=2))]),
        dx=0.005,
        dt=0.005,
        T_dur=6.0)
    m_reset = Model(name='',
        drift=DriftUrgencyGatedDip(snr=Fitted(8.540384994452571, minval=0.5, maxval=20), noise=Fitted(0.939630320448272, minval=0.2, maxval=2), t1=Fitted(0.376926843805476, minval=0, maxval=1), t1slope=Fitted(2.0550531206940854, minval=0, maxval=3), maxcoh=70, leak=Fitted(5.303231630655627, minval=-10, maxval=30), leaktarget=Fitted(0.07750000597007649, minval=-0.5, maxval=0.5), leaktargramp=Fitted(0.235498611069553, minval=0, maxval=3), dipstart=Fitted(-0.16006925975791975, minval=-0.25, maxval=0), dipstop=Fitted(0.002948614642891985, minval=0, maxval=0.1), diptype=2, dipparam=Fitted(7.237159473710119, minval=0, maxval=50), nd2=0),
        noise=NoiseUrgencyDip(noise=Fitted(0.939630320448272, minval=0.2, maxval=2), t1=Fitted(0.376926843805476, minval=0, maxval=1), t1slope=Fitted(2.0550531206940854, minval=0, maxval=3), dipstart=Fitted(-0.16006925975791975, minval=-0.25, maxval=0), dipstop=Fitted(0.002948614642891985, minval=0, maxval=0.1), diptype=2, nd2=0),
        bound=BoundDip(B=1, dipstart=Fitted(-0.16006925975791975, minval=-0.25, maxval=0), dipstop=Fitted(0.002948614642891985, minval=0, maxval=0.1), diptype=2),
        IC=ICPoint(x0=Fitted(0.07750000597007649, minval=-0.5, maxval=0.5)),
        overlay=OverlayChain(overlays=[OverlayNonDecision(nondectime=Fitted(0.21101574057013914, minval=0, maxval=0.3)), OverlayDipRatio(detect=Fitted(36.69568994991643, minval=2, maxval=50), diptype=2), OverlayPoissonMixture(pmixturecoef=Fitted(0.07870117978421114, minval=0, maxval=0.1), rate=Fitted(1.0578807371570935, minval=0.1, maxval=2))]),
        dx=0.005,
        dt=0.005,
        T_dur=6.0)
    m_pep = Model(name='',
        drift=DriftUrgencyGatedDip(snr=Fitted(8.832972019445862, minval=0.5, maxval=20), noise=Fitted(1.0078561963371921, minval=0.2, maxval=2), t1=Fitted(0.40100276591512546, minval=0, maxval=1), t1slope=Fitted(2.0989210837866445, minval=0, maxval=3), maxcoh=70, leak=Fitted(6.429409284269061, minval=-10, maxval=30), leaktarget=Fitted(0.07250151889268981, minval=-0.5, maxval=0.5), leaktargramp=Fitted(0.18332467537867733, minval=0, maxval=3), dipstart=Fitted(-0.12522642238252274, minval=-0.25, maxval=0), dipstop=Fitted(0.028692116593979435, minval=0, maxval=0.1), diptype=1, dipparam=0, nd2=0),
        noise=NoiseUrgencyDip(noise=Fitted(1.0078561963371921, minval=0.2, maxval=2), t1=Fitted(0.40100276591512546, minval=0, maxval=1), t1slope=Fitted(2.0989210837866445, minval=0, maxval=3), dipstart=Fitted(-0.12522642238252274, minval=-0.25, maxval=0), dipstop=Fitted(0.028692116593979435, minval=0, maxval=0.1), diptype=1, nd2=0),
        bound=BoundDip(B=1, dipstart=Fitted(-0.12522642238252274, minval=-0.25, maxval=0), dipstop=Fitted(0.028692116593979435, minval=0, maxval=0.1), diptype=1),
        IC=ICPoint(x0=Fitted(0.07250151889268981, minval=-0.5, maxval=0.5)),
        overlay=OverlayChain(overlays=[OverlayNonDecision(nondectime=Fitted(0.19756445939296025, minval=0, maxval=0.3)), OverlayDipRatio(detect=Fitted(9.216110069665165, minval=2, maxval=50), diptype=1), OverlayPoissonMixture(pmixturecoef=Fitted(0.06051383398198901, minval=0, maxval=0.1), rate=Fitted(1.0357620299786299, minval=0.1, maxval=2))]),
        dx=0.005,
        dt=0.005,
        T_dur=6.0)
else:
    m_pep = Model(name='',
      drift=DriftUrgencyGatedDip(snr=Fitted(4.505873697665654, minval=0.5, maxval=20), noise=Fitted(0.8646978315917069, minval=0.2, maxval=2), t1=Fitted(0.3055286858049485, minval=0, maxval=1), t1slope=Fitted(1.3181380911773535, minval=0, maxval=3), maxcoh=63, leak=Fitted(5.61039827833859, minval=-10, maxval=30), leaktarget=Fitted(0.017027948732443905, minval=-0.5, maxval=0.5), leaktargramp=Fitted(0.2497449262789112, minval=0, maxval=3), dipstart=Fitted(-0.14716771202266943, minval=-0.25, maxval=0), dipstop=Fitted(0.00891457419940691, minval=0, maxval=0.1), diptype=1, dipparam=0, nd2=0),
      noise=NoiseUrgencyDip(noise=Fitted(0.8646978315917069, minval=0.2, maxval=2), t1=Fitted(0.3055286858049485, minval=0, maxval=1), t1slope=Fitted(1.3181380911773535, minval=0, maxval=3), dipstart=Fitted(-0.14716771202266943, minval=-0.25, maxval=0), dipstop=Fitted(0.00891457419940691, minval=0, maxval=0.1), diptype=1, nd2=0),
      bound=BoundDip(B=1, dipstart=Fitted(-0.14716771202266943, minval=-0.25, maxval=0), dipstop=Fitted(0.00891457419940691, minval=0, maxval=0.1), diptype=1),
      IC=ICPoint(x0=Fitted(0.017027948732443905, minval=-0.5, maxval=0.5)),
      overlay=OverlayChain(overlays=[OverlayNonDecision(nondectime=Fitted(0.2099092842459185, minval=0, maxval=0.3)), OverlayDipRatio(detect=Fitted(11.526459113448741, minval=2, maxval=50), diptype=1), OverlayPoissonMixture(pmixturecoef=Fitted(0.03848253954263658, minval=0, maxval=0.1), rate=Fitted(0.3484094548380183, minval=0.1, maxval=2))]),
      dx=0.005,
      dt=0.005,
      T_dur=6.0)
    m_reset = Model(name='',
      drift=DriftUrgencyGatedDip(snr=Fitted(5.558893164532837, minval=0.5, maxval=20), noise=Fitted(1.1266769704065898, minval=0.2, maxval=2), t1=Fitted(0.2051644625489287, minval=0, maxval=1), t1slope=Fitted(1.3327003007885718, minval=0, maxval=3), maxcoh=63, leak=Fitted(12.481735596979597, minval=-10, maxval=30), leaktarget=Fitted(0.015907537602473848, minval=-0.5, maxval=0.5), leaktargramp=Fitted(0.1596782056231758, minval=0, maxval=3), dipstart=Fitted(-0.22654774186806043, minval=-0.25, maxval=0), dipstop=Fitted(0.0039057285122713475, minval=0, maxval=0.1), diptype=2, dipparam=Fitted(8.805973950181391, minval=0, maxval=50), nd2=0),
      noise=NoiseUrgencyDip(noise=Fitted(1.1266769704065898, minval=0.2, maxval=2), t1=Fitted(0.2051644625489287, minval=0, maxval=1), t1slope=Fitted(1.3327003007885718, minval=0, maxval=3), dipstart=Fitted(-0.22654774186806043, minval=-0.25, maxval=0), dipstop=Fitted(0.0039057285122713475, minval=0, maxval=0.1), diptype=2, nd2=0),
      bound=BoundDip(B=1, dipstart=Fitted(-0.22654774186806043, minval=-0.25, maxval=0), dipstop=Fitted(0.0039057285122713475, minval=0, maxval=0.1), diptype=2),
      IC=ICPoint(x0=Fitted(0.015907537602473848, minval=-0.5, maxval=0.5)),
      overlay=OverlayChain(overlays=[OverlayNonDecision(nondectime=Fitted(0.23378004824747164, minval=0, maxval=0.3)), OverlayDipRatio(detect=Fitted(2.1985943043271923, minval=2, maxval=50), diptype=2), OverlayPoissonMixture(pmixturecoef=Fitted(0.03404382751452179, minval=0, maxval=0.1), rate=Fitted(0.4317755989703897, minval=0.1, maxval=2))]),
      dx=0.005,
      dt=0.005,
      T_dur=6.0)
    m_gate = Model(name='',
      drift=DriftUrgencyGatedDip(snr=Fitted(4.982349578945338, minval=0.5, maxval=20), noise=Fitted(0.9493607207603991, minval=0.2, maxval=2), t1=Fitted(0.24989636700560622, minval=0, maxval=1), t1slope=Fitted(1.3263731969806372, minval=0, maxval=3), maxcoh=63, leak=Fitted(7.692866409205677, minval=-10, maxval=30), leaktarget=Fitted(0.018005919237749864, minval=-0.5, maxval=0.5), leaktargramp=Fitted(0.20753823987831005, minval=0, maxval=3), dipstart=Fitted(-0.24439145528487205, minval=-0.25, maxval=0), dipstop=Fitted(0.014273725045643261, minval=0, maxval=0.1), diptype=3, dipparam=0, nd2=0),
      noise=NoiseUrgencyDip(noise=Fitted(0.9493607207603991, minval=0.2, maxval=2), t1=Fitted(0.24989636700560622, minval=0, maxval=1), t1slope=Fitted(1.3263731969806372, minval=0, maxval=3), dipstart=Fitted(-0.24439145528487205, minval=-0.25, maxval=0), dipstop=Fitted(0.014273725045643261, minval=0, maxval=0.1), diptype=3, nd2=0),
      bound=BoundDip(B=1, dipstart=Fitted(-0.24439145528487205, minval=-0.25, maxval=0), dipstop=Fitted(0.014273725045643261, minval=0, maxval=0.1), diptype=3),
      IC=ICPoint(x0=Fitted(0.018005919237749864, minval=-0.5, maxval=0.5)),
      overlay=OverlayChain(overlays=[OverlayNonDecision(nondectime=Fitted(0.23301300378556872, minval=0, maxval=0.3)), OverlayDipRatio(detect=Fitted(4.8221145313327245, minval=2, maxval=50), diptype=3), OverlayPoissonMixture(pmixturecoef=Fitted(0.031182198519672435, minval=0, maxval=0.1), rate=Fitted(0.42210305081388555, minval=0.1, maxval=2))]),
      dx=0.005,
      dt=0.005,
      T_dur=6.0)


ax = c.ax("model_leak")
plot_model2(m_reset, ax)
plot_model2(m_reset, ax, corr=False)
ax.set_ylim(-.02, .09)
ax.set_xticks([-200, 0, 200])
#ax.set_xticks([-200, 0, 200])
ax.set_xlim(-400, 400)
ax.set_xlabel("Time (ms) from sample")
#ax.axvspan(100, 200, color='gray', alpha=.3, zorder=-1)
sns.despine(right=False, top=False, ax=ax)
if MONKEY == "Q":
    c.add_text("Dip", Point(-100, .1, ("model_leak", "axis_model_leak")), horizontalalignment="right")
    c.add_arrow(Point(-100, .1, ("model_leak", "axis_model_leak")), Point(110, .03, "model_leak"))
else:
    c.add_text("Dip", Point(-100, .1, ("model_leak", "axis_model_leak")), horizontalalignment="right")
    c.add_arrow(Point(-100, .1, ("model_leak", "axis_model_leak")), Point(100, .01, "model_leak"))


ax = c.ax("model_pep")
plot_model2(m_pep, ax)
plot_model2(m_pep, ax, corr=False)
ax.set_ylim(-.02, .09)
ax.set_xticks([-200, 0, 200])
ax.set_xlim(-400, 400)
ax.set_xlabel("Time (ms) from sample")
ax.set_ylabel("Predicted FEF activity")
#ax.axvspan(100, 200, color='gray', alpha=.3, zorder=-1)
sns.despine(right=False, top=False, ax=ax)


ax = c.ax("model_gate")
plot_model2(m_gate, ax)
plot_model2(m_gate, ax, corr=False)
ax.set_ylim(-.02, .09)
ax.set_xticks([-200, 0, 200])
ax.set_xlim(-400, 400)
ax.set_xlabel("Time (ms) from sample")
#ax.axvspan(100, 200, color='gray', alpha=.3, zorder=-1)
sns.despine(right=False, top=False, ax=ax)
c.add_text("Rebound", Point(.55, .85, "axis_model_gate"), horizontalalignment="right")
c.add_arrow(Point(.57, .85, "axis_model_gate"), Point(200, .01, "model_gate"))

c.add_text("Dip", Point(-100, .1, ("model_gate", "axis_model_gate")), horizontalalignment="right")
c.add_arrow(Point(-100, .1, ("model_gate", "axis_model_gate")), Point(70, .008, "model_gate"))


# Monkey Q
if MONKEY == "Q":
    BICs = [-3797.6498369898495, -4144.251824451733, -3967.380782208746, -4167.396005075162]
else:
    BICs = [6334.843349836494, 6375.163892758621, 6402.688529179219, 6263.523980891375]

ax = c.ax("compare")
#BICs = [-3546.9398923524864, -3868.637546262151, -3676.557901714192, -3771.608091795717]
BICs_plot = BICs[0]-np.asarray(BICs)[1:]
sns.barplot(BICs_plot, ["Pause", "Reset", "Motor"], color='k', ax=ax, orient="h")
ax.axvline(0, c='k', linewidth=1)
ax.yaxis.set_ticks_position('none') 
#ax.invert_xaxis()
ax.set_xlabel("$\Delta$BIC")
c.add_arrow(Point(0, 1.1, "axis_compare"), Point(1, 1.1, "axis_compare"))
c.add_text("Better model", Point(.42, 1.2, "axis_compare"), verticalalignment="center", horizontalalignment="center")
#c.add_text("Model comparison", )
sns.despine(ax=ax, left=True)


c.add_legend(Point(.1, .3, "axis_model_pep"), [("In-RF", {"color": (.8, .8, .8), "lw": 4}), ("Out-RF", {"color": 'r', "linestyle": "-", "lw": 1.5})], line_spacing=Vector(0, 1.5, "Msize"), sym_width=Vector(2, 0, "Msize"))


#################### Model RT distributions ####################


for modelname, model in [("gate", m_gate), ("pep", m_pep), ("leak", m_reset)]:
    ax = c.ax(f"model_{modelname}_rt")
    ax.cla()
    ps = 800
    for coh in [50, LC, MC, HC]:
        ax.plot(model.t_domain()*1000-800, scipy.convolve(ddm.solve_partial_conditions(model, conditions={"coherence": coh, "presample": ps, "highreward": [0,1]}).pdf_corr(), scipy.stats.norm(0,1).pdf(np.linspace(-3, 3, 50)), 'same')/np.sum(scipy.stats.norm(0,1).pdf(np.linspace(-3, 3, 50))), c=diplib.get_color(ps=ps, coh=coh))
        ax.set_xlim(-250, 250)
        ax.set_xticks([-200, 0, 200])
        ax.set_ylim(0, 1.2)
    ax.set_xlabel("Time (ms) from sample")
    sns.despine(right=False, top=False, ax=ax)

c.ax('model_pep_rt').set_ylabel("Predicted RT density")

if MONKEY == "Q":
    c.add_text("Dip", Point(0, .05, "model_pep_rt"), ha="right", va="center")
    c.add_arrow(Point(0, .05, "model_pep_rt"), Point(100, .12, "model_pep_rt"))
    c.add_text("Dip", Point(0, .05, "model_leak_rt"), ha="right", va="center")
    c.add_arrow(Point(0, .05, "model_leak_rt"), Point(100, .15, "model_leak_rt"))
    c.add_text("Dip", Point(-10, .05, "model_gate_rt"), ha="right", va="center")
    c.add_arrow(Point(-10, .05, "model_gate_rt"), Point(60, .11, "model_gate_rt"))
else:
    c.add_text("Dip", Point(0, .45, "model_pep_rt"), ha="right", va="center")
    c.add_arrow(Point(0, .45, "model_pep_rt"), Point(100, .25, "model_pep_rt"))
    c.add_text("Dip", Point(0, .45, "model_leak_rt"), ha="right", va="center")
    c.add_arrow(Point(0, .45, "model_leak_rt"), Point(100, .2, "model_leak_rt"))
    c.add_text("Dip", Point(-10, .45, "model_gate_rt"), ha="right", va="center")
    c.add_arrow(Point(0, .45, "model_gate_rt"), Point(60, .2, "model_gate_rt"))
#################### Data RT distribution ####################

ax = c.ax("data_rt")
ax.cla()
for coh in [LC, MC, HC]:
    T, activity70 = diplib.get_rt_conditional_activity(monkey=MONKEY, coh=coh, ps=800, smooth=5, time_range=(-400, 400), align="sample")
    ax.plot(T*1000, activity70, c=diplib.get_color(ps=800, coh=coh))

activity50 = diplib.get_rt_conditional_activity(monkey=MONKEY, coh=50, smooth=5, time_range=(400, 1200), align="presample")[1]
ax.plot(T*1000, activity50, c=diplib.get_color(ps=0, coh=50))
ax.set_xlim(-250, 250)
ax.set_xticks([-200, 0, 200])
if MONKEY == "Q":
    ax.set_ylim(0, 1.2)
else:
    ax.set_ylim(0, .24)

ax.set_xlabel("Time (ms) from sample")
ax.set_ylabel("Monkey RT density")
sns.despine(right=False, top=False, ax=ax)

if MONKEY == "Q":
    c.add_text("Dip", Point(40, .05, "data_rt"), ha="right", va="center")
    c.add_arrow(Point(40, .05, "data_rt"), Point(150, .11, "data_rt"))
else:
    c.add_text("Dip", Point(40, .12, "data_rt"), ha="right", va="center")
    c.add_arrow(Point(40, .12, "data_rt"), Point(110, .08, "data_rt"))

c.add_figure_labels([("a", "model_pep_rt", Vector(-.1, 1.5, "absolute")),
                     ("b", "model_leak_rt", Vector(-.1, 1.5, "absolute")),
                     ("c", "model_gate_rt", Vector(-.1, 1.5, "absolute")),
                     ("d", "compare"),
                     ("e", "model_pep_rt", Vector(-.1, 0, "absolute")),
                     ("f", "model_leak_rt", Vector(-.1, 0, "absolute")),
                     ("g", "model_gate_rt", Vector(-.1, 0, "absolute")),
                     ("h", "data_rt", Vector(-.1, 0, "absolute")),
                     ("i", "model_pep_sat", Vector(-.1, 0, "absolute")),
                     ("j", "model_leak_sat", Vector(-.1, 0, "absolute")),
                     ("k", "model_gate_sat", Vector(-.1, 0, "absolute")),
                     ("l", "model_pep", Vector(-.1, 0, "absolute")),
                     ("m", "model_leak", Vector(-.1, 0, "absolute")),
                     ("n", "model_gate", Vector(-.1, 0, "absolute")),
                     ("o", "data")])


if MONKEY == "Q":
    c.save(f"figure6.pdf")
else:
    c.save(f"figureS9.pdf")

