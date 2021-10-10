#from paranoid.ignore import *
from paranoid.decorators import accepts, returns, requires, ensures, paranoidclass
from paranoid.types import RangeOpenClosed, RangeOpen, Range, Positive0, NDArray, ParametersDict, Natural0, Set, Self, Number, Positive
import ddm
import numpy as np
import scipy
import scipy.stats
import sys



@paranoidclass
class Conditions(ParametersDict):
    """Valid conditions for our task."""
    def __init__(self):
        super().__init__({"coherence": Range(50, 100),
                          "presample": Natural0,
                          "highreward": Set([0, 1]),
                          "blocktype": Set([1, 2])})
    def test(self, v):
        super().test(v)
        assert isinstance(v, dict), "Non-dict passed"
        assert not set(v.keys()) - {'coherence', 'presample', 'highreward', 'blocktype'}, \
            "Invalid reward keys"
    def generate(self):
        print("Generating")
        for ps,coh,hr,bt in zip([0, 400, 800, 1000], [50, 53, 57, 60, 70], [0, 1], [1, 2]):
            yield {"presample": ps, "coherence": coh, "highreward": hr, "blocktype": bt}
            

# TODO separate out drift constant
@accepts(Range(50, 100), RangeOpenClosed(50, 100))
@requires("coh <= max_coh")
@returns(Range(0, 1))
@ensures("return == 0 <--> coh == 50")
@ensures("return == 1 <--> coh == max_coh")
# Monotonic increasing in coh, decreasing in exponent
@ensures("coh >= coh` and max_coh == max_coh` --> return >= return`")
def coh_transform(coh, max_coh):
    return (coh-50)/(max_coh-50)

@accepts(Positive0, Positive0, Positive0, Positive0)
@returns(Positive0)
@ensures("return >= 0")
# Monotonic in all variables, decreasing with t1, others increasing
@ensures("t >= t` and base >= base` and t1 <= t1` and slope >= slope` --> return >= return`")
def urgency(t, base, t1, slope):
    return base + ((t-t1)*slope if t>=t1 else 0)


# diptype 0 == none, 1 == pause, 2 == reset, 3 == saccade inhibition
@paranoidclass
class DriftUrgencyGatedDip(ddm.models.Drift):
    name = "Drift with piecewise linear urgency signal, reward/timing interaction bias, and coherence change transient"
    required_parameters = ["snr", "noise", "t1", "t1slope", "maxcoh", "leak", "leaktarget", "leaktargramp", "dipstart", "dipstop", "diptype", "dipparam", "nd2"]
    required_conditions = ["coherence", "presample", "highreward"]
    default_parameters = {"leaktargramp": 0, "dipparam": 0, "diptype": -1}

    def get_drift(self, t, x, conditions, **kwargs):
        if t < self.nd2:
            return .001
        # Coherence coefficient == coherence with a non-linear transform
        dipstart = min(self.dipstart, self.dipstop) + conditions["presample"]/1000
        dipstop = max(self.dipstart, self.dipstop) + conditions["presample"]/1000
        if self.diptype == 1 and dipstart < t and t < dipstop:
            return 0
        coh_coef = coh_transform(conditions["coherence"], self.maxcoh)
        is_past_delay = 1 if t-self.nd2 > conditions["presample"]/1000 else 0
        cur_urgency = self.snr * urgency(t-self.nd2, self.noise, self.t1, self.t1slope)
        leaktarg = self.leaktarget if conditions["highreward"] else -self.leaktarget
        leak = self.leak
        leaktargramp = self.leaktargramp if conditions["highreward"] else -self.leaktargramp
        if self.diptype == 2 and dipstart < t and t < dipstop:
            leak += self.dipparam
            leaktarg = 0
            leaktargramp = 0
        return coh_coef * (cur_urgency * is_past_delay) - leak*(x-(leaktarg+leaktargramp*(t-self.nd2)))
    @staticmethod
    def _test(v):
        assert v.snr in Positive(), "Invalid SNR"
        assert v.noise in Positive0(), "Invalid noise"
        assert v.t1 in Positive0(), "Invalid t1"
        assert v.t1slope in Positive0(), "Invalid t1slope"
        assert v.maxcoh in [63, 70], "Invalid maxcoh"
        assert v.leak in Positive0(), "Invalid leak"
        assert v.leaktarget in Number(), "Invalid leak"
        assert v.leaktargramp in Number(), "Invalid leak"


@paranoidclass
class NoiseUrgencyDip(ddm.models.Noise):
    name = "Noise with piecewise linear urgency signal"
    required_parameters = ["noise", "t1", "t1slope", "dipstart", "dipstop", "diptype", "nd2"]

    @accepts(Self, t=Positive0, conditions=Conditions)
    @returns(Positive)
    def get_noise(self, t, conditions, **kwargs):
        if t < self.nd2:
            return .001
        dipstart = min(self.dipstart, self.dipstop) + conditions["presample"]/1000
        dipstop = max(self.dipstart, self.dipstop) + conditions["presample"]/1000
        if self.diptype == 1 and dipstart < t and t < dipstop:
            return 0.001
        return urgency(t-self.nd2, self.noise, self.t1, self.t1slope) + .001
    @staticmethod
    def _test(v):
        assert v.noise in Positive0(), "Invalid noise"
        assert v.t1 in Positive0(), "Invalid t1"
        assert v.t1slope in Positive0(), "Invalid t1slope"



@paranoidclass
class ICPoint(ddm.models.InitialCondition):
    """Initial condition: a dirac delta function in the center of the domain."""
    name = "point_source"
    required_parameters = ["x0"]
    required_conditions = ["highreward"]
    @accepts(Self, NDArray(d=1, t=Number), RangeOpen(0, 1), Conditions)
    @returns(NDArray(d=1))
    @requires("x.size > 1")
    @requires("all(x[1:]-x[0:-1] - dx < 1e-8)")
    @requires("x[0] < self.x0 < x[-1]")
    @ensures("sum(return) == max(return)")
    @ensures("all((e in [0, 1] for e in return))")
    @ensures("self.x0 == 0 --> list(reversed(return)) == list(return)")
    @ensures("x.shape == return.shape")
    def get_IC(self, x, dx, conditions={}):
        start = np.round(self.x0/dx)
        if not conditions['highreward']:
            start = -start
        shift_i = int(start + (len(x)-1)/2)
        assert shift_i >= 0 and shift_i < len(x), "Invalid initial conditions"
        pdf = np.zeros(len(x))
        pdf[shift_i] = 1. # Initial condition at x=self.x0.
        return pdf
    
    @staticmethod
    def _test(v):
        assert v.x0 in Number(), "Invalid starting position"

    @staticmethod
    def _generate():
        yield ICPoint(x0=0)
        yield ICPoint(x0=.4)
        yield ICPoint(x0=-.9)

class BoundDip(ddm.Bound):
    name = "collapsing_exponential"
    required_parameters = ["B", "dipstart", "dipstop", "diptype"]
    required_conditions = ["coherence", "presample"]
    def get_bound(self, t, conditions, *args, **kwargs):
        dipstart = min(self.dipstart, self.dipstop) + conditions["presample"]/1000
        dipstop = max(self.dipstart, self.dipstop) + conditions["presample"]/1000
        if self.diptype == 3 and dipstart < t and t < dipstop:
            return self.B + 4*scipy.stats.beta.pdf(t, a=3, b=3, loc=dipstart, scale=dipstop-dipstart)
        return self.B

@paranoidclass
class LossLikelihoodAndDipSquaredError(ddm.LossFunction):
    """Likelihood loss function"""
    name = "Negative log likelihood"
    def setup(self, dt, T_dur, **kwargs):
        self.dt = dt
        self.T_dur = T_dur
        # Do the least squares component
        self.hists_corr = {}
        self.hists_err = {}
        for comb in self.sample.condition_combinations(required_conditions=self.required_conditions):
            self.hists_corr[frozenset(comb.items())] = np.histogram(self.sample.subset(**comb).corr, bins=int(T_dur/dt)+1, range=(0-dt/2, T_dur+dt/2))[0]/len(self.sample.subset(**comb))/dt # dt/2 (and +1) is continuity correction
            self.hists_err[frozenset(comb.items())] = np.histogram(self.sample.subset(**comb).err, bins=int(T_dur/dt)+1, range=(0-dt/2, T_dur+dt/2))[0]/len(self.sample.subset(**comb))/dt
        self.target = np.concatenate([s for i in sorted(self.hists_corr.keys()) for s in [self.hists_corr[i], self.hists_err[i]]])
        # Do the likelihood component
        self.hist_indexes = {}
        for comb in self.sample.condition_combinations(required_conditions=self.required_conditions):
            s = self.sample.subset(**comb)
            maxt = max(max(s.corr) if s.corr.size != 0 else -1, max(s.err) if s.err.size != 0 else -1)
            assert maxt <= self.T_dur, "Simulation time T_dur=%f not long enough for these data" % self.T_dur
            # Find the integers which correspond to the timepoints in
            # the pdfs.  Also don't group them into the first bin
            # because this creates bias.
            corr = [int(round(e/dt)) for e in s.corr]
            err = [int(round(e/dt)) for e in s.err]
            undec = self.sample.undecided
            self.hist_indexes[frozenset(comb.items())] = (corr, err, undec)
    @accepts(Self, ddm.Model)
    @returns(Number)
    @requires("model.dt == self.dt and model.T_dur == self.T_dur")
    def loss(self, model):
        assert model.dt == self.dt and model.T_dur == self.T_dur
        MIN_LIKELIHOOD = 1e-8 # Avoid log(0)
        sols = self.cache_by_conditions(model)
        loglikelihood = 0
        for k in sols.keys():
            # nans come from negative values in the pdfs, which in
            # turn come from the dx parameter being set too low.  This
            # comes up when fitting, because sometimes the algorithm
            # will "explore" and look at extreme parameter values.
            # For example, this arrises when variance is very close to
            # 0.  We will issue a warning now, but throwing an
            # exception may be the better way to handle this to make
            # sure it doesn't go unnoticed.
            if np.any(sols[k].pdf_corr()<0) or np.any(sols[k].pdf_err()<0):
                print("Warning: parameter values too extreme for dx.")
                return np.inf
            loglikelihood += np.sum(np.log(sols[k].pdf_corr()[self.hist_indexes[k][0]]+MIN_LIKELIHOOD))
            loglikelihood += np.sum(np.log(sols[k].pdf_err()[self.hist_indexes[k][1]]+MIN_LIKELIHOOD))
            if sols[k].prob_undecided() > 0:
                loglikelihood += np.log(sols[k].prob_undecided())*self.hist_indexes[k][2]
        this = np.concatenate([s for i in sorted(self.hists_corr.keys()) for s in [sols[i].pdf_corr(), sols[i].pdf_err()]])
        inds = np.concatenate([(s > i['presample']/1000) & (s < i['presample']/1000 + .4) & (i['presample'] != 0) for i in sorted(self.hists_corr.keys()) for s in [sols[i].model.t_domain(), sols[i].model.t_domain()]])
        print(inds.shape, this.shape)
        squarederror = np.sum((this[inds]-self.target[inds])**2)*self.dt**2

        return -loglikelihood + squarederror*100

def get_detect_prob(coh, param):
    return 2/(1+np.exp(-param*(coh-50)/50))-1

class OverlayDipRatio(ddm.Overlay):
    name = "Add a non-decision by shifting the histogram"
    required_parameters = ["detect", "diptype"]
    required_conditions = ["coherence"]
    def apply(self, solution):
        if self.diptype not in [1, 2, 3]:
            return solution
        corr = solution.corr
        err = solution.err
        m = solution.model
        cond = solution.conditions
        undec = solution.undec
        evolution = solution.evolution
        diptype = m.get_dependence("drift").diptype
        def set_dip_type(m, diptype):
            m.get_dependence("drift").diptype = diptype
            m.get_dependence("noise").diptype = diptype
            m.get_dependence("bound").diptype = diptype
            m.get_dependence("overlay").diptype = diptype
        set_dip_type(m, -1)
        ratio = get_detect_prob(cond['coherence'], self.detect)
        s = m.solve_numerical_implicit(conditions=cond, return_evolution=True)
        newcorr = corr * ratio + s.corr * (1-ratio)
        newerr = err * ratio + s.err * (1-ratio)
        newevo = evolution
        #newevo = evolution * ratio + s.evolution * (1-ratio)
        set_dip_type(m, diptype)
        return ddm.Solution(newcorr, newerr, m, cond, undec, newevo)
    def apply_trajectory(self, trajectory, model, rk4, seed, conditions={}):
        if self.diptype not in [1, 2, 3]:
            return trajectory
        prob = get_detect_prob(conditions['coherence'], self.detect)
        # We have a `prob` probability of detecting the dip.  If we
        # detected the dip, just use the given trajectory.  Otherwise,
        # simulate a new trajectory without the dip.
        if prob > np.random.rand():
            return trajectory
        diptype = model.get_dependence("drift").diptype
        def set_dip_type(m, diptype):
            m.get_dependence("drift").diptype = diptype
            m.get_dependence("noise").diptype = diptype
            m.get_dependence("bound").diptype = diptype
            m.get_dependence("overlay").diptype = diptype
        set_dip_type(model, -1)
        traj = model.simulate_trial(conditions=conditions, rk4=rk4, seed=seed, cutoff=True)
        set_dip_type(model, diptype)
        return traj



if __name__ == "__main__":
    MONKEY = sys.argv[1]
    DIPTYPE = int(sys.argv[2])
    
    assert MONKEY in ["Q", "P"]
    assert DIPTYPE in [1, 2, 3]
    
    # = "all", = "P", = "Q"
    FIT_USING = MONKEY
    
    
    if FIT_USING == "all":
        snr = ddm.Fittable(minval=0.5, maxval=20, default=9.243318909157688)
        leak = ddm.Fittable(minval=-10, maxval=30, default=9.46411355874963)
        x0 = ddm.Fittable(minval=-.5, maxval=.5, default=0.1294632585920082)
        leaktargramp = ddm.Fittable(minval=0, maxval=3, default=0)
        noise = ddm.Fittable(minval=.2, maxval=2, default=1.1520906498077081)
        t1 = ddm.Fittable(minval=0, maxval=1, default=0.34905555600815663)
        t1slope = ddm.Fittable(minval=0, maxval=3, default=1.9643425020687162)
    elif FIT_USING == "Q":
        
        snr=8.88568084926296
        noise=1.0548720586722578
        t1=0.36901214878189553
        t1slope=1.8788206931608151
        leak=7.168279198532247
        leaktargramp=0.2727682274028674
        x0=0
        pmixturecoef=0.07855503975842383
        rate=1.1033286141789573
        # BIC=-2437.43373081
    elif FIT_USING == "Qa":
        snr=9.060170270587802
        noise=1.0179900774673056
        t1=0.371031336294422
        t1slope=1.9465275695835973
        leak=6.998307948443117
        leaktargramp=0.12623536578615527
        x0=0.08250115239323992
        #nondectime=0.22057061297057912
    elif FIT_USING == "Pa":
        snr=5.686681836611886
        noise=1.1906524311529822
        t1=0.20137458605635317
        t1slope=1.337291735355154
        leak=13.866374271121522
        leaktargramp=0.16274830388625727
        x0=0.00804935818028657
        #nondectime=0.2416453496387595
    elif FIT_USING == "P":
        snr=5.537201726060179
        noise=1.1810931488493681
        t1=0.21601619680674877
        t1slope=1.3548843836932933
        leak=13.295294909173858
        leaktargramp=0.17745115949788842
        #nondectime=0.24382714587944565
        x0=0
        pmixturecoef=0.06402822265967034
        rate=0.15051296570284622
        #BIC=7414.12478212
    
    dipstart = ddm.Fittable(minval=-.4, maxval=0, default=-.2)
    dipstop = ddm.Fittable(minval=0, maxval=.5, default=.05)
    nondectime = ddm.Fittable(minval=0, maxval=.3, default=.1)
    nd2 = 0#ddm.Fittable(minval=0.0, maxval=0.3, default=0.1)
    detect = ddm.Fittable(minval=2, maxval=50, default=10)
    diptype = DIPTYPE
    dipparam = ddm.Fittable(minval=0, maxval=50) if diptype == 2 else 0
    m = ddm.Model(drift=    DriftUrgencyGatedDip(snr=snr,
                                                 noise=noise,
                                                 t1=t1,
                                                 t1slope=t1slope,
                                                 leak=leak,
                                                 maxcoh=(70 if MONKEY == "Q" else 63),
                                                 leaktarget=x0,
                                                 leaktargramp=leaktargramp,
                                                 dipstart=dipstart,
                                                 dipstop=dipstop,
                                                 diptype=diptype,
                                                 dipparam=dipparam,
                                                 nd2=nd2
                                                 ),
                  noise=         NoiseUrgencyDip(noise=noise,
                                                 t1=t1,
                                                 t1slope=t1slope,
                                                 dipstart=dipstart,
                                                 dipstop=dipstop,
                                                 diptype=diptype,
                                                 nd2=nd2
                                                 ),
                  IC=                    ICPoint(x0=x0),
                  bound=                BoundDip(B=1,
                                                 dipstart=dipstart,
                                                 dipstop=dipstop,
                                                 diptype=diptype
                                                 ),
                  #overlay=
                  overlay=ddm.OverlayChain(overlays=[   ddm.OverlayNonDecision(nondectime=nondectime),
                                                               OverlayDipRatio(detect=detect,
                                                                               diptype=diptype),
                                                     ddm.OverlayPoissonMixture(pmixturecoef=pmixturecoef,
                                                                               rate=rate)
                                           ]),
                  dx=0.002, dt=0.002, T_dur=3.0)
    
    #import ddm.plot
    import samplefactory
    s = samplefactory.SampleFactory(animal=(1 if MONKEY == "Q" else 2), zerocoh=False, testset=False)
    ddm.set_N_cpus(6)
    
    # from ddm import *
    # from ddm.models import *
    # import ddm.plot
    
    
    #m = ddm.plot.model_gui(sample=s, model=m)#, conditions={"highreward": [0, 1], "coherence": [53, 60, 70], "presample": [0, 400, 800]})
    
    #import matplotlib.pyplot as plt
    #plt.plot(m.simulate_trial(conditions={"coherence": 63, "presample": 800, "highreward": 0}))
    #plt.show()
    
    
    #ddm.plot.model_gui(sample=s, model=m)#, conditions={"highreward": [0, 1], "coherence": [53, 60, 70], "presample": [0, 400, 800]})
    
    ddm.functions.fit_adjust_model(model=m, sample=s,lossfunction=ddm.LossBIC)
    
    print("========== Final model ==========")
    print(f"Monkey {MONKEY} fit dip {DIPTYPE} using {FIT_USING}")
    print(m)
    with open(f"dip{DIPTYPE}-monkey{MONKEY}-fit{FIT_USING}.txt", "w") as f:
        f.write(repr(m))

