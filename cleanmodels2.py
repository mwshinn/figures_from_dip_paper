#from paranoid.ignore import *
from paranoid.decorators import accepts, returns, requires, ensures, paranoidclass
from paranoid.types import RangeOpenClosed, RangeOpen, Range, Positive0, NDArray, ParametersDict, Natural0, Set, Self, Number, Positive
import ddm
import numpy as np
import scipy

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
@accepts(Range(50, 100), RangeOpen(0, 10), RangeOpenClosed(50, 100))
@requires("coh <= max_coh")
@returns(Range(0, 1))
@ensures("return == 0 <--> coh == 50")
@ensures("return == 1 <--> coh == max_coh")
# Monotonic increasing in coh, decreasing in exponent
@ensures("coh >= coh` and exponent <= exponent` and max_coh == max_coh` --> return >= return`")
def coh_transform(coh, exponent, max_coh):
    coh_coef = (coh-50)/(max_coh-50)
    return coh_coef**exponent

@accepts(Positive0, Positive0, Positive0, Positive0)
@returns(Positive0)
@ensures("return >= 0")
# Monotonic in all variables, decreasing with t1, others increasing
@ensures("t >= t` and base >= base` and t1 <= t1` and slope >= slope` --> return >= return`")
def urgency(t, base, t1, slope):
    return base + ((t-t1)*slope if t>=t1 else 0)

@accepts(Positive0, Positive0, Positive0, Positive0)
@returns(Positive0)
@ensures("boost == boost` and return != 0 and return` != 0 --> return == return`")
@ensures("return != 0 --> return == boost/.1")
@ensures("boost == 0 --> return == 0")
@ensures("abs(t - t`) > .1 and return != 0 and presample == presample` and delay == delay` --> return` == 0")
def boost_signal(t, boost, presample, delay):
    duration = .1
    if presample == 0 or (not 0 < t - presample - delay < duration):
        return 0
    return boost/duration

@paranoidclass
class DriftUrgencyGated(ddm.models.Drift):
    name = "Drift with piecewise linear urgency signal, reward/timing interaction bias, and coherence change transient"
    required_parameters = ["snr", "noise", "t1", "t1slope", "cohexp", "maxcoh", "leak", "leaktarget", "leaktargramp"]
    required_conditions = ["coherence", "presample", "highreward"]
    default_parameters = {"leaktargramp": 0}

    def get_drift(self, t, x, conditions, **kwargs):
        # Coherence coefficient == coherence with a non-linear transform
        coh_coef = coh_transform(conditions["coherence"], self.cohexp, self.maxcoh)
        is_past_delay = 1 if t > conditions["presample"]/1000 else 0
        cur_urgency = self.snr * urgency(t, self.noise, self.t1, self.t1slope)
        leaktarg = self.leaktarget if conditions["highreward"] else -self.leaktarget
        leaktargramp = self.leaktargramp if conditions["highreward"] else -self.leaktargramp
        return coh_coef * (cur_urgency * is_past_delay) - self.leak*(x-(leaktarg+leaktargramp*t))
    @staticmethod
    def _test(v):
        assert v.snr in Positive(), "Invalid SNR"
        assert v.noise in Positive0(), "Invalid noise"
        assert v.t1 in Positive0(), "Invalid t1"
        assert v.t1slope in Positive0(), "Invalid t1slope"
        assert v.cohexp in Positive0(), "Invalid cohexp"
        assert v.maxcoh in [63, 70], "Invalid maxcoh"
        assert v.leak in Positive0(), "Invalid leak"
        assert v.leaktarget in Number(), "Invalid leak"
        assert v.leaktargramp in Number(), "Invalid leak"
    @staticmethod
    def _generate():
        yield DriftUrgencyGated(noise=1, snr=.2, t1=2, t1slope=2, delay=.3, cohexp=1, boost=.3, boosttime=.1, driftrew=.3, maxcoh=70, scaledrift=0)
        yield DriftUrgencyGated(noise=.2, snr=2, t1=.8, t1slope=1, delay=.2, cohexp=1.8, boost=3, boosttime=.1, driftrew=.8, maxcoh=63, scaledrift=1)


@paranoidclass
class NoiseUrgency(ddm.models.Noise):
    name = "Noise with piecewise linear urgency signal"
    required_parameters = ["noise", "t1", "t1slope"]

    @accepts(Self, t=Positive0, conditions=Conditions)
    @returns(Positive)
    def get_noise(self, t, conditions, **kwargs):
        return urgency(t, self.noise, self.t1, self.t1slope) + .001

    @staticmethod
    def _test(v):
        assert v.noise in Positive0(), "Invalid noise"
        assert v.t1 in Positive0(), "Invalid t1"
        assert v.t1slope in Positive0(), "Invalid t1slope"
    @staticmethod
    def _generate():
        yield NoiseUrgency(noise=1, t1=2, t1slope=2)
        yield NoiseUrgency(noise=.4, t1=.7, t1slope=1.01)

@paranoidclass
class OverlaySaccadeInhibition(ddm.models.Overlay):
    name = "After integrating, switch targets with some probability"
    required_parameters = ["pausemixturecoef", "pausenondecision", "pausedur", "sicohexp", "simaxcoh"]
    required_conditions = ["coherence", "presample"]
    @accepts(Self, ddm.Solution)
    @returns(ddm.Solution)
    def apply(self, solution):
        coh_coef = coh_transform(solution.conditions["coherence"], self.sicohexp, self.simaxcoh)
        mappingcoefcoh = coh_coef * self.pausemixturecoef

        dt = solution.model.dt
        pauseend = solution.conditions["presample"]/1000 + self.pausenondecision
        splitpoint1 = np.where(solution.model.t_domain() > pauseend - self.pausedur)[0][0]
        splitpoint2 = np.where(solution.model.t_domain() > pauseend)[0][0]
        firstpart_corr = solution.corr[0:splitpoint1]
        firstpart_err = solution.err[0:splitpoint1]
        middlepart_corr = solution.corr[splitpoint1:splitpoint2]
        middlepart_err = solution.err[splitpoint1:splitpoint2]
        lastpart_corr = solution.corr[splitpoint2:]
        lastpart_err = solution.err[splitpoint2:]

        middle_mass = np.sum(middlepart_corr + middlepart_err)
        last_mass = np.sum(lastpart_corr + lastpart_err)
        if middle_mass < 1e-10:
            middle_mass = 0
        if last_mass < 1e-5:
            last_mass = 1e-5
        corr = np.hstack([firstpart_corr,
                          middlepart_corr * (1 - mappingcoefcoh),
                          lastpart_corr * (1 + mappingcoefcoh * middle_mass / last_mass)])
        err = np.hstack([firstpart_err,
                         middlepart_err * (1 - mappingcoefcoh),
                         lastpart_err * (1 + mappingcoefcoh * middle_mass / last_mass)])
        return ddm.Solution(corr, err, solution.model, solution.conditions, pdf_undec=solution.undec)


@paranoidclass
class OverlayMappingError(ddm.models.Overlay):
    name = "After integrating, switch targets with some probability"
    required_parameters = ["mappingcoef"]
    required_conditions = ["highreward"]
    @accepts(Self, ddm.Solution)
    @returns(ddm.Solution)
    @ensures("return.conditions['highreward'] == 1 --> all(return.corr >= solution.corr)")
    @ensures("return.conditions['highreward'] == 0 --> all(return.corr <= solution.corr)")
    def apply(self, solution):
        if solution.conditions['highreward'] == 1:
            mismapped = self.mappingcoef * solution.err
            err = solution.err - mismapped
            corr = solution.corr + mismapped
            return ddm.Solution(corr, err, solution.model, solution.conditions)
        else:
            mismapped = self.mappingcoef * solution.corr
            corr = solution.corr - mismapped
            err = solution.err + mismapped
            return ddm.Solution(corr, err, solution.model, solution.conditions)

    @staticmethod
    def _test(v):
        assert v.mappingcoef in Range(0, 1), "Invalid mapping coefficient"
             
    @staticmethod
    def _generate():
        yield OverlayMappingError(mappingcoef=.4)
        yield OverlayMappingError(mappingcoef=0)
        yield OverlayMappingError(mappingcoef=1)
        yield OverlayMappingError(mappingcoef=.1)

@paranoidclass
class OverlayOnsetError(ddm.models.Overlay):
    name = "Choose high reward target automatically when coherence changes with some probability"
    # peak = scale*shape (mean of distribution), so scale = peak/shape
    # Also shifted by "meandelay" so that the mean occurs at `meandelay` after the presample
    # Shape should never be less than 1, and peak should generally never be less than .05 for numerical stability.
    required_parameters = ["onsetmixturecoef", "peak", "shape", "meandelay", "cohexp", "maxcoh", "delaypsadjust"]
    required_conditions = ["highreward", "presample", "coherence"]
    default_parameters = {"delaypsadjust": 0}
    @accepts(Self, ddm.Solution)
    @returns(ddm.Solution)
    def apply(self, solution):
        fullmeandelay = self.meandelay + solution.conditions["presample"]/1000 * self.delaypsadjust
        coh_coef = coh_transform(solution.conditions["coherence"], self.cohexp, self.maxcoh)
        mappingcoefcoh = coh_coef * self.onsetmixturecoef

        dt = solution.model.dt
        X = [i*dt - solution.conditions["presample"]/1000 - (fullmeandelay - self.peak) for i in range(0, len(solution.corr))]
        Y = np.asarray([self.gamma_pdf(x) if x > 0 else 0 for x in X])*dt
        sumY = np.sum(Y)
        if sumY > 1:
            print("Warning, renormalizing gamma from %f to 1" % sumY)
            Y /= sumY
        
        if solution.conditions["highreward"] == 1:
            corr = solution.corr * (1-mappingcoefcoh) + mappingcoefcoh*Y
            err = solution.err * (1-mappingcoefcoh)
        else:
            err = solution.err * (1-mappingcoefcoh) + mappingcoefcoh*Y
            corr = solution.corr * (1-mappingcoefcoh)
        
        return ddm.Solution(corr, err, solution.model, solution.conditions)

    @accepts(Self, Positive0())
    @returns(Positive0())
    def gamma_pdf(self, t):
        return 1/(scipy.special.gamma(self.shape)*((self.peak/self.shape)**self.shape)) * t**(self.shape-1) * np.exp(-t/(self.peak/self.shape))

    @staticmethod
    def _test(v):
        assert v.onsetmixturecoef in Range(0, 1), "Invalid onset mixture coefficient"
        assert v.peak in Positive0(), "Invalid peak"
        assert v.shape in Positive0(), "Invalid shape"
        assert v.meandelay in Positive0(), "Invalid meandelay"
        assert v.cohexp in Positive0(), "Invalid cohexp"

    @staticmethod
    def _generate():
        yield OverlayOnsetError(onsetmixturecoef=.4, peak=1, shape=1, meandelay=.2, cohexp=1, maxcoh=80)

# TODO URGENT this fails when peak is low or when mixture coefficient is 1
# (but not .99).  More likely to fail in high coherence case.  Also,
# onsetmixturecoef still scales early DDM stuff...
@paranoidclass
class OverlayOnsetOnlyError(ddm.models.Overlay):
    name = "Choose high reward target automatically when coherence changes with some probability"
    # peak = scale*shape (mean of distribution), so scale = peak/shape
    # Also shifted by "meandelay" so that the mean occurs at `meandelay` after the presample
    # Shape should never be less than 1, and peak should generally never be less than .05 for numerical stability.
    required_parameters = ["onsetmixturecoef", "peak", "shape", "meandelay", "cohexp", "maxcoh"]
    required_conditions = ["highreward", "presample", "coherence"]
    @accepts(Self, ddm.Solution)
    @returns(ddm.Solution)
    def apply(self, solution):
        coh_coef = coh_transform(solution.conditions["coherence"], self.cohexp, self.maxcoh)
        mappingcoefcoh = coh_coef * self.onsetmixturecoef

        dt = solution.model.dt
        X = [i*dt - solution.conditions["presample"]/1000 - (self.meandelay - self.peak) for i in range(0, len(solution.corr))]
        splitpoint = next(i for i,x in enumerate(X) if x > 0)
        firsthalf_corr = solution.corr[0:splitpoint]
        firsthalf_err = solution.err[0:splitpoint]
        weight = np.sum(firsthalf_corr) + np.sum(firsthalf_err)
        assert X[splitpoint] > 0 and (splitpoint == 0 or X[splitpoint-1] <= 0), "Invalid split"
        
        gammadist = np.asarray([self.gamma_pdf(x) for i,x in enumerate(X) if i >= splitpoint])*dt
        sum_gammadist = np.sum(gammadist)

        if sum_gammadist > 1:
            print("Warning, renormalizing gamma from %f to 1" % sum_gammadist)
            gammadist /= sum_gammadist
        highreward_detect_corr = np.hstack([firsthalf_corr, gammadist * (1-weight)])
        highreward_detect_err = np.hstack([firsthalf_err, 0*gammadist])
        lowreward_detect_corr = np.hstack([firsthalf_corr, 0*gammadist])
        lowreward_detect_err = np.hstack([firsthalf_err, gammadist * (1-weight)])

        if solution.conditions["highreward"] == 1:
            corr = solution.corr * (1-mappingcoefcoh) + mappingcoefcoh*highreward_detect_corr
            err = solution.err * (1-mappingcoefcoh) + mappingcoefcoh*highreward_detect_err
        else:
            err = solution.err * (1-mappingcoefcoh) + mappingcoefcoh*lowreward_detect_err
            corr = solution.corr * (1-mappingcoefcoh) + mappingcoefcoh*lowreward_detect_corr

        return ddm.Solution(corr, err, solution.model, solution.conditions)

    @accepts(Self, Positive0())
    @returns(Positive0())
    def gamma_pdf(self, t):
        return 1/(scipy.special.gamma(self.shape)*((self.peak/self.shape)**self.shape)) * t**(self.shape-1) * np.exp(-t/(self.peak/self.shape))

    @staticmethod
    def _test(v):
        assert v.onsetmixturecoef in Range(0, 1), "Invalid onset mixture coefficient"
        assert v.peak in Positive0(), "Invalid peak"
        assert v.shape in Positive0(), "Invalid shape"
        assert v.meandelay in Positive0(), "Invalid meandelay"
        assert v.cohexp in Positive0(), "Invalid cohexp"

    @staticmethod
    def _generate():
        yield OverlayOnsetOnlyError(onsetmixturecoef=.4, peak=1, shape=1, meandelay=.2, cohexp=1, maxcoh=80)


@paranoidclass
class OverlayDetectDiffusion(ddm.models.Overlay):
    name = "If a change is detected, initiate a second diffusion process (OU)"
    required_parameters = ["d2snr", "d2noise", "d2x0", "d2cohexp", "d2oux", "d2delay1", "d2delay2", "d2delay3", "d2switchdelay", "detect1", "detect2", "detect3", "detect4", "detect5", "detect6", "detect7", "detect8", "detect9", "d2maxcoh", "reset"]
    required_conditions = ["highreward", "presample", "coherence"]
    default_parameters = {"reset": 1}
    @staticmethod
    def _test(v):
        # TODO finish
        for p in ("detect"+str(i) for i in range(1, 10)):
            assert getattr(v, p) in Range(0, 1)
    def _generate():
        yield OverlayDetectDiffusion(d2snr=1, d2noise=1, d2x0=.1, d2cohexp=.8, d2oux=.2, d2delay1=.05, d2delay2=.1, d2delay3=.15, d2switchdelay=.1, detect1=0, detect2=.01, detect3=.1, detect4=.1, detect5=.5, detect6=.9, detect7=.2, detect8=.9, detect9=.95, d2maxcoh=70)
    @accepts(Self, ddm.Solution)
    @returns(ddm.Solution)
    def apply(self, solution):
        # Split the distribution into two halves: the first half will
        # be unmodified, but the second half will have some
        # probability of being captured by solution and some
        # probability of being replaced with a new diffusion process.
        delay = self.d2delay1 if solution.conditions["presample"] == 800 else self.d2delay2 if solution.conditions["presample"] == 400 else self.d2delay3
        splittime = solution.conditions["presample"]/1000 + delay - self.d2switchdelay
        t_domain = solution.model.t_domain()
        split_i = np.where(t_domain>splittime)[0][0] # Index for the split
        first_half_corr = solution.corr[:split_i]
        first_half_err = solution.err[:split_i]
        
        # Construct a histogram for the trials in which the monkey was
        # able to detect the change.  First, add a delay for detection
        # and changing objectives (i.e. rapid vs slow integration).
        # Then simulate the new faster diffusion across time points
        # startin with the delay.
        # start_i = np.where(t_domain > splittime + self.d2switchdelay)[0][0] # Index of first timepoint after delay
        # timepoints = t_domain[start_i:] - t_domain[start_i]
        # d2 = ddm.Model(drift=ddm.models.DriftLinear(t=0, drift=self.d2noise*self.d2snr*coh_transform(solution.conditions['coherence'], self.d2cohexp, 70), x=self.d2oux),
        #                noise=ddm.NoiseConstant(noise=self.d2noise),
        #                IC=ICPoint(x0=self.d2x0),
        #                dt=solution.model.dt, dx=solution.model.dx, T_dur=max(timepoints))
        # d2s = d2.solve(conditions={"highreward": solution.conditions["highreward"]})
        # diffused = (d2s.corr/d2.dt, d2s.err/d2.dt)
        # second_half_corr = np.hstack([t_domain[split_i:start_i]*0, diffused[0] * solution.model.dt])
        # second_half_err = np.hstack([t_domain[split_i:start_i]*0, diffused[1] * solution.model.dt])

        # Alternative construction
        starttime = splittime + self.d2switchdelay
        if self.reset == 1:
            icdist = ICPoint(x0=self.d2x0)
        elif self.reset == 0:
            prev_T_dur = solution.model.T_dur
            prev_overlay = solution.model._overlay
            solution.model._overlay = ddm.models.OverlayNone()
            solution.model.dependencies[4] = solution.model._overlay
            solution.model.T_dur = starttime
            icdist_pdf = solution.model.solve(conditions=solution.conditions).pdf_undec()
            solution.model.T_dur = prev_T_dur
            solution.model._overlay = prev_overlay
            solution.model.dependencies[4] = prev_overlay
            icdist = ddm.models.ICArbitrary(icdist_pdf/np.sum(icdist_pdf))
        coh_transformed = coh_transform(solution.conditions['coherence'], self.d2cohexp, self.d2maxcoh)
        if self.d2oux == 0:
            driftmod = ddm.models.drift.DriftConstant(drift=self.d2noise*self.d2snr*coh_transformed)
        else:
            driftmod = ddm.models.drift.DriftLinear(t=0, drift=self.d2noise*self.d2snr*coh_transformed, x=self.d2oux)
        d2 = ddm.Model(drift=driftmod,
                       noise=ddm.models.noise.NoiseConstant(noise=float(self.d2noise)), # Cast as float, otherwise np.isnumeric() doesn't work on grace
                       IC=icdist,
                       dt=solution.model.dt, dx=solution.model.dx, T_dur=solution.model.T_dur)
        d2s = d2.solve(conditions={"highreward": solution.conditions["highreward"]})
        start_i = np.where(t_domain > starttime)[0][0] - 1 # Index of first timepoint before the delay
        offset = starttime - t_domain[start_i]
        n_timepoints = len(t_domain[start_i:])
        assert offset >= 0

        s_corr_interp = np.interp(d2.t_domain()[0:n_timepoints], offset+d2.t_domain(), d2s.corr)
        s_err_interp = np.interp(d2.t_domain()[0:n_timepoints], offset+d2.t_domain(), d2s.err)
        # Normalize if numerical errors in interpolation make it not integrate to 1
        histsum = np.sum(s_corr_interp) + np.sum(s_err_interp)
        if histsum > 1:
            print("Warning, renormalizing interpolated histogram from %f to 1" % histsum)
            s_corr_interp /= histsum
            s_err_interp /= histsum
        second_half_corr = np.hstack([t_domain[split_i:start_i]*0, s_corr_interp])
        second_half_err = np.hstack([t_domain[split_i:start_i]*0, s_err_interp])
        
        # To combine the first and second halves, we need to make sure
        # they sum to 1.  Thus, find the sum of the first half and
        # adjust the scaling of the second half accordingly.
        first_half_ratio = (np.sum(first_half_corr) + np.sum(first_half_err))
        detect_corr = np.hstack([first_half_corr, second_half_corr * (1-first_half_ratio)])
        detect_err = np.hstack([first_half_err, second_half_err * (1-first_half_ratio)])

        # Join the two solutions
        cond = (solution.conditions["presample"], solution.conditions["coherence"])
        pdetect = self.detect1 if cond in [(0, 53), (0, 52)] else \
                  self.detect2 if cond in [(0, 60), (0, 57)] else \
                  self.detect3 if cond in [(0, 70), (0, 63)] else \
                  self.detect4 if cond in [(400, 53), (400, 52)] else \
                  self.detect5 if cond in [(400, 60), (400, 57)] else \
                  self.detect6 if cond in [(400, 70), (400, 63)] else \
                  self.detect7 if cond in [(800, 53), (800, 52)] else \
                  self.detect8 if cond in [(800, 60), (800, 57)] else \
                  self.detect9 if cond in [(800, 70), (800, 63)] else \
                  0 if cond[1] == 50 else \
                  np.nan

        corr = detect_corr*pdetect + solution.corr*(1-pdetect)
        err = detect_err*pdetect + solution.err*(1-pdetect)
        assert len(corr) == len(solution.corr)
        assert len(err) == len(solution.err)

        return ddm.Solution(corr, err, solution.model, solution.conditions)

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

# FOR COLLAPSING BOUNDS

@paranoidclass
class BoundCollapsingExponentialDelay(ddm.models.Bound):
    """Bound dependence: bound collapses exponentially over time.

    Takes three parameters: 

    `B` - the bound at time t = 0.
    `tau` - the time constant for the collapse, should be greater than
    zero.
    `t1` - the time at which the collapse begins in seconds
    """
    name = "collapsing_exponential"
    required_parameters = ["B", "tau", "t1"]

    @accepts(Self, Positive0, Conditions)
    @returns(Positive)
    @ensures("t > self.t1 --> return < self.B")
    @ensures("t <= self.t1 --> return == self.B")
    @ensures("self == self` and t > self.t1 and t > t` --> return < return`")
    def get_bound(self, t, conditions, **kwargs):
        if t <= self.t1:
            return self.B
        if t > self.t1:
            return self.B * np.exp(-self.tau*(t-self.t1))

    @staticmethod
    def _test(v):
        assert v.B in Positive(), "Invalid bound"
        assert v.tau in Positive(), "Invalid collapsing time constant"
        assert v.t1 in Positive0(), "Invalid t1"

    @staticmethod
    def _generate():
        yield BoundCollapsingExponentialDelay(B=2, tau=1, t1=.5)
        yield BoundCollapsingExponentialDelay(B=.5, tau=.1, t1=.1)
        yield BoundCollapsingExponentialDelay(B=1, tau=4, t1=1.1)

@paranoidclass
class OverlayPoissonRewardMixture(ddm.Overlay):
    name = "Poisson distribution mixture model (lapse rate)"
    required_parameters = ["pmixturecoef", "ratehr", "ratelr"]
    required_conditions = ["highreward"]
    @accepts(Self, ddm.Solution)
    @returns(ddm.Solution)
    def apply(self, solution):
        assert self.pmixturecoef >= 0 and self.pmixturecoef <= 1
        corr, err, m, cond = solution.corr, solution.err, solution.model, solution.conditions
        # These aren't real pdfs since they don't sum to 1, they sum
        # to 1/self.model.dt.  We can't just sum the correct and error
        # distributions to find this number because that would exclude
        # the non-decision trials.
        pdfsum = 1/m.dt
        # Pr = lambda ru, rr, P, t : (rr*P)/((rr+ru))*(1-numpy.exp(-1*(rr+ru)*t))
        # P0 = lambda ru, rr, P, t : P*numpy.exp(-(rr+ru)*t) # Nondecision
        # Pr' = lambda ru, rr, P, t : (rr*P)*numpy.exp(-1*(rr+ru)*t)
        # lapses_cdf = lambda t : 1-np.exp(-(2*self.rate)*t)
        lapses_hr = lambda t : self.ratehr*np.exp(-(self.ratehr+self.ratelr)*t) if t != 0 else 0
        lapses_lr = lambda t : self.ratelr*np.exp(-(self.ratehr+self.ratelr)*t) if t != 0 else 0
        X = [i*m.dt for i in range(0, len(corr))]
        if cond['highreward'] == 1:
            Y_corr = np.asarray(list(map(lapses_hr, X)))*m.dt
            Y_err = np.asarray(list(map(lapses_lr, X)))*m.dt
        else:
            Y_corr = np.asarray(list(map(lapses_lr, X)))*m.dt
            Y_err = np.asarray(list(map(lapses_hr, X)))*m.dt
        corr = corr*(1-self.pmixturecoef) + self.pmixturecoef*Y_corr # Assume numpy ndarrays, not lists
        err = err*(1-self.pmixturecoef) + self.pmixturecoef*Y_err
        #print(corr)
        #print(err)
        return ddm.Solution(corr, err, m, cond)
    @staticmethod
    def _test(v):
        assert v.pmixturecoef in Range(0, 1), "Invalid pmixture coef"
        assert v.ratehr in Positive(), "Invalid rate"
        assert v.ratelr in Positive(), "Invalid rate"
    @staticmethod
    def _generate():
        yield OverlayPoissonMixture(pmixturecoef=0, ratehr=1, ratelr=.3)
        yield OverlayPoissonMixture(pmixturecoef=.5, ratehr=.1, ratelr=1)
        yield OverlayPoissonMixture(pmixturecoef=.02, ratehr=10, ratelr=4)
        yield OverlayPoissonMixture(pmixturecoef=1, ratehr=1, ratelr=.01)


# FOR DATASET 2

# @paranoidclass
# class DriftUrgencyBoostJoint(ddm.Drift):
#     name = "Drift for a joint fit with piecewise linear urgency signal and coherence change transient"
#     required_parameters = ["basegain_a", "t1_a", "t1slope_a", "delay_a", "cohexp_a", "boost_a", "boosttime_a",
#                            "basegain_b", "t1_b", "t1slope_b", "delay_b", "cohexp_b", "boost_b", "boosttime_b", "maxcoh"]
#     required_conditions = ["coherence", "presample", "blocktype"]

#     @accepts(Self, t=Positive0, conditions=Conditions)
#     @returns(Positive0)
#     @ensures("conditions['blocktype'] == 1 and t < conditions['presample']/1000 + min(self.delay_a, self.boosttime_a) --> return == 0")
#     @ensures("conditions['blocktype'] == 2 and t < conditions['presample']/1000 + min(self.delay_b, self.boosttime_b) --> return == 0")
#     def get_drift(self, t, conditions, **kwargs):
#         if conditions["blocktype"] == 1:
#             # Coherence coefficient == coherence with a non-linear transform
#             coh_coef = coh_transform(conditions["coherence"], self.cohexp_a, self.maxcoh)
#             is_past_delay = 1 if t-self.delay_a > conditions["presample"]/1000 else 0
#             cur_urgency = urgency(t, self.basegain_a, self.t1_a, self.t1slope_a)
#             cur_boost = boost_signal(t, self.boost_a, conditions["presample"]/1000, self.boosttime_a)
#         elif conditions["blocktype"] == 2:
#             # Coherence coefficient == coherence with a non-linear transform
#             coh_coef = coh_transform(conditions["coherence"], self.cohexp_b, self.maxcoh)
#             is_past_delay = 1 if t-self.delay_b > conditions["presample"]/1000 else 0
#             cur_urgency = urgency(t, self.basegain_b, self.t1_b, self.t1slope_b)
#             cur_boost = boost_signal(t, self.boost_b, conditions["presample"]/1000, self.boosttime_b)
#         else:
#             raise ValueError("Invalid blocktype")
#         return coh_coef * (cur_urgency * is_past_delay + cur_boost)

#     @staticmethod
#     def _test(v):
#         assert v.basegain_a in Positive0(), "Invalid basegain_a"
#         assert v.t1_a in Positive0(), "Invalid t1_a"
#         assert v.t1slope_a in Positive0(), "Invalid t1slope_a"
#         assert v.delay_a in Positive0(), "Invalid delay_a"
#         assert v.cohexp_a in Positive0(), "Invalid cohexp_a"
#         assert v.basegain_b in Positive0(), "Invalid basegain_b"
#         assert v.t1_b in Positive0(), "Invalid t1_b"
#         assert v.t1slope_b in Positive0(), "Invalid t1slope_b"
#         assert v.delay_b in Positive0(), "Invalid delay_b"
#         assert v.cohexp_b in Positive0(), "Invalid cohexp_b"
#     @staticmethod
#     def _generate():
#         yield DriftUrgencyBoostJoint(basegain_a=1, t1_a=2, t1slope_a=2, delay_a=.3, cohexp_a=1, boost_a=.3, boosttime_a=.1, basegain_b=1, t1_b=2, t1slope_b=2, delay_b=.3, cohexp_b=1, boost_b=.3, boosttime_b=.1, maxcoh=80)
#         yield DriftUrgencyBoostJoint(basegain_a=.3, t1_a=.3, t1slope_a=.1, delay_a=.5, cohexp_a=1.3, boost_a=1, boosttime_a=.6, basegain_b=1.4, t1_b=1.1, t1slope_b=.2, delay_b=0, cohexp_b=.3, boost_b=.1, boosttime_b=.4, maxcoh=65)

# @paranoidclass
# class NoiseUrgencyJoint(ddm.Noise):
#     name = "Noise for joint fitting with piecewise linear urgency signal"
#     required_parameters = ["basegain_a", "t1_a", "t1slope_a", "noisescale_a", "basegain_b", "t1_b", "t1slope_b", "noisescale_b"]
#     required_conditions = ["blocktype"]
    
#     @accepts(Self, t=Positive0, conditions=Conditions)
#     @returns(Positive)
#     def get_noise(self, t, conditions, **kwargs):
#         if conditions["blocktype"] == 1:
#             return self.noisescale_a * urgency(t, self.basegain_a, self.t1_a, self.t1slope_a) + .001
#         elif conditions["blocktype"] == 2:
#             return self.noisescale_b * urgency(t, self.basegain_b, self.t1_b, self.t1slope_b) + .001
#         else:
#             raise ValueError("Invalid blocktype")
            

#     @staticmethod
#     def _test(v):
#         assert v.t1_a in Positive0(), "Invalid t1"
#         assert v.noisescale_a in Positive0(), "Invalid noisescale"
#         assert v.t1slope_a in Positive0(), "Invalid t1slope"
#         assert v.basegain_a in Positive0(), "Invalid basegain"
#         assert v.t1_b in Positive0(), "Invalid t1"
#         assert v.noisescale_b in Positive0(), "Invalid noisescale"
#         assert v.t1slope_b in Positive0(), "Invalid t1slope"
#         assert v.basegain_b in Positive0(), "Invalid basegain"
#     @staticmethod
#     def _generate():
#         yield NoiseUrgencyJoint(basegain_a=1, t1_a=2, t1slope_a=2, noisescale_a=.2, basegain_b=1, t1_b=2, t1slope_b=2, noisescale_b=.2)
#         yield NoiseUrgencyJoint(basegain_a=.3, t1_a=1, t1slope_a=.2, noisescale_a=.4, basegain_b=1.3, t1_b=1.1, t1slope_b=10, noisescale_b=.1)

class DriftUrgencyGatedJoint(ddm.models.Drift):
    name = "Drift with piecewise linear urgency signal, reward/timing interaction bias, and coherence change transient"
    required_parameters = ["snr_a", "noise_a", "t1_a", "t1slope_a", "cohexp_a", "leak_a", "leaktarget_a",
                           "snr_b", "noise_b", "t1_b", "t1slope_b", "cohexp_b", "leak_b", "leaktarget_b", "maxcoh"]
    required_conditions = ["coherence", "presample", "blocktype"]
    def get_drift(self, t, x, conditions, **kwargs):
        if conditions['blocktype'] == 1:
            leak = self.leak_a
            leaktarget = self.leaktarget_a
            cohexp = self.cohexp_a
            t1slope = self.t1slope_a
            t1 = self.t1_a
            noise = self.noise_a
            snr = self.snr_a
        elif conditions['blocktype'] == 2:
            leak = self.leak_b
            leaktarget = self.leaktarget_b
            cohexp = self.cohexp_b
            t1slope = self.t1slope_b
            t1 = self.t1_b
            noise = self.noise_b
            snr = self.snr_b
        # Coherence coefficient == coherence with a non-linear transform
        coh_coef = coh_transform(conditions["coherence"], cohexp, self.maxcoh)
        is_past_delay = 1 if t > conditions["presample"]/1000 else 0
        cur_urgency = snr * urgency(t, noise, t1, t1slope)
        return coh_coef * (cur_urgency * is_past_delay) - leak*(x-leaktarget)
    @staticmethod
    def _test(v):
        assert v.snr_a in Positive(), "Invalid SNR a"
        assert v.snr_b in Positive(), "Invalid SNR b"
        assert v.noise_a in Positive0(), "Invalid noise a"
        assert v.noise_b in Positive0(), "Invalid noise b"
        assert v.t1_a in Positive0(), "Invalid t1 a"
        assert v.t1_b in Positive0(), "Invalid t1 b"
        assert v.t1slope_a in Positive0(), "Invalid t1slope a"
        assert v.t1slope_b in Positive0(), "Invalid t1slope b"
        assert v.cohexp_a in Positive0(), "Invalid cohexp a"
        assert v.cohexp_b in Positive0(), "Invalid cohexp b"
        assert v.maxcoh_a in [63, 70], "Invalid maxcoh a"
        assert v.maxcoh_b in [63, 70], "Invalid maxcoh b"
        assert v.leak_a in Positive0(), "Invalid leak a"
        assert v.leak_b in Positive0(), "Invalid leak b"
        assert v.leaktarget_a in Number(), "Invalid leaktarget a"
        assert v.leaktarget_b in Number(), "Invalid leaktarget b"

@paranoidclass
class NoiseUrgencyJoint(ddm.models.Noise):
    name = "Noise with piecewise linear urgency signal"
    required_parameters = ["noise_a", "t1_a", "t1slope_a",
                           "noise_b", "t1_b", "t1slope_b"]
    @accepts(Self, t=Positive0, conditions=Conditions)
    @returns(Positive)
    def get_noise(self, t, conditions, **kwargs):
        if conditions["blocktype"] == 1:
            t1 = self.t1_a
            t1slope = self.t1slope_a
            noise = self.noise_a
        elif conditions["blocktype"] == 2:
            t1 = self.t1_b
            t1slope = self.t1slope_b
            noise = self.noise_b
        return urgency(t, noise, t1, t1slope) + .001

    @staticmethod
    def _test(v):
        assert v.noise_a in Positive0(), "Invalid noise a"
        assert v.t1_a in Positive0(), "Invalid t1 a"
        assert v.t1slope_a in Positive0(), "Invalid t1slope a"
        assert v.noise_b in Positive0(), "Invalid noise b"
        assert v.t1_b in Positive0(), "Invalid t1 b"
        assert v.t1slope_b in Positive0(), "Invalid t1slope b"

@paranoidclass
class OverlayPoissonMixtureJoint(ddm.models.Overlay):
    """An exponential mixture distribution.

    The output distribution should be pmixturecoef*100 percent exponential
    distribution and (1-umixturecoef)*100 percent of the distribution
    to which this overlay is applied.

    A mixture with the exponential distribution can be used to confer
    robustness when fitting using likelihood.

    Note that this is called OverlayPoissonMixture and not
    OverlayExponentialMixture because the exponential distribution is
    formed from a Poisson process, i.e. modeling a uniform lapse rate.

    Example usage:

      | overlay = OverlayPoissonMixture(pmixturecoef=.02, rate=1)
    """
    name = "Poisson distribution mixture model (lapse rate)"
    required_parameters = ["pmixturecoef_a", "rate_a", "pmixturecoef_b", "rate_b"]
    required_conditions = ["blocktype"]
    @staticmethod
    def _test(v):
        assert v.pmixturecoef_a in Range(0, 1), "Invalid mixture coef a"
        assert v.rate_a in Positive(), "Invalid rate a"
        assert v.pmixturecoef_b in Range(0, 1), "Invalid mixture coef b"
        assert v.rate_b in Positive(), "Invalid rate b"
    @accepts(Self, ddm.Solution)
    @returns(ddm.Solution)
    def apply(self, solution):
        if solution.conditions["blocktype"] == 1:
            pmixturecoef = self.pmixturecoef_a
            rate = self.rate_a
        elif solution.conditions["blocktype"] == 2:
            pmixturecoef = self.pmixturecoef_b
            rate = self.rate_b
        assert pmixturecoef >= 0 and pmixturecoef <= 1
        assert isinstance(solution, ddm.Solution)
        corr = solution.corr
        err = solution.err
        m = solution.model
        cond = solution.conditions
        undec = solution.undec
        # To make this work with undecided probability, we need to
        # normalize by the sum of the decided density.  That way, this
        # function will never touch the undecided pieces.
        norm = np.sum(corr)+np.sum(err)
        lapses = lambda t : 2*rate*np.exp(-1*rate*t)
        X = [i*m.dt for i in range(0, len(corr))]
        Y = np.asarray(list(map(lapses, X)))/len(X)
        Y /= np.sum(Y)
        corr = corr*(1-pmixturecoef) + .5*pmixturecoef*Y*norm # Assume numpy ndarrays, not lists
        err = err*(1-pmixturecoef) + .5*pmixturecoef*Y*norm
        #print(corr)
        #print(err)
        return ddm.Solution(corr, err, m, cond, undec)


@paranoidclass
class OverlayNonDecisionJoint(ddm.models.Overlay):
    """Add a non-decision time

    This shifts the reaction time distribution by `nondectime` seconds
    in order to create a non-decision time.

    Example usage:

      | overlay = OverlayNonDecision(nondectime=.2)
    """
    name = "Add a non-decision by shifting the histogram"
    required_parameters = ["nondectime_a", "nondectime_b"]
    required_conditions = ["blocktype"]
    @staticmethod
    def _test(v):
        assert v.nondectime_a in Number(), "Invalid non-decision time a"
        assert v.nondectime_b in Number(), "Invalid non-decision time b"
    @accepts(Self, ddm.Solution)
    @returns(ddm.Solution)
    @ensures("set(return.corr.tolist()) - set(solution.corr.tolist()).union({0.0}) == set()")
    @ensures("set(return.err.tolist()) - set(solution.err.tolist()).union({0.0}) == set()")
    @ensures("solution.prob_undecided() <= return.prob_undecided()")
    def apply(self, solution):
        if solution.conditions["blocktype"] == 1:
            nondectime = self.nondectime_a
        elif solution.conditions["blocktype"] == 2:
            nondectime = self.nondectime_b
        corr = solution.corr
        err = solution.err
        m = solution.model
        cond = solution.conditions
        undec = solution.undec
        shifts = int(nondectime/m.dt) # truncate
        newcorr = np.zeros(corr.shape, dtype=corr.dtype)
        newerr = np.zeros(err.shape, dtype=err.dtype)
        if shifts > 0:
            newcorr[shifts:] = corr[:-shifts]
            newerr[shifts:] = err[:-shifts]
        elif shifts < 0:
            newcorr[:shifts] = corr[-shifts:]
            newerr[:shifts] = err[-shifts:]
        else:
            newcorr = corr
            newerr = err
        return ddm.Solution(newcorr, newerr, m, cond, undec)

@paranoidclass
class OverlaySaccadeInhibitionJoint(ddm.models.Overlay):
    name = "After integrating, switch targets with some probability"
    required_parameters = ["pausemixturecoef_a", "pausenondecision_a", "pausedur_a", "sicohexp_a",
                           "pausemixturecoef_b", "pausenondecision_b", "pausedur_b", "sicohexp_b", "simaxcoh"]
    required_conditions = ["coherence", "presample", "blocktype"]
    @accepts(Self, ddm.Solution)
    @returns(ddm.Solution)
    def apply(self, solution):
        if solution.conditions["blocktype"] == 1:
            pausemixturecoef = self.pausemixturecoef_a
            pausenondecision = self.pausenondecision_a
            pausedur = self.pausedur_a
            sicohexp = self.sicohexp_a
        elif solution.conditions['blocktype'] == 2:
            pausemixturecoef = self.pausemixturecoef_b
            pausenondecision = self.pausenondecision_b
            pausedur = self.pausedur_b
            sicohexp = self.sicohexp_b
        coh_coef = coh_transform(solution.conditions["coherence"], sicohexp, self.simaxcoh)
        mappingcoefcoh = coh_coef * pausemixturecoef

        dt = solution.model.dt
        pauseend = solution.conditions["presample"]/1000 + pausenondecision
        splitpoint1 = np.where(solution.model.t_domain() > pauseend - pausedur)[0][0]
        splitpoint2 = np.where(solution.model.t_domain() > pauseend)[0][0]
        firstpart_corr = solution.corr[0:splitpoint1]
        firstpart_err = solution.err[0:splitpoint1]
        middlepart_corr = solution.corr[splitpoint1:splitpoint2]
        middlepart_err = solution.err[splitpoint1:splitpoint2]
        lastpart_corr = solution.corr[splitpoint2:]
        lastpart_err = solution.err[splitpoint2:]

        middle_mass = np.sum(middlepart_corr + middlepart_err)
        last_mass = np.sum(lastpart_corr + lastpart_err)
        if middle_mass < 1e-10:
            middle_mass = 0
        if last_mass < 1e-5:
            last_mass = 1e-5
        corr = np.hstack([firstpart_corr,
                          middlepart_corr * (1 - mappingcoefcoh),
                          lastpart_corr * (1 + mappingcoefcoh * middle_mass / last_mass)])
        err = np.hstack([firstpart_err,
                         middlepart_err * (1 - mappingcoefcoh),
                         lastpart_err * (1 + mappingcoefcoh * middle_mass / last_mass)])
        return ddm.Solution(corr, err, solution.model, solution.conditions, pdf_undec=solution.undec)

@paranoidclass
class BoundCollapsingExponentialDelayJoint(ddm.models.Bound):
    """Bound dependence: bound collapses exponentially over time.

    Takes three parameters: 

    `B` - the bound at time t = 0.
    `tau` - the time constant for the collapse, should be greater than
    zero.
    `t1` - the time at which the collapse begins in seconds
    """
    name = "collapsing_exponential"
    required_parameters = ["B", "tau_a", "t1_a", "tau_b", "t1_b"]
    required_conditions = ["blocktype"]

    @accepts(Self, Positive0, Conditions)
    @returns(Positive)
    def get_bound(self, t, conditions, **kwargs):
        if conditions["blocktype"] == 1:
            t1 = self.t1_a
            tau = self.tau_a
        elif conditions['blocktype'] == 2:
            t1 = self.t1_b
            tau = self.tau_b
        if t <= t1:
            return self.B
        if t > t1:
            return self.B * np.exp(-tau*(t-t1))

