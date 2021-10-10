# Potentially name this project dmglm?
import numpy as np
import paranoid as pns # Paranoid code
import sys

class Regressor:
    # This object should be immutable, because the "skip" method of
    # DesignMatrix doesn't make a copy of them when it derives a new
    # DesignMatrix from an existing DesignMatrix.
    name = None
    params = []
    def duration(self, *args, **kwargs):
        pass

Params = pns.Dict(k=pns.String, v=pns.Unchecked()) # Paranoid code
NumberList = pns.NDArray(d=1, t=pns.Number) # Paranoid code

class SpacedArray(pns.NDArray):
    """Evenly-spaced NDArray in ascending order"""
    def __init__(self):
        super().__init__(t=pns.Number, d=1)
    def test(self, v):
        super().test(v)
        assert len(np.unique(np.diff(v))) == 1, "Not evenly spaced"
        assert v[1] - v[0] > 0, "Not sorted ascending"
    def generate(self):
        yield np.arange(-10, 10, .1)
        yield np.linspace(0, 10, 25)

@pns.paranoidclass
class RegressorPoint(Regressor):
    def __init__(self, name, bins_after, bins_before=0):
        self.name = name
        self.bins_after = bins_after
        self.bins_before = bins_before
        self.params = [name + "_time"]
    @staticmethod
    def _test(v):
        assert v.bins_after in pns.Natural0()
        assert v.bins_before in pns.Natural0()
        assert v.name in pns.String()
    @pns.accepts(pns.Self)
    @pns.returns(pns.Natural1)
    def duration(self, **kwargs):
        return self.bins_after + self.bins_before
    @pns.accepts(pns.Self, params=Params, n_bins=pns.Natural1, _times=SpacedArray(),
                 _bintimes=SpacedArray())
    @pns.returns(pns.NDArray(d=2, t=pns.Set([0, 1])))
    def matrix(self, params, n_bins, _times, _bintimes, **kwargs):
        time = params[self.name + "_time"]
        M = np.zeros((n_bins, self.bins_before+self.bins_after))
        if time is None:
            return M
        index = next(i for i in range(0, len(_times)) if _bintimes[i] <= time and _bintimes[i+1] > time)
        M[(range(index-self.bins_before, index+self.bins_after), range(0, self.bins_before+self.bins_after))] = 1
        return M

@pns.paranoidclass
class RegressorConstant(Regressor):
    def __init__(self, name, binsize=1):
        self.name = name
        self.binsize = binsize
        self.params = []
    @staticmethod
    def _test(v):
        assert v.binsize in pns.Natural1()
        assert v.name in pns.String()
    @pns.accepts(pns.Self, n_bins=pns.Natural1)
    @pns.returns(pns.Natural1)
    def duration(self, n_bins, **kwargs):
        return n_bins // self.binsize
    @pns.accepts(pns.Self, params=Params, n_bins=pns.Natural1, 
                 _times=SpacedArray, _bintimes=SpacedArray)
    @pns.returns(pns.NDArray(d=2, t=pns.Set([0, 1])))
    def matrix(self, params, n_bins, _times, _bintimes, **kwargs):
        ident = np.eye(n_bins // self.binsize)
        const_mat = np.repeat(ident, self.binsize, axis=0)
        return const_mat

@pns.paranoidclass
class RegressorPointScaled(RegressorPoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params.append(self.name+"_val")
    def matrix(self, params, **kwargs):
        M = super().matrix(params=params, **kwargs)
        return M * params[self.name + "_val"]

@pns.paranoidclass
class RegressorConstantScaled(RegressorConstant):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params.append(self.name+"_val")
    def matrix(self, params, **kwargs):
        M = super().matrix(params=params, **kwargs)
        return M * params[self.name + "_val"]

@pns.paranoidclass
class DesignMatrix:
    def __init__(self, binsize=10, mintime=-1000, maxtime=1000):
        self._binsize = binsize
        self._mintime = mintime
        self._maxtime = maxtime
        self._times = np.arange(mintime, maxtime-binsize/2, binsize)+binsize/2
        self._bintimes = np.append(self._times-binsize/2, self._times[-1]+binsize/2)
        self._regressors = []

    @pns.accepts(pns.Self, pns.List(pns.Number))
    @pns.returns(pns.NDArray(t=pns.Positive0, d=1))
    def bin_spikes(self, spikes):
        return np.histogram(spikes, self._bintimes)[0]

    @pns.accepts(pns.Self)
    @pns.returns(pns.List(pns.String))
    def regressor_names(self):
        return [r.name for r in self._regressors]
    
    @pns.accepts(pns.Self)
    @pns.returns(pns.Natural1)
    def n_bins(self):
        return (self._maxtime - self._mintime) // self._binsize
    
    @pns.accepts(pns.Self)
    @pns.returns(pns.NDArray(d=2))
    def empty_matrix(self):
        n_regressor_timepoints = sum([r.duration(n_bins=self.n_bins()) for r in self._regressors])
        return np.zeros((0, n_regressor_timepoints))
    
    @pns.accepts(pns.Self, Regressor)
    def add_regressor(self, regressor):
        self._regressors.append(regressor)
    
    @pns.accepts(pns.Self, pns.List(pns.String))
    @pns.returns(pns.Self)
    @pns.ensures("len(self._regressors) == len(return._regressors) + len(regressors_to_skip)")
    def skip(self, regressors_to_skip):
        """Return a new DesignMatrix object with the given regressors removed."""
        new_designmatrix = self.__class__(binsize=self._binsize, mintime=self._mintime, maxtime=self._maxtime)
        for r in self._regressors:
            if r.name not in regressors_to_skip:
                new_designmatrix.add_regressor(r)
        return new_designmatrix
    
    @pns.accepts(pns.Self, Params)
    @pns.returns(pns.NDArray(d=2, t=pns.Number))
    @pns.ensures('min(return.shape) > 0')
    def build_matrix(self, params=dict()):
        assert all(k in params.keys() for k in [p for r in self._regressors for p in r.params]), "Bad parameters supplied: got %s but expected %s" % (str(params.keys()),set([p for r in self._regressors for p in r.params]))
        M = np.zeros((self.n_bins(), 0))
        for r in self._regressors:
            Mr = r.matrix(params=params, n_bins=self.n_bins(), _times=self._times, _bintimes=self._bintimes)
            M = np.concatenate([M, Mr], axis=1)
        return M
    
    @pns.accepts(pns.Self, pns.String, pns.NDArray(d=1, t=pns.Number))
    @pns.returns(pns.Tuple(SpacedArray, pns.NDArray(d=1, t=pns.Number)))
    @pns.ensures("len(return[0]) == len(return[1])")
    def get_regressor_from_output(self, name, output):
        start_index = 0
        for r in self._regressors:
            if r.name == name:
                break
            start_index += r.duration(n_bins=self.n_bins())
        r_length = r.duration(n_bins=self.n_bins())
        y = output[start_index:start_index+r_length]
        # This should be generalized
        if isinstance(r, RegressorConstant):
            cbinsize = self._binsize * r.binsize
            x = self._times
        else:
            x = np.arange(-r.bins_before*self._binsize, r.bins_after*self._binsize-self._binsize/2, self._binsize)+self._binsize/2
            #x = self._times[(self._times > -r.bins_before*self._binsize) & (self._times < r.bins_after*self._binsize)]
        return (x,y)

    @pns.accepts(pns.Self, pns.NDArray(d=1, t=pns.Number))
    @pns.returns(pns.NDArray(d=2, t=pns.Number))
    @pns.ensures("prediction.size == return.size")
    def get_trialwise_from_prediction(self, prediction):
        return np.reshape(prediction, (-1, len(self._times)))

@pns.paranoidclass
class MatrixBuilder:
    @pns.accepts(pns.Self, DesignMatrix)
    def __init__(self, design_matrix):
        self.dm = design_matrix
        self.trials = [] # Format: tuple of (params, trial_start, trial_stop)
        self.matrix = design_matrix.empty_matrix()
        self.y = np.asarray([])
    def add_spiketrain(self, params=dict(), spike_times=[], trial_start=None, trial_end=None):
        # Bin spikes to form a timeseries
        spikes = self.dm.bin_spikes(list(spike_times))
        # If the trial doesn't start or end right away, get the starting and ending times
        times = self.dm._times
        i_start = np.where(times>=trial_start)[0][0]
        i_end = np.where(times>trial_end)[0][0]
        # Insert the trial into the relevant places
        self.trials.append((params, i_start, i_end))
        self.y = np.concatenate([self.y, spikes[i_start:i_end]])
        X = self.dm.build_matrix(params)
        self.matrix = np.concatenate([self.matrix, X[i_start:i_end,:]], axis=0)
    def get_y(self):
        return self.y
    def get_x(self):
        return self.matrix
    def get_times(self):
        return dm._times
    def get_trials_from_predictor(self, predictor):
        i = 0
        all_trials = np.zeros((len(self.trials), len(self.dm._times)))
        for i,trial in enumerate(self.trials):
            pass # TODO
