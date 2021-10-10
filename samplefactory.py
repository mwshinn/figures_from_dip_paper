import pandas
import numpy as np
from ddm import Sample
import scipy
import scipy.io

def SampleFactory(animal=1, zerocoh=False, testset=False):
    limit = 2000 if animal == 1 else 3000
    behtab = pandas.read_pickle("test_set2.pkl" if testset else "training_set.pkl")
    cols = list(behtab.columns)
    cols[16] = 'choice_of_reward'
    behtab.columns = cols
    np.seterr(divide='ignore') # NOTE: hack to avoid bug in Pandas 0.18.1: https://github.com/pandas-dev/pandas/issues/13135
    behtab.query('animal == %i' % animal, inplace=True) # Single monkey
    behtab.query('preds_rt < %f' % limit, inplace=True) # Filter out long trials
    behtab.query('trialtype == 2 or trialtype == 3 or trialtype == 4', inplace=True) # Completed trials only # presample type == 3, didn't hold fixation == 4, no non-decision trials
    #behtab.query('choice_of_reward != -1 or correct_target_ds_perc == 50', inplace=True) # Inconsistent information for some trials
    behtab.query('choice_of_reward != -1', inplace=True) # When the monkey didn't make a choice
    # For 50% coherence trials, we define "correct" as going in the same direction as the high reward.
    #behtab['correct'] = np.logical_or(np.logical_and(behtab['correct'] == 1, behtab['correct_target_ds_perc'] != 50),
    #                                  np.logical_and(behtab[16] == 1, behtab['correct_target_ds_perc'] == 50))
    
    # Some 50% coherence trials have a presample of 800.  Set them to 0.
    behtab.loc[behtab['correct_target_ds_perc'] == 50, 'preds_interval'] = 0
    
    # For some reason "correct" and "reward" were overwritten with -1
    # in the matlab version of this dataset for 0 coherence.  At some
    # point I'll switch this to the SQLite version so this will work
    # for now.  In all honesty, it doesn't matter whether the random
    # number was assigned when the monkey did the task or whether it
    # is assigned now so I'm not too concerned about it.
    randvals1 = np.random.RandomState(seed=0).rand(len(behtab))
    correctind = np.logical_and(randvals1 >= .5, behtab['correct_target_ds_perc'] == 50)
    errorind = np.logical_and(randvals1 < .5, behtab['correct_target_ds_perc'] == 50)
    behtab.loc[correctind,'correct'] = True
    behtab.loc[errorind, 'correct'] = False
    hrindex = np.logical_and(behtab['correct_target_ds_perc'] == 50, behtab['correct'] == behtab['choice_of_reward'])
    lrindex = np.logical_and(behtab['correct_target_ds_perc'] == 50, behtab['correct'] != behtab['choice_of_reward'])
    behtab.loc[hrindex, 'reward'] = 1
    behtab.loc[lrindex, 'reward'] = 0
    assert np.all((behtab['reward'] == behtab['correct'])== behtab['choice_of_reward'])
    assert np.all((behtab['choice_of_reward'] == behtab['correct'])== behtab['reward'])
    
    correct = behtab[behtab['correct'] == True]
    error = behtab[behtab['correct'] == False]
    # The .values.tolist part is to get python types instead of numpy types.
    pdf = Sample((correct['preds_rt']/1000).values, (error['preds_rt']/1000).values, 0,
                 coherence=(correct['correct_target_ds_perc'].values, error['correct_target_ds_perc'].values),
                 highreward=(correct['reward'].values, error['reward'].values),
                 presample=(correct['preds_interval'].values, error['preds_interval'].values),
                 fileid=(correct['fileID'].values, error['fileID'].values),
                 correctloc=(correct['correct_target_loc'].values, error['correct_target_loc'].values),
                 correctcolor=(correct['correct_target_color'].values, error['correct_target_color'].values))
    if animal == 1:
        cohs = [53, 60, 70]
    else:
        cohs = [52, 57, 63]
    if zerocoh == True:
        cohs.append(50)
    pdf = pdf.subset(coherence=cohs)
    np.seterr(divide='raise') # NOTE: continuation of hack to avoid bug in Pandas 0.18.1
    return pdf



def SampleFactoryDS2(animal=1):
    # Load the data
    D = scipy.io.loadmat("tm_beh_data.mat", squeeze_me=True, chars_as_strings=True)
    behtab = pandas.DataFrame(D['tm_beh_data'])
    del D
    
    # Assign the column labels to be something meaningful
    BEHTAB_KEYS = ['animal', 'session', 'block', 'presample', 'logcolorratio', 'RT', 'RT_s', 'trialresult', 'choice', 'coh', 'RT_adj', 'presample_actual_dur', 'trialnum']
    cols = list(behtab.columns)
    cols[0:len(BEHTAB_KEYS)] = BEHTAB_KEYS
    behtab.columns = cols
    del cols
    
    # Convert columns of floats which should be ints to ints
    for c in behtab.columns:
        if numpy.sum(numpy.abs(behtab[c] - behtab[c].astype(int)))<1e-10:
            behtab[c] = behtab[c].astype(int)
    
    behtab = behtab.query("animal == @animal")
    
    shortblocks = []
    longblocks = []
    for block in list(set(behtab['block'])):
        trials = behtab.query('block == %i' % block)
        if len(trials) == len(trials.query('presample < 1000')):
            shortblocks.append(block)
        elif len(trials) == len(trials.query('presample > 600')):
            longblocks.append(block)
    
    assert set(behtab['block']) == set(shortblocks+longblocks)
    behshort = behtab.query("block in @shortblocks")
    behlong = behtab.query("block in @longblocks")
    assert len(behshort)+len(behlong) == len(behtab)
    
    scorrect = behshort.query('(choice == 1) == (coh > 50)')
    serror = behshort.query('(choice == 1) != (coh > 50)')
    def pt(x): # Pythonic types
        arr = np.asarray(x)
        if np.sum(np.abs(arr-np.round(arr))) == 0:
            arr = arr.astype(int)
        return arr.tolist()
    sample_short = Sample(scorrect['RT']/1000, serror['RT']/1000, 0,
                          coherence=(pt(50+numpy.abs(50-scorrect['coh'])), pt(50+numpy.abs(50-serror['coh']))),
                          presample=(pt(scorrect['presample']), pt(serror['presample'])),
                          block=(pt(np.ones(len(scorrect))), pt(numpy.ones(len(serror)))), # NOTE there was a bug where the correct trials here were *2
                          session=(scorrect['session'], serror['session']))
    assert len(scorrect) + len(serror) == len(behshort)
    lcorrect = behlong.query('(choice == 1) == (coh > 50)')
    lerror = behlong.query('(choice == 1) != (coh > 50)')
    sample_long = Sample(lcorrect['RT']/1000, lerror['RT']/1000, 0,
                         coherence=(pt(50+numpy.abs(50-lcorrect['coh'])), pt(50+numpy.abs(50-lerror['coh']))),
                         presample=(pt(lcorrect['presample']), pt(lerror['presample'])),
                         block=(pt(np.ones(len(lcorrect))*2), pt(numpy.ones(len(lerror))*2)),
                         session=(lcorrect['session'], lerror['session']))
    assert len(lcorrect) + len(lerror) == len(behlong)
    fullsample = sample_short + sample_long
    assert len(fullsample) == len(behtab)
    return fullsample

def SampleFactoryDS2(animal=1, zerocoh=False):
    # Load the data
    np.seterr(divide='ignore') # NOTE: hack to avoid bug in Pandas 0.18.1: https://github.com/pandas-dev/pandas/issues/13135
    D = scipy.io.loadmat("tm_beh_data.mat", squeeze_me=True, chars_as_strings=True)
    behtab = pandas.DataFrame(D['tm_beh_data'])
    D = D['tm_beh_data']
    
    # Assign the column labels to be something meaningful
    BEHTAB_KEYS = ['animal', 'session', 'block', 'presample', 'logcolorratio', 'RT', 'RT_s', 'trialresult', 'choice', 'coh', 'RT_adj', 'presample_actual_dur', 'trialnum']
    cols = list(behtab.columns)
    cols[0:len(BEHTAB_KEYS)] = BEHTAB_KEYS
    behtab.columns = cols
    behtab = behtab.query("animal == @animal")
    del cols
    
    ANIMAL = 0
    SESSION = 1
    BLOCK = 2
    PRESAMPLE = 3
    RT = 5
    CHOICE = 8
    COHERENCE = 9
    D = D[D[:,ANIMAL] == animal,:] # Only one monkey
    
    # Get list of blocks which have short vs long trials
    shortblocks = []
    longblocks = []
    # This loop could be relpaced with equivalent scipy code to remove
    # the pandas dependency, but I just copied it and pasted it from
    # before.
    for block in list(set(behtab['block'])):
        trials = behtab.query('block == %i' % block)
        if len(trials) == len(trials.query('presample < 1000')):
            shortblocks.append(block)
        elif len(trials) == len(trials.query('presample > 600')):
            longblocks.append(block)
    
    
    blocktype = isin(D[:,BLOCK], longblocks)+1
    #D[:,D.shape[1]] = np.isin(D[:,BLOCK], shortblocks)+1
    
    poscoh = D[:,COHERENCE] > 50
    correctresponses = (D[:,CHOICE] == 1) == poscoh
    D[:,COHERENCE] = (50+np.abs(50-D[:,COHERENCE]))
    D[:,PRESAMPLE] = D[:,PRESAMPLE]
    D[:,RT] = D[:,RT]/1000
    rts_and_conditions = np.asarray([D[:,RT], correctresponses, D[:,COHERENCE], D[:,PRESAMPLE], blocktype, D[:,SESSION]]).T
    column_labels = ["coherence", "presample", "blocktype", 'session']
    np.seterr(divide='raise') # NOTE: continuation of hack to avoid bug in Pandas 0.18.1
    
    return Sample.from_numpy_array(rts_and_conditions, column_labels)

def in1d(ar1, ar2, assume_unique=False, invert=False):
    """Backported for old numpy"""
    # Ravel both arrays, behavior for the first array could be different
    ar1 = np.asarray(ar1).ravel()
    ar2 = np.asarray(ar2).ravel()

    # Check if one of the arrays may contain arbitrary objects
    contains_object = ar1.dtype.hasobject or ar2.dtype.hasobject

    # This code is run when
    # a) the first condition is true, making the code significantly faster
    # b) the second condition is true (i.e. `ar1` or `ar2` may contain
    #    arbitrary objects), since then sorting is not guaranteed to work
    if len(ar2) < 10 * len(ar1) ** 0.145 or contains_object:
        if invert:
            mask = np.ones(len(ar1), dtype=bool)
            for a in ar2:
                mask &= (ar1 != a)
        else:
            mask = np.zeros(len(ar1), dtype=bool)
            for a in ar2:
                mask |= (ar1 == a)
        return mask

    # Otherwise use sorting
    if not assume_unique:
        ar1, rev_idx = np.unique(ar1, return_inverse=True)
        ar2 = np.unique(ar2)

    ar = np.concatenate((ar1, ar2))
    # We need this to be a stable sort, so always use 'mergesort'
    # here. The values from the first array should always come before
    # the values from the second array.
    order = ar.argsort(kind='mergesort')
    sar = ar[order]
    if invert:
        bool_ar = (sar[1:] != sar[:-1])
    else:
        bool_ar = (sar[1:] == sar[:-1])
    flag = np.concatenate((bool_ar, [invert]))
    ret = np.empty(ar.shape, dtype=bool)
    ret[order] = flag

    if assume_unique:
        return ret[:len(ar1)]
    else:
        return ret[rev_idx]


def isin(element, test_elements, assume_unique=False, invert=False):
    """Backported for old numpy"""
    element = np.asarray(element)
    return in1d(element, test_elements, assume_unique=assume_unique,
                invert=invert).reshape(element.shape)
