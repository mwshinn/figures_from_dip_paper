from diplib import get_color, make_gridlegend
from matplotlib import pyplot as plt
import tensorflow as tf
import sys
import scipy.signal
from canvas import Canvas, Vector, Point
import seaborn as sns

# ---------------------- Import the package ---------------------------
from psychrnn.tasks.perceptual_discrimination import PerceptualDiscrimination
from psychrnn.backend.models.basic import Basic


from psychrnn.tasks.task import Task
import numpy as np

tf.random.set_seed(0)
np.random.seed(0)

class ColorMatchTask(Task):
    # def __init__(self, dt, tau, T, N_batch):
    #     super(SimplePDM, self).__init__(2, 2, dt, tau, T, N_batch)
    def generate_trial_params(self, batch, trial):
        """"Define parameters for each trial.
        Using a combination of randomness, presets, and task attributes, define the necessary trial parameters.
        Args:
            batch (int): The batch number that this trial is part of.
            trial (int): The trial number of the trial within the batch.
        Returns:
            dict: Dictionary of trial parameters.
        """
        # ----------------------------------
        # Define parameters of a trial
        # ----------------------------------
        params = dict()
        params['coherence'] = np.random.choice([0, .25, .5, 1, 2, 4, 6, 8, 10])
        params['presample'] = np.random.choice([0, 200, 400])
        params['direction'] = np.random.choice([-1, 1])
        return params
    def trial_function(self, time, params):
        """ Compute the trial properties at the given time.
        Based on the params compute the trial stimulus (x_t), correct output (y_t), and mask (mask_t) at the given time.
        Args:
            time (int): The time within the trial (0 <= time < T).
            params (dict): The trial params produced generate_trial_params()
        Returns:
            tuple:
            x_t (ndarray(dtype=float, shape=(N_in,))): Trial input at time given params.
            y_t (ndarray(dtype=float, shape=(N_out,))): Correct trial output at time given params.
            mask_t (ndarray(dtype=bool, shape=(N_out,))): True if the network should train to match the y_t, False if the network should ignore y_t when training.
        """
        coh = params['coherence']
        direction = params['direction']
        ps = params['presample']
        # ----------------------------------
        # Initialize with noise
        # ----------------------------------
        x_t = np.random.randn(self.N_in)
        y_t = np.zeros(self.N_out)
        mask_t = np.ones(self.N_out)
        # ----------------------------------
        # Compute values
        # ----------------------------------
        if ps <= time:
            x_t[0] += coh*direction
            y_t[0] = direction
        return x_t, y_t, mask_t



def custom_pw_squared_error(predictions, y, mask):
    #split_i = tf.shape(y) - tf.math.count_nonzero(y, dtype=tf.dtypes.int32)
    least_squares_mask = tf.cast(tf.math.equal(y, 0), tf.float32)
    other_mask = tf.cast(tf.math.not_equal(y, 0), tf.float32)
    X = predictions * tf.sign(y) * -1
    response_phase = tf.reduce_mean(other_mask*tf.square(tf.abs(X)-1)*tf.cast(X<-1, tf.float32) + tf.sqrt(tf.abs((X+1)/2)+.0001)*tf.cast(-1<=X, tf.float32)*tf.cast(X<=1, tf.float32) + (1+tf.square(X-1))*tf.cast(1<X, tf.float32))
    initial_phase = tf.reduce_mean(least_squares_mask*tf.square(predictions))
    print_op = tf.print(response_phase, initial_phase, output_stream=sys.stderr, sep=',')
    with tf.control_dependencies([print_op]):
        retvar = response_phase + initial_phase
    return retvar


# ---------------------- Set up a basic model ---------------------------
cm = ColorMatchTask(dt = 10, tau = 100, T = 1000, N_batch = 128, N_in=1, N_out=1)
# np.asarray([cm.trial_function(i, {"coherence": 10, "presample": 15, "direction": 1, "onset": 50})[0] for i in range(0, 500)])


network_params = cm.get_task_params() # get the params passed in and defined in pd
network_params['name'] = 'model' # name the model uniquely if running mult models in unison
network_params['N_rec'] = 20 # set the number of recurrent units in the model
network_params['loss_function'] = 'myloss'
network_params['myloss'] = custom_pw_squared_error
model = Basic(network_params) # instantiate a basic vanilla RNN

# ---------------------- Train a basic model ---------------------------
model.train(cm) # train model to perform pd task

# ---------------------- Test the trained model ---------------------------
x = []
target_output = []
mask = []
trial_params = []
model_output = []
model_state = []
for i in range(0, 500):
    _x,_target_output,_mask, _trial_params = cm.get_trial_batch() # get pd task inputs and outputs
    _model_output, _model_state = model.test(_x) # run the model on input x
    x.append(_x)
    target_output.append(_target_output)
    mask.append(_mask)
    trial_params.append(_trial_params)
    model_output.append(_model_output)
    model_state.append(_model_state)

x = np.concatenate(x)
target_output = np.concatenate(target_output)
mask = np.concatenate(mask)
trial_params = np.concatenate(trial_params)
model_output = np.concatenate(model_output)
model_state = np.concatenate(model_state)


# ---------------------- Plot the results ---------------------------

c = Canvas(6.9, 5.2, "in")
# RNN Image
c.add_image("rnn.png", Point(1.1, 3.8, "in"), width=Vector(1, 0, "in"), ha="left", va="bottom", unitname="rnnimg")
c.add_arrow(Point(-.3, .5, "rnnimg"), Point(-.01, .5, "rnnimg"))
# c.add_text("Input", Point(-.6, .5, "rnnimg")+Vector(.05, .05, "cm"), ha="left", va="bottom")
c.add_arrow(Point(1.01, .5, "rnnimg"), Point(1.3, .5, "rnnimg"))
# c.add_text("Output", Point(1.05, .5, "rnnimg")+Vector(.05, .05, "cm"), ha="left", va="bottom")
c.add_grid(["in1", "in2", "in3"], 3, Point(.1, 3.9, "in"), Point(0.8, 4.7, "in"), spacing=Vector(1, .1, "in"), unitname="ingrid")
c.add_grid(["out1", "out2", "out3"], 3, Point(2.4, 3.9, "in"), Point(3.1, 4.7, "in"), spacing=Vector(1, .1, "in"), unitname="outgrid")

c.add_text("Evidence input", Point(.5, 1.05, "ingrid"), ha="center", va="bottom")
c.add_text("Motor output", Point(.5, 1.05, "outgrid"), ha="center", va="bottom")

c.add_axis("psychometric", Point(3.9, 4.0, "in"), Point(4.8, 4.9, "in"))
c.add_axis("chronometric", Point(5.6, 4.0, "in"), Point(6.5, 4.9, "in"))

c.add_grid([f"{l}{ps}" for l in ["rt", "unit"] for ps in [200, 400]],
           2, Point(.7, .5, "in"), Point(3.0, 3.3, "in"), size=Vector(1, 1, "in"))
c.add_grid([f"motor{ps}" for ps in [200, 400]],
           1, Point(4.2, 2.3, "in"), Point(6.5, 3.3, "in"), size=Vector(1, 1, "in"))

make_gridlegend(c, Point(4.7, 0.6, "in"), forrnn=True)

for num in range(0, 3):
    i = {0: 11, 1: 10, 2: 7}[num]
    # i = {0: 47, 1: 48, 2: 57}[num]
    ax = c.ax(f"out{num+1}")
    ax.cla()
    ax.plot(model_output[i], c='g')
    ax.axhline(0, c='k', linewidth=.5)
    ax.axvline(0, c='k', linewidth=.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlim(0, None)
    ax.axis("off")
    ax = c.ax(f"in{num+1}")
    ax.cla()
    ax.plot(x[i][:,0], c='g')
    ax.axhline(0, c='k', linewidth=.5)
    ax.axvline(0, c='k', linewidth=.5)
    ax.set_ylim(-10, 10)
    ax.set_xlim(0, None)
    ax.axis("off")


# Show psychometric function
def choice(x):
    try:
        ind = np.where(np.abs(x)>.8)[0][0]
    except IndexError:
        return 0
    if x[ind] > 0:
        return 1
    elif x[ind] < 0:
        return -1

def rt(x, params):
    try:
        ind = np.where(np.abs(x)>.8)[0][0]
    except IndexError:
        return np.inf
    return ind-params['presample']/10

# Psychometric

ax = c.ax("psychometric")
cohs = [0, .25, .5, 1, 2, 4, 6, 8, 10]
for ps in [0, 200, 400]:
    corrs = []
    for coh in cohs:
        inds = [i for i in range(0, len(trial_params)) if trial_params[i]['coherence'] == coh]
        choices = [choice(model_output[inds[i]]) == trial_params[inds[i]]['direction'] for i in range(0, len(inds)) if rt(model_output[inds[i]], trial_params[inds[i]]) <= np.inf and trial_params[inds[i]]['presample'] == ps]
        corr = sum(choices)
        corrs.append(corr/len(choices))
    ax.plot(np.asarray(cohs[0:5]), corrs[0:5], marker='o', color=get_color(ps=ps*2))

ax.set_xlabel("Coherence")
ax.set_ylabel("P(correct)")
ax.set_xticks([0, 1, 2])
sns.despine(ax=ax)

# Chronometric

ax = c.ax("chronometric")
cohs = [0, .25, .5, 1, 2, 4, 6, 8, 10]
for ps in [0, 200, 400]:
    all_rts = []
    for coh in cohs:
        inds = [i for i in range(0, len(trial_params)) if trial_params[i]['coherence'] == coh]
        rts = [rt(model_output[inds[i]], trial_params[inds[i]]) for i in range(0, len(inds)) if rt(model_output[inds[i]], trial_params[inds[i]]) >= 0 and choice(model_output[inds[i]]) == trial_params[inds[i]]['direction'] and trial_params[inds[i]]['presample'] == ps]
        rtvals = np.mean(rts)
        all_rts.append(rtvals)
    ax.plot(np.asarray(cohs[0:5]), np.asarray(all_rts[0:5])+ps/10, marker='o', color=get_color(ps=ps*2))

ax.set_xlabel("Coherence")
ax.set_ylabel("Presample-aligned\nRT (steps)")
ax.set_xticks([0, 1, 2])
sns.despine(ax=ax)

# No dip in RT distribution

bins = np.arange(-20, 100, 1)
for ps in [200, 400]:
    ax = c.ax(f"rt{ps}")
    ax.cla()
    for coh,cohname in [(0, 50), (.5, 53), (1, 60), (4, 70)]:
        rts = [rt(model_output[i], trial_params[i]) for i in range(0, len(trial_params)) if trial_params[i]['presample'] == ps and trial_params[i]['coherence'] == coh]
        trace = np.histogram(np.asarray(rts)[~np.isinf(rts)], bins=bins)
        ax.plot(bins[0:-1]+(bins[1]-bins[0])/2, scipy.signal.savgol_filter(trace[0], 3, 1)/len(model_output)*20, color=get_color(ps=2*ps,coh=cohname))
    sns.despine(ax=ax)
    ax.set_ylim(0, .017)
    ax.set_xlim(-15, 5)
    ax.set_xlabel("Time from\nsample (steps)")
    if ps == 400:
        ax.set_yticklabels([])
    else:
        ax.set_ylabel("RT histogram")


# No dip in mean output
for ps in [200, 400]:
    ax = c.ax(f"motor{ps}")
    ax.cla()
    for coh,cohname in [(0, 50), (.5, 53), (1, 60), (4, 70)]:
        trials = [i for i in range(0, len(trial_params)) if trial_params[i]['presample'] == ps and trial_params[i]['coherence'] == coh and trial_params[i]['direction'] == -1]
        ax.plot(np.linspace(1, 100, 100)-ps/10, np.mean(model_output[trials,:,0], axis=0), color=get_color(ps=ps*2, coh=cohname))
        trials = [i for i in range(0, len(trial_params)) if trial_params[i]['presample'] == ps and trial_params[i]['coherence'] == coh and trial_params[i]['direction'] == 1]
        ax.plot(np.linspace(1, 100, 100)-ps/10, np.mean(model_output[trials,:,0], axis=0), color=get_color(ps=ps*2, coh=cohname))
    ax.set_xlim(-15, 5)
    ax.set_ylim(-.45, .45)
    ax.set_xlabel("Time from\nsample (steps)")
    sns.despine(ax=ax)
    if ps == 400:
        ax.set_yticklabels([])
    else:
        ax.set_ylabel("Motor output")




# No dip in mean interneuron state
for ps in [200, 400]:
    ax = c.ax(f"unit{ps}")
    ax.cla()
    for coh,cohlabel in [(0, 50), (.5, 53), (1, 60), (2, 70)]:
        trials = [i for i in range(0, len(trial_params)) if trial_params[i]['presample'] == ps and trial_params[i]['coherence'] == coh and trial_params[i]['direction'] == 1]
        ax.plot(np.linspace(1, 100, 100)-ps/10, np.mean(model_state[trials,:,:], axis=(0,2)), color=get_color(ps=ps*2, coh=cohlabel))
        trials = [i for i in range(0, len(trial_params)) if trial_params[i]['presample'] == ps and trial_params[i]['coherence'] == coh and trial_params[i]['direction'] == -1]
        ax.plot(np.linspace(1, 100, 100)-ps/10, np.mean(model_state[trials,:,:], axis=(0,2)), color=get_color(ps=ps*2, coh=cohlabel))
    sns.despine(ax=ax)
    ax.set_xlim(-15, 5)
    ax.set_ylim(-.3, .3)
    ax.set_xlabel("Time from\nsample (steps)")
    if ps == 400:
        ax.set_yticklabels([])
    else:
        ax.set_ylabel("Mean unit activity")



c.add_figure_labels([("a", "in1", Vector(.5, .5, "cm")), ("b", "psychometric"), ("c", "chronometric"), ("d", "rt200"), ("e", "motor200"), ("f", "unit200")])

c.add_text("RNN RT distribution", (Point(0, 1, "axis_rt200") | Point(1, 1, "axis_rt400")) + Vector(0, .2, "cm"), weight="bold")
c.add_text("RNN motor output", (Point(0, 1, "axis_motor200") | Point(1, 1, "axis_motor400")) + Vector(0, .2, "cm"), weight="bold")
c.add_text("RNN mean hidden unit activity", (Point(0, 1, "axis_unit200") | Point(1, 1, "axis_unit400")) + Vector(0, .2, "cm"), weight="bold")
c.add_text("RNN schematic", Point(.5, 1, "rnnimg")+Vector(0, .2, "cm"), weight="bold")

c.save("figureRS1.pdf")

# ---------------------- Teardown the model -------------------------
#model.destruct()
