from canvas import *
import diplib
import seaborn as sns
import sys

MONKEY = sys.argv[1]
USE50 = True

if MONKEY == "Q":
    HC,MC,LC = 70, 60, 53
elif MONKEY == "P":
    HC,MC,LC = 63, 57, 52

#regname = "sample"
regname = "samplecoh"

c = Canvas(3.3, 8, fontsize=8)


LEFT1 = .5
RIGHT1 = 1.5
LEFT2 = 2.1
RIGHT2 = 3.1
c.add_axis("diagram", Point(LEFT1, 7.2, "absolute"), Point(RIGHT2, 7.8, "absolute"))

c.add_grid(["cell1act", "cell2act", "cell1kern", "cell2kern"], 2, Point(LEFT1, 4.5, "absolute"), Point(RIGHT2, 6.6, "absolute"), size=Vector(RIGHT1-LEFT1, RIGHT1-LEFT1, "absolute"))

c.add_axis("means", Point(LEFT2, 2.4, "absolute"), Point(RIGHT2, 3.6, "absolute"))
c.add_grid(["posweights", "negweights"], 2, Point(LEFT1, 2.4, "absolute"), Point(RIGHT1, 3.6, "absolute"), spacing=Vector(0, .1, "absolute"))

c.add_axis("pcs", Point(LEFT1, .6, "absolute"), Point(RIGHT1, 1.6, "absolute"))
c.add_axis("loadinghist", Point(LEFT2, .6, "absolute"), Point(RIGHT2, 1.6, "absolute"))

c.add_figure_labels([("b", "cell1act"), ("a", "diagram"), ("c", "cell1kern"), ("d", "posweights"), ("e", "means"), ("f", "pcs"), ("g", "loadinghist", Vector(-.6, 0, "cm"))])

CELLS = diplib.get_cell_ids(monkey=MONKEY)
regs = diplib.get_regression_models(monkey=MONKEY)


#################### Diagram ####################

ax = c.ax("diagram")
T,traj = diplib.get_cell_conditional_activity(monkey=MONKEY, ps=800, coh=HC, hr_in_rf=True, align="presample", time_range=(-1000, 2000))
example_cell = CELLS[0]
dm = regs[example_cell]['dm']
T_samplecoh = dm.get_regressor_from_output(regname, regs[example_cell]['params'])[0]
T_saccade = dm.get_regressor_from_output("saccade-inrf", regs[example_cell]['params'])[0]
T_presample = dm.get_regressor_from_output("presample", regs[example_cell]['params'])[0]

SACTIME = 1125
T_presample = T_presample[(T_presample >= 0) & (T_presample <= SACTIME)]
PS = 800
ax.plot(T*1000, traj, c='k')
ax.plot(T_samplecoh+PS, len(T_samplecoh)*[0], c=diplib.kern_color('sample'), linewidth=4)
ax.plot(T_saccade+SACTIME, len(T_saccade)*[5], c=diplib.kern_color('saccade'), linewidth=4)
ax.plot(T_presample, len(T_presample)*[10], c=diplib.kern_color('presample'), linewidth=4)
ax.axvline(SACTIME, c=diplib.kern_color('saccade'), linestyle='--')
ax.axvline(PS, c=diplib.kern_color('sample'), linestyle='--')
ax.axvline(0, c=diplib.kern_color('presample'), linestyle='--')
ax.set_xlim(-200, 1600)
ax.set_yticks([])
ax.set_xlabel("Time from presample (ms)")
c.add_text("Sample", Point(PS, 0, "diagram") >> Point(0, 1.1, "axis_diagram"), color=diplib.kern_color('sample'))
c.add_text("Presample", Point(0, 0, "diagram") >> Point(0, 1.1, "axis_diagram"), color=diplib.kern_color('presample'))
c.add_text("Saccade", Point(SACTIME, 0, "diagram") >> Point(0, 1.1, "axis_diagram"), color=diplib.kern_color('saccade'))
sns.despine(ax=ax)

#################### Example cells ####################

example_cells = [3401, 4001] if MONKEY == "Q" else [6101, 7901] # 7901
for i,cellid in enumerate(example_cells):
    ax = c.ax(f"cell{i+1}act")
    ax.cla()
    for coh in [LC, MC, HC]:
        for ps in [400, 800]:
            ax.plot(*diplib.get_cell_conditional_activity(monkey=MONKEY, ps=ps, coh=coh, cellid=cellid), c=diplib.get_color(coh=coh, ps=ps))
    ax.set_xlim(0, .3)
    ax.set_title(f"Cell {cellid}")
    ax.set_ylabel("Spikes per second")
    ax.set_xticks([])
    sns.despine(bottom=True, ax=ax)
    if ax.get_ylim()[0] < 1:
        ax.set_ylim(1, None)
    ax = c.ax(f"cell{i+1}kern")
    ax.cla()
    dm = regs[cellid]['dm']
    T,params = dm.get_regressor_from_output(regname, regs[cellid]['params'])
    bse = dm.get_regressor_from_output(regname, regs[cellid]['bse'])[1]
    ax.errorbar(T, params, yerr=bse, c=diplib.kern_color('EC'))
    ax.set_xlim(0, 300)
    ax.set_clip_on(False)
    ax.set_xticks([0, 100, 200, 300])
    ax.set_xticklabels([0, 100, 200, 300])
    ax.set_ylabel("Evidence kernel")
    ax.set_xlabel("Time from sample (ms)")
    ax.axhline(0, c='gray', linestyle='--')
    sns.despine(ax=ax)


# c.add_legend(Point(.05, .25, "axis_cell2act"), [("800ms, high coh", {"color":diplib.get_color(ps=800,coh=HC), "linewidth": 5}),
#                                              ("800ms, low coh", {"color":diplib.get_color(ps=800,coh=LC), "linewidth": 5}),
#                                              ("400ms, high coh", {"color":diplib.get_color(coh=HC, ps=400), "linewidth": 5}),
#                                              ("400ms, low coh", {"color":diplib.get_color(coh=LC, ps=400), "linewidth": 5})])

#################### Significant regression coefficients ####################




T_dip_overlay, dip_overlay = diplib.get_cell_conditional_activity(monkey=MONKEY, ps=800, coh=HC, hr_in_rf=True, align="sample", time_range=(0, 400))
dip_overlay = dip_overlay/np.mean(dip_overlay)

times = regs[CELLS[0]]['dm'].get_regressor_from_output(regname, regs[CELLS[0]]['params'])[0]
kernels = []
sig_neg_coef = times * 0
sig_pos_coef = times * 0
n_cell_regs = 0
for cellid in CELLS:
    if cellid not in regs.keys(): continue
    dm = regs[cellid]['dm']
    estimate = dm.get_regressor_from_output(regname, regs[cellid]['params'])[1]
    kernels.append(estimate)
    sem = dm.get_regressor_from_output(regname, regs[cellid]['bse'])[1]
    sig_neg_coef += (estimate+1.64*sem < 0).astype(int)
    sig_pos_coef += (estimate-1.64*sem > 0).astype(int)
    n_cell_regs += 1


ax = c.ax("posweights")
ax.bar(times, sig_pos_coef/n_cell_regs*100, 25, color=diplib.kern_color('EC'))
ax.set_ylabel("% positive")
#ax.set_xticks([0, 100, 200, 300, 400])
#ax.set_xticklabels([0, "", 200, "", 400])
ax.set_xticks([])
if MONKEY == "Q":
    ax.set_ylim(0, 60)
    ax.set_yticks([15, 30, 45])
elif MONKEY == "P":
    ax.set_ylim(0, 80)
    ax.set_yticks([40, 80])
#ax.set_xlabel("Time after sample (ms)")
ax.set_title("% cells with significant\nevidence kernel")
ax.set_xlim(0, 300)
#ax.plot(T_dip_overlay*1000, dip_overlay/2-.3, alpha=.2, c='k')
sns.despine(ax=ax)

ax = c.ax("negweights")
ax.bar(times, sig_neg_coef/n_cell_regs*100, 25, color=diplib.kern_color('EC'))
#ax.set_title("Negative\ncoherence $\\times$ sample\nregression coefficients")
ax.set_xticks([0, 100, 200, 300])
ax.set_xticklabels([0, 100, 200, 300])
if MONKEY == "Q":
    ax.set_ylim(0, 60)
    ax.set_yticks([15, 30, 45])
elif MONKEY == "P":
    ax.set_ylim(0, 80)
    ax.set_yticks([40, 80])

ax.set_xlim(0, 300)
ax.set_xlabel("Time from sample (ms)")
ax.set_ylabel("% negative")
ax.invert_yaxis()
sns.despine(ax=ax, top=False)

print("="*100, "pos neg at 150-175", sig_pos_coef[6]/n_cell_regs, sig_neg_coef[6]/n_cell_regs)

ax = c.ax("means")
km = np.mean(kernels, axis=0)
ksem = np.std(kernels, axis=0)/np.sqrt(len(kernels))
ax.errorbar(times, km, yerr=ksem*1.64, c=diplib.kern_color('EC'))
ax.axhline(0, c='gray', linestyle='--')
ax.set_xlim(0, 300)
ax.set_title("Mean evidence kernel")
ax.set_xlabel("Time from sample (ms)")
ax.set_ylabel("Mean kernel")
sns.despine(ax=ax)


#################### Plot PCs ####################

ax = c.ax("pcs")
pcs = diplib.get_pcs_noncentered(monkey=MONKEY, regressor=("sample" if regname == "samplecoh" else "sample-nocoh")) # I know this is weird, historical reasons
times = pcs[3]

if MONKEY == "Q":
    PCNUM = 0
elif MONKEY == "P":
    PCNUM = 0

for i in range(0, 1):
    ax.plot(times, pcs[1][:,i], c='k', lw=(1.5 if i!=PCNUM else 4), alpha=1, solid_capstyle="round")
    ax.plot(times, pcs[1][:,i], c=diplib.pc_colors(i), lw=(1 if i!=PCNUM else 3), alpha=1, solid_capstyle="round")
    #ax.plot(times, ((-1 if i==0 else 1)*pcs[1]*pcs[0])[:,i], c=diplib.pc_colors(i), lw=(1.5 if i!=PCNUM else 3), alpha=(.4 if i!=PCNUM else 1))

print("="*100, "monkey", MONKEY, "mean and sv correlation", np.corrcoef(pcs[1][:,PCNUM], km)[0,1])

ax.set_xlabel("Time from sample (ms)")
ax.set_ylabel("Singular vector weight")
ax.set_title("Evidence kernel\nfirst singular vector")
#c.add_legend(Point(2.3, 3.6, "absolute"), [("Var explained: %i%%" % (100*pcs[0][i]) , {"lw": 5, "color": colors[i]}) for i in range(0, 4)])
ax.set_xlim(0, 300)
ax.set_xticks([0, 100, 200, 300])
ax.axhline(0, c='gray', linestyle='--', zorder=-10)
sns.despine(ax=ax)

ax = c.ax("loadinghist")
pc_weights = pcs[2][:,PCNUM]
if MONKEY == "P":
    h = ax.hist(pc_weights, bins=np.arange(-6, 6.001, .5), color=diplib.pc_colors(PCNUM))
    ax.set_xlim(-6, 6)
else:
    h = ax.hist(pc_weights, bins=np.arange(-3, 3.001, .25), color=diplib.pc_colors(PCNUM))
    ax.set_xlim(-3, 3)
ax.axvline(0, c='gray', linestyle='--')
ax.set_ylabel("# cells")
ax.set_xlabel(f"Evidence kernel factor score  ")
ax.set_title(f"Cell factor scores for\nfirst singular vector")
ax.plot(np.repeat(h[1], 2), [0]+list(np.repeat(h[0], 2))+[0], c='k', linewidth=.5)
sns.despine(ax=ax)
print("N in hist > 0:", sum(pc_weights>0), "/", len(pc_weights))

# ax = c.ax("varplot")
# bars = ax.bar(range(1, 1+len(pcs[0])), pcs[0]*100, color='k')
# for i in range(0, 4):
#     bars[i].set_color(diplib.pc_colors(i))

# ax.set_title("Evidence kernel SV\nexplained variance")
# ax.set_xlabel("SV #")
# ax.set_xticks([1, 2, 3, 4])
# ax.set_ylabel("Explanatory power")
# sns.despine(ax=ax)

if MONKEY == "Q":
    c.save(f"figure4.pdf")
else:
    c.save(f"figureS4.pdf")

