"""Generate the 5 ILLUSTRATIVE experiment figures used as stand-ins in the report.

These plots are NOT measured results. No log or dataset is read: every curve,
bar and error value is a hard-coded assumption, some with added random noise
(np.random.seed(21)). They were drawn so the report layout could be completed
before Experiments 2, 3, 4, 6 and 8 were run, and each one is captioned as
illustrative in experiments_and_results.tex. Replace them with plots made from
real logs once the experiments are carried out.

Run from docs/report/:  python3 tools/gen_all_missing.py
Writes Figures/report/exp0{2,3,4,6,8}_*.png
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch, FancyArrowPatch
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec

np.random.seed(21)
BLUE   = '#0090B5'
ORANGE = '#E07B2A'
GREEN  = '#3A8A4A'
RED    = '#C0392B'
PURPLE = '#7B68EE'
GRAY   = '#888888'

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,
                     'axes.titlesize':10,'axes.labelsize':9,
                     'xtick.labelsize':8,'ytick.labelsize':8,
                     'legend.fontsize':8,'figure.dpi':150})

# ══════════════════════════════════════════════════════
# FIG 1 – Exp02: Pose-estimation accuracy
# ══════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

ranges  = [0.20, 0.35, 0.50]
yaw_lbl = ['−20°', '0°', '+20°']
light   = ['Bright', 'Normal', 'Dim']

# translation error (m) per range, averaged over yaw and light
te_bright = np.array([0.008, 0.012, 0.021])
te_normal = np.array([0.011, 0.016, 0.028])
te_dim    = np.array([0.018, 0.025, 0.041])
ye_bright = np.array([1.2,  1.8,  3.1])   # yaw error (deg)
ye_normal = np.array([1.6,  2.3,  4.0])
ye_dim    = np.array([2.5,  3.6,  6.2])

ax = axes[0]
for vals, lbl, clr, ls in [(te_bright,'Bright',BLUE,'-'),
                             (te_normal,'Normal',ORANGE,'--'),
                             (te_dim,   'Dim',   RED,  ':')]:
    ax.plot(ranges, vals*100, color=clr, lw=2, ls=ls, marker='o', ms=6, label=lbl)
ax.axhline(y=1.5, color=GRAY, lw=1, ls='--', alpha=0.7)
ax.text(0.51, 1.6, 'capture range', color=GRAY, fontsize=7.5)
ax.set_xlabel('Camera-to-Socket Range (m)')
ax.set_ylabel('Translation Error (cm)')
ax.set_title('(a) Translation Error vs. Range', fontweight='bold')
ax.legend(title='Lighting')
ax.grid(alpha=0.25, ls='--')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

ax2 = axes[1]
x = np.arange(3)
w = 0.25
for i,(vals,lbl,clr) in enumerate([(ye_bright,'Bright',BLUE),
                                     (ye_normal,'Normal',ORANGE),
                                     (ye_dim,   'Dim',   RED)]):
    bars = ax2.bar(x + i*w - w, vals, w, color=clr, alpha=0.8, label=lbl)
    for b,v in zip(bars,vals):
        ax2.text(b.get_x()+b.get_width()/2, v+0.15, f'{v:.1f}°',
                 ha='center', fontsize=7, color=clr, fontweight='bold')
ax2.set_xlabel('Camera-to-Socket Range (m)')
ax2.set_ylabel('Yaw Error (degrees)')
ax2.set_title('(b) Yaw Error vs. Range', fontweight='bold')
ax2.set_xticks(x); ax2.set_xticklabels(['0.20 m','0.35 m','0.50 m'])
ax2.legend(title='Lighting')
ax2.grid(axis='y', alpha=0.25, ls='--')
ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

fig.tight_layout(pad=2)
fig.savefig('Figures/report/exp02_pose_error.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('exp02 done')

# ══════════════════════════════════════════════════════
# FIG 2 – Exp03: Navigation performance
# ══════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
scenarios = ['Direct\nPath', 'Static\nObstacle', 'Moving\nObstacle']
n = 10
sr  = [9, 9, 8]   # successes out of 10
pos_err_mean = [0.072, 0.089, 0.113]  # m
pos_err_std  = [0.018, 0.022, 0.031]
hdg_err_mean = [2.8, 3.4, 5.1]        # deg
hdg_err_std  = [0.7, 0.9, 1.4]
x = np.arange(3)
w = 0.35

ax = axes[0]
bars1 = ax.bar(x - w/2, [v/10*100 for v in sr], w, color=BLUE, alpha=0.8, label='Success Rate (%)')
ax2b  = ax.twinx()
tt = [44, 61, 79]
bars2 = ax2b.bar(x + w/2, tt, w, color=ORANGE, alpha=0.8, label='Travel Time (s)')
ax.set_ylabel('Navigation Success Rate (%)', color=BLUE)
ax2b.set_ylabel('Mean Travel Time (s)', color=ORANGE)
ax.set_ylim(0, 105); ax2b.set_ylim(0, 120)
ax.tick_params(axis='y', labelcolor=BLUE)
ax2b.tick_params(axis='y', labelcolor=ORANGE)
ax.set_xticks(x); ax.set_xticklabels(scenarios)
ax.set_title('(a) Success Rate and Travel Time', fontweight='bold')
for b,v in zip(bars1,[90,90,80]): ax.text(b.get_x()+b.get_width()/2,v+1,f'{v}%',ha='center',fontsize=8,color=BLUE,fontweight='bold')
for b,v in zip(bars2,tt): ax2b.text(b.get_x()+b.get_width()/2,v+1,f'{v}s',ha='center',fontsize=8,color=ORANGE,fontweight='bold')
ax.legend([Line2D([0],[0],color=BLUE,lw=6,alpha=0.8),Line2D([0],[0],color=ORANGE,lw=6,alpha=0.8)],
          ['Success Rate (%)','Travel Time (s)'], loc='lower right')
ax.spines['top'].set_visible(False); ax2b.spines['top'].set_visible(False)
ax.grid(axis='y',alpha=0.2,ls='--',zorder=0)

ax3 = axes[1]
for i,(pm,ps,clr,lbl) in enumerate([(pos_err_mean,pos_err_std,BLUE,'Position Error (m)'),
                                      ([v/57.3 for v in hdg_err_mean],[v/57.3 for v in hdg_err_std],
                                       ORANGE,'Heading Error (rad)')]):
    ax3.bar(x + i*0.32 - 0.16, pm, 0.3, yerr=ps, color=clr, alpha=0.8,
            capsize=4, label=lbl)
ax3.set_xticks(x); ax3.set_xticklabels(scenarios)
ax3.set_ylabel('Final Approach Error')
ax3.set_title('(b) Final Position and Heading Error (95% CI)', fontweight='bold')
ax3.legend()
ax3.grid(axis='y',alpha=0.25,ls='--')
ax3.spines['top'].set_visible(False); ax3.spines['right'].set_visible(False)

fig.tight_layout(pad=2)
fig.savefig('Figures/report/exp03_nav_results.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('exp03 done')

# ══════════════════════════════════════════════════════
# FIG 3 – Exp04: Arm workspace error map + orientation
# ══════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

# 5×3 workspace grid (left, centre, right) × (near, mid, far)
gx = np.array([-0.12, -0.06, 0.00, 0.06, 0.12])
gy = np.array([0.15, 0.25, 0.35])
GX, GY = np.meshgrid(gx, gy)
# error increases toward edges and far reach
err_map = (0.004 + 0.003*(GX/0.12)**2 + 0.004*(GY/0.35)**2
           + 0.003*np.abs(GX/0.12) + np.random.normal(0,0.001,GX.shape))
err_map = np.clip(err_map*1000, 3, 18)  # in mm

ax = axes[0]
sc = ax.scatter(GX.ravel(), GY.ravel(), c=err_map.ravel(),
                s=400, cmap='RdYlGn_r', vmin=3, vmax=18, zorder=5,
                edgecolors='k', linewidths=0.5)
for xi,yi,ei in zip(GX.ravel(), GY.ravel(), err_map.ravel()):
    ax.text(xi, yi, f'{ei:.0f}', ha='center', va='center',
            fontsize=7, fontweight='bold', color='white' if ei>11 else 'black')
cb = fig.colorbar(sc, ax=ax, shrink=0.85)
cb.set_label('Mean Position Error (mm)')
ax.set_xlabel('Lateral offset (m)'); ax.set_ylabel('Forward reach (m)')
ax.set_title('(a) Workspace Error Map — Plug-Tip Position', fontweight='bold')
ax.text(-0.12, 0.08, 'LEFT', ha='center', fontsize=7.5, color=BLUE, fontstyle='italic')
ax.text(0.00,  0.08, 'CENTRE', ha='center', fontsize=7.5, color=GREEN, fontstyle='italic')
ax.text(0.12,  0.08, 'RIGHT', ha='center', fontsize=7.5, color=ORANGE, fontstyle='italic')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

ax2 = axes[1]
zones = ['Left', 'Centre', 'Right']
ori_data = [np.random.normal(3.1, 0.7, 15), np.random.normal(1.8, 0.5, 15),
            np.random.normal(3.4, 0.8, 15)]
bp = ax2.boxplot(ori_data, patch_artist=True, widths=0.5,
                 boxprops=dict(facecolor=BLUE, alpha=0.7),
                 medianprops=dict(color='white', lw=2),
                 whiskerprops=dict(color=BLUE),
                 capprops=dict(color=BLUE),
                 flierprops=dict(marker='o', color=BLUE, ms=4, alpha=0.5))
ax2.set_xticklabels(zones)
ax2.set_ylabel('Plug Orientation Error (degrees)')
ax2.set_title('(b) Passive-Wrist Orientation Error by Zone', fontweight='bold')
ax2.axhline(y=5, color=RED, ls='--', lw=1.2, alpha=0.7)
ax2.text(3.4, 5.2, 'spec limit', color=RED, fontsize=7.5)
ax2.grid(axis='y', alpha=0.25, ls='--')
ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
for i,(d,z) in enumerate(zip(ori_data,zones)):
    ax2.text(i+1, np.median(d)+0.2, f'{np.median(d):.1f}°',
             ha='center', fontsize=8, color=BLUE, fontweight='bold')

fig.tight_layout(pad=2)
fig.savefig('Figures/report/exp04_arm_results.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('exp04 done')

# ══════════════════════════════════════════════════════
# FIG 4 – Exp06: Domain-randomization robustness
# ══════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
conditions = ['Nominal\nSimulation', 'Camera Noise\n& Latency', 'Friction &\nSocket Offset', 'All\nPerturbations']
acc_nodr = [85.2, 52.4, 48.8, 28.1]
acc_dr   = [87.4, 79.2, 73.6, 65.3]
drop_nodr = [0, 32.8, 36.4, 57.1]
drop_dr   = [0,  8.2, 13.8, 22.1]

x = np.arange(4); w = 0.32
ax = axes[0]
b1 = ax.bar(x - w/2, acc_nodr, w, color=ORANGE, alpha=0.8, label='No DR Policy')
b2 = ax.bar(x + w/2, acc_dr,   w, color=BLUE,   alpha=0.8, label='DR Policy')
for b,v in zip(b1,acc_nodr): ax.text(b.get_x()+b.get_width()/2,v+0.5,f'{v:.0f}%',ha='center',fontsize=7.5,color=ORANGE,fontweight='bold')
for b,v in zip(b2,acc_dr):   ax.text(b.get_x()+b.get_width()/2,v+0.5,f'{v:.0f}%',ha='center',fontsize=7.5,color=BLUE,fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(conditions, fontsize=8)
ax.set_ylabel('Terminal Accuracy (%)')
ax.set_title('(a) Accuracy under Perturbation Conditions', fontweight='bold')
ax.set_ylim(0, 100); ax.legend(loc='lower left')
ax.grid(axis='y',alpha=0.25,ls='--')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

ax2 = axes[1]
fail_cats_nodr = [[0,0,0,0],[20,12,6,5],[18,14,10,7],[40,22,18,14],[7,4,10,7]]
fail_cats_dr   = [[0,0,0,0],[ 4, 5,3,3],[ 2, 4, 5,5],[14,10,12,9],[1,3, 6,5]]
cat_labels = ['Timeout','Target\nNot Reached','Oscillation','Face-plate\nCross','Joint\nLimit']
cat_colors = [GRAY, ORANGE, RED, PURPLE, GREEN]
bottom_nodr = np.zeros(4); bottom_dr = np.zeros(4)
for cats_n, cats_d, lbl, clr in zip(fail_cats_nodr, fail_cats_dr, cat_labels, cat_colors):
    ax2.bar(x - w/2, cats_n, w, bottom=bottom_nodr, color=clr, alpha=0.8,
            label=f'{lbl}' if x[0]==0 else '')
    ax2.bar(x + w/2, cats_d, w, bottom=bottom_dr,   color=clr, alpha=0.5)
    bottom_nodr += np.array(cats_n); bottom_dr += np.array(cats_d)
ax2.set_xticks(x); ax2.set_xticklabels(conditions, fontsize=8)
ax2.set_ylabel('Failure Count (per 250 episodes)')
ax2.set_title('(b) Failure Taxonomy (solid=No DR, faded=DR)', fontweight='bold')
ax2.legend(loc='upper left', fontsize=7.5)
ax2.grid(axis='y',alpha=0.25,ls='--')
ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

fig.tight_layout(pad=2)
fig.savefig('Figures/report/exp06_dr_robustness.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('exp06 done')

# ══════════════════════════════════════════════════════
# FIG 5 – Exp08: End-to-end mission timeline
# ══════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 1, figsize=(12, 6.5))

# ── Panel (a): top-down map with trajectory ──────────
ax = axes[0]
ax.set_xlim(0,12); ax.set_ylim(0,8); ax.set_aspect('equal')
ax.set_facecolor('#F8F8F8')
# walls / room outline
for rect in [([0,0],[12,0]),([12,0],[12,8]),([12,8],[0,8]),([0,8],[0,0])]:
    ax.plot([rect[0][0],rect[1][0]],[rect[0][1],rect[1][1]],'k-',lw=2)
# obstacle (pillar)
pillar = plt.Rectangle((5.5,3.5),1,1, color='#444444')
ax.add_patch(pillar)
ax.text(6,4,'Pillar',ha='center',va='center',color='white',fontsize=7)
# charging station
station = plt.Rectangle((10.5,3.5),1.2,1.2,color=BLUE,alpha=0.6)
ax.add_patch(station)
ax.text(11.1,4.1,'Socket',ha='center',va='center',color='white',fontsize=7,fontweight='bold')
# trajectory: smooth curve from start to socket
t = np.linspace(0,1,200)
tx = 1.5 + 9.0*t + 0.3*np.sin(4*np.pi*t)*(1-t)
ty = 4.0 + 0.8*np.sin(np.pi*t) + 0.2*np.sin(8*np.pi*t)*(1-t)
ax.plot(tx, ty, color=BLUE, lw=2, label='Base trajectory')
# phase markers
phases = [(0,'Start',GREEN),(0.35,'Socket\nSearch',ORANGE),(0.70,'Base\nRefine',PURPLE),(0.95,'Insertion',RED)]
for frac,lbl,clr in phases:
    idx = int(frac*199)
    ax.scatter(tx[idx],ty[idx],color=clr,s=80,zorder=10)
    ax.text(tx[idx]+0.2,ty[idx]+0.3,lbl,fontsize=7.5,color=clr,fontweight='bold')
ax.scatter([1.5],[4.0],s=120,color=GREEN,marker='o',zorder=11,label='Start')
ax.scatter([10.8],[4.1],s=150,color=RED,marker='*',zorder=11,label='Goal')
ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
ax.set_title('(a) Top-Down Map: Full Base Trajectory from Start to Charging Station', fontweight='bold')
ax.legend(loc='upper left',fontsize=8)

# ── Panel (b): state-machine timeline ────────────────
ax2 = axes[1]
phases_tl = [
    ('SLAM / Navigation',       0,  42, BLUE),
    ('Socket Search',          42,  12, ORANGE),
    ('Base Refinement',        54,   8, PURPLE),
    ('Arm Alignment',          62,  11, GREEN),
    ('Insertion',              73,   9, RED),
]
mission_outcomes = ['SUCCESS' for _ in range(3)] + ['NAV RECOVERY'] + ['SUCCESS']
for y_idx, (lbl, start, dur, clr) in enumerate(phases_tl):
    ax2.barh(y_idx, dur, left=start, color=clr, alpha=0.8, height=0.6,
             edgecolor='white', linewidth=0.8)
    ax2.text(start + dur/2, y_idx, f'{dur}s', ha='center', va='center',
             color='white', fontsize=8.5, fontweight='bold')
ax2.set_yticks(range(len(phases_tl)))
ax2.set_yticklabels([p[0] for p in phases_tl])
ax2.set_xlabel('Time (seconds)')
ax2.set_title('(b) State-Machine Phase Timeline — Single Representative Mission (82 s total, outcome: SUCCESS)',
              fontweight='bold')
ax2.set_xlim(0, 95)
ax2.axvline(82, color='k', ls='--', lw=1.2)
ax2.text(83, 4.4, 'Mission end\n82 s', fontsize=7.5)
ax2.grid(axis='x', alpha=0.25, ls='--')
ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

fig.tight_layout(pad=2.5)
fig.savefig('Figures/report/exp08_timeline.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('exp08 done')
print('All figures saved.')
