"""Rebuild the manuscript's full figure sequence from recorded results.

Run from an environment with numpy and matplotlib. This script intentionally
uses the implementation's index-radius band mask, not an idealized FFT ring.
"""
from pathlib import Path
import json
import csv
import argparse
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import ListedColormap

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "source_data"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=HERE / "output")
OUTPUT = parser.parse_args().output.resolve()
OUTPUT.mkdir(parents=True, exist_ok=True)

def read_rows(name):
    with (DATA / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))

from audit_panel_alignment import require_matplotlib_panel_alignment

NAVY, BLUE, LIGHT, INK, MID, GRID = "#174d70", "#2c79a1", "#e2f0f6", "#18313f", "#58707d", "#d4e0e6"
ORANGE, PALE = "#bc6534", "#f3f6f7"
mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "svg.fonttype": "none", "pdf.fonttype": 42,
    "axes.linewidth": 0.7, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.facecolor": "white",
})


def export(fig, name, *, multi=False):
    out = OUTPUT / name
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig, json_out=str(out) + ".alignment.json",
                                       tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    fig.savefig(str(out) + ".svg")
    fig.savefig(str(out) + ".pdf")
    fig.savefig(str(out) + ".tiff", dpi=600)
    fig.savefig(str(out) + "_preview.png", dpi=300)
    plt.close(fig)


def box(ax, xy, wh, title, detail="", *, active=False):
    x, y = xy; w, h = wh
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.017",
                                facecolor=LIGHT if active else PALE,
                                edgecolor=BLUE if active else "#9aaeb9", linewidth=1.0,
                                transform=ax.transAxes))
    ax.text(x+w/2, y+h/2+(0.028 if detail else 0), title, transform=ax.transAxes,
            ha="center", va="center", fontsize=8, color=INK, fontweight="bold" if active else "normal")
    if detail:
        ax.text(x+w/2, y+h/2-0.033, detail, transform=ax.transAxes,
                ha="center", va="center", fontsize=7.5, color=INK)


def arrow(ax, start, end, *, active=False):
    ax.add_patch(FancyArrowPatch(start, end, transform=ax.transAxes,
                 arrowstyle="-|>", mutation_scale=8, lw=1.0,
                 color=BLUE if active else MID, shrinkA=0, shrinkB=0))


def fig1():
    # Two aligned modality rows merge into one left-to-right output path.
    fig, ax = plt.subplots(figsize=(7.18, 2.60))
    fig.subplots_adjust(left=.025, right=.975, bottom=.06, top=.94)
    ax.axis("off")
    x = [.02, .165, .33, .515, .755, .89]
    w = [.11, .13, .14, .205, .105, .095]
    yt, yb, h = .68, .20, .22
    for row, y in [("TEXT",yt),("IMAGE",yb)]:
        ax.text(x[0], y+h+.045, row, transform=ax.transAxes, fontsize=8,
                fontweight="bold", color=MID, va="bottom")
    box(ax,(x[0],yt),(w[0],h),"Text","tokens")
    box(ax,(x[1],yt),(w[1],h),"Text encoder","word vectors")
    box(ax,(x[2],yt),(w[2],h),"1D rFFT","filter")
    box(ax,(x[0],yb),(w[0],h),"Image","patches")
    box(ax,(x[1],yb),(w[1],h),"Patch encoder","14 × 14 grid")
    box(ax,(x[2],yb),(w[2],h),"rFFT2 + gains","index-radius bands",active=True)
    box(ax,(x[3],.36),(w[3],.36),"Bidirectional gate","spectral energy exchange")
    box(ax,(x[4],.40),(w[4],.28),"Inverse FFT","residual + pool")
    box(ax,(x[5],.40),(w[5],.28),"JS + MLP","prediction")
    for y in (yt,yb):
        arrow(ax,(x[0]+w[0]+.005,y+h/2),(x[1]-.005,y+h/2))
        arrow(ax,(x[1]+w[1]+.005,y+h/2),(x[2]-.005,y+h/2))
    arrow(ax,(x[2]+w[2]+.005,yt+h/2),(x[3]-.005,.635))
    arrow(ax,(x[2]+w[2]+.005,yb+h/2),(x[3]-.005,.445))
    arrow(ax,(x[3]+w[3]+.006,.54),(x[4]-.006,.54))
    arrow(ax,(x[4]+w[4]+.006,.54),(x[5]-.006,.54))
    ax.text(.02,.055,"Blue = added 2D band operation",transform=ax.transAxes,
            fontsize=7.5,color=MID)
    export(fig,"fig1_architecture")


def fig2():
    h, wr = 14, 8
    fy = np.arange(h)[:,None] / (h-1)
    fx = np.arange(wr)[None,:] / (wr-1)
    radius = np.sqrt(fy**2 + fx**2)
    fig, axes = plt.subplots(1,2,figsize=(3.50,2.90))
    fig.subplots_adjust(left=.11,right=.98,bottom=.13,top=.93,wspace=.29)
    colors=["#dbeef5","#94c4d8","#5595ba","#174d70"]
    for ax,B in zip(axes,(2,4)):
        mask=np.minimum((radius*B).astype(int),B-1)
        ax.imshow(mask,origin="lower",interpolation="nearest",aspect="equal",
                  cmap=ListedColormap(colors[:B]),vmin=0,vmax=B-1)
        ax.set_xticks(np.arange(-.5,wr,1),minor=True)
        ax.set_yticks(np.arange(-.5,h,1),minor=True)
        ax.grid(which="minor",color="white",lw=.42)
        ax.tick_params(which="minor",bottom=False,left=False)
        ax.set_xticks([0,3,7]);ax.set_yticks([0,6,13])
        ax.tick_params(labelsize=7.5,length=2)
        ax.set_title(f"B = {B}",fontsize=8,fontweight="bold",pad=5)
        ax.set_xlabel("Horizontal index",fontsize=8)
        ax.set_ylabel("Vertical index",fontsize=8)
        ax.plot(0,0,marker="o",ms=3.2,color=ORANGE)
        for spine in ax.spines.values():spine.set_visible(False)
    export(fig,"fig2_band_partition",multi=True)


def fig3():
    rows = read_rows("figure3_matched_differences.csv")
    runs = {(int(r["fold_zero_based"]), int(r["seed"])): r for r in rows}
    assert len(runs) == 15
    fig,ax=plt.subplots(figsize=(3.50,2.75))
    fig.subplots_adjust(left=.275,right=.97,bottom=.19,top=.95)
    for fold in range(5):
        for seed,offset in [(30,-.14),(31,0),(32,.14)]:
            delta=float(runs[fold,seed]["difference_percentage_points"])
            ax.plot(delta,5-fold+offset,"o",ms=3.2,color=BLUE,mec="white",mew=.35)
    # Recorded group-bootstrap output; do not estimate uncertainty from the 15 dots.
    bootstrap = read_rows("figure3_group_bootstrap.csv")[0]
    est,lo,hi = [round(100 * float(bootstrap[k]), 3)
                 for k in ("difference", "ci95_low", "ci95_high")]
    ax.errorbar(est,6.1,xerr=[[est-lo],[hi-est]],fmt="D",ms=4.5,color=NAVY,capsize=2.5,lw=1.2)
    ax.axvline(0,color=MID,lw=.8)
    ax.set_yticks([1,2,3,4,5,6.1],["Fold 5","Fold 4","Fold 3","Fold 2","Fold 1","Group bootstrap"])
    ax.set_xlim(-2.9,2.9);ax.set_ylim(.5,6.55)
    ax.set_xlabel("Spectral − FFN macro F1 (percentage points)",fontsize=8)
    ax.tick_params(length=2.5)
    ax.spines["left"].set_visible(False)
    export(fig,"fig3_decisive_ablation")


def fig4():
    # Counts cross-checked against Table 5 (sum = 7,723).
    counts=np.zeros((2,2),dtype=int)
    for row in read_rows("figure4_prediction_overlap.csv"):
        counts[1-int(row["fsru_correct"]),1-int(row["fsru_2d_correct"])]=int(row["n_samples"])
    assert counts.sum()==7723
    fig,ax=plt.subplots(figsize=(3.50,2.70))
    fig.subplots_adjust(left=.25,right=.96,bottom=.13,top=.95)
    from matplotlib.colors import ListedColormap
    ax.imshow([[0,1],[1,2]],cmap=ListedColormap(["#dcebf2","#9ec7d8","#f0d2bf"]),vmin=0,vmax=2)
    for i in range(2):
        for j in range(2):
            ax.text(j,i,f"{counts[i,j]:,}\n{100*counts[i,j]/7723:.1f}%",ha="center",va="center",fontsize=8,color=INK)
    ax.set_xticks([0,1],["Correct","Wrong"]);ax.set_yticks([0,1],["Correct","Wrong"])
    ax.set_xlabel("FSRU-2D correctness",fontsize=8);ax.set_ylabel("FSRU correctness",fontsize=8)
    ax.tick_params(length=0,pad=5)
    ax.plot([.5,.5],[-.5,1.5],color="white",lw=3,solid_capstyle="butt")
    ax.plot([-.5,1.5],[.5,.5],color="white",lw=3,solid_capstyle="butt")
    for spine in ax.spines.values():spine.set_visible(False)
    export(fig,"fig4_error_overlap")


def fig5():
    rows = read_rows("figure5_band_gains.csv")
    assert len(rows) == 15
    low=np.array([float(r["low_band_gain"]) for r in rows])
    high=np.array([float(r["high_band_gain"]) for r in rows])
    ratios=high/low
    baseline=read_rows("figure5_baseline_gain_ratio.csv")[0]
    baseline_mean=float(baseline["mean"])
    baseline_sd=float(baseline["sample_sd"])
    fig,axes=plt.subplots(1,2,figsize=(7.18,2.75))
    fig.subplots_adjust(left=.075,right=.97,bottom=.22,top=.86,wspace=.30)
    ax=axes[0]
    for i,(a,b) in enumerate(zip(low,high)):
        ax.plot([0,1],[a,b],color="#b6cbd5",lw=.7,zorder=1)
        ax.plot(0,a,"o",ms=3,color=BLUE,zorder=2)
        ax.plot(1,b,"o",ms=3,color=ORANGE,zorder=2)
    ax.set_xticks([0,1],["Low band","High band"]);ax.set_xlim(-.25,1.25)
    ax.set_ylabel("Mean gain magnitude",fontsize=8)
    ax.set_title("a  Paired gains across 15 models",loc="left",fontsize=8,fontweight="bold")
    ax=axes[1]
    ax.axhline(1,color=MID,lw=.7,ls="--")
    ax.scatter(np.arange(1,16),ratios,s=10,color=BLUE)
    ax.hlines(baseline_mean,1,15,color=ORANGE,lw=1.3)
    ax.fill_between([1,15],[baseline_mean-baseline_sd]*2,[baseline_mean+baseline_sd]*2,color=ORANGE,alpha=.13)
    ax.set_xlim(.5,15.5);ax.set_ylim(.68,1.47)
    ax.set_xticks([1,5,10,15]);ax.set_xlabel("FSRU-2D model index",fontsize=8)
    ax.set_ylabel("High / low gain ratio",fontsize=8)
    ax.set_title("b  Band preference varies by model",loc="left",fontsize=8,fontweight="bold")
    fig.text(.51,.96,"FSRU baseline ratio: 0.994 ± 0.009 (orange line ± SD)",fontsize=7.5,
             ha="center",va="top",color=MID)
    for ax in axes:
        ax.grid(axis="y",color=GRID,lw=.6)
        ax.tick_params(length=2.5)
    export(fig,"fig5_gain_mechanism",multi=True)


if __name__=="__main__":
    fig1();fig2();fig3();fig4();fig5()
    print("Rendered five figures in",OUTPUT)
