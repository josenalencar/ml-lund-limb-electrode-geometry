"""
ML x Lund — final geometric analysis (post-audit, N=12).

Reads the audited placements (../data/lund_picks_corrected.json) and computes:
  (1) Frontal cardiac AXIS shift (ML vs Lund) — hexaxial method.
  (2) WCT displacement (ML - Lund) in frontal / sagittal / transverse planes.
  (3) LIMB-LEAD amplitude comparison (ML vs Lund) — lead-vector magnitude ratio
      and predicted amplitude change for a swept frontal cardiac dipole.

Convention (raw): +X = patient's left, +Y = posterior, +Z = superior. WCT=(RA+LA+LL)/3.
Geometry-only subjects -> the cardiac dipole is held FIXED; only electrodes move.

Outputs -> ../results/{final_geometry.csv, SUMMARY.md}
           ../figures/analysis/{wct_shift_*.png, axis_shift.png, limb_lead_comparison.png}
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJ    = Path(__file__).resolve().parents[1]
DATA    = PROJ / "data"
RESULTS = PROJ / "results"
FIGA    = PROJ / "figures" / "analysis"
RESULTS.mkdir(parents=True, exist_ok=True); FIGA.mkdir(parents=True, exist_ok=True)
import charles_io
DS = charles_io.data_root()

ORDER = [
    ("Charles pt027","charles","pt027"),("Charles pt028","charles","pt028"),
    ("Charles pt029","charles","pt029"),
    ("ECGI P001","ecgi","P001"),("ECGI P002","ecgi","P002"),("ECGI P004","ecgi","P004"),
    ("ECGI P006","ecgi","P006"),("ECGI P010","ecgi","P010"),("ECGI P020","ecgi","P020"),
    ("ECGI P023","ecgi","P023"),("ECGI P024","ecgi","P024"),("ECGI P027","ecgi","P027"),
]
LIMB = ["I","II","III","aVR","aVL","aVF"]

def force_Nx3(a):
    a = np.asarray(a)
    return a.T if (a.ndim == 2 and a.shape[0] == 3 and a.shape[1] > 3) else a

def load_charles(pid):
    num = pid[2:]; suf = num.lstrip("0") or "0"
    g = sio.loadmat(str(DS/f"{num}/Meshes/2012-07-{suf}Subject_daltorso_registered.mat"),
                    squeeze_me=True, struct_as_record=False)["geom"]
    h = sio.loadmat(str(DS/f"{num}/Meshes/2012-07-{suf}Subject_myocardium_registered.mat"),
                    squeeze_me=True, struct_as_record=False)
    hs = h[[k for k in h if not k.startswith("__")][0]]
    return force_Nx3(g.node), force_Nx3(hs.node if hasattr(hs,"node") else hs)

def load_ecgi(pid):
    t = sio.loadmat(str(DS/f"ECGI_Challenge_2026_Training/Geometries/{pid}_torso.mat"),
                    squeeze_me=True, struct_as_record=False)["torso"]
    e = sio.loadmat(str(DS/f"ECGI_Challenge_2026_Training/Geometries/{pid}_EndoEpi.mat"),
                    squeeze_me=True, struct_as_record=False)
    es = e[[k for k in e if not k.startswith("__")][0]]
    return force_Nx3(t.node), force_Nx3(es.node if hasattr(es,"node") else es)

def leads(RA, LA, LL):
    wct = (RA + LA + LL) / 3.0
    return wct, {"I":LA-RA,"II":LL-RA,"III":LL-LA,"aVR":RA-wct,"aVL":LA-wct,"aVF":LL-wct}

def frontal_angle(v):  # hexaxial: I=0, aVF=+90
    return float(np.degrees(np.arctan2(-v[2], v[0])))

def unwrap(d):
    while d > 180: d -= 360
    while d <= -180: d += 360
    return d

def apparent_axis(I_vec, aVF_vec, theta_deg):
    th = np.radians(theta_deg)
    H = np.array([np.cos(th), 0.0, -np.sin(th)])
    iI = I_vec/np.linalg.norm(I_vec); iF = aVF_vec/np.linalg.norm(aVF_vec)
    return float(np.degrees(np.arctan2(H@iF, H@iI)))

AMPSWEEP = np.arange(0, 360, 10)           # full frontal circle for amplitude comparison

def main():
    picks = json.loads((DATA/"lund_picks_corrected.json").read_text())  # audited placements
    rows, geom = [], {}
    for label, cohort, pid in ORDER:
        torso, heart = (load_charles(pid) if cohort=="charles" else load_ecgi(pid))
        c = torso.mean(0); hc = heart.mean(0)
        p = {k: np.array(picks[label][k], float) for k in picks[label]}
        wct_L, lead_L = leads(p["lund_RA"], p["lund_LA"], p["lund_LL"])
        wct_M, lead_M = leads(p["ml_RA"],   p["ml_LA"],   p["ml_LL"])

        d = wct_M - wct_L
        dX, dYpost, dZ = float(d[0]), float(d[1]), float(d[2]); dYant = -dYpost
        row = {"patient":label,"dX_left_mm":dX,"dY_anterior_mm":dYant,"dZ_sup_mm":dZ,
               "WCT_3D_mm":float(np.linalg.norm(d)),
               "WCT_frontal_mm":float(np.hypot(dX,dZ)),
               "WCT_sagittal_mm":float(np.hypot(dYant,dZ)),
               "WCT_transverse_mm":float(np.hypot(dX,dYpost))}
        # lead rotations + apparent axis
        for k in LIMB:
            row[f"rot_{k}"] = unwrap(frontal_angle(lead_M[k]) - frontal_angle(lead_L[k]))
        row["axis_Lund_60"] = apparent_axis(lead_L["I"], lead_L["aVF"], 60)
        row["axis_ML_60"]   = apparent_axis(lead_M["I"], lead_M["aVF"], 60)
        row["axis_shift_60"]= unwrap(row["axis_ML_60"] - row["axis_Lund_60"])

        # ---- (3) LIMB-LEAD amplitude comparison ----
        for k in LIMB:
            vL, vM = lead_L[k], lead_M[k]
            magL, magM = float(np.linalg.norm(vL)), float(np.linalg.norm(vM))
            row[f"mag_Lund_{k}"] = magL
            row[f"mag_ML_{k}"]   = magM
            row[f"magratio_{k}"] = magM/magL if magL else np.nan
            # amplitude of each lead for unit frontal dipoles swept over the circle
            aL, aM = [], []
            for t in AMPSWEEP:
                th = np.radians(t); H = np.array([np.cos(th),0.0,-np.sin(th)])
                aL.append(H@vL); aM.append(H@vM)
            aL, aM = np.array(aL), np.array(aM)
            row[f"amp_pctdiff_{k}"] = 100.0*np.mean(np.abs(aM-aL))/(np.mean(np.abs(aL))+1e-9)
        rows.append(row)
        geom[label] = {"wctL":wct_L-c, "wctM":wct_M-c, "heart":hc-c}

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS/"final_geometry.csv", index=False)
    def ms(c): return df[c].mean(), df[c].std()

    # ---- WCT plane figures ----
    planes = [("frontal",0,2,"X — lateral (mm) [+left]","Z — sup/inf (mm)"),
              ("sagittal",1,2,"Y — post/ant (mm) [+post]","Z — sup/inf (mm)"),
              ("transverse",0,1,"X — lateral (mm) [+left]","Y — post/ant (mm) [+post]")]
    for name,a,b,xl,yl in planes:
        fig,ax=plt.subplots(figsize=(8.5,8.5))
        for g in geom.values():
            ax.annotate("",xy=(g["wctM"][a],g["wctM"][b]),xytext=(g["wctL"][a],g["wctL"][b]),
                        arrowprops=dict(arrowstyle="->",color="#0d6b6e",lw=1.6,alpha=.8))
            ax.scatter([g["wctL"][a]],[g["wctL"][b]],s=22,c="blue",zorder=4)
            ax.scatter([g["wctM"][a]],[g["wctM"][b]],s=22,c="red",zorder=4)
            ax.scatter([g["heart"][a]],[g["heart"][b]],s=40,c="crimson",marker="*",alpha=.5,zorder=3)
        ax.axhline(0,color="grey",lw=.5,ls="--"); ax.axvline(0,color="grey",lw=.5,ls="--")
        ax.set_aspect("equal"); ax.grid(alpha=.25); ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.set_title(f"WCT shift  Lund(blue)→ML(red)  —  {name.upper()} (n=12)\npink star = heart center")
        fig.tight_layout(); fig.savefig(FIGA/f"wct_shift_{name}.png",dpi=130); plt.close(fig)

    # ---- axis-shift figure ----
    fig,ax=plt.subplots(figsize=(10,5)); o=df.sort_values("axis_shift_60")
    ax.bar(range(len(o)),o["axis_shift_60"],color="#c2701c")
    ax.axhline(o["axis_shift_60"].mean(),color="black",ls="--",
               label=f"mean={o['axis_shift_60'].mean():+.1f}°")
    ax.set_xticks(range(len(o))); ax.set_xticklabels(o["patient"],rotation=45,ha="right",fontsize=8)
    ax.set_ylabel("frontal QRS-axis shift ML−Lund (°)\n[fixed dipole +60°]")
    ax.set_title("Frontal cardiac-axis shift, ML vs Lund (hexaxial)"); ax.legend(); ax.grid(axis="y",alpha=.3)
    fig.tight_layout(); fig.savefig(FIGA/"axis_shift.png",dpi=130); plt.close(fig)

    # ---- limb-lead comparison figure (magnitude ratio + amplitude %diff) ----
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13,5))
    mr=[df[f"magratio_{k}"].mean() for k in LIMB]; mrsd=[df[f"magratio_{k}"].std() for k in LIMB]
    ax1.bar(LIMB,mr,yerr=mrsd,capsize=4,color="#0d6b6e")
    ax1.axhline(1.0,color="black",ls="--",lw=1); ax1.set_ylabel("lead-vector magnitude ratio  ML / Lund")
    ax1.set_title("Limb-lead amplitude scale (ML vs Lund)\n>1 = ML inflates, <1 = ML deflates"); ax1.grid(axis="y",alpha=.3)
    pd_=[df[f"amp_pctdiff_{k}"].mean() for k in LIMB]; pdsd=[df[f"amp_pctdiff_{k}"].std() for k in LIMB]
    ax2.bar(LIMB,pd_,yerr=pdsd,capsize=4,color="#c2701c")
    ax2.set_ylabel("mean |ΔA| / |A_Lund|  (%)"); ax2.set_title("Predicted limb-lead amplitude change\n(swept frontal dipole)")
    ax2.grid(axis="y",alpha=.3)
    fig.tight_layout(); fig.savefig(FIGA/"limb_lead_comparison.png",dpi=130); plt.close(fig)

    # ---- SUMMARY.md ----
    L=["# ML × Lund — final analysis (N=12, audited placements)\n",
       "Picks: `data/lund_picks_corrected.json`. Convention +X=left, +Y=posterior, +Z=superior. WCT=(RA+LA+LL)/3.\n",
       "## 1. Frontal cardiac axis (hexaxial; fixed dipole, electrodes move)\n"]
    m,s=ms("axis_shift_60")
    L.append(f"- **Apparent QRS-axis shift ML−Lund @+60°: {m:+.1f}° ± {s:.1f}°** "
             f"(range {df['axis_shift_60'].min():+.1f}..{df['axis_shift_60'].max():+.1f}°).")
    L.append("\n**Limb-lead vector rotation (ML−Lund):**")
    L.append("| Lead | "+" | ".join(LIMB)+" |"); L.append("|---|"+"---|"*6)
    L.append("| mean° | "+" | ".join(f"{ms(f'rot_{k}')[0]:+.1f}" for k in LIMB)+" |")
    L.append("| SD | "+" | ".join(f"{ms(f'rot_{k}')[1]:.1f}" for k in LIMB)+" |")

    L.append("\n## 2. WCT displacement (ML − Lund), all planes\n")
    for col,n in [("dX_left_mm","dX (+left)"),("dY_anterior_mm","dY (+anterior)"),("dZ_sup_mm","dZ (+superior)")]:
        mm,ss=ms(col); L.append(f"- {n}: **{mm:+.1f} ± {ss:.1f} mm**")
    L.append("")
    for col,n in [("WCT_frontal_mm","Frontal (X–Z)"),("WCT_sagittal_mm","Sagittal (Y–Z)"),
                  ("WCT_transverse_mm","Transverse (X–Y)"),("WCT_3D_mm","3-D total")]:
        mm,ss=ms(col); L.append(f"- {n}: **{mm:.1f} ± {ss:.1f} mm**")

    L.append("\n## 3. Limb-lead amplitude comparison (ML vs Lund)\n")
    L.append("| Lead | mag ratio ML/Lund | amplitude %Δ |")
    L.append("|---|---:|---:|")
    for k in LIMB:
        r,_=ms(f"magratio_{k}"); pdm,_=ms(f"amp_pctdiff_{k}")
        L.append(f"| {k} | {r:.2f} | {pdm:.0f}% |")
    L.append("\n(mag ratio >1 = ML inflates that lead's scale vs Lund; <1 = deflates. "
             "%Δ = mean absolute amplitude change over a swept frontal dipole.)")

    L.append("\n## Per-patient table (key columns)\n")
    cols=["patient","dX_left_mm","dY_anterior_mm","dZ_sup_mm","WCT_3D_mm","axis_shift_60"]
    L.append("| "+" | ".join(cols)+" |"); L.append("|"+"---|"*len(cols))
    for _,r in df.iterrows():
        L.append("| "+" | ".join(r[c] if c=="patient" else f"{r[c]:+.1f}" for c in cols)+" |")
    (RESULTS/"SUMMARY.md").write_text("\n".join(L))

    # console
    print("WCT (ML-Lund):")
    for c,n in [("dX_left_mm","dX+left"),("dY_anterior_mm","dY+ant"),("dZ_sup_mm","dZ+sup"),
                ("WCT_frontal_mm","|front|"),("WCT_sagittal_mm","|sag|"),("WCT_transverse_mm","|trans|"),("WCT_3D_mm","|3D|")]:
        mm,ss=ms(c); print(f"  {n:<9}{mm:+7.1f} +/- {ss:5.1f}")
    print(f"axis shift @+60: {ms('axis_shift_60')[0]:+.1f} +/- {ms('axis_shift_60')[1]:.1f}")
    print("limb-lead mag ratio (ML/Lund) & amp %diff:")
    for k in LIMB:
        print(f"  {k:<3} ratio={ms(f'magratio_{k}')[0]:.2f}  ampΔ={ms(f'amp_pctdiff_{k}')[0]:.0f}%")
    print(f"\nWrote results + 5 figures under {PROJ}")

if __name__ == "__main__":
    main()
