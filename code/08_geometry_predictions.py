"""
Geometry-based predictions (no rhythm, no demographics needed) — all 12 subjects.

  (1) Table 1  — dataset / geometry characteristics (cohort, #electrodes, torso
      AP-depth/width/height, heart size, MI status, pacing). No age/BMI/sex exist
      in these de-identified datasets, so this describes what IS available.
  (2) Chest AP-depth regression — does the anterior WCT shift (ML−Lund) scale with
      torso antero-posterior depth?
  (3) Modeled R-progression / transition zone — project a canonical ventricular
      QRS vector onto V1–V9 (audited precordials) and find the R/S transition under
      Lund vs ML; report the predicted shift. Includes posterior leads V7–V9 and a
      predicted per-lead contamination ranking. H is a stated model assumption; the
      ML−Lund SHIFT (common-mode WCT offset) is robust to H (sweep reported).

V-lead source: ECGI = audited v14 picks (data/v14_ecgi_picks.json);
               Charles = find_anatomical_electrodes on the 120-ch vest.
Limb/WCT source: audited data/lund_picks_corrected.json.

Outputs -> ../results/{table1_dataset.csv, table1_dataset.md, geom_predictions.csv}
           ../figures/analysis/{ap_depth_regression.png, predicted_transition_shift.png,
                                 predicted_contam_by_lead.png}
"""
from __future__ import annotations
from pathlib import Path
import json, sys, importlib.util
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJ   = Path(__file__).resolve().parents[1]
DATA   = PROJ / "data"; RESULTS = PROJ / "results"; FIGA = PROJ / "figures" / "analysis"
import charles_io
DS     = charles_io.data_root()
import charles_io as wdp
spec = importlib.util.spec_from_file_location("ac", str(Path(__file__).parent / "02_apply_corrections.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

VANT = ["V1","V2","V3","V4","V5","V6"]; VPOST = ["V7","V8","V9"]; VALL = VANT + VPOST
VAUD = json.loads((DATA / "vlead_picks_audited.json").read_text())   # hand-audited V1-V9
PICKS= json.loads((DATA / "lund_picks_corrected.json").read_text())
MI_CHARLES = {"pt027": 1}  # MI.txt: 027 MI+; 028/029 not listed (unknown)

# canonical normal QRS vector (body frame +X=left, +Y=posterior, +Z=superior):
# leftward, slightly posterior, inferior  (mean QRS axis ~+45° frontal)
H0 = np.array([1.0, 0.3, -0.6]); H0 = H0 / np.linalg.norm(H0)

def vleads(label, cohort, pid):
    """Return (torso, heart, Vdict{V:xyz or None}) using the hand-audited V1-V9."""
    if cohort == "charles":
        g = wdp.load_charles_geometry(DS / pid[2:]); torso, heart = g.torso_pts, g.heart_pts
    else:
        torso, heart = m.load_ecgi(pid)
    av = VAUD[label]
    V = {v: (np.array(av[v], float) if v in av else None) for v in VALL}
    return torso, heart, V

def wct(d, pre):  # pre = "ml" or "lund"
    return (np.array(d[f"{pre}_RA"]) + np.array(d[f"{pre}_LA"]) + np.array(d[f"{pre}_LL"])) / 3.0

def transition(defls):
    """leads (1..n) with deflection values -> interpolated lead index of R/S=0 crossing."""
    xs = np.arange(1, len(defls) + 1); y = np.array(defls)
    for i in range(len(y) - 1):
        if y[i] == 0: return float(xs[i])
        if y[i] * y[i+1] < 0:
            return float(xs[i] + (0 - y[i]) / (y[i+1] - y[i]))
    return float(xs[0] if y[0] > 0 else xs[-1])  # no crossing -> edge

def main():
    rows = []; contam = {v: [] for v in VALL}
    trans_shift = []; sweep_shifts = {a: [] for a in (0, 30, 45, 60, 90)}
    for seq, label, cohort, pid in m.ORDER:
        torso, heart, V = vleads(label, cohort, pid)
        d = PICKS[label]; w_ml, w_lu = wct(d, "ml"), wct(d, "lund")
        ap = float(torso[:,1].max() - torso[:,1].min())
        width = float(torso[:,0].max() - torso[:,0].min())
        height = float(torso[:,2].max() - torso[:,2].min())
        hbb = heart.max(0) - heart.min(0); heart_diag = float(np.linalg.norm(hbb))
        dd = w_ml - w_lu; dY_ant = float(-dd[1])
        n_v_good = sum(1 for v in VALL if V[v] is not None)
        rows.append(dict(seq=seq, patient=label, cohort=cohort.upper(),
                         torso_AP_mm=ap, torso_width_mm=width, torso_height_mm=height,
                         heart_diag_mm=heart_diag,
                         WCT_anterior_shift_mm=dY_ant,
                         WCT_3D_shift_mm=float(np.linalg.norm(dd)),
                         n_precordials_usable=n_v_good,
                         MI=("yes" if MI_CHARLES.get(pid)==1 else ("n/a" if cohort=="charles" else "unknown")),
                         rhythm="paced"))
        # modeled R-progression: deflection per available anterior lead
        defl_lu, defl_ml, leads_used = [], [], []
        for v in VANT:
            if V[v] is None: continue
            leads_used.append(v)
            defl_lu.append(float(H0 @ (V[v] - w_lu)))
            defl_ml.append(float(H0 @ (V[v] - w_ml)))
        if len(defl_lu) >= 3:
            t_lu, t_ml = transition(defl_lu), transition(defl_ml)
            trans_shift.append(t_ml - t_lu)
        # sweep H frontal axis for robustness of the shift
        for axdeg in sweep_shifts:
            th = np.radians(axdeg); Hs = np.array([np.cos(th), 0.30, -np.sin(th)]); Hs /= np.linalg.norm(Hs)
            dl = [float(Hs @ (V[v]-w_lu)) for v in VANT if V[v] is not None]
            dm = [float(Hs @ (V[v]-w_ml)) for v in VANT if V[v] is not None]
            if len(dl) >= 3: sweep_shifts[axdeg].append(transition(dm) - transition(dl))
        # predicted per-lead contamination = |common-mode offset| / |native (Lund) deflection|
        offset = abs(float(H0 @ (w_lu - w_ml)))
        for v in VALL:
            if V[v] is None: continue
            native = abs(float(H0 @ (V[v] - w_lu)))
            contam[v].append(100.0 * offset / (native + 1e-6))

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "geom_predictions.csv", index=False)

    # ---- Table 1 ----
    def med_iqr(s): return f"{s.median():.0f} [{s.quantile(.25):.0f}–{s.quantile(.75):.0f}]"
    t1 = ["# Table 1 — dataset / geometry characteristics (N=12)\n",
          "No age/BMI/sex in these de-identified datasets; this describes the in-silico cohort. "
          "All recordings are **paced** (no sinus). Real clinical Table 1 → prospective volunteer study.\n",
          "| Characteristic | All (N=12) | Charles (3) | ECGI (9) |", "|---|---|---|---|"]
    for col, name, unit in [("torso_AP_mm","Torso AP depth","mm"),
                            ("torso_width_mm","Torso width","mm"),
                            ("torso_height_mm","Torso height","mm"),
                            ("heart_diag_mm","Heart bbox diagonal","mm"),
                            ("WCT_anterior_shift_mm","WCT anterior shift ML−Lund","mm"),
                            ("n_precordials_usable","Usable precordials (of 9)","")]:
        a, c, e = df[col], df[df.cohort=="CHARLES"][col], df[df.cohort=="ECGI"][col]
        t1.append(f"| {name} ({unit}) | {med_iqr(a)} | {med_iqr(c)} | {med_iqr(e)} |")
    t1.append(f"| MI status | 027 MI+; 028/029 n/a; ECGI unknown |  |  |")
    t1.append(f"| Rhythm | paced (all) |  |  |")
    (RESULTS / "table1_dataset.md").write_text("\n".join(t1))
    df[["patient","cohort","torso_AP_mm","torso_width_mm","torso_height_mm","heart_diag_mm",
        "WCT_anterior_shift_mm","n_precordials_usable","MI","rhythm"]].to_csv(
        RESULTS / "table1_dataset.csv", index=False)

    # ---- AP-depth regression ----
    x = df["torso_AP_mm"].values; y = df["WCT_anterior_shift_mm"].values
    lr = stats.linregress(x, y)
    fig, ax = plt.subplots(figsize=(7.5,6))
    ax.scatter(x, y, c=["#1f4e79" if c=="CHARLES" else "#c2701c" for c in df.cohort], s=70, zorder=3)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, lr.intercept + lr.slope*xs, "k--",
            label=f"slope={lr.slope:.2f}  r={lr.rvalue:.2f}  p={lr.pvalue:.3f}")
    for _,r in df.iterrows(): ax.annotate(r.patient.split()[-1], (r.torso_AP_mm, r.WCT_anterior_shift_mm), fontsize=7)
    ax.set_xlabel("torso antero-posterior depth (mm)"); ax.set_ylabel("anterior WCT shift  ML−Lund (mm)")
    ax.set_title("Anterior WCT shift vs chest AP-depth (n=12)\nnavy=Charles, orange=ECGI")
    ax.legend(); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(FIGA/"ap_depth_regression.png", dpi=130); plt.close(fig)

    # ---- predicted transition-zone shift ----
    ts = np.array(trans_shift)
    fig, ax = plt.subplots(figsize=(8,5))
    ax.hist(ts, bins=8, color="#0d6b6e", alpha=.85)
    ax.axvline(ts.mean(), color="black", ls="--", label=f"mean shift = {ts.mean():+.2f} lead-units")
    ax.set_xlabel("predicted R/S transition shift  ML−Lund (precordial-lead units; + = toward V6/leftward)")
    ax.set_ylabel("subjects"); ax.set_title("Modeled precordial transition-zone shift, ML vs Lund\n"
                  f"(canonical QRS vector; robust across H sweep: " +
                  ", ".join(f"{a}°:{np.mean(sweep_shifts[a]):+.2f}" for a in sweep_shifts) + ")")
    ax.legend(); ax.grid(axis="y", alpha=.3); fig.tight_layout()
    fig.savefig(FIGA/"predicted_transition_shift.png", dpi=130); plt.close(fig)

    # ---- predicted per-lead contamination (median) ----
    med = [np.median(contam[v]) if contam[v] else np.nan for v in VALL]
    fig, ax = plt.subplots(figsize=(9,5))
    cols = ["#0d6b6e"]*6 + ["#7b2cbf"]*3
    ax.bar(VALL, med, color=cols)
    for i,mv in enumerate(med):
        if not np.isnan(mv): ax.text(i, mv, f"{mv:.0f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("predicted contamination  |WCT offset| / |native| (%, median)")
    ax.set_title("Predicted per-lead contamination (modeled), anterior V1–V6 (teal) + posterior V7–V9 (purple)")
    ax.grid(axis="y", alpha=.3); fig.tight_layout(); fig.savefig(FIGA/"predicted_contam_by_lead.png", dpi=130); plt.close(fig)

    # console
    print("Table 1 (geometry) written. AP-depth regression: "
          f"slope={lr.slope:.2f} mm/mm, r={lr.rvalue:.2f}, p={lr.pvalue:.3f}")
    print(f"Predicted transition shift ML-Lund: mean {ts.mean():+.2f} lead-units (n={len(ts)})")
    print("Sweep (frontal axis -> mean shift):", {a: round(float(np.mean(sweep_shifts[a])),2) for a in sweep_shifts})
    print("Predicted median contamination by lead:")
    for v, mv in zip(VALL, med): print(f"  {v}: {mv:.0f}%" if not np.isnan(mv) else f"  {v}: n/a")
    print(f"\nWrote results + 3 figures + Table 1 under {PROJ}")

if __name__ == "__main__":
    main()
