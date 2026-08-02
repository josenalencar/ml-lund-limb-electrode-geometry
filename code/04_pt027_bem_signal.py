"""
pt027 SIGNAL-LEVEL simulation — ML vs Lund (audited placements).

Uses pt027's BEM forward matrix to reconstruct the body-surface potential field
from each paced beat, then reads the potential at the AUDITED ML and Lund
electrode nodes (from data/lund_picks_corrected.json, mapped to nearest torso
node). Computes — from SIGNALS, not geometry —:

  * Precordial contamination = ΔWCT = WCT_ML − WCT_Lund over the QRS, and the
    per-precordial % of QRS energy that is pure WCT-shift artefact
    (V_ML − V_Lund = WCT_Lund − WCT_ML, identical across V1..V6).
  * Limb-lead signed QRS integrals under ML vs Lund (signal-level amplitude check).

Adapts code/v23_papouchado_integral.py (which did ML vs WA) to ML vs Lund.
N = 1 patient (pt027) — the only subject with a forward matrix.

Outputs -> ../results/{pt027_bem_signal.csv, pt027_bem_summary.md}
           ../figures/analysis/{pt027_precordial_contamination.png, pt027_limb_signal_comparison.png}
"""
from __future__ import annotations
import sys, re, json
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJ      = Path(__file__).resolve().parents[1]
DATA      = PROJ / "data"
RESULTS   = PROJ / "results"
FIGA      = PROJ / "figures" / "analysis"
RESULTS.mkdir(parents=True, exist_ok=True); FIGA.mkdir(parents=True, exist_ok=True)
DS        = None  # resolved in main() from MLLUND_DATA_ROOT

import charles_io as wdp

LIMB = ("I", "II", "III", "aVR", "aVL", "aVF")
VC   = ("V1", "V2", "V3", "V4", "V5", "V6")

def qrs_integral(signal, qs, qe):
    seg = signal[qs:qe + 1]
    return float(seg.sum()) if len(seg) else 0.0

def nearest_node(torso_pts, xyz):
    d = np.linalg.norm(torso_pts - np.asarray(xyz, float), axis=1)
    return int(d.argmin()), float(d.min())

def main():
    DS = wdp.data_root()
    pt27 = DS / "027"
    geom = wdp.load_charles_geometry(pt27)
    torso_pts = geom.torso_pts
    A = sio.loadmat(pt27 / "FwdInvTransforms/2012-07-27Subject_forward_matrix.mat",
                    squeeze_me=True, struct_as_record=False)["A"]
    if A.shape[0] < A.shape[1]: A = A.T
    assert A.shape[0] == len(torso_pts), \
        f"forward-matrix rows ({A.shape[0]}) != torso nodes ({len(torso_pts)})"
    ch_idx = geom.extra["channels_into_torso"]
    A_m = A[ch_idx, :]
    lam = np.linalg.svd(A_m, full_matrices=False, compute_uv=False)[0] / 100.0
    inv = np.linalg.inv(A_m.T @ A_m + lam**2 * np.eye(A.shape[1])) @ A_m.T

    # --- AUDITED ML & Lund picks -> nearest torso node (same daltorso frame) ---
    picks = json.loads((DATA / "lund_picks_corrected.json").read_text())["Charles pt027"]  # audited placements
    nodes = {"ML": {}, "Lund": {}}; snaps = {}
    print("Mapping audited picks -> nearest torso node (distance in mm):")
    for cfg, pre in (("ML", "ml"), ("Lund", "lund")):
        for el in ("RA", "LA", "LL"):
            n, d = nearest_node(torso_pts, picks[f"{pre}_{el}"])
            nodes[cfg][el] = n; snaps[f"{cfg}_{el}"] = d
            print(f"  {cfg:<4} {el}: node {n:5d}  (snap {d:5.1f} mm)")
    n_torso = len(torso_pts)
    maxsnap = max(np.linalg.norm(torso_pts[nodes[c][e]] - np.array(picks[f'{p}_{e}']))
                  for c, p in (("ML","ml"),("Lund","lund")) for e in ("RA","LA","LL"))
    if maxsnap > 40:
        print(f"WARNING: max snap distance {maxsnap:.1f} mm — frame/units may mismatch.")

    events, _ = wdp.load_charles_events(pt27)
    mesh_frame = wdp.build_heart_frame(geom.heart_pts, geom.torso_pts)
    pre_map = wdp.find_anatomical_electrodes(geom.electrode_xyz, geom.torso_pts, mesh_frame)
    def chan(name):
        m = pre_map[name]
        return m["idx"] if isinstance(m, dict) else m
    # hand-audited precordials -> nearest recording channel (native V = recorded signal there)
    audV = json.loads((DATA / "vlead_picks_audited.json").read_text())["Charles pt027"]
    def nchan(xyz): return int(np.linalg.norm(geom.electrode_xyz - np.array(xyz), axis=1).argmin())
    vchans = {v: nchan(audV[v]) for v in VC if v in audV}

    rows = []
    for ev in events:
        src = inv @ ev.signal
        body = A @ src
        leads_pre = wdp.derive_12_leads(ev.signal, pre_map)
        qs, qe = wdp.detect_qrs(leads_pre)

        def lead6(cfg):
            ra, la, ll = (body[nodes[cfg]["RA"]], body[nodes[cfg]["LA"]], body[nodes[cfg]["LL"]])
            wct = (ra + la + ll) / 3.0
            return dict(I=la-ra, II=ll-ra, III=ll-la, aVR=ra-wct, aVL=la-wct, aVF=ll-wct, WCT=wct)
        lML, lLU = lead6("ML"), lead6("Lund")
        qML = wdp.detect_qrs({k:v for k,v in lML.items() if k!="WCT"})
        qLU = wdp.detect_qrs({k:v for k,v in lLU.items() if k!="WCT"})

        pace = None
        m = re.search(r"Pace(\d+)", ev.event_id)
        if m and ev.ventricle == "LV": pace = int(m.group(1))
        row = dict(event_id=ev.event_id, ventricle=ev.ventricle, pace=pace, qrs_len=int(qe-qs+1))

        # limb-lead signed QRS integrals (signal-level), both configs
        for l in LIMB:
            row[f"{l}_int_Lund"] = qrs_integral(lLU[l], *qLU)
            row[f"{l}_int_ML"]   = qrs_integral(lML[l], *qML)

        # contamination = ΔWCT over the QRS (V_ML - V_Lund = WCT_Lund - WCT_ML)
        dwct = (lML["WCT"][qs:qe+1] - lLU["WCT"][qs:qe+1]).astype(float)
        row["dWCT_energy"] = float(np.sum(dwct**2))
        row["dWCT_signed_int"] = float(dwct.sum())
        row["dWCT_abs_int"] = float(np.abs(dwct).sum())
        # per-precordial native QRS energy (raw electrode potential) for % contamination
        for v in VC:
            seg = ev.signal[vchans[v]][qs:qe+1].astype(float)   # native V = recorded signal at audited position
            row[f"{v}_energy"] = float(np.sum(seg**2))
            row[f"{v}_pct_contam"] = 100.0 * row["dWCT_energy"] / (row[f"{v}_energy"] + 1e-9)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "pt027_bem_signal.csv", index=False)
    n = len(df)

    # ---- precordial % contamination ----
    pct = {v: df[f"{v}_pct_contam"].mean() for v in VC}
    pctmed = {v: df[f"{v}_pct_contam"].median() for v in VC}

    # ---- limb-lead signal comparison (ML vs Lund) ----
    limb_stats = {}
    for l in LIMB:
        aML, aLU = df[f"{l}_int_ML"], df[f"{l}_int_Lund"]
        ratio = aML.abs().mean() / (aLU.abs().mean() + 1e-9)
        cc = float(np.corrcoef(aML, aLU)[0,1]) if aML.std() and aLU.std() else np.nan
        pdiff = 100.0 * (aML - aLU).abs().mean() / (aLU.abs().mean() + 1e-9)
        limb_stats[l] = dict(mean_abs_Lund=float(aLU.abs().mean()),
                             mean_abs_ML=float(aML.abs().mean()),
                             ratio=float(ratio), corr=cc, pct_diff=float(pdiff))

    # ---- figures ----
    fig, ax = plt.subplots(figsize=(8,5))
    ax.bar(VC, [pctmed[v] for v in VC], color="#0d6b6e")
    for i,v in enumerate(VC): ax.text(i, pctmed[v], f"{pctmed[v]:.0f}%", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("% of QRS energy that is WCT-shift artefact (median)")
    ax.set_title(f"pt027 BEM — precordial contamination, ML vs Lund (n={n} beats, MEDIAN)\n"
                 "= ΔWCT energy / native precordial QRS energy  (mean is inflated by small-denominator beats)")
    ax.grid(axis="y", alpha=.3); fig.tight_layout()
    fig.savefig(FIGA / "pt027_precordial_contamination.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9,5))
    x = np.arange(len(LIMB)); w=0.38
    ax.bar(x-w/2, [limb_stats[l]["mean_abs_Lund"] for l in LIMB], w, label="Lund", color="#1f4e79")
    ax.bar(x+w/2, [limb_stats[l]["mean_abs_ML"]   for l in LIMB], w, label="ML",   color="#c2701c")
    ax.set_xticks(x); ax.set_xticklabels(LIMB)
    ax.set_ylabel("mean |signed QRS integral|  (signal units)")
    ax.set_title(f"pt027 BEM — limb-lead amplitude, ML vs Lund (n={n} beats)")
    ax.legend(); ax.grid(axis="y", alpha=.3); fig.tight_layout()
    fig.savefig(FIGA / "pt027_limb_signal_comparison.png", dpi=130); plt.close(fig)

    # ---- summary ----
    we = df["dWCT_energy"]; ql = df["qrs_len"]
    L = [f"# pt027 — signal-level BEM simulation, ML vs Lund (n={n} paced beats)\n",
         "Forward matrix: `027/FwdInvTransforms/2012-07-27Subject_forward_matrix.mat`. "
         "Audited placements from `data/lund_picks_corrected.json` (nearest torso node).",
         "Adapted from `code/v23_papouchado_integral.py` (was ML vs WA).\n",
         "## Precordial contamination (ML vs Lund)\n",
         "`V_ML − V_Lund = WCT_Lund − WCT_ML` — identical across V1–V6; % = ΔWCT energy / native QRS energy.\n",
         "| Lead | % contamination (MEDIAN, reported) | mean (inflated — see caveat) |",
         "|---|---:|---:|"]
    for v in VC:
        L.append(f"| {v} | **{pctmed[v]:.1f}%** | {pct[v]:.1f}% |")
    L.append(f"\nΔWCT QRS energy: mean {we.mean():.3f} ± {we.std():.3f}; "
             f"QRS length {ql.mean():.0f} ± {ql.std():.0f} samples.\n")
    L.append("## Limb-lead amplitude, ML vs Lund (signed QRS integral)\n")
    L.append("| Lead | mean|Lund| | mean|ML| | ratio ML/Lund | corr(ML,Lund) | %Δ |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for l in LIMB:
        s = limb_stats[l]
        L.append(f"| {l} | {s['mean_abs_Lund']:.2f} | {s['mean_abs_ML']:.2f} | "
                 f"{s['ratio']:.2f} | {s['corr']:.2f} | {s['pct_diff']:.0f}% |")
    L.append("\n## Caveats (read before quoting numbers)\n"
             f"- **Report the MEDIAN, not the mean.** Means for V4–V6 exceed 100% because a few beats "
             f"have near-zero native precordial QRS energy (tiny denominator). Median V1≈{pctmed['V1']:.0f}%, "
             f"V2–V6 ≈ {min(pctmed[v] for v in VC[1:]):.0f}–{max(pctmed[v] for v in VC[1:]):.0f}% is the robust statistic.\n"
             f"- **Coarse forward mesh ({n_torso} torso nodes) + very-lateral audited Lund picks** → electrode "
             f"picks snap to the nearest torso node by up to **{max(snaps.values()):.0f} mm** "
             f"(ML RA/LA ≈{snaps['ML_RA']:.0f} mm; Lund RA ≈{snaps['Lund_RA']:.0f} mm). The signal-level "
             "result uses the snapped on-surface nodes, not the exact off-surface picks — so it is not a "
             "perfect match to the geometry analysis (which used the picks as-is). Lateral Lund electrodes "
             "should be re-audited ONTO the torso surface for a clean signal-level number.\n"
             "- **N=1 patient** (only pt027 has a forward matrix); **paced beats** (higher inter-beat variance "
             "than sinus). This is a signal-level confirmation of the geometric predictions in `FINDINGS.md`, "
             "not a population result.")
    (RESULTS / "pt027_bem_summary.md").write_text("\n".join(L))

    # console
    print(f"\nn beats = {n}")
    print("precordial % contamination (mean):")
    for v in VC: print(f"  {v}: {pct[v]:5.1f}%   (median {pctmed[v]:.1f}%)")
    print("limb-lead ML/Lund |integral| ratio & corr:")
    for l in LIMB:
        s = limb_stats[l]; print(f"  {l:<3} ratio={s['ratio']:.2f}  corr={s['corr']:+.2f}  %Δ={s['pct_diff']:.0f}%")
    print(f"\nWrote CSV + summary + 2 figures under {PROJ}")

if __name__ == "__main__":
    main()
