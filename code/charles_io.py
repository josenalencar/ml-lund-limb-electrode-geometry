"""
Readers and 12-lead derivation for the Charles University / Petr Stovicek (PSTOV)
body-surface potential-mapping datasets distributed through the EDGAR repository.

This module is the subset of the author's larger lead-mapping pipeline that the
Mason-Likar versus Lund analysis depends on. It provides:

  load_charles_geometry      torso and myocardial surfaces, electrode coordinates
  load_charles_events        per-pace body-surface recordings, averaged over runs
  parse_carto_file           CARTO pacing-site coordinates
  build_heart_frame          principal anatomical axes of the ventricular surface
  build_carto_frame          the same frame anchored on CARTO points
  find_anatomical_electrodes map recording channels onto standard lead positions
  derive_12_leads            standard 12 leads from a body-surface potential array
  detect_qrs                 QRS window from the multi-lead RMS envelope

Coordinate convention: +X = patient's left, +Y = posterior, +Z = superior.
Body-surface potentials are in millivolts, sampled at 2000 Hz, as documented in
the dataset ReadMe.

The location of the source data is read from the MLLUND_DATA_ROOT environment
variable, which must point at the directory holding the per-subject folders
(027, 028, 029) and, where applicable, ECGI_Challenge_2026_Training.
"""
from __future__ import annotations
import os, re, warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import scipy.io as sio
warnings.filterwarnings("ignore", category=RuntimeWarning)


def data_root() -> Path:
    """Resolve the source-data directory, or fail with an actionable message."""
    env = os.environ.get("MLLUND_DATA_ROOT")
    if not env:
        raise RuntimeError(
            "Set MLLUND_DATA_ROOT to the directory containing the Charles/PSTOV "
            "subject folders (027, 028, 029) and, if used, the "
            "ECGI_Challenge_2026_Training directory. Both are obtained from the "
            "EDGAR repository at https://edgar.sci.utah.edu (free account). "
            "Example:  export MLLUND_DATA_ROOT=/path/to/Charles_PSTOV-12-07-27"
        )
    root = Path(env).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"MLLUND_DATA_ROOT does not exist: {root}")
    return root


@dataclass
class HeartFrame:
    """An anatomically-normalised reference frame for one patient's heart.
    The base→apex axis is the long axis (PC1 of heart vertices, sign chosen so
    +long_axis points from apex toward base in chest coordinates).
    The septum→lateral axis (PC2) is signed so +radial points away from the
    body's midline.
    The anterior→inferior axis (PC3) is what remains.
    """
    center: np.ndarray            # (3,) heart centroid in torso coords
    long_axis: np.ndarray         # unit vector, points from apex to base
    radial_axis: np.ndarray       # unit vector, points septum→lateral
    ap_axis: np.ndarray           # unit vector, points anterior→posterior (or vice versa)
    long_range: tuple             # (apex_proj, base_proj) projection limits
    radial_range: tuple
    ap_range: tuple

    def classify(self, point_xyz: np.ndarray) -> dict:
        rel = point_xyz - self.center
        long_p   = float(rel @ self.long_axis)
        radial_p = float(rel @ self.radial_axis)
        ap_p     = float(rel @ self.ap_axis)
        lo, hi = self.long_range
        rng = hi - lo
        frac_base = (long_p - lo) / rng if rng else 0.5  # 0 apex, 1 base
        # tertiles
        if frac_base < 1/3:  long_bin = "apical"
        elif frac_base < 2/3: long_bin = "mid"
        else: long_bin = "basal"
        radial_max = max(abs(self.radial_range[0]), abs(self.radial_range[1]))
        if radial_max < 1e-6:
            radial_bin = "mid-radial"
        elif radial_p > 0.25 * radial_max: radial_bin = "lateral"
        elif radial_p < -0.25 * radial_max: radial_bin = "septal"
        else: radial_bin = "mid-radial"
        ap_max = max(abs(self.ap_range[0]), abs(self.ap_range[1]))
        if ap_max < 1e-6:
            ap_bin = "mid-AP"
        elif ap_p > 0.25 * ap_max: ap_bin = "posterior"
        elif ap_p < -0.25 * ap_max: ap_bin = "anterior"
        else: ap_bin = "mid-AP"
        return dict(
            long_axis_pos=long_p, radial_pos=radial_p, ap_pos=ap_p,
            apex_to_base_frac=frac_base,
            long_axis=long_bin, radial=radial_bin, ap=ap_bin,
            anatomic_summary=f"{long_bin} {ap_bin} ({radial_bin})",
        )


def build_heart_frame(heart_pts: np.ndarray, torso_pts: np.ndarray) -> HeartFrame:
    """heart_pts: (N,3). torso_pts: (M,3). Build a heart-anchored frame.

    Algorithm:
      - PC1 of heart → long axis
      - Sign long_axis so apex is the *narrowest* heart slice along it
        (the apex is anatomically pointy, base is broader). We compute the
        cross-section spread perpendicular to the long axis at low and high
        projection percentiles and pick the orientation where the low end has
        larger spread (that end = base).
      - For radial: PC2 of heart → radial. Sign chosen so + points away from
        body midline (i.e. so heart "lateral" side is +).
      - AP axis = PC3.
    """
    hc = heart_pts.mean(axis=0)
    H = heart_pts - hc
    # SVD
    _, _, Vt = np.linalg.svd(H, full_matrices=False)
    axes = Vt        # rows: principal axes

    long_axis = axes[0]
    # Compute spread perpendicular to long_axis at high vs low projection ends
    proj = H @ long_axis
    perp_basis = axes[1:3]
    perp = H @ perp_basis.T
    lo_mask = proj < np.percentile(proj, 25)
    hi_mask = proj > np.percentile(proj, 75)
    lo_spread = np.linalg.norm(perp[lo_mask].std(axis=0))
    hi_spread = np.linalg.norm(perp[hi_mask].std(axis=0))
    # base end is broader. If high end is broader, leave long_axis as-is (so +=base).
    # If low end is broader, flip.
    if lo_spread > hi_spread:
        long_axis = -long_axis
    # Recompute proj with possibly flipped sign
    proj = H @ long_axis

    radial_axis = axes[1]
    # Sign radial so + points away from torso midline.
    body_midline = torso_pts.mean(axis=0)
    heart_offset = hc - body_midline
    # If projection of heart_offset onto radial_axis is positive, heart side is +radial
    # We want + radial to mean "lateral" relative to body — so it should agree with
    # whichever side of midline the heart sits.
    if heart_offset @ radial_axis < 0:
        radial_axis = -radial_axis

    ap_axis = axes[2]
    # Sign ap_axis so + points toward back of torso.
    # The back-of-torso direction in torso coords: torso extent along ap_axis,
    # vs heart sits anterior in chest.
    torso_centered = torso_pts - body_midline
    torso_ap_extent = torso_centered @ ap_axis
    # The back of torso is in one direction. We use heart offset which sits *anterior*.
    # If heart offset projection onto ap_axis is positive, that's the anterior side,
    # so flip so + ap means posterior.
    if heart_offset @ ap_axis > 0:
        ap_axis = -ap_axis

    rng_long   = (float(proj.min()), float(proj.max()))
    radial_p = H @ radial_axis
    rng_radial = (float(radial_p.min()), float(radial_p.max()))
    ap_p = H @ ap_axis
    rng_ap = (float(ap_p.min()), float(ap_p.max()))

    return HeartFrame(
        center=hc, long_axis=long_axis, radial_axis=radial_axis, ap_axis=ap_axis,
        long_range=rng_long, radial_range=rng_radial, ap_range=rng_ap,
    )


# --------------------------------------------------------------------------- #
# 12-lead derivation from arbitrary BSPM grid
# --------------------------------------------------------------------------- #
def find_anatomical_electrodes(
    electrode_xyz: np.ndarray,
    torso_pts: np.ndarray,
    heart_frame: HeartFrame,
) -> dict:
    """ANATOMICALLY-FAITHFUL placement of V1–V9 / RA / LA / LL on the electrode
    grid using cross-section parameterisation at the heart-Z level.

    Algorithm:
      1. Slice the torso mesh at heart-level (within ±15% of heart long-axis extent).
      2. Within that slice, compute the body-cross-section centroid (not the heart).
      3. Determine the anterior direction: heart is anatomically anterior of the
         body centroid in the chest, so anterior = (heart_center - body_centroid)
         projected into the slice plane.
      4. Compute the angle of each electrode around the body-centroid, with
         anterior = 0°, left = +90°, posterior = ±180°, right = −90°.
      5. Place V1..V9 at canonical anatomical angles:
            V1 = −15°  (4th ICS, right sternal border)
            V2 = +15°  (4th ICS, left sternal border)
            V3 = +35°  (between V2 and V4)
            V4 = +60°  (5th ICS, midclavicular line)
            V5 = +75°  (5th ICS, anterior axillary line)
            V6 = +90°  (5th ICS, midaxillary line)
            V7 = +120° (posterior axillary line)
            V8 = +155° (midscapular line)
            V9 = +175° (paravertebral line)
      6. For each angle, snap to the nearest electrode in the chest-Z slice.
      7. RA / LA / LL: top corners and bottom of torso (Mason-Likar-equivalent
         on the 120-electrode vest; not true wrist/ankle, which the vest cannot
         record from).
    """
    rel = electrode_xyz - heart_frame.center
    long_p   = rel @ heart_frame.long_axis
    radial_p = rel @ heart_frame.radial_axis
    ap_p     = rel @ heart_frame.ap_axis
    long_extent = heart_frame.long_range[1] - heart_frame.long_range[0]

    # === Anterior direction in the slice plane ===
    body_centroid = torso_pts.mean(axis=0)
    anterior_vec = heart_frame.center - body_centroid
    anterior_vec = anterior_vec - (anterior_vec @ heart_frame.long_axis) * heart_frame.long_axis
    if np.linalg.norm(anterior_vec) < 1e-6:
        anterior_vec = -heart_frame.ap_axis
    anterior_vec = anterior_vec / np.linalg.norm(anterior_vec)
    left_vec = heart_frame.radial_axis

    # Each electrode's chest-level angle (0=anterior, +90=left, ±180=back, -90=right)
    rel_to_body = electrode_xyz - body_centroid
    a_comp = rel_to_body @ anterior_vec
    l_comp = rel_to_body @ left_vec
    angle_deg = np.degrees(np.arctan2(l_comp, a_comp))
    a_long = rel_to_body @ heart_frame.long_axis     # head-foot position rel. body centroid

    # === Chest-level Z slice for V-lead candidates ===
    # Tight constraint: V1-V9 should sit at 4th/5th ICS, near heart center,
    # NOT span the whole torso. Use ±25% of long-axis extent (≈ ±half-a-heart-height).
    chest_mask = np.abs(long_p) < long_extent * 0.25
    chest_idx = np.where(chest_mask)[0]
    if len(chest_idx) < 9:
        # widen progressively until we have at least 9 candidates
        for k in (0.35, 0.45, 0.55, 0.7, 1.0):
            chest_mask = np.abs(long_p) < long_extent * k
            chest_idx = np.where(chest_mask)[0]
            if len(chest_idx) >= 9:
                break
        else:
            chest_idx = np.arange(len(electrode_xyz))

    # Canonical anatomic angles (matching docstring above; v5 had a 5°-10°
    # posterior drift baked in that pushed V6 BEHIND the midaxillary line —
    # corrected v6).
    canonical_angles = {
        "V1": -15.0,    # 4th ICS, right sternal border
        "V2": +15.0,    # 4th ICS, left sternal border
        "V3": +35.0,    # between V2 and V4
        "V4": +60.0,    # 5th ICS, midclavicular line
        "V5": +75.0,    # 5th ICS, anterior axillary line
        "V6": +90.0,    # 5th ICS, midaxillary line
        "V7": +120.0,   # posterior axillary line
        "V8": +155.0,   # midscapular line
        "V9": +175.0,   # paravertebral line
    }

    # For V7-V9 (posterior leads), the back coverage of typical vests is sparser,
    # so allow a wider Z slice (40% of long-axis extent vs 25% for V1-V6).
    chest_mask_wider = np.abs(long_p) < long_extent * 0.40
    chest_idx_wider = np.where(chest_mask_wider)[0]
    if len(chest_idx_wider) < 9: chest_idx_wider = np.arange(len(electrode_xyz))

    used = set()
    v_indices = {}
    for n in ("V1","V2","V3","V4","V5","V6","V7","V8","V9"):
        target = canonical_angles[n]
        pool = chest_idx_wider if n in ("V7","V8","V9") else chest_idx
        d = np.abs(((angle_deg[pool] - target + 180) % 360) - 180)
        order = pool[np.argsort(d)]
        for cand in order:
            if int(cand) not in used:
                v_indices[n] = int(cand)
                used.add(int(cand))
                break
        else:
            v_indices[n] = int(order[0])

    # === Limb electrodes (Mason-Likar-equivalent positions on the torso vest) ===
    # RA: high body (large long_p), right side (negative radial_p)
    score_ra = +long_p - 2.0 * radial_p
    score_la = +long_p + 2.0 * radial_p
    basal_thr = np.percentile(long_p, 70)
    basal_mask = long_p > basal_thr
    if basal_mask.sum() < 2:
        basal_mask = np.ones_like(long_p, dtype=bool)
    ra_pool = np.where(basal_mask)[0]
    ra_idx = int(ra_pool[np.argmax(score_ra[ra_pool])])
    la_idx = int(ra_pool[np.argmax(score_la[ra_pool])])
    if la_idx == ra_idx:
        score2 = score_la.copy(); score2[ra_idx] = -np.inf
        la_idx = int(np.argmax(score2))
    # LL: low body (apical end), slightly left of midline
    apex_thr = np.percentile(long_p, 30)
    apex_mask = long_p < apex_thr
    if apex_mask.sum() < 1:
        apex_mask = np.ones_like(long_p, dtype=bool)
    ll_pool = np.where(apex_mask)[0]
    # Prefer slightly-left of midline (radial_p ≈ 0 or slightly positive)
    ll_idx = int(ll_pool[np.argmin(np.abs(radial_p[ll_pool] - 0))])

    mapping = {}
    for name, idx in v_indices.items():
        mapping[name] = {
            "idx": int(idx),
            "elec_xyz": electrode_xyz[idx].tolist(),
            "axial_angle_deg": float(angle_deg[idx]),
            "long_p": float(long_p[idx]),
            "radial_p": float(radial_p[idx]),
            "ap_p": float(ap_p[idx]),
        }
    for name, idx in (("RA", ra_idx), ("LA", la_idx), ("LL", ll_idx)):
        mapping[name] = {
            "idx": int(idx),
            "elec_xyz": electrode_xyz[idx].tolist(),
            "axial_angle_deg": float(angle_deg[idx]),
            "long_p": float(long_p[idx]),
            "radial_p": float(radial_p[idx]),
            "ap_p": float(ap_p[idx]),
        }
    return mapping


def derive_12_leads(potvals: np.ndarray, mapping: dict) -> dict:
    """potvals: (n_channels, n_time) BSPM signal. Returns dict of up to 15 leads:
       I, II, III, aVR, aVL, aVF, V1-V6, plus V7-V9 if posterior coverage exists."""
    RA = potvals[mapping["RA"]["idx"]]
    LA = potvals[mapping["LA"]["idx"]]
    LL = potvals[mapping["LL"]["idx"]]
    WCT = (RA + LA + LL) / 3.0
    leads = {
        "I":   LA - RA,
        "II":  LL - RA,
        "III": LL - LA,
        "aVR": RA - (LA + LL) / 2,
        "aVL": LA - (RA + LL) / 2,
        "aVF": LL - (RA + LA) / 2,
    }
    for v in ("V1", "V2", "V3", "V4", "V5", "V6"):
        leads[v] = potvals[mapping[v]["idx"]] - WCT
    for v in ("V7", "V8", "V9"):
        if v in mapping:
            leads[v] = potvals[mapping[v]["idx"]] - WCT
    return leads


def detect_qrs(leads: dict, pad: int = 6) -> tuple[int, int]:
    stack = np.stack(list(leads.values()))
    v = stack.var(axis=0)
    thr = v.max() * 0.10
    above = v > thr
    best, start, cur = (0, 0, 0), None, 0
    for i, a in enumerate(above):
        if a:
            if start is None: start = i
            cur = i - start + 1
            if cur > best[2]: best = (start, i, cur)
        else:
            start = None; cur = 0
    s, e = best[0], best[1]
    s = max(0, s - pad)
    e = min(len(v)-1, e + pad)
    return s, e



@dataclass
class PatientGeometry:
    patient_id: str
    dataset: str
    heart_pts: np.ndarray             # (N,3)
    torso_pts: np.ndarray             # (M,3)
    electrode_xyz: np.ndarray         # (n_channels, 3)
    extra: dict = field(default_factory=dict)

@dataclass
class Event:
    patient_id: str
    dataset: str
    event_id: str                     # e.g. "LV_Pace01"
    ventricle: str                    # "LV" / "RV" / "PVC"
    pacing_xyz: Optional[np.ndarray]  # site coords, may be in HEART frame or TORSO frame
    pacing_in_torso_coords: bool      # if False, pacing_xyz is in heart-mesh coords
    signal: np.ndarray                # (n_channels, n_time) mV
    sampling_hz: int
    n_repeats: int                    # how many runs were averaged
    pacing_xyz_origin_label: str = "" # source of pacing coord


# ---- Charles_PSTOV --------------------------------------------------------- #
def parse_carto_file(path: Path) -> dict:
    """Returns {('LV', 1): (x,y,z), ('RV', 1): ...}"""
    out = {}
    section = None
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s: continue
            if s.startswith("LV"): section = "LV"; continue
            if s.startswith("RV"): section = "RV"; continue
            if s.startswith("Pac#") or s.startswith("CARTO"): continue
            parts = re.split(r"\s+", s)
            if len(parts) >= 4 and section in ("LV", "RV"):
                try:
                    pid = int(parts[0])
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    out[(section, pid)] = np.array([x, y, z])
                except ValueError:
                    pass
    return out


def load_charles_geometry(patient_dir: Path) -> PatientGeometry:
    pid = patient_dir.name
    mesh_dir = patient_dir / "Meshes"
    # Heart and torso meshes (registered)
    h = sio.loadmat([f for f in mesh_dir.glob("*myocardium_registered.mat")][0],
                    squeeze_me=True, struct_as_record=False)["geom"]
    t = sio.loadmat([f for f in mesh_dir.glob("*daltorso_registered.mat")][0],
                    squeeze_me=True, struct_as_record=False)["geom"]
    # geom.node is (3, N); transpose
    heart_pts = h.node.T if h.node.shape[0] == 3 else h.node
    torso_pts = t.node.T if t.node.shape[0] == 3 else t.node
    # 120-channel BSPM electrode positions: indices into torso mesh
    ch_idx = t.channels.astype(int) - 1   # MATLAB 1-based → Python 0-based
    electrode_xyz = torso_pts[ch_idx]
    return PatientGeometry(
        patient_id=pid, dataset="Charles_PSTOV",
        heart_pts=heart_pts, torso_pts=torso_pts, electrode_xyz=electrode_xyz,
        extra={"channels_into_torso": ch_idx},
    )


def build_carto_frame(carto_points: np.ndarray) -> HeartFrame:
    """Build an anatomical frame from the CARTO LV/RV pacing-site points themselves.
    PCA gives long-axis (apex-base), then septum-lateral, then AP. Sign conventions
    fixed by heuristics on point-cloud shape (base end has broader scatter)."""
    hc = carto_points.mean(axis=0)
    P = carto_points - hc
    _, _, Vt = np.linalg.svd(P, full_matrices=False)
    axes = Vt
    long_axis = axes[0]
    proj = P @ long_axis
    perp = P @ axes[1:3].T
    lo_mask = proj < np.percentile(proj, 25)
    hi_mask = proj > np.percentile(proj, 75)
    if np.linalg.norm(perp[lo_mask].std(axis=0)) > np.linalg.norm(perp[hi_mask].std(axis=0)):
        long_axis = -long_axis
    proj = P @ long_axis
    radial_axis = axes[1]
    # CARTO sign convention: positive x is usually LATERAL for LV (left). Use the
    # spread of LV paces along radial as anchor — we'll orient so radial PCA matches
    # any +x preference, but if the user-supplied labels disagree, no harm — what we
    # need is consistency within each patient.
    rad_proj = P @ radial_axis
    # Flip so the side with MORE LV paces (CARTO target side) is + radial
    # (since LV mapping targets are biased toward the lateral wall)
    if rad_proj.mean() < 0:
        radial_axis = -radial_axis
    ap_axis = axes[2]
    return HeartFrame(
        center=hc, long_axis=long_axis, radial_axis=radial_axis, ap_axis=ap_axis,
        long_range=(float(proj.min()), float(proj.max())),
        radial_range=(float((P@radial_axis).min()), float((P@radial_axis).max())),
        ap_range=(float((P@ap_axis).min()), float((P@ap_axis).max())),
    )


def load_charles_events(patient_dir: Path) -> tuple[list[Event], HeartFrame]:
    pid = patient_dir.name
    interv = patient_dir / "Interventions"
    carto_file = next((patient_dir / "Docs").glob("*CARTOPacingSites.txt"))
    sites = parse_carto_file(carto_file)
    # Build CARTO-anchored frame for this patient
    carto_points = np.array([v for v in sites.values()])
    carto_frame = build_carto_frame(carto_points)
    events: list[Event] = []
    for pace_dir in sorted(interv.glob("intervention*VentPace*")):
        kind = "LV" if "Left" in pace_dir.name else "RV"
        m = re.search(r"Pace(\d+)", pace_dir.name)
        if not m: continue
        n = int(m.group(1))
        site = sites.get((kind, n))
        run_files = sorted(pace_dir.glob("*.mat"))
        if not run_files: continue
        # Average across runs
        sigs = []
        for f in run_files:
            mat = sio.loadmat(f, squeeze_me=True, struct_as_record=False)
            pv = mat["bspm"].potvals   # (120, T)
            sigs.append(pv)
        T_min = min(s.shape[1] for s in sigs)
        arr = np.stack([s[:, :T_min] for s in sigs]).mean(axis=0)  # (120, T_min)
        events.append(Event(
            patient_id=pid, dataset="Charles_PSTOV",
            event_id=f"{kind}_Pace{n:02d}",
            ventricle=kind,
            pacing_xyz=site,
            pacing_in_torso_coords=False,   # CARTO coords are in heart-electrical-system, NOT torso CT
            # 2000 Hz is documented in the dataset ReadMe
            signal=arr, sampling_hz=2000, n_repeats=len(run_files),
            pacing_xyz_origin_label="CARTOPacingSites.txt",
        ))
    return events, carto_frame


