"""
The 12-subject cohort and the surface-mesh readers shared by the analysis scripts.

SUBJECTS lists every model in a fixed order: (sequence, label, cohort, subject id).
The label is the key used throughout data/lund_picks_corrected.json.

Source meshes are located through charles_io.data_root(). Charles/PSTOV surfaces are
registered MATLAB structures; ECGI Challenge surfaces are per-subject torso and
endocardial-epicardial files. Both are returned as (N, 3) arrays in millimetres,
with +X = patient's left, +Y = posterior, +Z = superior.
"""
from __future__ import annotations
import numpy as np
import scipy.io as sio
import charles_io

SUBJECTS = [
    ("001", "Charles pt027", "charles", "pt027"),
    ("002", "Charles pt028", "charles", "pt028"),
    ("003", "Charles pt029", "charles", "pt029"),
    ("004", "ECGI P001", "ecgi", "P001"),
    ("005", "ECGI P002", "ecgi", "P002"),
    ("006", "ECGI P004", "ecgi", "P004"),
    ("007", "ECGI P006", "ecgi", "P006"),
    ("008", "ECGI P010", "ecgi", "P010"),
    ("009", "ECGI P020", "ecgi", "P020"),
    ("010", "ECGI P023", "ecgi", "P023"),
    ("011", "ECGI P024", "ecgi", "P024"),
    ("012", "ECGI P027", "ecgi", "P027"),
]

# Retained for scripts that iterate the cohort without the sequence field.
ORDER = SUBJECTS


def force_Nx3(a):
    """Return an (N, 3) array, transposing the (3, N) layout some meshes use."""
    a = np.asarray(a)
    return a.T if (a.ndim == 2 and a.shape[0] == 3 and a.shape[1] > 3) else a


def _first_struct(mat):
    return mat[[k for k in mat if not k.startswith("__")][0]]


def load_charles(pid):
    """Torso and myocardial surfaces for a Charles/PSTOV subject."""
    root = charles_io.data_root()
    num = pid[2:]
    suf = num.lstrip("0") or "0"
    g = sio.loadmat(str(root / f"{num}/Meshes/2012-07-{suf}Subject_daltorso_registered.mat"),
                    squeeze_me=True, struct_as_record=False)["geom"]
    h = _first_struct(sio.loadmat(
        str(root / f"{num}/Meshes/2012-07-{suf}Subject_myocardium_registered.mat"),
        squeeze_me=True, struct_as_record=False))
    return force_Nx3(g.node), force_Nx3(h.node if hasattr(h, "node") else h)


def load_ecgi(pid):
    """Torso and ventricular surfaces for an ECGI Challenge subject."""
    root = charles_io.data_root()
    base = root / "ECGI_Challenge_2026_Training/Geometries"
    t = sio.loadmat(str(base / f"{pid}_torso.mat"),
                    squeeze_me=True, struct_as_record=False)["torso"]
    e = _first_struct(sio.loadmat(str(base / f"{pid}_EndoEpi.mat"),
                                  squeeze_me=True, struct_as_record=False))
    return force_Nx3(t.node), force_Nx3(e.node if hasattr(e, "node") else e)


def load_surfaces(cohort, pid):
    """Dispatch on cohort."""
    return load_charles(pid) if cohort == "charles" else load_ecgi(pid)
