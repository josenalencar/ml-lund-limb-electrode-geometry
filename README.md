# Mason-Likar vs Lund limb-electrode geometry — derived data and analysis code

Derived data and analysis code supporting:

> de Alencar JN. *Geometric distortion of the limb-lead system under Mason-Likar electrode placement.* Journal of Electrocardiology (under review).

The study quantifies the displacement of the Wilson central terminal (WCT) produced by moving
the limb electrodes from a proximal (Lund) placement to the Mason-Likar placement, and the
common-mode offset that displacement adds to every precordial lead.

## What is in this repository

This repository contains **only derived quantities and analysis code**. It does **not**
redistribute any source geometry or signal data. See *Obtaining the source data* below.

```
data/
  lund_picks_corrected.json    Mason-Likar and Lund limb-electrode coordinates placed on each
                               of the 12 torso surfaces, after manual audit. Six 3-D points per
                               subject (ml_RA, ml_LA, ml_LL, lund_RA, lund_LA, lund_LL), in the
                               coordinate frame of that subject's source mesh.
  vlead_picks_audited.json     Audited precordial electrode coordinates V1-V9 for the subject
                               used in the signal-level analysis.

results/
  final_geometry.csv           Per-subject WCT displacement (components, 3-D magnitude, and
                               per-plane magnitudes), per-lead rotations, direction-only frontal
                               axis shift, and limb-lead vector magnitude ratios. N = 12.
  table1_dataset.csv           Per-subject torso antero-posterior depth, width and height, heart
                               bounding-box diagonal, and anterior WCT shift. N = 12.
  pt027_bem_signal.csv         Per-beat signed QRS integrals for the six limb leads under both
                               montages, the WCT-difference waveform energy, and per-precordial
                               contamination fractions. Subject pt027, 32 paced beats.
  bem_v1v9_direction.csv       Per-beat R- and S-wave amplitudes in V1-V9 under both montages,
                               peak of the common-mode offset, and per-lead contamination
                               fractions. Subject pt027, 32 paced beats.

code/
  charles_io.py                Readers for the Charles/PSTOV datasets, 12-lead derivation and
                               QRS detection. Shared by the scripts below.
  03_final_analysis.py         WCT displacement, lead rotations, frontal axis, limb-lead vectors.
  04_pt027_bem_signal.py       Boundary-element forward simulation, limb leads and precordial
                               contamination for pt027.
  08_geometry_predictions.py   Cohort size measures and the body-habitus regression.
  15_bem_v1v9_direction.py     Per-lead V1-V9 analysis across the 32 beats.
```

## Reproducing the results

Every statistic reported in the manuscript can be read directly from `results/*.csv`, with no
source data required. To regenerate those files, obtain the source meshes and recordings as
described below, then:

```bash
pip install -r requirements.txt
export MLLUND_DATA_ROOT=/path/to/Charles_PSTOV-12-07-27
python code/03_final_analysis.py        # cohort geometry, N = 12
python code/08_geometry_predictions.py  # size measures and habitus regression
python code/04_pt027_bem_signal.py      # forward simulation, limb leads and V1-V6
python code/15_bem_v1v9_direction.py    # V1-V9 per-lead analysis
```

`MLLUND_DATA_ROOT` must point at the directory holding the per-subject folders (`027`, `028`,
`029`) and the `ECGI_Challenge_2026_Training` directory. The scripts fail with an explicit
message if it is unset.

## Units and conventions

- Electrode and mesh coordinates: **millimetres**.
- Body-surface potentials: **millivolts** (the source `.mat` files carry an explicit `unit` field).
- Sampling rate of the pt027 body-surface recordings: **2000 Hz**, as documented in the dataset
  ReadMe. QRS windows are reported in milliseconds using that rate.
- Coordinate frame: right-handed, +X = patient's left, +Y = posterior, +Z = superior.
- WCT position is the centroid of the three limb-electrode positions; WCT potential is the mean
  of the three limb-electrode potentials.

## Obtaining the source data

Both source datasets are hosted by the **Experimental Data and Geometric Analysis Repository
(EDGAR)**, https://edgar.sci.utah.edu, which is open to the public and requires only a free
self-service account.

1. **Charles University / Petr Stovicek (PSTOV) cases pt027, pt028, pt029** — the Charles
   University repository within EDGAR. Subject pt027 is the only case with a boundary-element
   forward matrix, and is therefore the only subject in the signal-level analysis.
2. **ECGI Challenge 2026 training set** (P001, P002, P004, P006, P010, P020, P023, P024, P027) —
   the Challenge release hosted on EDGAR, downloaded for this work in **May 2026**. Access is not
   restricted to Challenge participants. The release is a community distribution rather than a
   versioned archive: it carries no DOI, no version tag and no published data descriptor. These
   nine subjects are drawn from a cohort of thirteen patients with spontaneous premature
   ventricular contractions studied by 128-lead body-surface potential mapping at the Institute of
   Measurement Science, Slovak Academy of Sciences, described in Ondrusova B, Tino P,
   Svehlikova J. *A two-step inverse solution for a single dipole cardiac source.* Front Physiol
   2023;14:1264690. https://doi.org/10.3389/fphys.2023.1264690

## Acknowledgements

This work used data from the Experimental Data and Geometric Analysis Repository (EDGAR),
curated by the Center for Integrative Biomedical Computing at the Scientific Computing and
Imaging Institute, University of Utah. Please cite:

> Aras K, Good W, Tate J, Burton B, Brooks D, Coll-Font J, et al. Experimental Data and Geometric
> Analysis Repository — EDGAR. *J Electrocardiol* 2015;48:975-81.
> https://doi.org/10.1016/j.jelectrocard.2015.08.008

The author gratefully acknowledges the data collection performed by Petr Stovicek, S. Havranek
and J. Simek, from the Second Department of Medicine, Department of Cardiovascular Medicine,
First Faculty of Medicine, Charles University in Prague and General University Hospital in Prague,
Czech Republic, and by D. Wichterle, from the Department of Cardiology, Institute for Clinical and
Experimental Medicine, Prague, Czech Republic.

The author further acknowledges the Institute of Measurement Science of the Slovak Academy of
Sciences, Bratislava, for the patient geometries and electrode arrays released through the ECGI
Challenge 2026, and the Consortium for ECG Imaging for organising that release.

## Known limitations of the deposited quantities

- The Lund electrode positions are anatomic landmark picks placed and audited on mesh surfaces,
  not physical electrode recordings. The source publications define the Lund placement
  anatomically rather than metrically, and these picks lie towards the lateral extreme of that
  range, which widens the Mason-Likar-to-Lund separation.
- The pt027 forward model is defined on a 352-node torso surface with a median internode spacing
  of about 33 mm, so electrode positions are snapped to the nearest node, by up to 41 mm. The WCT
  displacement realised in the signal-level analysis is 71 mm anteriorly, against 74 mm for the
  same subject's positions as placed.
- All 32 beats are paced and come from a single subject. They characterise the stability of the
  effect across activation sequences, not its variability across people.

## Licence

- Code (`code/`): MIT.
- Derived data (`data/`, `results/`): Creative Commons Attribution 4.0 International (CC BY 4.0).

Neither licence extends to the source EDGAR or ECGI Challenge datasets, which remain subject to
the terms of their respective providers.

## Contact

José Nunes de Alencar, MD — Instituto Dante Pazzanese de Cardiologia, São Paulo, Brazil
jose.alencar@dantepazzanese.org.br
