"""
EXTENDED pt027 BEM signal analysis — V1..V9 + precordial R/S direction, over the
32 REAL paced beats (promotes the V7-V9 contamination and the precordial
up/down direction from the single synthetic beat to real-source, n=32).

Same BEM forward matrix and audited placements as 04_pt027_bem_signal.py.
For each beat:
  - reconstruct body-surface field  src = inv@signal ; body = A@src
  - WCT_ML, WCT_Lund from limb-electrode nodes
  - precordial V_i = Phi_i - WCT  (Phi_i = body at nearest torso node to audited V_i)
      computed under BOTH montages -> R, S, peak-to-peak over the QRS window
  - common-mode offset dWCT = WCT_Lund - WCT_ML ; per-lead contamination = dWCT energy / native energy
Outputs -> ../results/bem_v1v9_direction.{csv,md}
"""
from __future__ import annotations
import sys, re, json
from pathlib import Path
import numpy as np, pandas as pd, scipy.io as sio

PROJ = Path(__file__).resolve().parents[1]; DATA = PROJ/"data"; RES = PROJ/"results"
import charles_io as wdp

DS = None  # resolved in main() from MLLUND_DATA_ROOT
LIMB=("I","II","III","aVR","aVL","aVF"); VALL=("V1","V2","V3","V4","V5","V6","V7","V8","V9")

def nnode(torso,xyz): return int(np.linalg.norm(torso-np.asarray(xyz,float),axis=1).argmin())
def RS(w,qs,qe):
    seg=w[qs:qe+1]
    return float(max(seg.max(),0.0)), float(min(seg.min(),0.0)), float(seg.max()-seg.min())

def main():
    DS=wdp.data_root()
    pt=DS/"027"; geom=wdp.load_charles_geometry(pt); torso=geom.torso_pts
    A=sio.loadmat(pt/"FwdInvTransforms/2012-07-27Subject_forward_matrix.mat",
                  squeeze_me=True,struct_as_record=False)["A"]
    if A.shape[0]<A.shape[1]: A=A.T
    ch=geom.extra["channels_into_torso"]; A_m=A[ch,:]
    lam=np.linalg.svd(A_m,full_matrices=False,compute_uv=False)[0]/100.0
    inv=np.linalg.inv(A_m.T@A_m+lam**2*np.eye(A.shape[1]))@A_m.T

    picks=json.loads((DATA/"lund_picks_corrected.json").read_text())["Charles pt027"]
    nodes={c:{e:nnode(torso,picks[f"{p}_{e}"]) for e in ("RA","LA","LL")}
           for c,p in (("ML","ml"),("Lund","lund"))}
    audV=json.loads((DATA/"vlead_picks_audited.json").read_text())["Charles pt027"]
    vnode={v:nnode(torso,audV[v]) for v in VALL if v in audV}

    events,_=wdp.load_charles_events(pt)

    rows=[]
    for ev in events:
        body=A@(inv@ev.signal)
        def wct(cfg):
            n=nodes[cfg]; return (body[n["RA"]]+body[n["LA"]]+body[n["LL"]])/3.0
        wML, wLU = wct("ML"), wct("Lund")
        # QRS window from the Lund limb leads
        limbLU=dict(I=body[nodes["Lund"]["LA"]]-body[nodes["Lund"]["RA"]],
                    II=body[nodes["Lund"]["LL"]]-body[nodes["Lund"]["RA"]],
                    III=body[nodes["Lund"]["LL"]]-body[nodes["Lund"]["LA"]],
                    aVR=body[nodes["Lund"]["RA"]]-wLU, aVL=body[nodes["Lund"]["LA"]]-wLU,
                    aVF=body[nodes["Lund"]["LL"]]-wLU)
        qs,qe=wdp.detect_qrs(limbLU)
        dwct=(wLU-wML)[qs:qe+1].astype(float); dE=float(np.sum(dwct**2))
        row=dict(event_id=ev.event_id, qs=int(qs), qe=int(qe), dWCT_energy=dE,
                 dWCT_peak=float(np.max(np.abs(dwct))))
        for v in vnode:
            phi=body[vnode[v]]
            rL,sL,pL=RS(phi-wLU,qs,qe); rM,sM,pM=RS(phi-wML,qs,qe)
            nativeE=float(np.sum(((phi-wLU)[qs:qe+1])**2))
            row.update({f"{v}_R_Lund":rL,f"{v}_S_Lund":sL,f"{v}_ptp_Lund":pL,
                        f"{v}_R_ML":rM,f"{v}_S_ML":sM,f"{v}_ptp_ML":pM,
                        f"{v}_contam_pct":100.0*dE/(nativeE+1e-12)})
        rows.append(row)
    df=pd.DataFrame(rows); df.to_csv(RES/"bem_v1v9_direction.csv",index=False); n=len(df)

    L=[f"# pt027 BEM — V1-V9 precordial direction & contamination, {n} REAL paced beats\n",
       f"Offset peak {df['dWCT_peak'].mean():.3f}±{df['dWCT_peak'].std():.3f} mV.\n",
       "## Per-lead contamination % (median across beats) and R/S change ML vs Lund\n",
       "| Lead | contam% (median) | R Lund→ML (mean) | S Lund→ML (mean) | ptp ML/Lund |",
       "|---|---:|---:|---:|---:|"]
    for v in VALL:
        if f"{v}_contam_pct" not in df: continue
        cm=df[f"{v}_contam_pct"].median()
        rL,rM=df[f"{v}_R_Lund"].mean(),df[f"{v}_R_ML"].mean()
        sL,sM=df[f"{v}_S_Lund"].mean(),df[f"{v}_S_ML"].mean()
        pr=df[f"{v}_ptp_ML"].mean()/(df[f"{v}_ptp_Lund"].mean()+1e-12)
        L.append(f"| {v} | {cm:.0f}% | {rL:.3f}→{rM:.3f} | {sL:.3f}→{sM:.3f} | {pr:.2f} |")
    (RES/"bem_v1v9_direction.md").write_text("\n".join(L))
    print("\n".join(L)); print(f"\nWrote bem_v1v9_direction.{{csv,md}} (n={n})")

if __name__=="__main__": main()
