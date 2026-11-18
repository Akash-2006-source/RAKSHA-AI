"""Surrogate safety measures (TTC, PET, DRAC) + features at a given prediction horizon.

For each scenario we build a feature row from a window of frames that ends
`horizon` seconds BEFORE the event time (t_event = time of minimum clearance).
Experiment 3 sweeps horizon in {0,1,2,3,4,5}.
"""
import numpy as np
import pandas as pd

DT = 0.1
WINDOW = 1.5  # seconds of history summarised
CAP = 30.0    # TTC / PET cap (s)


def _pair_measures(g):
    ax, ay, bx, by = g.ax.values, g.ay.values, g.bx.values, g.by.values
    avx, avy, bvx, bvy = g.avx.values, g.avy.values, g.bvx.values, g.bvy.values
    rad = g.ra.values + g.rb.values
    rx, ry = bx - ax, by - ay
    rvx, rvy = avx - bvx, avy - bvy  # closing velocity of A toward B
    dist = np.hypot(rx, ry)
    clr = np.maximum(dist - rad, 0.01)
    closing = (rx * rvx + ry * rvy) / np.maximum(dist, 1e-3)  # >0 means approaching
    # 2D TTC: time until separation < rad assuming constant velocities
    vv = rvx ** 2 + rvy ** 2
    b = rx * rvx + ry * rvy
    c = rx ** 2 + ry ** 2 - rad ** 2
    disc = b ** 2 - vv * c
    with np.errstate(divide="ignore", invalid="ignore"):
        ttc = np.where((vv > 1e-6) & (disc >= 0) & (b > 0), (b - np.sqrt(np.maximum(disc, 0))) / vv, CAP)
    ttc = np.clip(np.where(c < 0, 0.0, ttc), 0, CAP)
    drac = np.where(closing > 0, closing ** 2 / (2 * clr), 0.0)
    return dist, clr, closing, ttc, drac


def _pet(g):
    """Post-encroachment time at the origin (crossing) / headway proxy (following).
    Computed from the full scenario: time gap between A and B occupying the conflict area."""
    ax, ay, bx, by = g.ax.values, g.ay.values, g.bx.values, g.by.values
    t = g.t.values
    rad = (g.ra.values + g.rb.values)
    if np.allclose(ay, by):  # following: headway based PET = gap / follower speed
        gap = np.maximum(bx - ax - rad, 0.01)
        spd = np.maximum(g.avx.values, 0.5)
        return np.clip(gap / spd, 0, CAP)
    inA = np.abs(ax) < 1.5 + g.ra.values
    inB = np.abs(by) < 1.5 + g.rb.values
    pet = np.full(len(t), CAP)
    if inA.any() and inB.any():
        ta0, ta1 = t[inA][0], t[inA][-1]; tb0, tb1 = t[inB][0], t[inB][-1]
        val = (tb0 - ta1) if tb0 >= ta1 else ((ta0 - tb1) if ta0 >= tb1 else 0.0)
        # PET only becomes "known" after both have passed; for a predictive setting
        # we expose it as a projected PET from current kinematics instead:
    # projected PET: |ETA_A - ETA_B| to the conflict point at constant velocity
    etaA = np.where(avx_ok(g.avx.values), -ax / np.maximum(g.avx.values, 0.1), CAP)
    etaB = np.where(avx_ok(g.bvy.values), -by / np.maximum(g.bvy.values, 0.1), CAP)
    ok = (etaA > 0) & (etaB > 0)
    pet = np.where(ok, np.abs(etaA - etaB), CAP)
    return np.clip(pet, 0, CAP)


def avx_ok(v):
    return v > 0.1


def extract(traj: pd.DataFrame, meta: pd.DataFrame, horizon: float = 0.0) -> pd.DataFrame:
    rows = []
    meta_i = meta.set_index("scenario")
    for sid, g in traj.groupby("scenario", sort=False):
        m = meta_i.loc[sid]
        dist, clr, closing, ttc, drac = _pair_measures(g)
        pet = _pet(g)
        end = int(round(m.t_event / DT)) - int(round(horizon / DT))
        end = max(end, int(WINDOW / DT))
        s = slice(end - int(WINDOW / DT), end + 1)
        speedA = np.hypot(g.avx.values, g.avy.values)
        speedB = np.hypot(g.bvx.values, g.bvy.values)
        accA = np.gradient(speedA, DT); accB = np.gradient(speedB, DT)
        rows.append(dict(
            scenario=sid,
            ttc_min=ttc[s].min(), ttc_last=ttc[end], pet_min=pet[s].min(), pet_last=pet[end],
            drac_max=drac[s].max(), drac_last=drac[end],
            dist_last=dist[end], clr_last=clr[end], closing_last=closing[end],
            speed_a=speedA[end], speed_b=speedB[end], rel_speed=abs(speedA[end] - speedB[end]),
            acc_a=accA[end], acc_b=accB[end], acc_a_min=accA[s].min(), acc_b_min=accB[s].min(),
            jerk_a=np.abs(np.diff(accA[s])).mean() / DT,
            crossing=int(not np.allclose(g.ay.values, g.by.values)),
        ))
    f = pd.DataFrame(rows).set_index("scenario")
    f = f.join(meta_i[["type_a", "type_b", "event"]])
    return f.reset_index()

