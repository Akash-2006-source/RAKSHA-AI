"""Experiments 1-4 for RAKSHA. Outputs figures + results.json to reports/."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from features import extract

traj = pd.read_csv("data/processed/trajectories.csv.gz")
meta = pd.read_csv("data/processed/scenarios.csv")
OUT = {}

TTC_ONLY = ["ttc_last"]
KIN = ["dist_last", "clr_last", "closing_last", "speed_a", "speed_b", "rel_speed", "acc_a", "acc_b",
       "acc_a_min", "acc_b_min", "jerk_a"]
SSM = ["ttc_min", "ttc_last", "pet_min", "pet_last", "drac_max", "drac_last"]
CTX = ["crossing", "ta_bus", "ta_car", "ta_motorcycle", "ta_truck", "tb_bus", "tb_car", "tb_motorcycle",
       "tb_pedestrian", "tb_truck"]


def prep(h):
    f = extract(traj, meta, horizon=h)
    f = pd.concat([f, pd.get_dummies(f.type_a, prefix="ta").astype(int),
                   pd.get_dummies(f.type_b, prefix="tb").astype(int)], axis=1)
    for c in CTX:
        if c not in f: f[c] = 0
    return f


def models():
    return {
        "LogReg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "RandomForest": RandomForestClassifier(300, min_samples_leaf=3, class_weight="balanced", n_jobs=-1, random_state=0),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                                 colsample_bytree=0.8, eval_metric="logloss", random_state=0),
    }


def split(f):
    return train_test_split(f, test_size=0.3, stratify=f.event, random_state=0)


def score(y, p):
    best = max(np.linspace(0.05, 0.95, 19), key=lambda t: f1_score(y, p > t))
    return dict(auc=roc_auc_score(y, p), ap=average_precision_score(y, p), f1=f1_score(y, p > best))


HORIZON = 2.0  # predict 2 s before closest approach (h=0 would leak the label via clearance)
f0 = prep(HORIZON)
tr, te = split(f0)
print("event rate", f0.event.mean())
OUT["event_rate"] = float(f0.event.mean())

# ---- Exp 1: TTC-only vs ML
FULL = SSM + KIN + CTX
exp1 = {}
# TTC rule baseline: smaller TTC => riskier (score = -ttc)
exp1["TTC threshold rule"] = score(te.event, 1 - te.ttc_last / 30)
for name, m in models().items():
    m.fit(tr[TTC_ONLY], tr.event); exp1[f"{name} (TTC only)"] = score(te.event, m.predict_proba(te[TTC_ONLY])[:, 1])
fitted = {}
for name, m in models().items():
    m.fit(tr[FULL], tr.event); fitted[name] = m
    exp1[f"{name} (full)"] = score(te.event, m.predict_proba(te[FULL])[:, 1])
OUT["exp1"] = exp1

# ---- Exp 2: feature-group ablation with XGBoost
groups = {"TTC": ["ttc_min", "ttc_last"], "TTC+PET": ["ttc_min", "ttc_last", "pet_min", "pet_last"],
          "TTC+PET+DRAC": SSM, "TTC+PET+DRAC+kinematics": SSM + KIN, "All + context": FULL}
exp2 = {}
for g, cols in groups.items():
    m = models()["XGBoost"].fit(tr[cols], tr.event)
    exp2[g] = score(te.event, m.predict_proba(te[cols])[:, 1])
OUT["exp2"] = exp2

# ---- Exp 3: lead time (train and test at each horizon; TTC rule as baseline)
exp3 = {}
for h in [1, 2, 3, 4, 5]:
    f = prep(float(h)); a, b = split(f)
    m = models()["XGBoost"].fit(a[FULL], a.event)
    exp3[h] = dict(xgb=score(b.event, m.predict_proba(b[FULL])[:, 1])["auc"],
                   ttc_rule=roc_auc_score(b.event, 1 - b.ttc_last / 30))
OUT["exp3"] = exp3

# ---- Exp 4: vehicle-combination risk
combo = f0.assign(combo=f0.type_a + " vs " + f0.type_b)
c = combo.groupby("combo").event.agg(["mean", "count"]).query("count>=40").sort_values("mean", ascending=False)
OUT["exp4"] = c.round(3).to_dict("index")
m = fitted["XGBoost"]
te_c = te.assign(p=m.predict_proba(te[FULL])[:, 1], combo=te.type_a + " vs " + te.type_b)
OUT["exp4_model_auc_by_kind"] = {k: float(roc_auc_score(g.event, g.p)) for k, g in te_c.groupby("crossing") if g.event.nunique() > 1}

# ---- figures
fig, ax = plt.subplots(figsize=(8, 4.5))
labels = list(exp1); ax.barh(labels, [exp1[k]["auc"] for k in labels], color="#c0392b")
ax.set_xlabel("ROC-AUC (test)"); ax.set_title("Exp 1: TTC-only vs ML"); ax.set_xlim(0.5, 1)
plt.tight_layout(); plt.savefig("reports/exp1.png", dpi=130); plt.close()

fig, ax = plt.subplots(figsize=(7, 4)); hs = list(exp3)
ax.plot(hs, [exp3[h]["xgb"] for h in hs], "o-", label="XGBoost (full features)")
ax.plot(hs, [exp3[h]["ttc_rule"] for h in hs], "s--", label="TTC-only rule")
ax.set_xlabel("Lead time before closest approach (s)"); ax.set_ylabel("ROC-AUC"); ax.legend(); ax.grid(alpha=.3)
ax.set_title("Exp 3: early prediction"); plt.tight_layout(); plt.savefig("reports/exp3.png", dpi=130); plt.close()

fig, ax = plt.subplots(figsize=(8, 5)); c2 = c.head(15)[::-1]
ax.barh(c2.index, c2["mean"], color="#e67e22"); ax.set_xlabel("Near-miss rate"); ax.set_title("Exp 4: risk by vehicle pair")
plt.tight_layout(); plt.savefig("reports/exp4.png", dpi=130); plt.close()

fig, axs = plt.subplots(1, 3, figsize=(13, 4))
for a_, (n, mm) in zip(axs, fitted.items()):
    ConfusionMatrixDisplay(confusion_matrix(te.event, mm.predict_proba(te[FULL])[:, 1] > 0.5)).plot(ax=a_, colorbar=False)
    a_.set_title(n)
plt.tight_layout(); plt.savefig("reports/confusion.png", dpi=130); plt.close()

expl = shap.TreeExplainer(m)
sv = expl.shap_values(te[FULL].iloc[:600])
plt.figure(); shap.summary_plot(sv, te[FULL].iloc[:600], show=False); plt.tight_layout()
plt.savefig("reports/shap.png", dpi=130); plt.close()
imp = pd.Series(np.abs(sv).mean(0), index=FULL).sort_values(ascending=False)
OUT["top_shap"] = imp.head(8).round(3).to_dict()

te_c[["scenario", "type_a", "type_b", "crossing", "event", "p"]].to_csv("data/processed/test_predictions.csv", index=False)
joblib.dump(dict(model=m, features=FULL), "data/processed/xgb_model.joblib")
json.dump(OUT, open("reports/results.json", "w"), indent=2, default=float)
print(json.dumps(OUT, indent=1, default=float))

