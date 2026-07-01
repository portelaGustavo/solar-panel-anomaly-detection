# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
# ---

# %% [markdown]
# # v6_35 — Dados/tabela da seção 3.5 (classificação em duas etapas)
#
# Versão enxuta focada na seção 3.5: compara modelo único (12 classes) vs duas etapas
# (anomalia vs No-Anomaly, depois tipo), para RF e XGBoost, sobre as 181 features.

# %% [markdown]
# ## Setup, dataset e extração de features

# %%
import json
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from skimage.feature import local_binary_pattern, hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, f1_score, recall_score
from xgboost import XGBClassifier
from tqdm import tqdm

try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "InfraredSolarModules"
DATASET_URL = "https://github.com/RaptorMaps/InfraredSolarModules/raw/master/2020-02-14_InfraredSolarModules.zip"


def garantir_dataset():
    if (DATA_DIR / "module_metadata.json").exists():
        return
    zip_path = BASE_DIR / "2020-02-14_InfraredSolarModules.zip"
    if not zip_path.exists():
        urllib.request.urlretrieve(DATASET_URL, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(BASE_DIR)


garantir_dataset()
df = pd.DataFrame.from_dict(json.load(open(DATA_DIR / "module_metadata.json")), orient="index")

GABOR_KERNELS = [cv2.getGaborKernel((9, 9), 2.0, th, 4.0, 0.5, 0, ktype=cv2.CV_32F)
                 for th in np.deg2rad([0, 45, 90, 135])]


def extrair_features(img):
    img = img.astype(np.uint8)
    total = img.size
    f = {}
    # --- base (conjunto inicial, secao 3.3) ---
    f["mean_int"] = float(img.mean())
    f["std_int"] = float(img.std())
    f["max_int"] = float(img.max())
    f["min_int"] = float(img.min())
    f["p90_int"] = float(np.percentile(img, 90))
    otsu_t, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, hot = cv2.threshold(img, max(otsu_t, 180), 255, cv2.THRESH_BINARY)
    hot = cv2.morphologyEx(hot, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    f["hot_fraction"] = float(hot.sum() / 255) / total
    contornos, _ = cv2.findContours(hot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = [c for c in contornos if cv2.contourArea(c) >= 2]
    f["num_blobs"] = float(len(blobs))
    if blobs:
        maior = max(blobs, key=cv2.contourArea)
        area = cv2.contourArea(maior)
        x, y, w, h = cv2.boundingRect(maior)
        f["largest_area"] = float(area)
        f["largest_extent"] = float(area / (w * h)) if w * h > 0 else 0.0
        f["largest_aspect"] = float(w / h) if h > 0 else 0.0
    else:
        f["largest_area"] = f["largest_extent"] = f["largest_aspect"] = 0.0
    f["row_cov"] = float((hot.sum(axis=1) > 0).mean())
    f["col_cov"] = float((hot.sum(axis=0) > 0).mean())
    f["dark_fraction"] = float((img < (img.mean() - img.std())).mean())
    f["edge_density"] = float((cv2.Canny(img, 50, 150) > 0).mean())
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    f["grad_mean"] = float(np.sqrt(gx ** 2 + gy ** 2).mean())
    imgf = img.astype(np.float32)
    f["sym_lr"] = float(np.abs(imgf - np.fliplr(imgf)).mean())
    f["sym_tb"] = float(np.abs(imgf - np.flipud(imgf)).mean())
    hist = cv2.calcHist([img], [0], None, [8], [0, 256]).flatten()
    hist = hist / hist.sum()
    for i, hv in enumerate(hist):
        f[f"hist{i}"] = float(hv)
    # --- grupos adicionais (secao 3.4) ---
    f["skew_int"] = float(skew(imgf.ravel())) if img.std() > 0 else 0.0
    f["kurt_int"] = float(kurtosis(imgf.ravel())) if img.std() > 0 else 0.0
    hu = np.zeros(7)
    if blobs:
        mk = np.zeros_like(img)
        cv2.drawContours(mk, [maior], -1, 255, -1)
        huv = cv2.HuMoments(cv2.moments(mk)).flatten()
        hu = np.array([-np.sign(v) * np.log10(abs(v) + 1e-30) for v in huv])
    for i in range(7):
        f[f"hu_{i}"] = float(hu[i])
    hh, ww = img.shape
    hs, ws = hh // 3, ww // 3
    for i in range(3):
        for j in range(3):
            y0, y1 = i * hs, (hh if i == 2 else (i + 1) * hs)
            x0, x1 = j * ws, (ww if j == 2 else (j + 1) * ws)
            f[f"grid_{i}{j}"] = float(img[y0:y1, x0:x1].mean())
    lbp = local_binary_pattern(img, P=8, R=1, method="uniform")
    lh, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)
    for k, v in enumerate(lh):
        f[f"lbp_{k}"] = float(v)
    for i, kern in enumerate(GABOR_KERNELS):
        resp = cv2.filter2D(imgf, cv2.CV_32F, kern)
        f[f"gabor{i}_mean"] = float(resp.mean())
        f[f"gabor{i}_std"] = float(resp.std())
    for k, v in enumerate(hog(imgf, orientations=8, pixels_per_cell=(8, 8),
                              cells_per_block=(1, 1), feature_vector=True, channel_axis=None)):
        f[f"hog_{k}"] = float(v)
    return f


# Nomes do grupo base (conjunto inicial da secao 3.3)
BASE_FEATS = ["mean_int", "std_int", "max_int", "min_int", "p90_int", "hot_fraction",
              "num_blobs", "largest_area", "largest_extent", "largest_aspect", "row_cov",
              "col_cov", "dark_fraction", "edge_density", "grad_mean", "sym_lr", "sym_tb",
              "hist0", "hist1", "hist2", "hist3", "hist4", "hist5", "hist6", "hist7"]

registros, y = [], []
for _, row in tqdm(df.iterrows(), total=df.shape[0]):
    img = cv2.imread(str(DATA_DIR / row["image_filepath"]), cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    registros.append(extrair_features(img))
    y.append(row["anomaly_class"])
y = np.array(y)
FEAT_NAMES = list(registros[0].keys())
X_full = np.array([[r[k] for k in FEAT_NAMES] for r in registros])
print(f"Total de imagens: {len(y)} | features extraidas: {len(FEAT_NAMES)} | base: {len(BASE_FEATS)}")

# Split unico reutilizado em todas as secoes
idx = np.arange(len(y))
idx_tr, idx_te = train_test_split(idx, test_size=0.3, random_state=42, stratify=y)
y_tr, y_te = y[idx_tr], y[idx_te]
classes = sorted(set(y))
X = X_full  # usa todas as 181 features (melhor conjunto da secao 3.4)
Xtr, Xte = X[idx_tr], X[idx_te]

# Rotulos binarios (etapa 1) e mascaras de anomalia
y_bin_tr = np.where(y_tr == "No-Anomaly", "No-Anomaly", "Anomalia")
y_bin_te = np.where(y_te == "No-Anomaly", "No-Anomaly", "Anomalia")
mask_anom_tr = y_tr != "No-Anomaly"
mask_anom_te = y_te != "No-Anomaly"


def treina_prediz(Xa, ya, Xb, model_name):
    """Treina (RF ou XGB) em (Xa, ya) e prediz Xb. Abstrai o encoding do XGB."""
    if model_name == "RF":
        m = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)
        m.fit(Xa, ya)
        return m.predict(Xb)
    lo = LabelEncoder().fit(ya)
    m = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.3, tree_method="hist",
                      random_state=42, n_jobs=-1, eval_metric="mlogloss")
    m.fit(Xa, lo.transform(ya), sample_weight=compute_sample_weight("balanced", ya))
    return lo.inverse_transform(m.predict(Xb))


def avalia(pred):
    return accuracy_score(y_te, pred) * 100, f1_score(y_te, pred, average="macro") * 100


# %% [markdown]
# ## 3.5 Classificacao em duas etapas (RF e XGB)
#
# Etapa 1: anomalia vs No-Anomaly. Etapa 2: tipo (so nas anomalias). Compara modelo unico
# (12 classes de uma vez) vs duas etapas, para RF e XGBoost.

# %%
resultados = []
detector = []
for mn in ["RF", "XGB"]:
    # Modelo unico (12 classes)
    pred_uni = treina_prediz(Xtr, y_tr, Xte, mn)
    acc_u, f1_u = avalia(pred_uni)

    # Etapa 1 (binario) + etapa 2 (tipo)
    pred_bin = treina_prediz(Xtr, y_bin_tr, Xte, mn)
    pred_tipo = treina_prediz(Xtr[mask_anom_tr], y_tr[mask_anom_tr], Xte, mn)
    pred_two = np.where(pred_bin == "Anomalia", pred_tipo, "No-Anomaly")
    acc_t, f1_t = avalia(pred_two)

    # Metricas do detector binario (etapa 1)
    acc_bin = accuracy_score(y_bin_te, pred_bin) * 100
    rec_anom = recall_score(y_bin_te, pred_bin, pos_label="Anomalia") * 100

    resultados.append({"modelo": mn, "abordagem": "unico", "acc": acc_u, "f1": f1_u})
    resultados.append({"modelo": mn, "abordagem": "duas etapas", "acc": acc_t, "f1": f1_t})
    detector.append({"modelo": mn, "acc_bin": acc_bin, "recall_anom": rec_anom})

tab = pd.DataFrame(resultados)
det = pd.DataFrame(detector)
print("\n=== 3.5 Unico vs duas etapas ===")
print(tab.to_string(index=False))
print("\n=== Detector binario (etapa 1) ===")
print(det.to_string(index=False))

print("\n--- LaTeX (comparacao) ---")
for r in resultados:
    print(f"{r['modelo']} & {r['abordagem']} & {r['acc']:.1f}\\% & {r['f1']:.1f}\\% \\\\")
print("\n--- LaTeX (detector) ---")
for d in detector:
    print(f"{d['modelo']} & {d['acc_bin']:.1f}\\% & {d['recall_anom']:.1f}\\% \\\\")

# %% [markdown]
# ## 3.5 (extra) Ajuste de limiar do detector binario (XGB) — trade-off precision/recall

# %%
from sklearn.metrics import precision_score

le_bin = LabelEncoder().fit(y_bin_tr)
xgb_b = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.3, tree_method="hist",
                      random_state=42, n_jobs=-1, eval_metric="mlogloss")
xgb_b.fit(Xtr, le_bin.transform(y_bin_tr), sample_weight=compute_sample_weight("balanced", y_bin_tr))
idx_anom = list(le_bin.classes_).index("Anomalia")
proba = xgb_b.predict_proba(Xte)[:, idx_anom]
ytrue = (y_bin_te == "Anomalia").astype(int)

print("\n=== Limiar | Recall | Precision (Anomalia, XGB) ===")
print("--- LaTeX (limiar) ---")
for t in [0.50, 0.40, 0.30, 0.20, 0.10]:
    pred_t = (proba >= t).astype(int)
    rec = recall_score(ytrue, pred_t) * 100
    pre = precision_score(ytrue, pred_t, zero_division=0) * 100
    print(f"{t:.2f} & {rec:.1f}\\% & {pre:.1f}\\% \\\\")
