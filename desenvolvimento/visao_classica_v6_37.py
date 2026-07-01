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
# # v6_36 — Figuras/tabela da seção 3.6 (análise do XGBoost)
#
# Treina o XGBoost único nas 181 features e gera as matrizes de confusão (por coluna e
# por linha) e a tabela das maiores confusões por classe.

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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
from skimage.feature import local_binary_pattern, hog
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
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
le = LabelEncoder().fit(y)
REL = BASE_DIR / "Relatorio_visao_computacional"
Xtr, Xte = X_full[idx_tr], X_full[idx_te]


def treina_xgb(cols_idx):
    m = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.3, tree_method="hist",
                      random_state=42, n_jobs=-1, eval_metric="mlogloss")
    m.fit(Xtr[:, cols_idx], le.transform(y_tr), sample_weight=compute_sample_weight("balanced", y_tr))
    pred = le.inverse_transform(m.predict(Xte[:, cols_idx]))
    return m, accuracy_score(y_te, pred) * 100, f1_score(y_te, pred, average="macro") * 100


# %% [markdown]
# ## 3.7 Otimizacao por remocao de features (XGBoost)

# %%
from sklearn.inspection import permutation_importance

todas = list(range(len(FEAT_NAMES)))
m_full, acc_full, f1_full = treina_xgb(todas)
print(f"Completo ({len(FEAT_NAMES)} feat): acc {acc_full:.1f}%  F1 macro {f1_full:.1f}%")

perm = permutation_importance(m_full, Xte, le.transform(y_te), n_repeats=5,
                              random_state=42, scoring="f1_macro", n_jobs=-1)
imp = pd.DataFrame({"feature": FEAT_NAMES, "imp": perm.importances_mean}).sort_values("imp")
manter = [FEAT_NAMES.index(f) for f in imp[imp["imp"] > 0]["feature"]]
n_rem = len(FEAT_NAMES) - len(manter)
print(f"Features com importancia <= 0 (removidas): {n_rem}")

m_red, acc_red, f1_red = treina_xgb(manter)
print(f"Reduzido ({len(manter)} feat): acc {acc_red:.1f}%  F1 macro {f1_red:.1f}%")
print(f"delta: acc {acc_red-acc_full:+.1f}  F1 macro {f1_red-f1_full:+.1f}")

print("\n--- LaTeX ---")
print(f"Completo & {len(FEAT_NAMES)} & {acc_full:.1f}\\% & -- & {f1_full:.1f}\\% & -- \\\\")
print(f"Reduzido & {len(manter)} & {acc_red:.1f}\\% & {acc_red-acc_full:+.1f} & {f1_red:.1f}\\% & {f1_red-f1_full:+.1f} \\\\")

# %%
# Grafico: 12 features mais e menos importantes
extremos = pd.concat([imp.head(12), imp.tail(12)])
plt.figure(figsize=(9, 9))
cores = ["crimson" if v <= 0 else "steelblue" for v in extremos["imp"]]
plt.barh(extremos["feature"], extremos["imp"], color=cores)
plt.axvline(0, color="black", lw=0.8)
plt.title("Importancia por permutacao (vermelho = inutil/atrapalha)")
plt.xlabel("Queda no F1 macro ao embaralhar a feature")
plt.tight_layout()
plt.savefig(REL / "importancia_features.png", dpi=130)
plt.close()
print("[salvo]", REL / "importancia_features.png")
