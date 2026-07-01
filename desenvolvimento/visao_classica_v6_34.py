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
# # v6_34 — Dados/tabela da seção 3.4 (refinamento incremental de features)
#
# Versão enxuta do gerador, focada apenas na seção 3.4: usa o XGBoost (melhor modelo
# da 3.3) e mede o ganho ao acrescentar grupos de features em etapas.

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
from sklearn.metrics import accuracy_score, f1_score
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


def acc_f1(cols, model_name):
    """Treina RF ou XGBoost no subconjunto de colunas e devolve (acc, F1 macro)."""
    Xc = X_full[:, [FEAT_NAMES.index(c) for c in cols]]
    Xtr, Xte = Xc[idx_tr], Xc[idx_te]
    if model_name == "RF":
        m = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)
        m.fit(Xtr, y_tr)
        pred = m.predict(Xte)
    else:  # XGB
        m = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.3, tree_method="hist",
                          random_state=42, n_jobs=-1, eval_metric="mlogloss")
        m.fit(Xtr, le.transform(y_tr), sample_weight=compute_sample_weight("balanced", y_tr))
        pred = le.inverse_transform(m.predict(Xte))
    return accuracy_score(y_te, pred) * 100, f1_score(y_te, pred, average="macro") * 100


# Grupos de features por prefixo
def por_prefixo(prefixos):
    return [n for n in FEAT_NAMES if any(n.startswith(p) for p in prefixos)]


GRUPOS = {
    "grade+LBP": por_prefixo(["grid_", "lbp_"]),
    "skew/kurt": ["skew_int", "kurt_int"],
    "Hu": por_prefixo(["hu_"]),
    "Gabor+HOG": por_prefixo(["gabor", "hog_"]),
}

# %% [markdown]
# ## 3.4 Refinamento incremental das caracteristicas (RF e XGBoost)
#
# Parte do conjunto base e acrescenta grupos de features em etapas, medindo acc e F1
# macro a cada passo, para os dois melhores modelos (RF e XGBoost). A coluna delta mostra
# a variacao em relacao ao conjunto base.

# %%
estagios = [("base", [])]
for nome_grp in ["grade+LBP", "skew/kurt", "Hu", "Gabor+HOG"]:
    estagios.append((nome_grp, GRUPOS[nome_grp]))


def benchmark_incremental(model_name):
    cols_acum, linhas = list(BASE_FEATS), []
    for nome_grp, novas in estagios:
        cols_acum = cols_acum + novas
        acc, f1m = acc_f1(cols_acum, model_name)
        rotulo = "base" if nome_grp == "base" else f"+ {nome_grp}"
        linhas.append({"etapa": rotulo, "n_feat": len(cols_acum), "acc": acc, "f1": f1m})
    base_acc, base_f1 = linhas[0]["acc"], linhas[0]["f1"]
    for r in linhas:
        r["d_acc"] = r["acc"] - base_acc
        r["d_f1"] = r["f1"] - base_f1
    return linhas


def imprime(model_name, linhas):
    print(f"\n=== 3.4 {model_name} ===")
    print(f"{'etapa':16s} {'n':>4s} {'acc':>6s} {'dacc':>6s} {'f1':>6s} {'df1':>6s}")
    for r in linhas:
        print(f"{r['etapa']:16s} {r['n_feat']:4d} {r['acc']:6.1f} {r['d_acc']:+6.1f} {r['f1']:6.1f} {r['d_f1']:+6.1f}")
    print(f"--- LaTeX ({model_name}) ---")
    for i, r in enumerate(linhas):
        if i == 0:
            print(f"{r['etapa']} & {r['n_feat']} & {r['acc']:.1f}\\% & -- & {r['f1']:.1f}\\% & -- \\\\")
        else:
            print(f"{r['etapa']} & {r['n_feat']} & {r['acc']:.1f}\\% & {r['d_acc']:+.1f} & "
                  f"{r['f1']:.1f}\\% & {r['d_f1']:+.1f} \\\\")


for mn in ["RF", "XGB"]:
    imprime(mn, benchmark_incremental(mn))
