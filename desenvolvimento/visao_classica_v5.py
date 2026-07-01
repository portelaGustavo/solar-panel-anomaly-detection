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
# # Visão Clássica v5 — Análise de confusão de classes (XGBoost)
#
# Fixamos o **XGBoost** (melhor modelo) e mantemos todas as features (otimização de
# features fica para o final). O foco aqui é a **confusão entre classes**: quais o
# modelo mais troca entre si (top 5 por classe), para decidir quais vale agrupar.
#
# Pipeline: features → duas etapas → modelo único → análise de confusão.

# %% [markdown]
# ## 1. Setup e download do dataset

# %%
import json
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
from skimage.feature import local_binary_pattern, hog
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score)

try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:  # notebook (Jupyter/Colab)
    BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "InfraredSolarModules"
DATASET_URL = "https://github.com/RaptorMaps/InfraredSolarModules/raw/master/2020-02-14_InfraredSolarModules.zip"


def garantir_dataset():
    if (DATA_DIR / "module_metadata.json").exists():
        print(f"Dataset ja presente em {DATA_DIR}")
        return
    zip_path = BASE_DIR / "2020-02-14_InfraredSolarModules.zip"
    if not zip_path.exists():
        print(f"Baixando dataset de {DATASET_URL} ...")
        urllib.request.urlretrieve(DATASET_URL, zip_path)
    print("Extraindo dataset ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(BASE_DIR)
    print(f"Dataset pronto em {DATA_DIR}")


garantir_dataset()

# %% [markdown]
# ## 2. Leitura do JSON (12 classes, sem agrupamento)

# %%
with open(DATA_DIR / "module_metadata.json", "r") as f:
    metadados = json.load(f)
df = pd.DataFrame.from_dict(metadados, orient="index")
print(df.head())
print("\n--- CONTAGEM DE CLASSES ---")
print(df["anomaly_class"].value_counts())

# %% [markdown]
# ## 3. Extração de features
#
# Intensidade (média, desvio, skew, kurtosis), região quente (Otsu), momentos de Hu,
# bordas (Canny), textura (Sobel), simetria, histograma, grade espacial 3x3, LBP,
# Gabor (textura direcional) e HOG (formas/trincas).

# %%
HOT_FLOOR = 200
# Banco de Gabor: 4 orientacoes (textura direcional)
GABOR_KERNELS = [cv2.getGaborKernel((9, 9), 2.0, th, 4.0, 0.5, 0, ktype=cv2.CV_32F)
                 for th in np.deg2rad([0, 45, 90, 135])]


def extrair_features(img):
    img = img.astype(np.uint8)
    total = img.size
    f = {}

    # Intensidade
    f["mean_int"] = float(img.mean())
    f["std_int"] = float(img.std())
    f["max_int"] = float(img.max())
    f["min_int"] = float(img.min())
    f["p90_int"] = float(np.percentile(img, 90))
    # Forma da distribuicao de intensidade
    flat = img.flatten().astype(np.float32)
    f["skew_int"] = float(skew(flat)) if img.std() > 0 else 0.0
    f["kurt_int"] = float(kurtosis(flat)) if img.std() > 0 else 0.0

    # Regiao quente (Otsu com piso)
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

    # Momentos de Hu do maior blob (forma invariante a escala/rotacao), log-transformados
    hu = np.zeros(7)
    if blobs:
        mask = np.zeros_like(img)
        cv2.drawContours(mask, [maior], -1, 255, -1)
        huv = cv2.HuMoments(cv2.moments(mask)).flatten()
        hu = np.array([-np.sign(v) * np.log10(abs(v) + 1e-30) for v in huv])
    for i in range(7):
        f[f"hu_{i}"] = float(hu[i])

    # Regiao escura, bordas, textura
    f["dark_fraction"] = float((img < (img.mean() - img.std())).mean())
    f["edge_density"] = float((cv2.Canny(img, 50, 150) > 0).mean())
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    f["grad_mean"] = float(np.sqrt(gx ** 2 + gy ** 2).mean())

    # Simetria
    imgf = img.astype(np.float32)
    f["sym_lr"] = float(np.abs(imgf - np.fliplr(imgf)).mean())
    f["sym_tb"] = float(np.abs(imgf - np.flipud(imgf)).mean())

    # Histograma (8 bins)
    hist = cv2.calcHist([img], [0], None, [8], [0, 256]).flatten()
    hist = hist / hist.sum()
    for i, hv in enumerate(hist):
        f[f"hist{i}"] = float(hv)

    # Grade 3x3 (localizacao do calor)
    h, w = img.shape
    hs, ws = h // 3, w // 3
    for i in range(3):
        for j in range(3):
            y0, y1 = i * hs, (h if i == 2 else (i + 1) * hs)
            x0, x1 = j * ws, (w if j == 2 else (j + 1) * ws)
            f[f"grid_{i}{j}"] = float(img[y0:y1, x0:x1].mean())

    # LBP (textura)
    lbp = local_binary_pattern(img, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)
    for k, v in enumerate(lbp_hist):
        f[f"lbp_{k}"] = float(v)

    # Gabor: mean/std da resposta de cada orientacao (textura direcional)
    imgf32 = img.astype(np.float32)
    for i, kern in enumerate(GABOR_KERNELS):
        resp = cv2.filter2D(imgf32, cv2.CV_32F, kern)
        f[f"gabor{i}_mean"] = float(resp.mean())
        f[f"gabor{i}_std"] = float(resp.std())

    # HOG: histograma de gradientes orientados (formas, trincas)
    hog_vec = hog(imgf32, orientations=8, pixels_per_cell=(8, 8),
                  cells_per_block=(1, 1), feature_vector=True, channel_axis=None)
    for k, v in enumerate(hog_vec):
        f[f"hog_{k}"] = float(v)

    return f


_amostra = cv2.imread(str(DATA_DIR / df.iloc[0]["image_filepath"]), cv2.IMREAD_GRAYSCALE)
FEAT_NAMES = list(extrair_features(_amostra).keys())
print(f"{len(FEAT_NAMES)} features por imagem")

# %% [markdown]
# ## 4. Matriz de features de todo o dataset

# %%
from tqdm import tqdm

registros, y = [], []
for _, row in tqdm(df.iterrows(), total=df.shape[0]):
    img = cv2.imread(str(DATA_DIR / row["image_filepath"]), cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    registros.append(extrair_features(img))
    y.append(row["anomaly_class"])

X = np.array([[r[k] for k in FEAT_NAMES] for r in registros])
y = np.array(y)
print("X:", X.shape, "| y:", y.shape)

# Split unico, reutilizado nas duas etapas e na comparacao
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
classes = sorted(set(y))

# %%
# Preparacao dos rotulos para as duas etapas (model-agnostico)
y_bin_tr = np.where(y_tr == "No-Anomaly", "No-Anomaly", "Anomalia")
y_bin_te = np.where(y_te == "No-Anomaly", "No-Anomaly", "Anomalia")
mask_anom_tr = y_tr != "No-Anomaly"
mask_anom_te = y_te != "No-Anomaly"

# %% [markdown]
# ## 5. Classificação com XGBoost
#
# Etapa 1 binária, etapa 2 isolada, combinada (relatório + matriz) e o modelo único.
# XGBoost precisa de rótulos numéricos (`LabelEncoder`) e usa `sample_weight` para
# compensar o desbalanceamento (não tem `class_weight`).

# %%
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight


def make_xgb():
    return XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.3,
                         tree_method="hist", random_state=42, n_jobs=-1, eval_metric="mlogloss")


# %% [markdown]
# ### 5.1 XGBoost — Etapa 1: Anomalia vs No-Anomaly

# %%
le_bin = LabelEncoder().fit(y_bin_tr)
xgb_bin = make_xgb()
xgb_bin.fit(X_tr, le_bin.transform(y_bin_tr), sample_weight=compute_sample_weight("balanced", y_bin_tr))
pred_bin_xgb = le_bin.inverse_transform(xgb_bin.predict(X_te))

print(f"Etapa 1 XGB (binaria) — acuracia: {accuracy_score(y_bin_te, pred_bin_xgb) * 100:.1f}%")
print(f"Recall de 'Anomalia': {recall_score(y_bin_te, pred_bin_xgb, pos_label='Anomalia') * 100:.1f}%")
print("\n--- Relatorio etapa 1 (XGB) ---")
print(classification_report(y_bin_te, pred_bin_xgb, zero_division=0))

# %% [markdown]
# ### 5.2 XGBoost — Etapa 2 isolada (tipo, gate perfeito)

# %%
le_tipo = LabelEncoder().fit(y_tr[mask_anom_tr])
xgb_tipo = make_xgb()
xgb_tipo.fit(X_tr[mask_anom_tr], le_tipo.transform(y_tr[mask_anom_tr]),
             sample_weight=compute_sample_weight("balanced", y_tr[mask_anom_tr]))
pred_tipo_iso_xgb = le_tipo.inverse_transform(xgb_tipo.predict(X_te[mask_anom_te]))

print(f"Etapa 2 isolada XGB — acuracia: {accuracy_score(y_te[mask_anom_te], pred_tipo_iso_xgb) * 100:.1f}% | "
      f"F1 macro: {f1_score(y_te[mask_anom_te], pred_tipo_iso_xgb, average='macro') * 100:.1f}%\n")
print("--- Relatorio por tipo de anomalia (XGB, gate perfeito) ---")
print(classification_report(y_te[mask_anom_te], pred_tipo_iso_xgb, zero_division=0))

# %% [markdown]
# ### 5.3 XGBoost — Predição combinada (duas etapas) + matriz

# %%
pred_xgb_final = np.array(pred_bin_xgb, dtype=object)
m_xgb = pred_bin_xgb == "Anomalia"
pred_xgb_final[m_xgb] = le_tipo.inverse_transform(xgb_tipo.predict(X_te[m_xgb]))

print(f"XGB duas etapas — acuracia: {accuracy_score(y_te, pred_xgb_final) * 100:.1f}% | "
      f"F1 macro: {f1_score(y_te, pred_xgb_final, average='macro') * 100:.1f}%\n")
print("--- Relatorio por classe (XGB duas etapas) ---")
print(classification_report(y_te, pred_xgb_final, zero_division=0))

# %%
matriz_xgb2 = confusion_matrix(y_te, pred_xgb_final, labels=classes, normalize="pred")
plt.figure(figsize=(11, 8))
sns.heatmap(matriz_xgb2, annot=True, fmt=".0%", cmap="Greens", vmin=0, vmax=1,
            xticklabels=classes, yticklabels=classes)
plt.title("Matriz de Confusao - XGB duas etapas (normalizada por coluna)", fontsize=14)
plt.ylabel("Classe Verdadeira")
plt.xlabel("Previsao (coluna soma 100%)")
plt.xticks(rotation=45, ha="right")
plt.show()

# %% [markdown]
# ### 5.4 XGBoost — Modelo único (12 classes) + matriz

# %%
le = LabelEncoder().fit(y)
xgb_unico = make_xgb()
xgb_unico.fit(X_tr, le.transform(y_tr), sample_weight=compute_sample_weight("balanced", y_tr))
pred_xgb_unico = le.inverse_transform(xgb_unico.predict(X_te))

print(f"XGB unico — acuracia: {accuracy_score(y_te, pred_xgb_unico) * 100:.1f}% | "
      f"F1 macro: {f1_score(y_te, pred_xgb_unico, average='macro') * 100:.1f}%\n")
print("--- Relatorio por classe (XGB unico) ---")
print(classification_report(y_te, pred_xgb_unico, zero_division=0))

# %%
matriz_xgb1 = confusion_matrix(y_te, pred_xgb_unico, labels=classes, normalize="pred")
plt.figure(figsize=(11, 8))
sns.heatmap(matriz_xgb1, annot=True, fmt=".0%", cmap="Greens", vmin=0, vmax=1,
            xticklabels=classes, yticklabels=classes)
plt.title("Matriz de Confusao - XGB unico (normalizada por coluna)", fontsize=14)
plt.ylabel("Classe Verdadeira")
plt.xlabel("Previsao (coluna soma 100%)")
plt.xticks(rotation=45, ha="right")
plt.show()

# %% [markdown]
# ## 6. Resumo XGBoost: único vs duas etapas

# %%
def metricas(nome, pred):
    return {"config": nome,
            "acuracia": round(accuracy_score(y_te, pred) * 100, 1),
            "f1_macro": round(f1_score(y_te, pred, average="macro") * 100, 1)}


resumo = pd.DataFrame([
    metricas("XGB unico", pred_xgb_unico),
    metricas("XGB duas etapas", pred_xgb_final),
]).set_index("config")
print(resumo.sort_values("f1_macro", ascending=False))

# %% [markdown]
# ## 7. Análise de confusão entre classes
#
# Quais classes o modelo (XGB único) mais troca entre si. Lido da matriz de confusão
# **normalizada por linha** (recall): para cada classe verdadeira, qual % foi prevista
# como cada outra. Ajuda a decidir quais classes vale agrupar.

# %%
cm = confusion_matrix(y_te, pred_xgb_unico, labels=classes)
cm_norm = cm / cm.sum(axis=1, keepdims=True)  # por linha (recall)

plt.figure(figsize=(11, 8))
sns.heatmap(cm_norm, annot=True, fmt=".0%", cmap="Oranges", vmin=0, vmax=1,
            xticklabels=classes, yticklabels=classes)
plt.title("Confusao entre classes - XGB unico (normalizada por linha / recall)", fontsize=14)
plt.ylabel("Classe Verdadeira (linha soma 100%)")
plt.xlabel("Previsao")
plt.xticks(rotation=45, ha="right")
plt.show()

# %%
# Para cada classe: as 5 classes com que mais se confunde (fora a diagonal)
print("Classe verdadeira (acerto proprio) -> top 5 confusoes:")
for i, real in enumerate(classes):
    fora = cm_norm[i].copy()
    fora[i] = 0
    ordem = np.argsort(fora)[::-1][:5]
    tops = ", ".join(f"{classes[j]} {fora[j]*100:.0f}%" for j in ordem if fora[j] > 0)
    print(f"  {real:16s} (acerto {cm_norm[i, i]*100:3.0f}%) -> {tops}")

# %%
# Ranking dos pares de confusao mais fortes (real != previsto)
pares = []
for i, real in enumerate(classes):
    for j, prev in enumerate(classes):
        if i != j and cm[i, j] > 0:
            pares.append({"real": real, "previsto": prev,
                          "n": int(cm[i, j]), "pct_da_real": round(cm_norm[i, j] * 100, 1)})
conf = pd.DataFrame(pares).sort_values("pct_da_real", ascending=False)
print("\nTop 15 confusoes (% da classe verdadeira):")
print(conf.head(15).to_string(index=False))

# %% [markdown]
# ## 8. Agrupar classes irmãs e comparar
#
# As confusões mais fortes são entre pares "irmãos" (mesmo defeito, grau diferente):
# `Diode`↔`Diode-Multi`, `Cell`↔`Cell-Multi`, `Hot-Spot`↔`Hot-Spot-Multi`. Agrupamos os
# três pares (12 → 9 classes) e re-treinamos o XGBoost único.
#
# *Obs.: comparar 12 vs 9 classes não é maçã-com-maçã (problema fica mais fácil), mas é
# justificável fisicamente e mostra o ganho de eliminar a confusão entre irmãos.*

# %%
substituicoes = {"Diode-Multi": "Diode", "Cell-Multi": "Cell", "Hot-Spot-Multi": "Hot-Spot"}
y_grp = np.array([substituicoes.get(c, c) for c in y])
classes_grp = sorted(set(y_grp))
print(f"Classes: {len(classes)} -> {len(classes_grp)} ({classes_grp})")

Xg_tr, Xg_te, yg_tr, yg_te = train_test_split(X, y_grp, test_size=0.3, random_state=42, stratify=y_grp)
le_g = LabelEncoder().fit(y_grp)
xgb_grp = make_xgb()
xgb_grp.fit(Xg_tr, le_g.transform(yg_tr), sample_weight=compute_sample_weight("balanced", yg_tr))
pred_grp = le_g.inverse_transform(xgb_grp.predict(Xg_te))

print(f"\nXGB 12 classes — acuracia: {accuracy_score(y_te, pred_xgb_unico) * 100:.1f}% | "
      f"F1 macro: {f1_score(y_te, pred_xgb_unico, average='macro') * 100:.1f}%")
print(f"XGB 9 classes  — acuracia: {accuracy_score(yg_te, pred_grp) * 100:.1f}% | "
      f"F1 macro: {f1_score(yg_te, pred_grp, average='macro') * 100:.1f}%\n")
print("--- Relatorio por classe (9 classes agrupadas) ---")
print(classification_report(yg_te, pred_grp, zero_division=0))

# %%
# F1 por classe no cenario agrupado, comparando os pares que foram fundidos
f1_grp_arr = f1_score(yg_te, pred_grp, average=None, labels=classes_grp)
f1_12 = pd.Series(f1_score(y_te, pred_xgb_unico, average=None, labels=classes), index=classes)
print("Pares fundidos: F1 antes (separados) -> depois (juntos):")
for base, multi in [("Cell", "Cell-Multi"), ("Diode", "Diode-Multi"), ("Hot-Spot", "Hot-Spot-Multi")]:
    depois = dict(zip(classes_grp, f1_grp_arr))[base]
    print(f"  {base:9s}: {f1_12[base]:.2f} + {multi:14s} {f1_12[multi]:.2f}  ->  {base} {depois:.2f}")

# %%
# Matriz de confusao do cenario agrupado (normalizada por coluna)
matriz_grp = confusion_matrix(yg_te, pred_grp, labels=classes_grp, normalize="pred")
plt.figure(figsize=(9, 7))
sns.heatmap(matriz_grp, annot=True, fmt=".0%", cmap="Purples", vmin=0, vmax=1,
            xticklabels=classes_grp, yticklabels=classes_grp)
plt.title("Matriz de Confusao - XGB 9 classes agrupadas (por coluna)", fontsize=14)
plt.ylabel("Classe Verdadeira")
plt.xlabel("Previsao (coluna soma 100%)")
plt.xticks(rotation=45, ha="right")
plt.show()
