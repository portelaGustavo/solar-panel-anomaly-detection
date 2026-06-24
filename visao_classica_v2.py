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
# # Visão Clássica v2 — Detecção de Anomalias em Painéis Solares
#
# Classificação de anomalias em painéis solares por visão computacional clássica
# (sem deep learning). O pipeline:
#
# 1. **Pré-processamento e visualização** das imagens infravermelhas (24x40 px).
# 2. **Extração de features** por imagem: intensidade, forma da região quente
#    (limiar Otsu), bordas (Canny), textura (gradiente Sobel), simetria e histograma.
# 3. **Classificação** com RandomForest treinado sobre essas features, comparado
#    a uma regra manual como baseline.
# 4. **Avaliação** por classe (precision/recall/F1), matriz de confusão e importância
#    das features.
#
# Tudo roda em CPU (não precisa de GPU). Imagens aparecem inline a cada etapa.

# %% [markdown]
# ## 1. Setup e download do dataset
#
# Baixa e extrai em Python puro (funciona no Colab sem `!wget`/`!unzip`). Pula se já existe.

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from tqdm import tqdm

try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:  # notebook (Jupyter/Colab): __file__ nao existe
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
# ## 2. Leitura do JSON e criação do DataFrame

# %%
with open(DATA_DIR / "module_metadata.json", "r") as f:
    metadados = json.load(f)

df = pd.DataFrame.from_dict(metadados, orient="index")
print(df.head())
print("\n--- CONTAGEM DE CLASSES ---")
print(df["anomaly_class"].value_counts())

# %% [markdown]
# ## 3. Visualização das amostras por classe
#
# Imagens infravermelhas em tons de cinza (24x40 px). Defeitos térmicos aparecem
# como regiões mais claras (quentes); outras anomalias (sombra, vegetação) como regiões escuras.


# %%
def visualizar_amostras(df, classe_nome, num_amostras=5):
    df_filtrado = df[df["anomaly_class"] == classe_nome]
    amostras = df_filtrado.sample(min(len(df_filtrado), num_amostras))
    fig, axes = plt.subplots(1, num_amostras, figsize=(15, 3))
    fig.suptitle(f"Amostras da classe: {classe_nome}", fontsize=16)
    if num_amostras == 1:
        axes = [axes]
    for ax, caminho_imagem in zip(axes, amostras["image_filepath"]):
        img = cv2.imread(str(DATA_DIR / caminho_imagem), cv2.IMREAD_GRAYSCALE)
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


# %%
for classe in ["No-Anomaly", "Hot-Spot", "Offline-Module", "Cell"]:
    visualizar_amostras(df, classe, num_amostras=5)

# %% [markdown]
# ## 4. Pipeline clássico: threshold → morfologia → contornos
#
# Demonstração numa imagem `Cell` (ponto quente): binariza, limpa com morfologia,
# extrai contornos e desenha a caixa do defeito.

# %%
amostra = df[df["anomaly_class"] == "Cell"].sample(1).iloc[0]
img_original = cv2.imread(str(DATA_DIR / amostra["image_filepath"]), cv2.IMREAD_GRAYSCALE)

limiar = 200
_, img_binaria = cv2.threshold(img_original, limiar, 255, cv2.THRESH_BINARY)
kernel = np.ones((2, 2), np.uint8)
img_erodida = cv2.erode(img_binaria, kernel, iterations=1)
img_dilatada = cv2.dilate(img_erodida, kernel, iterations=1)

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, img, titulo in zip(
    axes,
    [img_original, img_binaria, img_erodida, img_dilatada],
    ["Original", f"Binaria (>{limiar})", "Apos Erosao", "Apos Dilatacao"],
):
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    ax.set_title(titulo)
    ax.axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Extração de features
#
# Aqui extraímos **~25 features** por imagem,
# cobrindo vários tipos de anomalia (não só ponto quente):
#
# - **Intensidade**: média, desvio, máx, mín, percentil 90
# - **Região quente** (limiar Otsu adaptativo): fração, nº de blobs, área/extensão/aspecto do maior, cobertura por linha/coluna
# - **Região escura**: fração de pixels frios → sombra / vegetação
# - **Bordas** (Canny): densidade → trincas (Cracking)
# - **Textura** (gradiente Sobel): rugosidade → sujeira (Soiling)
# - **Simetria** esquerda-direita / cima-baixo
# - **Histograma** (8 bins): distribuição térmica


# %%
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

    # Regiao quente via Otsu (adaptativo) com piso absoluto
    otsu_t, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t = max(otsu_t, 180)
    _, hot = cv2.threshold(img, t, 255, cv2.THRESH_BINARY)
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

    # Regiao escura (sombra / vegetacao): pixels bem abaixo da media
    f["dark_fraction"] = float((img < (img.mean() - img.std())).mean())

    # Bordas (trincas) e textura (sujeira)
    f["edge_density"] = float((cv2.Canny(img, 50, 150) > 0).mean())
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    f["grad_mean"] = float(np.sqrt(gx ** 2 + gy ** 2).mean())

    # Simetria
    imgf = img.astype(np.float32)
    f["sym_lr"] = float(np.abs(imgf - np.fliplr(imgf)).mean())
    f["sym_tb"] = float(np.abs(imgf - np.flipud(imgf)).mean())

    # Histograma (8 bins, normalizado)
    hist = cv2.calcHist([img], [0], None, [8], [0, 256]).flatten()
    hist = hist / hist.sum()
    for i, hv in enumerate(hist):
        f[f"hist{i}"] = float(hv)

    return f


FEAT_NAMES = list(extrair_features(img_original).keys())
print(f"{len(FEAT_NAMES)} features:", FEAT_NAMES)

# %%
# Visualiza a regiao quente (Otsu) que alimenta as features
amostra_feat = df[df["anomaly_class"] == "Hot-Spot-Multi"].sample(1).iloc[0]
img_feat = cv2.imread(str(DATA_DIR / amostra_feat["image_filepath"]), cv2.IMREAD_GRAYSCALE)
otsu_t, _ = cv2.threshold(img_feat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
_, hot_vis = cv2.threshold(img_feat, max(otsu_t, 180), 255, cv2.THRESH_BINARY)
edges_vis = cv2.Canny(img_feat, 50, 150)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, im, tt in zip(axes, [img_feat, hot_vis, edges_vis],
                      [f"Original ({amostra_feat['anomaly_class']})", "Regiao quente (Otsu)", "Bordas (Canny)"]):
    ax.imshow(im, cmap="gray")
    ax.set_title(tt)
    ax.axis("off")
plt.show()

# %% [markdown]
# ## 6. Matriz de features de todo o dataset
#
# Extrai as features das 20.000 imagens (rápido, CPU). `X` = features, `y` = classe.

# %%
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

# %% [markdown]
# ## 7. Baseline: regra manual
#
# Regra `if/else` na mão, para comparar com o classificador treinado. A ordem
# dos testes segue o princípio **do mais extremo para o mais genérico**: a primeira
# condição que casa decide, então casos abrangentes (catch-all) ficam por último.
# Lógica por *quanto do módulo está quente*: nada → tudo quente → banda → pontos.


# %%
def classificar_regra(f):
    # 1. Nada quente -> modulo nominal (maioria, curto-circuito no topo)
    if f["max_int"] < 185 and f["hot_fraction"] < 0.12:
        return "No-Anomaly"
    # 2. Calor muito espalhado (mais extremo): modulo quase todo quente
    if f["hot_fraction"] >= 0.28 or f["largest_area"] >= 260:
        return "Hot-Spot-Multi" if f["num_blobs"] >= 2 else "Offline-Module"
    # 3. Mancha unica grande -> modulo desligado (aquecido por inteiro)
    if f["largest_area"] >= 190:
        return "Offline-Module"
    # 4. Banda quente (diodo de bypass aquece 1/3 ~ 2/3 do modulo)
    if f["row_cov"] >= 0.5:
        return "Diode-Multi"
    if f["row_cov"] >= 0.4 and f["largest_area"] >= 110:
        return "Diode"
    # 5. Pontos quentes localizados
    if f["num_blobs"] >= 2:
        return "Cell-Multi"
    return "Cell"


y_regra = [classificar_regra(r) for r in registros]
print(f"Baseline regra manual — acuracia: {accuracy_score(y, y_regra) * 100:.1f}% | "
      f"F1 macro: {f1_score(y, y_regra, average='macro') * 100:.1f}%")

# %% [markdown]
# ## 8. Classificador treinado: RandomForest
#
# Treina sobre as features. `class_weight='balanced'` compensa o desbalanceamento.
# Split estratificado 70/30.

# %%
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
clf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)
clf.fit(X_tr, y_tr)
pred = clf.predict(X_te)

print(f"RandomForest — acuracia: {accuracy_score(y_te, pred) * 100:.1f}% | "
      f"F1 macro: {f1_score(y_te, pred, average='macro') * 100:.1f}%\n")
print("--- Relatorio por classe ---")
print(classification_report(y_te, pred, zero_division=0))

# %% [markdown]
# ## 9. Resultados: matriz de confusão, importância das features e amostras previstas

# %%
classes = sorted(set(y))
matriz = confusion_matrix(y_te, pred, labels=classes)
plt.figure(figsize=(11, 8))
sns.heatmap(matriz, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
plt.title("Matriz de Confusao - RandomForest", fontsize=15)
plt.ylabel("Classe Verdadeira")
plt.xlabel("Previsao")
plt.xticks(rotation=45, ha="right")
plt.show()

# %%
# Importancia das features
imp = sorted(zip(FEAT_NAMES, clf.feature_importances_), key=lambda x: x[1], reverse=True)
nomes, vals = zip(*imp)
plt.figure(figsize=(10, 6))
sns.barplot(x=list(vals), y=list(nomes), color="steelblue")
plt.title("Importancia das features (RandomForest)")
plt.xlabel("Importancia")
plt.tight_layout()
plt.show()

# %%
# Amostras aleatorias: classe real vs prevista pelo RandomForest (verde = acerto)
amostra_pred = df.sample(8, random_state=0)
fig, axes = plt.subplots(2, 4, figsize=(14, 6))
for ax, (_, r) in zip(axes.ravel(), amostra_pred.iterrows()):
    img = cv2.imread(str(DATA_DIR / r["image_filepath"]), cv2.IMREAD_GRAYSCALE)
    vec = np.array([[extrair_features(img)[k] for k in FEAT_NAMES]])
    pr = clf.predict(vec)[0]
    cor = "green" if pr == r["anomaly_class"] else "red"
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    ax.set_title(f"real: {r['anomaly_class']}\nprev: {pr}", color=cor, fontsize=9)
    ax.axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 10. Comparação de classificadores: RandomForest vs Gradient Boosting vs SVM
#
# Treinamos três classificadores sobre as **mesmas features** e o mesmo split, para
# ver qual separa melhor:
#
# - **RandomForest**: floresta de árvores por votação (bagging).
# - **Gradient Boosting** (`HistGradientBoosting`): árvores em sequência, cada uma
#   corrige o erro da anterior. Costuma ser forte em features tabulares.
# - **SVM**: busca a fronteira de maior margem entre as classes. Precisa de features
#   **normalizadas** (por isso vai num pipeline com `StandardScaler`).
#
# Todos com balanceamento de classes (o dataset é desbalanceado).

# %%
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

modelos = {
    "RF": RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1),
    "GB": HistGradientBoostingClassifier(random_state=42),
    "SVM": make_pipeline(StandardScaler(), SVC(class_weight="balanced", random_state=42)),
}

resultados = []
preds = {}
for nome, modelo in modelos.items():
    if nome == "GB":  # balanceia via sample_weight
        modelo.fit(X_tr, y_tr, sample_weight=compute_sample_weight("balanced", y_tr))
    else:
        modelo.fit(X_tr, y_tr)
    p = modelo.predict(X_te)
    preds[nome] = p
    resultados.append({
        "modelo": nome,
        "acuracia": accuracy_score(y_te, p) * 100,
        "f1_macro": f1_score(y_te, p, average="macro") * 100,
    })

tabela = pd.DataFrame(resultados).set_index("modelo").round(1)
print(tabela)

# %% [markdown]
# ## 11. Desempenho por classe de cada modelo
#
# F1-score de cada classe em cada modelo. A coluna `melhor` aponta qual modelo vence
# naquela classe. Assim dá para ver onde cada método é forte e onde é fraco.

# %%
f1_por_classe = {nome: f1_score(y_te, p, average=None, labels=classes) for nome, p in preds.items()}
tab_f1 = pd.DataFrame(f1_por_classe, index=classes).round(2)
tab_f1["melhor"] = tab_f1.idxmax(axis=1)
print(tab_f1)
