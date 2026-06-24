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
# Nossa versão do método clássico (sem deep learning). Em relação à v1,
# acrescentamos: **mais features térmicas**, **regra multiclasse** (não só 3 classes)
# e **métricas por classe** (precision/recall/F1).
#
# Roda no Colab e no local. As imagens aparecem inline a cada etapa.

# %% [markdown]
# ## 1. Setup e download do dataset
#
# Baixa e extrai o dataset em Python puro (funciona no Colab sem `!wget`/`!unzip`).
# Se o dataset já existe (ex.: rodando local), não baixa de novo.

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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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
#
# O `module_metadata.json` mapeia cada imagem para sua classe de anomalia.

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
# como regiões mais claras (quentes).


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
        if img is None:
            ax.set_title("Erro ao carregar")
            ax.axis("off")
            continue
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
# Numa imagem da classe `Cell` (ponto quente): binarizamos pelo limiar, limpamos
# com erosão/dilatação e extraímos contornos da região quente. Por fim desenhamos
# a caixa do defeito detectado.

# %%
amostra = df[df["anomaly_class"] == "Cell"].sample(1).iloc[0]
img_original = cv2.imread(str(DATA_DIR / amostra["image_filepath"]), cv2.IMREAD_GRAYSCALE)

limiar = 200
_, img_binaria = cv2.threshold(img_original, limiar, 255, cv2.THRESH_BINARY)
kernel = np.ones((2, 2), np.uint8)
img_erodida = cv2.erode(img_binaria, kernel, iterations=1)
img_dilatada = cv2.dilate(img_erodida, kernel, iterations=1)

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
titulos = ["Original", f"Binaria (>{limiar})", "Apos Erosao", "Apos Dilatacao"]
for ax, img, titulo in zip(axes, [img_original, img_binaria, img_erodida, img_dilatada], titulos):
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    ax.set_title(titulo)
    ax.axis("off")
plt.tight_layout()
plt.show()

# %%
contornos, _ = cv2.findContours(img_dilatada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
img_resultado = cv2.cvtColor(img_original, cv2.COLOR_GRAY2BGR)
anomalia_detectada = False
for contorno in contornos:
    area = cv2.contourArea(contorno)
    if area > 5:
        anomalia_detectada = True
        x, y, w, h = cv2.boundingRect(contorno)
        cv2.rectangle(img_resultado, (x, y), (x + w, y + h), (255, 0, 0), 1)
        print(f"Area do defeito: {area} pixels.")
if not anomalia_detectada:
    print("No-Anomaly. Nenhuma mancha quente significativa.")
    cv2.rectangle(img_resultado, (0, 0),
                  (img_original.shape[1] - 1, img_original.shape[0] - 1), (0, 255, 0), 1)

plt.figure(figsize=(4, 6))
plt.imshow(cv2.cvtColor(img_resultado, cv2.COLOR_BGR2RGB))
plt.title("Deteccao Final")
plt.axis("off")
plt.show()

# %% [markdown]
# ## 5. Extração de features térmicas (nossa adição)
#
# A v1 usava só a **área do maior blob**. Aqui extraímos **7 features** por imagem.
# A ideia: a *forma* e a *fração* da região quente separam as classes (ponto isolado
# vs banda de 1/3 do módulo vs módulo inteiro quente).
#
# - `mean_int`, `max_int`: intensidade média e máxima (No-Anomaly tem max baixo)
# - `hot_fraction`: fração de pixels quentes
# - `num_blobs`: nº de manchas quentes (1 = Cell, várias = Cell-Multi)
# - `largest_area`: área da maior mancha
# - `row_cov`, `col_cov`: cobertura por linha/coluna (detecta banda do diodo)

# %%
HOT_FLOOR = 200  # intensidade minima para um pixel ser considerado "quente"


def extrair_features(img):
    total = img.size
    mean_int = float(img.mean())
    max_int = int(img.max())

    _, hot = cv2.threshold(img, HOT_FLOOR, 255, cv2.THRESH_BINARY)
    hot = cv2.morphologyEx(hot, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    hot_fraction = float(hot.sum() / 255) / total

    contornos, _ = cv2.findContours(hot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = [c for c in contornos if cv2.contourArea(c) >= 2]
    num_blobs = len(blobs)
    largest_area = max((cv2.contourArea(c) for c in blobs), default=0.0)

    row_cov = float((hot.sum(axis=1) > 0).mean())
    col_cov = float((hot.sum(axis=0) > 0).mean())

    return {
        "mean_int": mean_int,
        "max_int": max_int,
        "hot_fraction": hot_fraction,
        "num_blobs": num_blobs,
        "largest_area": largest_area,
        "row_cov": row_cov,
        "col_cov": col_cov,
    }


# Demonstra as features numa imagem
print("Features da amostra acima:")
for k, v in extrair_features(img_original).items():
    print(f"  {k:14s} = {v}")

# %% [markdown]
# ## 6. Regra de classificação multiclasse
#
# Limiares derivados do diagnóstico das médias de cada feature por classe.
# Cobre as classes com assinatura térmica clara. As classes não-térmicas
# (Shadowing, Vegetation, Soiling, Cracking) são difíceis por este método.


# %%
def classificar(f):
    if f["max_int"] < 185 and f["hot_fraction"] < 0.12:
        return "No-Anomaly"
    if f["row_cov"] >= 0.5 and f["hot_fraction"] >= 0.22:
        return "Diode-Multi"
    if f["row_cov"] >= 0.4 and f["largest_area"] >= 110:
        return "Diode"
    if f["hot_fraction"] >= 0.28 or f["largest_area"] >= 260:
        return "Hot-Spot-Multi"
    if f["largest_area"] >= 190:
        return "Offline-Module"
    if f["num_blobs"] >= 2:
        return "Cell-Multi"
    return "Cell"


# %% [markdown]
# ## 7. Avaliação: acurácia, F1 por classe e matriz de confusão
#
# Roda a regra sobre todo o dataset. Atenção: acurácia global engana em dataset
# desbalanceado, por isso olhamos o relatório por classe.

# %%
y_verdadeiro, y_previsto = [], []
for _, row in tqdm(df.iterrows(), total=df.shape[0]):
    img = cv2.imread(str(DATA_DIR / row["image_filepath"]), cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    y_verdadeiro.append(row["anomaly_class"])
    y_previsto.append(classificar(extrair_features(img)))

print(f"\nAcuracia geral: {accuracy_score(y_verdadeiro, y_previsto) * 100:.2f}%\n")
print("--- Relatorio por classe ---")
print(classification_report(y_verdadeiro, y_previsto, zero_division=0))

# %%
classes_unicas = sorted(set(y_verdadeiro) | set(y_previsto))
matriz = confusion_matrix(y_verdadeiro, y_previsto, labels=classes_unicas)
plt.figure(figsize=(12, 8))
sns.heatmap(matriz, annot=True, fmt="d", cmap="Blues",
            xticklabels=classes_unicas, yticklabels=classes_unicas)
plt.title("Matriz de Confusao - Visao Classica v2", fontsize=16)
plt.ylabel("Classe Verdadeira", fontsize=12)
plt.xlabel("Previsao do Algoritmo", fontsize=12)
plt.xticks(rotation=45, ha="right")
plt.show()
