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

# %%
"""Classical Computer Vision approach for solar panel anomaly detection.

Local-runnable port of visao_classica.ipynb (originally written on Google Colab).
Changes vs notebook:
  - Removed Colab shell magics (!wget / !unzip / !pip). Dataset is read from the
    local ``InfraredSolarModules/`` folder next to this script.
  - Figures are saved to ``outputs/`` (headless friendly) and shown only when a
    display is available.
  - Fixed a syntax error in the original notebook (a comment was missing its '#').

Pipeline: threshold -> morphology -> contours -> largest-blob-area heuristic.
"""

# %%
import json
import os
from pathlib import Path

# %%
import cv2
import numpy as np
import pandas as pd
import matplotlib

# %%
if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")  # headless: render to files instead of a window
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

# %%
HOT_FLOOR = 200  # intensidade absoluta minima para um pixel ser "quente"


# %%
def extrair_features(img):
    """Extrai features termicas de uma imagem (grayscale 24x40).

    Ideia: defeitos termicos = regioes quentes. A *forma* e a *fracao* da regiao
    quente separam as classes (ponto isolado vs banda de 1/3 vs modulo inteiro).
    """
    total = img.size
    mean_int = float(img.mean())
    max_int = int(img.max())

    # Mascara de pixels quentes (limiar absoluto, robusto a imagem uniforme)
    _, hot = cv2.threshold(img, HOT_FLOOR, 255, cv2.THRESH_BINARY)
    kernel = np.ones((2, 2), np.uint8)
    hot = cv2.morphologyEx(hot, cv2.MORPH_OPEN, kernel)
    hot_fraction = float(hot.sum() / 255) / total

    contornos, _ = cv2.findContours(hot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = [c for c in contornos if cv2.contourArea(c) >= 2]
    num_blobs = len(blobs)
    largest_area = max((cv2.contourArea(c) for c in blobs), default=0.0)

    # Cobertura por linha/coluna: detecta "banda" (diodo aquece ~1/3, 2/3 do modulo)
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

# %%
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:  # rodando em notebook (Jupyter/Colab): __file__ nao existe
    BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "InfraredSolarModules"
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)


# %%
def save_fig(name):
    path = OUT_DIR / name
    plt.savefig(path, bbox_inches="tight", dpi=120)
    print(f"[saved] {path}")


# %%
def load_metadata():
    with open(DATA_DIR / "module_metadata.json", "r") as f:
        metadados = json.load(f)
    df = pd.DataFrame.from_dict(metadados, orient="index")
    print(df.head())
    print("\n")
    print(df["anomaly_class"].value_counts())
    return df


# %%
def visualizar_amostras(df, classe_nome, num_amostras=5):
    df_filtrado = df[df["anomaly_class"] == classe_nome]
    # Pega amostras aleatorias  (was a bare line in the notebook -> SyntaxError; fixed)
    amostras = df_filtrado.sample(min(len(df_filtrado), num_amostras))

    fig, axes = plt.subplots(1, num_amostras, figsize=(15, 3))
    fig.suptitle(f"Amostras da classe: {classe_nome}", fontsize=16)
    if num_amostras == 1:
        axes = [axes]

    for ax, caminho_imagem in zip(axes, amostras["image_filepath"]):
        caminho_corrigido = DATA_DIR / caminho_imagem
        img = cv2.imread(str(caminho_corrigido), cv2.IMREAD_GRAYSCALE)
        if img is None:
            ax.set_title("Erro ao carregar")
            ax.axis("off")
            print(f"Aviso: nao foi possivel carregar {caminho_corrigido}")
            continue
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.axis("off")
    plt.tight_layout()
    save_fig(f"amostras_{classe_nome}.png")
    plt.close(fig)


# %%
def teste_threshold_morfologia(df):
    """Single-image demo: threshold -> morphology -> contour -> bounding box."""
    amostra = df[df["anomaly_class"] == "Cell"].sample(1).iloc[0]
    caminho_imagem = DATA_DIR / amostra["image_filepath"]
    img_original = cv2.imread(str(caminho_imagem), cv2.IMREAD_GRAYSCALE)

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
    save_fig("pipeline_morfologia.png")
    plt.close(fig)

    contornos, _ = cv2.findContours(img_dilatada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_resultado = cv2.cvtColor(img_original, cv2.COLOR_GRAY2BGR)
    anomalia_detectada = False
    area_minima = 5
    print("--- Resultados da Analise Classica (amostra) ---")
    for contorno in contornos:
        area = cv2.contourArea(contorno)
        if area > area_minima:
            anomalia_detectada = True
            x, y, w, h = cv2.boundingRect(contorno)
            cv2.rectangle(img_resultado, (x, y), (x + w, y + h), (255, 0, 0), 1)
            print(f"Area do defeito: {area} pixels.")
    if not anomalia_detectada:
        print("No-Anomaly. Nenhuma mancha quente significativa.")
        cv2.rectangle(img_resultado, (0, 0),
                      (img_original.shape[1] - 1, img_original.shape[0] - 1), (0, 255, 0), 1)

    fig = plt.figure(figsize=(4, 6))
    plt.imshow(cv2.cvtColor(img_resultado, cv2.COLOR_BGR2RGB))
    plt.title("Deteccao Final")
    plt.axis("off")
    save_fig("deteccao_final.png")
    plt.close(fig)


# %%
def classificar(f):
    """Regra multiclasse baseada nas features termicas.

    Limiares derivados do diagnostico por classe (medias de cada feature).
    Cobre as classes com assinatura termica clara; classes nao-termicas
    (Shadowing, Vegetation, Soiling, Cracking) sao dificeis por este metodo.
    """
    # Sem regiao quente de verdade -> modulo nominal
    if f["max_int"] < 185 and f["hot_fraction"] < 0.12:
        return "No-Anomaly"
    # Banda quente cobrindo muitas linhas -> diodo de bypass (aquece 1/3 ~ 2/3)
    if f["row_cov"] >= 0.5 and f["hot_fraction"] >= 0.22:
        return "Diode-Multi"
    if f["row_cov"] >= 0.4 and f["largest_area"] >= 110:
        return "Diode"
    # Calor muito espalhado / mancha grande
    if f["hot_fraction"] >= 0.28 or f["largest_area"] >= 260:
        return "Hot-Spot-Multi"
    if f["largest_area"] >= 190:
        return "Offline-Module"
    # Pontos quentes localizados
    if f["num_blobs"] >= 2:
        return "Cell-Multi"
    return "Cell"


# %%
def avaliar_dataset(df):
    """Run the feature-based heuristic over the dataset; report accuracy, per-class metrics, confusion matrix."""
    y_verdadeiro, y_previsto = [], []

    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        caminho_imagem = DATA_DIR / row["image_filepath"]
        img_original = cv2.imread(str(caminho_imagem), cv2.IMREAD_GRAYSCALE)
        if img_original is None:
            continue
        previsao = classificar(extrair_features(img_original))
        y_verdadeiro.append(row["anomaly_class"])
        y_previsto.append(previsao)

    acuracia = accuracy_score(y_verdadeiro, y_previsto)
    print(f"\nAcuracia geral da Visao Classica: {acuracia * 100:.2f}%\n")
    print("--- Relatorio por classe ---")
    print(classification_report(y_verdadeiro, y_previsto, zero_division=0))

    classes_unicas = sorted(set(y_verdadeiro) | set(y_previsto))
    matriz = confusion_matrix(y_verdadeiro, y_previsto, labels=classes_unicas)
    fig = plt.figure(figsize=(12, 8))
    sns.heatmap(matriz, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes_unicas, yticklabels=classes_unicas)
    plt.title("Matriz de Confusao - Visao Classica", fontsize=16)
    plt.ylabel("Classe Verdadeira", fontsize=12)
    plt.xlabel("Previsao do Algoritmo", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    save_fig("matriz_confusao_classica.png")
    plt.close(fig)


# %%
def main():
    if not (DATA_DIR / "module_metadata.json").exists():
        raise SystemExit(
            f"Dataset nao encontrado em {DATA_DIR}. "
            "Extraia ../InfraredSolarModules/2020-02-14_InfraredSolarModules.zip para esta pasta."
        )
    df = load_metadata()
    for classe in ["No-Anomaly", "Hot-Spot", "Offline-Module", "Cell"]:
        visualizar_amostras(df, classe, num_amostras=5)
    teste_threshold_morfologia(df)
    avaliar_dataset(df)


# %%
if __name__ == "__main__":
    main()
