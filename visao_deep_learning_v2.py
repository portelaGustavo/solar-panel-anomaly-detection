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
"""Deep Learning (ResNet34 / fastai) — NOSSA versao (v2).

Port local-rodavel do notebook original do colega (visao_deep_learning_4.ipynb).
Mantido separado do original para comparacao. Edite aqui localmente; para treinar
na GPU, de push e rode o .ipynb pareado no Google Colab (git pull).

Mudancas vs original:
  - Sem magics de Colab (!wget / !unzip / !pip). Le o dataset da pasta local
    ``InfraredSolarModules/`` ao lado deste arquivo.
  - Figuras salvas em ``outputs/`` e pesos em ``models/``.
  - Hiperparametros via env var para teste rapido em CPU:
      EPOCHS_FROZEN / EPOCHS_FINETUNE / IMG_SIZE / SAMPLE_N
    Ex.: EPOCHS_FROZEN=1 EPOCHS_FINETUNE=1 IMG_SIZE=128 SAMPLE_N=600 python visao_deep_learning_v2.py

Original reporta ~67%. Pendencias dos autores: pesos por classe, reprodutibilidade.
"""

# %%
import os
from pathlib import Path

# %%
import matplotlib
if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
import pandas as pd
from fastai.vision.all import (
    ImageDataLoaders, Resize, aug_transforms, vision_learner, resnet34,
    accuracy, ClassificationInterpretation, set_seed,
)

# %%
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:  # rodando em notebook (Jupyter/Colab): __file__ nao existe
    BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "InfraredSolarModules"
OUT_DIR = BASE_DIR / "outputs"
MODELS_DIR = BASE_DIR / "models"
OUT_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# %%
EPOCHS_FROZEN = int(os.environ.get("EPOCHS_FROZEN", "4"))
EPOCHS_FINETUNE = int(os.environ.get("EPOCHS_FINETUNE", "10"))
IMG_SIZE = int(os.environ.get("IMG_SIZE", "224"))          # menor = mais rapido (ex.: 128)
SAMPLE_N = int(os.environ.get("SAMPLE_N", "0"))            # 0 = dataset inteiro; >0 = subconjunto p/ teste


# %%
def save_fig(name):
    path = OUT_DIR / name
    plt.savefig(path, bbox_inches="tight", dpi=120)
    print(f"[saved] {path}")
    plt.close()


# %%
def carregar_dados_balanceados():
    with open(DATA_DIR / "module_metadata.json", "r") as f:
        metadados = json.load(f)
    df = pd.DataFrame.from_dict(metadados, orient="index")

    # Consolida classes redundantes
    substituicoes = {"Cell-Multi": "Cell", "Diode-Multi": "Diode", "Hot-Spot-Multi": "Hot-Spot"}
    df["anomaly_class"] = df["anomaly_class"].replace(substituicoes)

    # Balanceamento: reduz No-Anomaly para 1800, mantem o resto
    df_no_anomaly = df[df["anomaly_class"] == "No-Anomaly"].sample(n=1800, random_state=42)
    df_anomalias = df[df["anomaly_class"] != "No-Anomaly"]
    df_bal = pd.concat([df_no_anomaly, df_anomalias]).sample(frac=1, random_state=42).reset_index(drop=True)

    if SAMPLE_N > 0:
        df_bal = df_bal.sample(n=min(SAMPLE_N, len(df_bal)), random_state=42).reset_index(drop=True)
        print(f"[teste] usando subconjunto de {len(df_bal)} imagens")

    print(df_bal.head())
    print("\n--- CONTAGEM DE CLASSES ---")
    print(df_bal["anomaly_class"].value_counts())
    return df_bal


# %%
def main():
    if not (DATA_DIR / "module_metadata.json").exists():
        raise SystemExit(
            f"Dataset nao encontrado em {DATA_DIR}. "
            "Extraia ../InfraredSolarModules/2020-02-14_InfraredSolarModules.zip para esta pasta "
            "(ou, no Colab, baixe via !wget)."
        )

    set_seed(42, reproducible=True)
    df = carregar_dados_balanceados()

    dls = ImageDataLoaders.from_df(
        df,
        path=str(DATA_DIR),
        fn_col="image_filepath",
        label_col="anomaly_class",
        item_tfms=Resize(IMG_SIZE),
        batch_tfms=aug_transforms(do_flip=True, max_rotate=0.0, max_lighting=0.0, max_warp=0.0),
    )
    dls.show_batch()
    save_fig("dl_batch.png")

    # Treinamento inicial (backbone congelado)
    learn = vision_learner(dls, resnet34, metrics=accuracy, path=BASE_DIR, model_dir=MODELS_DIR)
    learn.fit_one_cycle(EPOCHS_FROZEN)
    learn.save("modelo-resnet34-etapa1")

    # lr_find para escolher a faixa de aprendizado do fine tuning
    learn.unfreeze()
    learn.lr_find()
    save_fig("dl_lr_find.png")

    # Fine tuning com learning rate discriminativo
    learn.load("modelo-resnet34-etapa1")
    learn.unfreeze()
    learn.fit_one_cycle(EPOCHS_FINETUNE, lr_max=slice(1e-5, 1e-3))

    learn.show_results(max_n=9, figsize=(10, 10))
    save_fig("dl_resultados.png")

    interp = ClassificationInterpretation.from_learner(learn)
    interp.plot_confusion_matrix(figsize=(7, 7))
    save_fig("dl_matriz_confusao.png")

    interp.plot_top_losses(k=9, figsize=(15, 11))
    save_fig("dl_top_losses.png")

    learn.export(MODELS_DIR / "modelo_termografico_final.pkl")
    print(f"[saved] {MODELS_DIR / 'modelo_termografico_final.pkl'}")


# %%
if __name__ == "__main__":
    main()
