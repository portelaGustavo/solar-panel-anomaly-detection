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
# # Visão Deep Learning v2 — ResNet34 (fastai)
#
# Nossa versão do método de IA. Mesma receita da v1 (ResNet34, transfer
# learning), mas roda no Colab e no local, com download automático do dataset.
#
# **Use GPU no Colab**: Ambiente de execução → Alterar tipo → GPU (T4).
# As imagens (batch, lr_find, matriz de confusão, top losses) aparecem inline.

# %% [markdown]
# ## 1. Setup e download do dataset
#
# Baixa e extrai em Python puro (sem `!wget`/`!unzip`). Pula se já existe.

# %%
import json
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
from fastai.vision.all import (
    ImageDataLoaders, Resize, aug_transforms, vision_learner, resnet34,
    accuracy, ClassificationInterpretation, set_seed,
)

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
# ## 2. Leitura, consolidação de classes e balanceamento
#
# - Consolida classes redundantes (`Cell-Multi`→`Cell`, etc.).
# - Reduz `No-Anomaly` para 1800 imagens (era metade do dataset) e mantém o resto.

# %%
with open(DATA_DIR / "module_metadata.json", "r") as f:
    metadados = json.load(f)
df = pd.DataFrame.from_dict(metadados, orient="index")

substituicoes = {"Cell-Multi": "Cell", "Diode-Multi": "Diode", "Hot-Spot-Multi": "Hot-Spot"}
df["anomaly_class"] = df["anomaly_class"].replace(substituicoes)

df_no_anomaly = df[df["anomaly_class"] == "No-Anomaly"].sample(n=1800, random_state=42)
df_anomalias = df[df["anomaly_class"] != "No-Anomaly"]
df = pd.concat([df_no_anomaly, df_anomalias]).sample(frac=1, random_state=42).reset_index(drop=True)

print(df.head())
print("\n--- CONTAGEM DE CLASSES ---")
print(df["anomaly_class"].value_counts())

# %% [markdown]
# ## 3. DataLoaders e visualização do batch
#
# Redimensiona para 224x224 e aplica flip. Mostra um batch já processado.

# %%
set_seed(42, reproducible=True)
dls = ImageDataLoaders.from_df(
    df,
    path=str(DATA_DIR),
    fn_col="image_filepath",
    label_col="anomaly_class",
    item_tfms=Resize(224),
    batch_tfms=aug_transforms(do_flip=True, max_rotate=0.0, max_lighting=0.0, max_warp=0.0),
)
dls.show_batch()

# %% [markdown]
# ## 4. Treinamento inicial (backbone congelado)
#
# ResNet34 pré-treinada, treina só a cabeça por 4 épocas.

# %%
learn = vision_learner(dls, resnet34, metrics=accuracy)
learn.fit_one_cycle(4)
learn.save("modelo-resnet34-etapa1")

# %% [markdown]
# ## 5. `lr_find` — melhor faixa de learning rate
#
# Descongela a rede e procura a faixa de aprendizado para o fine tuning.

# %%
learn.unfreeze()
learn.lr_find()

# %% [markdown]
# ## 6. Fine tuning com learning rate discriminativo
#
# Treina a rede inteira por 10 épocas, LR menor nas camadas iniciais.

# %%
learn.load("modelo-resnet34-etapa1")
learn.unfreeze()
learn.fit_one_cycle(10, lr_max=slice(1e-5, 1e-3))

# %% [markdown]
# ## 7. Resultados: previsões, matriz de confusão e top losses

# %%
learn.show_results(max_n=9, figsize=(10, 10))

# %%
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(7, 7))

# %%
interp.plot_top_losses(k=9, figsize=(15, 11))

# %% [markdown]
# ## 8. Exporta o modelo final
#
# Salva arquitetura + pesos em `.pkl` para inferência depois.

# %%
learn.export(BASE_DIR / "modelo_termografico_final.pkl")
print("Modelo exportado.")
