# Melhorias Futuras — Visão Clássica

Ideias para melhorar o desempenho do método clássico (features + classificador).
Ordenadas por impacto esperado. Estado atual: **RandomForest ~62% acc / 42% F1 macro**
com 25 features (DL de referência ~67%). Gargalo = classes raras (support baixo) e
classes não-térmicas (Shadowing, Vegetation, Soiling, Cracking).

## A. Melhorar precisão

### 1. Mais features (maior alavanca)
As features são o limite, não o modelo. Faltam sinais para as classes difíceis:

| Feature | Pega qual classe |
|---------|------------------|
| **Grade espacial** (dividir imagem em 3x3, intensidade por célula) | **Diode** (aquece o terço de baixo) — informa *onde* está o calor |
| **LBP** (Local Binary Patterns) | textura → **Soiling** |
| **HOG** / histograma de orientação de gradiente | trincas, formas |
| **Filtros de Gabor** | textura direcional |
| **Momentos de Hu** do maior blob | geometria da mancha quente |
| **connectedComponentsWithStats** | áreas / centróides / contagem por blob |
| **Skewness / kurtosis** do histograma | forma da distribuição térmica |
| **Contraste / razão quente-frio** | intensidade relativa |

### 2. Pré-processamento
- **CLAHE** (equalização adaptativa de histograma) antes de extrair features → normaliza
  o brilho entre imagens.
- Normalizar features (ajuda SVM / kNN; RF e árvores não precisam).

### 3. Lidar com o desbalanceamento (classes raras vão mal)
- **SMOTE** / oversampling das classes raras (biblioteca `imbalanced-learn`).
- Undersample da classe `No-Anomaly` (é metade do dataset).
- `class_weight='balanced_subsample'` no RandomForest.
- **Data augmentation** térmico (flip / rotação) para gerar mais amostras das raras.

### 4. Estratégia de classes (alto impacto, baixo custo)
- **Classificação em duas etapas**: 1º `anomalia vs No-Anomaly`, depois `qual tipo`.
  Tira o peso da classe gigante.
- **Agrupar** classes raras parecidas (`Hot-Spot` + `Hot-Spot-Multi`,
  `Diode` + `Diode-Multi`) → menos classes, mais amostras por classe, F1 sobe.

### 5. Modelo
- **XGBoost / LightGBM** (costuma superar RF / HistGradientBoosting com tuning).
- **Ensemble** (stacking / voting) combinando RF + GB + SVM.

### 6. Seleção de features
- Remover features de baixa importância (ruído) → pode melhorar a generalização.

## B. Mais métricas de avaliação

| Métrica | Por que adicionar |
|---------|-------------------|
| **Matriz de confusão normalizada** (% por linha) | ver *com quem* cada classe é confundida |
| **Balanced accuracy** | acurácia que pesa todas as classes igual |
| **MCC** (Matthews Correlation Coefficient) | ótima para desbalanceado, num só número |
| **Cohen's kappa** | acerto além do acaso |
| **ROC-AUC / PR-AUC** (one-vs-rest, via `predict_proba`) | qualidade independente do limiar |
| **Top-2 accuracy** | a classe certa está entre as 2 mais prováveis? |
| **Cross-validation** (média ± desvio) | número mais confiável que um único split |

## C. Análise das features (sobre a extração)

| Análise | O que mostra |
|---------|--------------|
| **Permutation importance** | importância mais confiável que a nativa do RF |
| **Matriz de correlação** entre features | acha features redundantes para cortar |
| **Mutual information** feature ↔ classe | quais features mais separam as classes |
| **Boxplots por classe** de cada feature | ver visualmente quais features separam bem |

## Ordem prática recomendada
1. **Grade espacial 3x3** + **agrupar classes raras** → ganho rápido e grande.
2. **Matriz de confusão normalizada** + **MCC / balanced accuracy** → enxergar melhor os erros.
3. Depois: **LBP / HOG**, **SMOTE**, **XGBoost**.
