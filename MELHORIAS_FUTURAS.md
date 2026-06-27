# Melhorias Futuras — Visão Clássica

Ideias para melhorar o desempenho do método clássico (features + classificador).
Ordenadas por impacto esperado. DL de referência ~67%. Gargalo = classes raras
(support baixo) e classes não-térmicas (Shadowing, Vegetation, Soiling, Cracking).

## Já implementado (progresso)

| Estágio | Acurácia | F1 macro |
|---------|----------|----------|
| Regra manual (baseline) | ~32% | ~10% |
| RandomForest, 25 features | 61.9% | 41.9% |
| + features grade 3x3 + LBP (44 feat) | 65.1% | 43.3% |
| + agrupar Diode + Diode-Multi (11 classes) | 66.7% | 46.7% |
| + agrupar Hot-Spot + Hot-Spot-Multi (10 classes) | 67.3% | 50.5% |

**v3 (12 classes, sem agrupamento, 181 features: base+Hu+skew/kurt+Gabor+HOG):**

| Config | Acurácia | F1 macro |
|--------|----------|----------|
| RF único | 68.0% | 48.3% |
| RF duas etapas | 69.6% | 48.9% |
| XGB duas etapas | 73.6% | 53.7% |
| **XGB único** | **74.2%** | **53.9%** |

Evolução das features no XGB único: 53 feat = 47.9% F1 → +Gabor/HOG (181 feat) = **53.9%** F1.

Feito também: comparação RF vs GradientBoosting vs SVM (RF ganhou); GridSearchCV
(sem ganho, removido). As demais ideias abaixo ainda **não** foram implementadas.

## A. Melhorar precisão

### 1. Mais features (maior alavanca)
As features são o limite, não o modelo. Faltam sinais para as classes difíceis:

| Feature | Pega qual classe |
|---------|------------------|
| ~~**Grade espacial** (3x3, intensidade por célula)~~ ✅ feito | **Diode** — informa *onde* está o calor |
| ~~**LBP** (Local Binary Patterns)~~ ✅ feito | textura → **Soiling** |
| ~~**HOG** / histograma de orientação de gradiente~~ ✅ usado (v3) | trincas, formas → **+4.8 F1 macro** (maior ganho de feature) |
| ~~**Filtros de Gabor**~~ ✅ usado (v3) | textura direcional → +1.6 F1 macro |
| ~~**Momentos de Hu** do maior blob~~ ✅ usado (v3) | geometria da mancha → +1.3 F1 macro (XGB único) |
| ~~**connectedComponentsWithStats**~~ ❌ descartado | só +0.5 F1, redundante com num_blobs/largest_area |
| ~~**Skewness / kurtosis** do histograma~~ ✅ usado (v3) | forma da distribuição → +1 a +2.7 F1 macro |
| ~~**Contraste / razão quente-frio**~~ ❌ descartado | redundante com max/min/std/hot_fraction, ganho nulo |

### 2. Pré-processamento
- ~~**CLAHE** (equalização adaptativa de histograma)~~ ❌ descartado → **piorou ~4-5 F1 macro**.
  Equalizar contraste destrói a intensidade absoluta, que é o sinal central em térmica
  (anomalia = quente em valor absoluto). Não serve para este problema.
- Normalizar features (ajuda SVM / kNN; RF e árvores não precisam).

### 3. Lidar com o desbalanceamento (classes raras vão mal)
- **SMOTE** / oversampling das classes raras (biblioteca `imbalanced-learn`).
- Undersample da classe `No-Anomaly` (é metade do dataset).
- `class_weight='balanced_subsample'` no RandomForest.
- **Data augmentation** térmico (flip / rotação) para gerar mais amostras das raras.

### 3b. Capitalizar o detector binário (etapa 1) — alto valor prático
A etapa 1 (anomalia vs No-Anomaly) é forte (XGB: 88% acc, 84% recall) e resolve o problema
prático principal: "esse painel precisa de inspeção?". Roda em CPU/visão clássica → deployável
em campo sem GPU (vantagem que a CNN não tem). Enquadrar o trabalho como **dois produtos**:
(1) detector de falha (clássico, forte) e (2) classificador de tipo (mais difícil, DL ajuda).
- **Ajuste de limiar da etapa 1** (`predict_proba` + threshold) → para triagem, priorizar
  **recall** (não perder falha) trocando por mais alarme falso. Mostrar curva precision-recall
  e escolher o ponto de operação. **(próximo a implementar)**

### 4. Estratégia de classes (alto impacto, baixo custo)
- ~~**Classificação em duas etapas**: 1º `anomalia vs No-Anomaly`, depois `qual tipo`~~
  ✅ feito (v3) → +acurácia no RF, mas F1 macro ~igual; com XGBoost até piora. Gargalo = recall
  da etapa 1 (~78%). Não compensou.
- ~~**Agrupar** classes raras parecidas (`Hot-Spot`+`Hot-Spot-Multi`, `Diode`+`Diode-Multi`)~~
  ✅ feito (v2) → F1 macro subiu de 43% para 50%. (Resta avaliar agrupar mais, ex.: `Cell`+`Cell-Multi`.)

### 5. Modelo
- ~~**XGBoost / LightGBM**~~ ✅ usado (v3) → **melhor modelo**: XGB único 70.5% acc / ~47% F1 macro,
  bateu RF e SVM. LightGBM ainda não testado.
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
