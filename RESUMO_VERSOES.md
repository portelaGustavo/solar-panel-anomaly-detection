# Resumo das versões (visão clássica)

Cada `visao_classica_vN` é uma evolução do método clássico (sem deep learning) para
detecção de anomalias em painéis solares. Resumo do conteúdo de cada uma.

## Visão geral

| Versão | Foco | Modelo(s) | Classes | Melhor F1 macro |
|--------|------|-----------|---------|-----------------|
| v2 | Features + comparação de modelos + agrupamento | Regra, RF, GB, SVM | 12 → agrupado | ~50% |
| v3 | Duas etapas + XGBoost + limiar | RF e XGBoost | 12 | ~54% |
| v4 | Análise de confusão + de features | XGBoost | 12 | (análise) |
| v5 | Confusão detalhada + agrupamento | XGBoost | 12 → 9 | ~62% |

---

## v2 — Base clássica, features e comparação de modelos
Notebook mais "didático", passo a passo do método clássico.
- Setup, leitura do JSON, visualização de amostras por classe.
- Pipeline clássico: threshold → morfologia → contornos.
- Extração de features (intensidade, região quente, etc.) + adição de **grade 3x3 e LBP**.
- Baseline com **regra manual** (if/else) vs **classificador treinado**.
- Comparação de **RandomForest vs Gradient Boosting vs SVM** + desempenho por classe.
- Experimentos de **agrupamento** de classes irmãs (Diode, Hot-Spot).

## v3 — Classificação em duas etapas + XGBoost
Muda a estratégia de classificação e introduz o XGBoost.
- Mesmas features da v2, ampliadas (skew/kurtosis, momentos de Hu, **Gabor e HOG**).
- **Duas etapas**: etapa 1 (anomalia vs No-Anomaly) → etapa 2 (qual tipo).
- Avaliação da **etapa 2 isolada** (portão perfeito) e da combinada.
- Compara **RandomForest vs XGBoost**, único vs duas etapas (XGBoost ganha).
- Relatórios por classe + matrizes de confusão normalizadas.
- **Ajuste de limiar** do detector de falha (curva precision-recall, recall até ~95%).

## v4 — Análise de confusão e de features
Foca em entender o modelo (só XGBoost, sem RF).
- Mesmo pipeline da v3 (features + duas etapas + único), apenas XGBoost.
- **Análise de confusão entre classes** (matriz por linha, quais classes se trocam).
- **Importância de features por permutação**: identifica features inúteis/que atrapalham.
- Experimento de **remover as features candidatas** e comparar (sem perda relevante).

## v5 — Confusão detalhada + agrupamento de classes
Versão mais enxuta e focada no agrupamento.
- Só XGBoost, todas as features (otimização de features fica para o fim).
- **Análise de confusão** mostrando as **5 maiores confusões por classe**.
- **Agrupamento dos pares irmãos** (Diode+Diode-Multi, Cell+Cell-Multi,
  Hot-Spot+Hot-Spot-Multi): 12 → 9 classes, com comparação antes/depois.

---

## Evolução do desempenho (F1 macro, clássico)
Regra manual ~10% → RF + features ~42% → XGBoost + Gabor/HOG (v3) ~54% →
agrupamento de irmãos (v5) ~62%. Referência deep learning ~67%.
