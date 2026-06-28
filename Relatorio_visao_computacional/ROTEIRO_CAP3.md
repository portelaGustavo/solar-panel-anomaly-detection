# Roteiro de desenvolvimento — Capítulo 3 (Solução com visão clássica)

Plano de escrita das novas subseções do Cap. 3 do `relatorio.tex`, na ordem narrativa
acordada. Para cada uma: objetivo, figuras/tabelas necessárias, notebook-fonte dos dados e
o que o **v6** (a criar depois) precisará gerar de forma unificada e limpa para o relatório.

Legenda de status: ✅ já no .tex | ✍️ a escrever | 📊 dado/figura a coletar (v6).

---

## Já existente no .tex
- **3.1 Preparação do ambiente e análise exploratória** ✅
- **3.2 Pipeline de análise inicial** ✅ (heurística: threshold → morfologia → contornos → regra)

A partir daqui, o roteiro continua de onde a heurística parou.

---

## 3.3 Extração de features e classificadores ✍️
**Narrativa:** a heurística da 3.2 só separa poucas classes. Evoluímos para um vetor de
features por imagem + classificadores de aprendizado de máquina.
- Descrever as features iniciais (intensidade, região quente via Otsu, forma do maior blob,
  cobertura por linha/coluna, região escura, bordas, textura, simetria, histograma).
- Classificadores avaliados: **RandomForest**, **Gradient Boosting**, **SVM**, **XGBoost**.
- **Benchmark inicial** (tabela: acurácia + F1 macro de cada modelo).

**Figuras/Tabelas:**
- Tabela: benchmark dos 4 modelos (acc, F1 macro) no conjunto de features inicial.

**Fonte:** v2 (RF/GB/SVM), v3 (RF vs XGBoost).
**v6 gera:** 📊 um único benchmark com os 4 modelos no MESMO conjunto de features e split.

---

## 3.4 Refinamento incremental das features ✍️
**Narrativa:** adicionamos grupos de features em etapas, medindo o impacto a cada passo.
Ordem seguida:
1. grade 3x3 + LBP
2. skewness / kurtosis
3. momentos de Hu
4. Gabor + HOG

**Figuras/Tabelas:**
- Tabela/gráfico: evolução nº de features × F1 macro (XGBoost) a cada grupo adicionado.
- Comentário de qual classe cada grupo ajudou (ex.: HOG → trincas; Gabor → textura).

**Fonte:** v2 (grade+LBP), v3 (skew/kurt, Hu, Gabor, HOG).
**v6 gera:** 📊 a progressão limpa (treina XGBoost acumulando cada grupo, registra F1 macro).

---

## 3.5 Classificação em duas etapas ✍️
**Narrativa:** em vez de classificar as 12 de uma vez, separar em duas etapas:
etapa 1 = anomalia vs No-Anomaly; etapa 2 = qual tipo (só nas anomalias).
- Comparar com o benchmark (modelo único vs duas etapas).
- Destacar o **detector binário** (etapa 1) como resultado de alto valor prático.
- Mostrar a **etapa 2 isolada** (gate perfeito) separando qualidade do tipo vs do portão.

**Figuras/Tabelas:**
- Tabela: único vs duas etapas (acc, F1 macro) para RF e XGBoost.
- (Opcional) relatório binário da etapa 1 + curva precision-recall do detector.

**Fonte:** v3.
**v6 gera:** 📊 a tabela comparativa única vs duas etapas.

---

## 3.6 Análise do melhor modelo (XGBoost) ✍️
**Narrativa:** o benchmark aponta o XGBoost como melhor. Analisamos seus erros.
- **Matriz de confusão normalizada por coluna (precision).**
- **Matriz de confusão entre classes (normalizada por linha / recall).**
- **Tabela das maiores confusões** por classe (com quem cada uma mais se confunde).

**Figuras/Tabelas:**
- Figura: matriz de confusão normalizada por coluna (precision).
- Figura: matriz de confusão normalizada por linha (recall).
- Tabela: top confusões por classe (ex.: Diode-Multi→Diode 37%, Cell→No-Anomaly 24%...).

**Fonte:** v4 (matriz por coluna) e v5 (matriz por linha + tabela top-5 confusões).
**v6 gera:** 📊 as duas matrizes + a tabela de confusões do XGBoost final.

---

## 3.7 Otimização por remoção de features ✍️
**Narrativa:** usamos muitas features (~181, maioria HOG). Medimos a importância por
permutação e removemos as inúteis/que atrapalham, mantendo o desempenho com menos features.
- Mostrar quantas features têm importância ≤ 0 (candidatas a remover).
- Benchmark: conjunto completo vs reduzido (acc, F1 macro + nº de features).

**Figuras/Tabelas:**
- Figura: importância por permutação (melhores e piores features).
- Tabela: completo vs reduzido (acc, F1 macro, nº features).

**Fonte:** final do v4 (importância + experimento de remoção).
**v6 gera:** 📊 a importância e a comparação completo vs reduzido.

---

## 3.8 Agrupamento de classes irmãs — **(OPCIONAL)** ✍️
> Marcado como opcional. Decidir depois se entra no relatório. A tabela de confusões (3.6)
> já cobre a necessidade de mostrar quais classes se confundem.

**Narrativa (se incluir):** agrupar pares irmãos (Diode+Diode-Multi, Cell+Cell-Multi,
Hot-Spot+Hot-Spot-Multi), 12→9 classes, e medir o ganho.
- Benchmark antes/depois (F1 macro ~54% → ~62%) + F1 dos pares fundidos.

**Fonte:** v5.
**v6 gera (se usado):** 📊 a comparação 12 vs 9 classes.

---

## Próximo passo
Quando o roteiro estiver aprovado, criar o **v6** (`visao_classica_v6`): um notebook único
que roda o pipeline final e gera, de forma limpa e reproducível, todas as tabelas e figuras
📊 listadas acima, na ordem do Cap. 3. Depois, escrever as subseções no `relatorio.tex`
referenciando essas figuras/tabelas.
