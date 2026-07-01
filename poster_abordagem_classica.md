# Poster — Abordagem clássica (texto atualizado)

## Versão padrão (recomendada)

**Abordagem clássica:**

Extração manual de características e classificação estatística. Foram extraídos 181 atributos por imagem, cobrindo descritores de intensidade, forma (limiar de Otsu), bordas (Canny), textura (Sobel, LBP e Gabor), simetria, momentos de Hu e distribuição espacial por grade 3x3. Avaliaram-se quatro classificadores (Random Forest, Gradient Boosting, SVM e XGBoost), com o XGBoost obtendo o melhor desempenho.

Essa abordagem resultou em F1-score macro de 53,9% e acurácia de 74,2% (12 classes). Como detector binário de falha (anomalia vs normal), atingiu 88,4% de acurácia.

---

## Versão curta (sem o detector binário)

**Abordagem clássica:**

Extração manual de características e classificação estatística. Foram extraídos 181 atributos por imagem, cobrindo descritores de intensidade, forma (limiar de Otsu), bordas (Canny), textura (Sobel, LBP e Gabor), simetria, momentos de Hu e distribuição espacial por grade 3x3. A classificação multiclasse foi executada com o algoritmo XGBoost, o melhor entre os avaliados.

Essa abordagem resultou em F1-score macro de 53,9% e acurácia de 74,2%.

---

## Versão comparável ao Deep Learning (9 classes)

**Abordagem clássica:**

Extração manual de características e classificação estatística. Foram extraídos 181 atributos por imagem, cobrindo descritores de intensidade, forma (limiar de Otsu), bordas (Canny), textura (Sobel, LBP e Gabor), simetria, momentos de Hu e distribuição espacial por grade 3x3. Avaliaram-se quatro classificadores (Random Forest, Gradient Boosting, SVM e XGBoost), com o XGBoost obtendo o melhor desempenho.

Essa abordagem resultou em F1-score macro de 53,9% (12 classes). Com o agrupamento das classes irmãs (9 categorias, mesmo critério do Deep Learning), o F1-score macro sobe para 61,7%.
