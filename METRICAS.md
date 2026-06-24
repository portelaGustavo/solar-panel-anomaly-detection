# Métricas de Classificação (referência)

Explicação das métricas do `classification_report` do scikit-learn, usadas para
avaliar os modelos deste projeto. Exemplos com números reais do RandomForest.

## Os 4 tijolos (matriz de confusão)

Para uma classe (ex.: `Cell`), todo resultado cai em uma de 4 contagens:

| Sigla | Nome | O que é |
|-------|------|---------|
| **TP** | Verdadeiro Positivo | era Cell, previu Cell ✅ |
| **FP** | Falso Positivo | não era Cell, previu Cell ❌ (alarme falso) |
| **FN** | Falso Negativo | era Cell, previu outra ❌ (deixou passar) |
| **TN** | Verdadeiro Negativo | não era Cell, não previu Cell ✅ |

## As 4 colunas do relatório

### Precision (precisão)
```
precision = TP / (TP + FP)
```
Dos que o modelo **previu** como a classe, quantos eram mesmo. Alta = poucos alarmes
falsos. Responde: *"quando ele diz Cell, dá pra confiar?"*

### Recall (revocação / sensibilidade)
```
recall = TP / (TP + FN)
```
Dos que **eram** a classe, quantos ele achou. Alto = deixa passar pouco.
Responde: *"ele pega todos os Cell?"*

### F1-score
```
f1 = 2 * (precision * recall) / (precision + recall)
```
Média **harmônica** de precision e recall. Só fica alto se **os dois** forem bons
(se um for baixo, o F1 cai). Resume precision + recall num número.

### Support
Quantas imagens **daquela classe** existem no conjunto avaliado. **Não é métrica de
qualidade**, é o tamanho da amostra. Serve para:
- contexto: F1 alto com support 50 é menos confiável que com support 3000;
- peso: usado no cálculo do `weighted avg`.

## As 3 linhas de baixo

| Linha | O que é |
|-------|---------|
| **accuracy** | acertos totais / total de imagens. Um número só. |
| **macro avg** | média **simples** entre as classes. Toda classe pesa igual (rara = comum). |
| **weighted avg** | média **ponderada pelo support**. Classe grande domina. |

## Exemplo real (RandomForest, teste com 6000 imagens)

```
                precision    recall  f1-score   support
          Cell       0.46      0.45      0.46       563
    No-Anomaly       0.82      0.77      0.79      3000
      Hot-Spot       0.26      0.13      0.18        75
      ...
      accuracy                           0.62      6000
     macro avg       0.43      0.43      0.42      6000
  weighted avg       0.63      0.62      0.62      6000
```

- `Cell`: acerta ~46% do que chama de Cell e pega ~45% dos Cell reais.
- `No-Anomaly`: support 3000 (classe gigante), F1 0.79 (fácil e bem feito).
- `Hot-Spot`: support 75 (raro), recall 0.13 → **deixa passar 87%**. Modelo fraco aqui.

## Qual olhar?

| Situação | Métrica principal |
|----------|-------------------|
| Dataset **equilibrado** | accuracy já basta |
| Dataset **desbalanceado** (nosso caso) | **macro avg** (revela classes raras) + **support** |

**Por que macro importa aqui:** metade do dataset é `No-Anomaly`. A acurácia (62%) e o
weighted (62%) ficam inflados porque a classe gigante puxa para cima. O **macro F1 (42%)**
ignora o tamanho e mostra a verdade: o modelo vai bem nas classes grandes/fáceis e
tropeça nas raras.
