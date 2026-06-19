# 🔍 Detecção de Fraudes em Transações Bancárias

Projeto de ciência de dados para identificação automática de transações fraudulentas utilizando Machine Learning com Python.

---

## 📋 Sobre o projeto

Fraudes em cartões de crédito causam prejuízos bilionários ao sistema financeiro global. Este projeto aplica técnicas de análise de dados e aprendizado de máquina para detectar padrões suspeitos em transações bancárias, utilizando um dataset real com mais de 284 mil registros.

**Abordagem:**
- Análise exploratória dos dados (EDA) com Pandas
- Visualizações com Matplotlib
- Modelo de classificação com Árvore de Decisão (Scikit-learn)
- Avaliação com métricas adequadas para dados desbalanceados

---

## 📁 Estrutura do repositório

```
fraud-detection/
├── deteccao_fraudes.py         # Script principal
├── grafico_analise_fraudes.png # Gráficos gerados pela análise
├── matriz_confusao.png         # Matriz de confusão do modelo
└── README.md
```

> **Nota:** O arquivo `creditcard.csv` não está incluído neste repositório por conta do tamanho (144 MB). Veja as instruções de instalação abaixo para obtê-lo.

---

## 🗃️ Base de dados

**Credit Card Fraud Detection** — disponibilizado pelo Machine Learning Group da Universidade Livre de Bruxelas (ULB) via Kaggle.

- 📥 Download: [kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- 284.807 transações realizadas por portadores de cartões europeus em setembro de 2013
- Apenas 492 fraudes (0,172% do total) — dataset altamente desbalanceado
- As colunas V1 a V28 são componentes PCA (dados originais anonimizados por privacidade)
- As colunas `Time`, `Amount` e `Class` são as únicas não transformadas

---

## ⚙️ Instalação e uso

**Pré-requisitos:** Python 3.8+

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/fraud-detection.git
cd fraud-detection
```

2. Instale as dependências:
```bash
pip install pandas matplotlib scikit-learn
```

3. Baixe o dataset no link acima e coloque o arquivo `creditcard.csv` na raiz do projeto.

4. Execute o script:
```bash
python deteccao_fraudes.py
```

### Google Colab

Também é possível rodar diretamente no Colab sem instalação local:

1. Faça upload do `creditcard.csv` no ambiente do Colab
2. Cole e execute o conteúdo de `deteccao_fraudes.py`

---

## 📊 Resultados

O modelo de Árvore de Decisão (`max_depth=5`) treinado com 80% dos dados obteve os seguintes resultados no conjunto de teste:

| Métrica | Normal | Fraude |
|---|---|---|
| Precision | ~1.00 | ~0.85 |
| Recall | ~1.00 | ~0.76 |
| F1-score | ~1.00 | ~0.80 |

> Os valores exatos variam conforme a execução, mas ficam próximos aos indicados acima.

**Sobre as métricas:**
- **Precision:** das transações classificadas como fraude, quantas realmente eram?
- **Recall:** de todas as fraudes reais, quantas o modelo encontrou?
- **F1-score:** média harmônica entre Precision e Recall — principal métrica para dados desbalanceados

---

## 🚧 Possíveis melhorias

- [ ] Aplicar técnicas de balanceamento (SMOTE / undersampling)
- [ ] Testar outros algoritmos (Random Forest, XGBoost, Regressão Logística)
- [ ] Otimizar hiperparâmetros com GridSearchCV
- [ ] Analisar importância das features

---

## 🛠️ Tecnologias utilizadas

- [Python 3](https://www.python.org/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [Scikit-learn](https://scikit-learn.org/)

---

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.
