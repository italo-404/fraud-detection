import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


# --- 1. Carregamento dos dados ---

df = pd.read_csv("creditcard.csv")

print(f"Total de transações: {len(df):,}")
print(f"Colunas: {list(df.columns)}\n")
print(df.head())


# --- 2. Análise exploratória (EDA) ---

contagem = df["Class"].value_counts()
percentual_fraude = (contagem[1] / len(df)) * 100

print(f"\nTransações normais: {contagem[0]:,}")
print(f"Transações fraudulentas: {contagem[1]:,} ({percentual_fraude:.4f}%)")
print("\nValores nulos:\n", df.isnull().sum())
print("\nEstatísticas de 'Amount':\n", df["Amount"].describe().round(2))
print("\nValor médio por classe:\n", df.groupby("Class")["Amount"].mean())


# --- 3. Visualizações ---

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Análise de Fraudes em Transações Bancárias", fontsize=14, fontweight="bold")

cores = ["#4CAF50", "#F44336"]
valores = [contagem[0], contagem[1]]

axes[0].pie(valores, labels=["Normal", "Fraude"], colors=cores, autopct="%1.2f%%", startangle=90)
axes[0].set_title("Proporção: Normal vs. Fraude")

axes[1].bar(["Normal", "Fraude"], valores, color=cores)
axes[1].set_title("Quantidade de Transações")
axes[1].set_ylabel("Número de transações")
for i, v in enumerate(valores):
    axes[1].text(i, v + 500, f"{v:,}", ha="center", fontweight="bold")

normais = df[df["Class"] == 0]["Amount"]
fraudes = df[df["Class"] == 1]["Amount"]
axes[2].hist(normais, bins=50, color=cores[0], alpha=0.6, label="Normal", range=(0, 500))
axes[2].hist(fraudes, bins=50, color=cores[1], alpha=0.6, label="Fraude", range=(0, 500))
axes[2].set_title("Distribuição do Valor (até $500)")
axes[2].set_xlabel("Valor da transação")
axes[2].set_ylabel("Frequência")
axes[2].legend()

plt.tight_layout()
plt.savefig("grafico_analise_fraudes.png", dpi=150, bbox_inches="tight")
plt.show()


# --- 4. Preparação para o modelo ---

X = df.drop(columns=["Class"])
y = df["Class"]

# stratify=y garante que a proporção de fraudes seja mantida em treino e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTreino: {len(X_treino):,} registros | Teste: {len(X_teste):,} registros")


# --- 5. Treinamento do modelo ---

# max_depth limita a complexidade da árvore e reduz overfitting
modelo = DecisionTreeClassifier(max_depth=5, random_state=42)
modelo.fit(X_treino, y_treino)


# --- 6. Avaliação ---

y_pred = modelo.predict(X_teste)

print("\nRelatório de Classificação:")
print(classification_report(y_teste, y_pred, target_names=["Normal", "Fraude"]))

cm = confusion_matrix(y_teste, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Fraude"])
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("Matriz de Confusão")
plt.tight_layout()
plt.savefig("matriz_confusao.png", dpi=150, bbox_inches="tight")
plt.show()

vn, fp, fn, vp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
acuracia = (vn + vp) / (vn + fp + fn + vp) * 100
print(f"\nAcurácia: {acuracia:.2f}%")
print(f"Fraudes detectadas: {vp} de {vp + fn} ({vp / (vp + fn) * 100:.1f}%)")
