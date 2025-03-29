import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

file_path = "C:/Users/ianaj/av1-AI/data/EMGsDataset.csv"
data = pd.read_csv(file_path, header=None)
data = data.T

X = data.iloc[:, :2].values  
y = data.iloc[:, 2].values  

one_hot_encoder = OneHotEncoder(sparse_output=False)
Y_one_hot = one_hot_encoder.fit_transform(y.reshape(-1, 1))  

assert X.shape == (50000, 2) 
assert Y_one_hot.shape == (50000, 5) 

X_transposed = X.T  
Y_transposed = Y_one_hot.T  

print("Shapes for MQO:")
print(f"X (N×p): {X.shape}")
print(f"Y (N×C): {Y_one_hot.shape}")

print("\nShapes for Bayesian Gaussian models:")
print(f"X (p×N): {X_transposed.shape}")
print(f"Y (C×N): {Y_transposed.shape}")

# Definir cores das classes
class_colors = {
    0: 'blue',
    1: 'green',
    2: 'red',
    3: 'purple',
    4: 'orange'
}

sample_size = 1000
sample_indices = np.random.choice(X.shape[0], sample_size, replace=False)

plt.figure(figsize=(10, 6))
for i in sample_indices:
    plt.scatter(X[i, 0], X[i, 1], color=class_colors[np.argmax(Y_one_hot[i])], alpha=0.5)

plt.title("Gráfico de Dispersão das Classes de EMG")
plt.xlabel("Corrugador do Supercílio (Sensor 1)")
plt.ylabel("Zigomático Maior (Sensor 2)")
plt.grid(True)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lambdas = [0, 0.25, 0.5, 0.75, 1]

# Resultados para uma única execução
def evaluate_model(lambda_val):
    clf = GaussianNB(var_smoothing=lambda_val)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return accuracy_score(y_test, y_pred)

results = {lambda_val: evaluate_model(lambda_val) for lambda_val in lambdas}

print("\nAcurácia do modelo para diferentes valores de λ:")
for lambda_val, accuracy in results.items():
    print(f"λ = {lambda_val}: {accuracy:.4f}")

plt.plot(lambdas, list(results.values()), marker='o')
plt.title("Acurácia do Classificador Gaussiano Regularizado")
plt.xlabel("λ (Lambda)")
plt.ylabel("Acurácia")
plt.grid(True)
plt.show()

R = 500
accuracy_results = {lambda_val: [] for lambda_val in lambdas}

for r in range(R):
    if r % 50 == 0:
        print(f"Rodada {r + 1}/{R}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)
    
    for lambda_val in lambdas:
        acc = evaluate_model(lambda_val)
        accuracy_results[lambda_val].append(acc)

stats = {lambda_val: {
    "Média": np.mean(accuracy_results[lambda_val]),
    "Desvio Padrão": np.std(accuracy_results[lambda_val]),
    "Máximo": np.max(accuracy_results[lambda_val]),
    "Mínimo": np.min(accuracy_results[lambda_val])
} for lambda_val in lambdas}

df_stats = pd.DataFrame(stats).T
print("\nResultados da Validação por Monte Carlo:")
print(df_stats)

plt.figure(figsize=(10, 6))
plt.boxplot([accuracy_results[lambda_val] for lambda_val in lambdas], labels=[f"λ = {val}" for val in lambdas])
plt.title("Distribuição da Acurácia dos Modelos por Monte Carlo")
plt.ylabel("Acurácia")
plt.xlabel("Modelos")
plt.grid(True)
plt.show()
