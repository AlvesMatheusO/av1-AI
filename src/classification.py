import numpy as np
import matplotlib.pyplot as plt

def classification(file_path="../data/EMGsDataset.csv", R=500):

    # -------------------------------------------------------
    # 1) Carregamento e organização dos dados
    # -------------------------------------------------------
    data = np.loadtxt(file_path, delimiter=',')
    data = data.T  # Transpõe para ter 50000 linhas e 3 colunas
    X = data[:, :2]  # Sinais dos sensores (Corrugador do Supercílio e Zigomático Maior)
    y = data[:, 2]   # Rótulos (classes: 1 a 5)
    y = y.astype(int)  # Garantir que os rótulos sejam inteiros

    print("Shapes dos dados:")
    print(f"X (N×p): {X.shape}")  # Espera-se (50000, 2)
    print(f"y (N,): {y.shape}")   # Espera-se (50000,)

    # -------------------------------------------------------
    # 2) Visualização dos dados
    # -------------------------------------------------------
    # Definindo cores para as classes (assumindo classes de 1 a 5)
    class_colors = {
        1: 'blue',
        2: 'green',
        3: 'red',
        4: 'purple',
        5: 'orange'
    }

    sample_size = 1000
    sample_indices = np.random.choice(X.shape[0], sample_size, replace=False)

    plt.figure(figsize=(10, 6))
    for i in sample_indices:
        plt.scatter(X[i, 0], X[i, 1], color=class_colors[y[i]], alpha=0.5)
    plt.title("Gráfico de Dispersão das Classes de EMG")
    plt.xlabel("Sensor 1: Corrugador do Supercílio")
    plt.ylabel("Sensor 2: Zigomático Maior")
    plt.grid(True)
    plt.show()

    # -------------------------------------------------------
    # 3) Função de divisão de dados (train/test)
    # -------------------------------------------------------
    def train_test_split(X, y, test_size=0.2):
        N = X.shape[0]
        indices = np.random.permutation(N)
        train_size = int(N * (1 - test_size))
        return X[indices[:train_size]], X[indices[train_size:]], y[indices[:train_size]], y[indices[train_size:]]

    # -------------------------------------------------------
    # 4) Implementação do Gaussian Naive Bayes
    # -------------------------------------------------------
    def gaussian_naive_bayes_fit(X, y, var_smoothing):
        """
        Ajusta o modelo Gaussiano Ingênuo:
          - Para cada classe, estima a média e a variância (com adição de var_smoothing)
          - Calcula a priori das classes.
        """
        classes = np.unique(y)
        model = {'classes': classes, 'means': {}, 'variances': {}, 'priors': {}}
        for c in classes:
            X_c = X[y == c]
            model['means'][c] = np.mean(X_c, axis=0)
            model['variances'][c] = np.var(X_c, axis=0) + var_smoothing
            model['priors'][c] = X_c.shape[0] / float(X.shape[0])
        return model

    def gaussian_naive_bayes_predict(X, model):
        """
        Para cada amostra em X, calcula a log-probabilidade de cada classe
        e retorna a classe com maior probabilidade.
        """
        classes = model['classes']
        predictions = []
        for i in range(X.shape[0]):
            x_i = X[i]
            best_log_prob = -np.inf
            best_class = None
            for c in classes:
                mean = model['means'][c]
                var = model['variances'][c]
                prior = model['priors'][c]
                # Cálculo do log da função densidade gaussiana:
                log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * var)) - np.sum((x_i - mean)**2 / (2 * var))
                log_posterior = np.log(prior) + log_likelihood
                if log_posterior > best_log_prob:
                    best_log_prob = log_posterior
                    best_class = c
            predictions.append(best_class)
        return np.array(predictions)

    def accuracy_score(y_true, y_pred):
        return np.mean(y_true == y_pred)

    # Função para avaliar o modelo para um dado var_smoothing (λ)
    def evaluate_model(var_smoothing):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = gaussian_naive_bayes_fit(X_train, y_train, var_smoothing)
        y_pred = gaussian_naive_bayes_predict(X_test, model)
        return accuracy_score(y_test, y_pred)

    # Avaliação para diferentes valores de λ
    lambdas = [0, 0.25, 0.5, 0.75, 1]
    results = {lam: evaluate_model(lam) for lam in lambdas}

    print("\nAcurácia do modelo para diferentes valores de λ:")
    for lam, acc in results.items():
        print(f"λ = {lam}: {acc:.4f}")

    plt.figure()
    plt.plot(lambdas, list(results.values()), marker='o')
    plt.title("Acurácia do Classificador Gaussiano (Naive Bayes) Regularizado")
    plt.xlabel("λ (Lambda)")
    plt.ylabel("Acurácia")
    plt.grid(True)
    plt.show()

    # -------------------------------------------------------
    # 5) Validação Monte Carlo
    # -------------------------------------------------------
    accuracy_results = {lam: [] for lam in lambdas}
    for r in range(R):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        for lam in lambdas:
            model = gaussian_naive_bayes_fit(X_train, y_train, lam)
            y_pred = gaussian_naive_bayes_predict(X_test, model)
            acc = accuracy_score(y_test, y_pred)
            accuracy_results[lam].append(acc)

    def calculate_stats(acc_list):
        return {
            "Média": np.mean(acc_list),
            "Desvio Padrão": np.std(acc_list),
            "Máximo": np.max(acc_list),
            "Mínimo": np.min(acc_list)
        }

    stats = {lam: calculate_stats(accuracy_results[lam]) for lam in lambdas}
    
    print("\nResultados da Validação por Monte Carlo:")
    for lam in lambdas:
        print(f"λ = {lam}: {stats[lam]}")

    # Visualização: Boxplot da acurácia para cada valor de λ
    plt.figure(figsize=(10,6))
    plt.boxplot([accuracy_results[lam] for lam in lambdas], labels=[f"λ = {lam}" for lam in lambdas])
    plt.title("Distribuição da Acurácia dos Modelos por Monte Carlo")
    plt.xlabel("λ (Lambda)")
    plt.ylabel("Acurácia")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    classification_file = "../data/EMGsDataset.csv"
    classification(classification_file)
