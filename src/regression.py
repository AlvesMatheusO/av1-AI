import numpy as np
import matplotlib.pyplot as plt

def regression(nome="../data/atividade_enzimatica.csv", R=500):

    # -------------------------------------------------------
    # 1) Visualização inicial dos dados
    # -------------------------------------------------------
    data = np.loadtxt(nome, delimiter=',')
    x = data[:, :2]  # temperatura e pH
    y = data[:, 2]   # atividade enzimática

    plt.figure(figsize=(6,5))
    scatter = plt.scatter(x[:,0], x[:,1], c=y, cmap='viridis', edgecolor='k')
    cbar = plt.colorbar(scatter)
    cbar.set_label("Atividade Enzimática")
    plt.xlabel("Temperatura")
    plt.ylabel("pH")
    plt.title("Visualização dos Dados - Atividade Enzimática")
    

    # -------------------------------------------------------
    # 2) Organização dos dados
    # -------------------------------------------------------
    y = y.reshape(-1, 1)  # transforma y em (N,1)
    
    # -------------------------------------------------------
    # 3) Cálculo dos coeficientes dos modelos
    # -------------------------------------------------------
    # Modelo da Média
    mediaY = np.mean(y)
    beta_media = np.array([mediaY, 0, 0]).reshape(-1, 1)

    # MQO Tradicional e MQO Regularizado
    X_aug = np.hstack((np.ones((x.shape[0], 1)), x))
    lambdas = [0, 0.25, 0.5, 0.75, 1] 
    beta_regs = {}
    for lam in lambdas:
        I = np.eye(X_aug.shape[1])
        I[0, 0] = 0  
        beta_reg = np.linalg.pinv(X_aug.T @ X_aug + lam * I) @ (X_aug.T @ y)
        beta_regs[lam] = beta_reg

    print("\n=== COEFICIENTES (usando todo o dataset) ===\n")
    print("Modelo da Média:")
    print(f"  Coeficientes: {beta_media.ravel()}\n")
    print("MQO Tradicional (λ = 0):")
    print(f"  Coeficientes: {beta_regs[0].ravel()}\n")
    print("MQO Regularizado (Tikhonov):")
    for lam in lambdas[1:]:
        print(f"  λ = {lam} -> Coeficientes: {beta_regs[lam].ravel()}")

    # -------------------------------------------------------
    # 4) Validação por Monte Carlo
    # -------------------------------------------------------
    N = len(y)                  # numero total de amostras
    train_size = int(0.8 * N)   # 80% para treinamento

    rss_media = []
    rss_mqo = []
    rss_regs = {lam: [] for lam in lambdas}

    for r in range(R):

        indices = np.random.permutation(N)
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]

        X_train, y_train = X_aug[train_indices], y[train_indices]
        X_test, y_test = X_aug[test_indices], y[test_indices]

        # Modelo da Média
        media_train = np.mean(y_train)
        y_pred_media = np.full_like(y_test, media_train)
        rss_media.append(np.sum((y_test - y_pred_media)**2))

        # MQO Tradicional
        beta_mqo_train = np.linalg.pinv(X_train.T @ X_train) @ (X_train.T @ y_train)
        y_pred_mqo = X_test @ beta_mqo_train
        rss_mqo.append(np.sum((y_test - y_pred_mqo)**2))

        # MQO Regularizado para cada λ
        for lam in lambdas:
            I = np.eye(X_train.shape[1])
            I[0, 0] = 0  # não regulariza o intercepto
            beta_reg_train = np.linalg.pinv(X_train.T @ X_train + lam * I) @ (X_train.T @ y_train)
            y_pred_reg = X_test @ beta_reg_train
            rss_regs[lam].append(np.sum((y_test - y_pred_reg)**2))

    # -------------------------------------------------------
    # 5) Cálculo e exibição das estatísticas finais do RSS
    # -------------------------------------------------------
    def calculate_statistics(rss_list):
        return {
            "Média": np.mean(rss_list),
            "Desvio-Padrão": np.std(rss_list),
            "Maior Valor": np.max(rss_list),
            "Menor Valor": np.min(rss_list)
        }

    stats_media = calculate_statistics(rss_media)
    stats_mqo = calculate_statistics(rss_mqo)
    stats_regs = {lam: calculate_statistics(rss_regs[lam]) for lam in lambdas}

    print("\n=== ESTATÍSTICAS FINAIS (RSS) - Validação Monte Carlo (R = {}) ===\n".format(R))
    print("Modelo da Média:")
    print(stats_media)
    print("\nMQO Tradicional (λ = 0):")
    print(stats_regs[0])
    print("\nMQO Regularizado (Tikhonov):")
    for lam in lambdas[1:]:
        print(f"  λ = {lam}:")
        print(stats_regs[lam])
    plt.show()

if __name__ == "__main__":
    regression()
