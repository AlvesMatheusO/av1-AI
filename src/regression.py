import numpy as np
import matplotlib.pyplot as plt

def regression(nome="C:/Users/ianaj/av1-AI/data/atividade_enzimatica.csv"):
    ## 1 Visualização inicial dos dados 
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
    
    ## 2 Organização da dimensão
    y = y.reshape(-1,1)

    ## 3 MQO tradicional, MQO regularizado e Média de valores observáveis.
    # Modelo da média:
    mediaY = np.mean(y)
    beta_media = np.array([mediaY, 0, 0]).reshape(-1, 1)

    # Modelo de MQO Tradicional
    X_aug = np.hstack((np.ones((x.shape[0], 1)), x))
    beta_mqo = np.linalg.pinv(X_aug.T @ X_aug) @ (X_aug.T @ y)

    # Modelo de MQO Regularizado
    lambdas = [0, 0.25, 0.5, 0.75, 1]
    beta_regs = {}
    for lamb in lambdas:
        I = np.eye(X_aug.shape[1])
        I[0, 0] = 0
        beta_reg = np.linalg.pinv(X_aug.T @ X_aug + lamb * I) @ (X_aug.T @ y)
        beta_regs[lamb] = beta_reg

    print("\n=== RESULTADOS ===\n")
    print("Modelo da Média:")
    print(f"  Coeficientes: {beta_media.ravel()}\n")
    print("MQO Tradicional:")
    print(f"  Coeficientes: {beta_mqo.ravel()}\n")
    print("MQO Regularizado (Tikhonov):")
    
    for lamb in lambdas:
        print(f" λ = {lamb} -> beta: {beta_regs[lamb].ravel()}")    

    plt.show()

    return X_aug, y, mediaY, beta_mqo, beta_regs, lambdas

if __name__ == "__main__":
    X_aug, y, mediaY, beta_mqo, beta_regs, lambdas = regression()

    ## 4 modelo de MQO Regularizado

    lambdas = [0, 0.25, 0.5, 0.75, 1]
    beta_regs = {}
    for lamb in lambdas:
        I = np.eye(X_aug.shape[1])
        I[0, 0] = 0
        beta_reg = np.linalg.pinv(X_aug.T @ X_aug + lamb * I) @ (X_aug.T @ y)
        beta_regs[lamb] = beta_reg


    print("\n=== RESULTADOS ===\n")

    print("Modelo da Média:")
    print(f"  Coeficientes: {beta_media.ravel()}\n")

    print("MQO Tradicional:")
    print(f"  Coeficientes: {beta_mqo.ravel()}\n")

    print("MQO Regularizado (Tikhonov):")
    
    for lamb in lambdas:
        print(f" λ = {lamb} -> beta: {beta_regs[lamb].ravel()}")    


    


    plt.show()

    ## 5 Validação por Monte Carlo
    R = 500  # Número de rodadas
    N = len(y)  # Número total de amostras
    train_size = int(0.8 * N)  # 80% para treinamento

    # Listas para armazenar os resultados de RSS
    rss_media = []
    rss_mqo = []
    rss_regs = {lamb: [] for lamb in lambdas}

    for r in range(R):
        # Particionamento aleatório dos dados
        indices = np.random.permutation(N)
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]

        # Dados de treinamento e teste
        X_train, y_train = X_aug[train_indices], y[train_indices]
        X_test, y_test = X_aug[test_indices], y[test_indices]

        # Modelo da Média
        y_pred_media = np.full_like(y_test, mediaY)
        rss_media.append(np.sum((y_test - y_pred_media) ** 2))

        # Modelo MQO Tradicional
        beta_mqo_train = np.linalg.pinv(X_train.T @ X_train) @ (X_train.T @ y_train)
        y_pred_mqo = X_test @ beta_mqo_train
        rss_mqo.append(np.sum((y_test - y_pred_mqo) ** 2))

        # Modelos MQO Regularizados
        for lamb in lambdas:
            I = np.eye(X_train.shape[1])
            I[0, 0] = 0  # Não regularizamos o intercepto
            beta_reg_train = np.linalg.pinv(X_train.T @ X_train + lamb * I) @ (X_train.T @ y_train)
            y_pred_reg = X_test @ beta_reg_train
            rss_regs[lamb].append(np.sum((y_test - y_pred_reg) ** 2))

    # Cálculo das estatísticas finais
    def calculate_statistics(rss_list):
        return {
            "Média": np.mean(rss_list),
            "Desvio-Padrão": np.std(rss_list),
            "Maior Valor": np.max(rss_list),
            "Menor Valor": np.min(rss_list)
        }

    stats_media = calculate_statistics(rss_media)
    stats_mqo = calculate_statistics(rss_mqo)
    stats_regs = {lamb: calculate_statistics(rss_regs[lamb]) for lamb in lambdas}

    # Exibição dos resultados
    print("\n=== ESTATÍSTICAS FINAIS ===\n")
    print("Modelo da Média:")
    print(stats_media)

    print("\nMQO Tradicional:")
    print(stats_mqo)

    print("\nMQO Regularizado (Tikhonov):")
    for lamb in lambdas:
        print(f" λ = {lamb}:")
        print(stats_regs[lamb])
