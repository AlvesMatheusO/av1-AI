import numpy as np
import matplotlib.pyplot as plt

def regression(nome="../data/atividade_enzimatica.csv"):

    ## 1 Visualizacão inicial dos dados 
    data = np.loadtxt(nome, delimiter=',')

    x = data[:, :2]  # temperatura e pH
    y = data[:, 2]   # atividade enzimática

    plt.figure(figsize=(6,5))
    scatter = plt.scatter(x[:,0], x[:,1], c=y, cmap='viridis', edgecolor='k')
    cbar = plt.colorbar(scatter)
    cbar.set_label("Atividade Enzimatica")
    plt.xlabel("Temperatura")
    plt.ylabel("pH")
    plt.title("Visualização dos Dados - Atividade Enzimática")
   


    ## 2 Organização da dimensão
    y = y.reshape(-1,1)
    
    ## 3 MQO tradicional, MQO regularizado e Média de valores observaveis.
    #modelo de media:
    mediaY = np.mean(y)
    beta_media = np.array([mediaY, 0, 0]).reshape(-1, 1)

    #modelo de MQO Tradicional
    X_aug = np.hstack((np.ones((x.shape[0], 1)), x))
    beta_mqo = np.linalg.pinv(X_aug.T @ X_aug) @ (X_aug.T @ y)

    #modelo de MQO Regularizado
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

if __name__ == "__main__":
    regression()
