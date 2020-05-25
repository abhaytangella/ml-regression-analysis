# ML Final Project
# Abhay Tangella, 19 April 2020

# General packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

# LinReg packages
from sklearn.linear_model import LinearRegression

# KRR packages
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics.pairwise import laplacian_kernel
from sklearn.metrics.pairwise import polynomial_kernel
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.metrics.pairwise import sigmoid_kernel
from sklearn.metrics.pairwise import additive_chi2_kernel

# KNR packages
from sklearn.neighbors import KNeighborsRegressor

# NN packages
from sklearn.neural_network import MLPRegressor

# Various sklearn packages
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


def load_data():
    print("Loading data")
    df = pd.read_csv("weather.csv")

    # Remove weather column
    df = df.drop(columns=['weather'])

    # Want to predict temperature
    labels = df['temperature']

    # Remove temperature column from dataset
    # df = pd.get_dummies(data=df, columns=['weather'], prefix=['weather'])
    # print(df.head(5))
    df = df.drop(columns='temperature')

    return labels, df

def get_test_train(labels, df):
    train_x, test_x, train_y, test_y = train_test_split(df, labels, test_size=0.25)

    scaler = StandardScaler()
    scaler.fit(train_x)
    train_x = scaler.transform(train_x)
    test_x = scaler.transform(test_x)


    print('Training Features Shape:', train_x.shape)
    print('Training Labels Shape:', train_y.shape)
    print('Testing Features Shape:', test_x.shape)
    print('Testing Labels Shape:', test_y.shape)

    return train_x, test_x, train_y, test_y

def choose_krr_alpha(train_x, test_x, train_y, test_y):
    alphas = [0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 2.0]
    alpha_scores = []
    best_a_score = 0.0
    best_a = ""

    for a in alphas:
        krr = KernelRidge(kernel="laplacian", alpha=a)
        krr.fit(train_x, train_y)
        krr.predict(test_x)
        score = krr.score(test_x, test_y)
        if score > best_a_score:
            best_a_score = score
            best_a = a
        alpha_scores.append(score)
    
    print(alpha_scores)
    print("Best alpha: " + str(best_a))
    print("Score received: " + str(best_a_score))

    plt.plot(alphas, alpha_scores)
    plt.xlabel('Alpha')
    plt.ylabel('Score')
    plt.title('Tuning Alpha Hyperparameter for KRR')
    plt.show()

def choose_krr_gamma(train_x, test_x, train_y, test_y):
    gammas = [0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 2.0]
    gamma_scores = []
    best_g_score = 0.0
    best_g = ""

    for g in gammas:
        krr = KernelRidge(kernel="laplacian", gamma=g)
        krr.fit(train_x, train_y)
        krr.predict(test_x)
        score = krr.score(test_x, test_y)
        if score > best_g_score:
            best_g_score = score
            best_g = g
        gamma_scores.append(score)
    
    print(gamma_scores)
    print("Best gamma: " + str(best_g))
    print("Score received: " + str(best_g_score))

    plt.plot(gammas, gamma_scores)
    plt.xlabel('Gamma')
    plt.ylabel('Score')
    plt.title('Tuning Gamma Hyperparameter for KRR')
    plt.show()

def choose_krr_kernel(train_x, test_x, train_y, test_y):
    kernels = ['linear', 'rbf', 'laplacian', 'polynomial', 'sigmoid']
    kernel_scores = []
    best_k_score = 0.0
    best_k = ""

    for k in kernels:
        krr = KernelRidge(kernel=k)
        krr.fit(train_x, train_y)
        krr.predict(test_x)
        score = krr.score(test_x, test_y)
        if score > best_k_score:
            best_k_score = score
            best_k = k
        kernel_scores.append(score)
    
    print(kernel_scores)
    print("Best kernel: " + str(best_k))
    print("Score received: " + str(best_k_score))

    plt.bar(kernels, kernel_scores)
    plt.xlabel('Kernel')
    plt.ylabel('Score')
    plt.xticks(np.arange(len(kernels)), kernels)
    plt.title('Tuning Kernel Hyperparameter for KRR')
    plt.show()


def tune_krr_hyperparams(train_x, test_x, train_y, test_y):
    print("Starting KRR hyperparameters tuning")
    krr = KernelRidge()
    kernel_types = ['linear', 'rbf', 'laplacian', 'polynomial', 'sigmoid']
    # kernel_types = ['linear', 'rbf', 'laplacian']
    alphas = [0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 2.0]
    gammas = [0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 2.0]
    params = {
        "alpha": alphas,
        "kernel": kernel_types,
        "gamma": gammas
    }
    krr = GridSearchCV(krr, params, n_jobs=-1)
    krr.fit(train_x, train_y)
    print(krr.best_params_)
    return krr.best_params_

def krr_predict(krr_params, train_x, test_x, train_y, test_y):
    print("Starting KRR prediction")
    a = krr_params['alpha']
    g = krr_params['gamma']
    k = krr_params['kernel']

    krr = KernelRidge(alpha=a, kernel=k, gamma=g)
    krr.fit(train_x, train_y)
    print("KRR Score: ", krr.score(test_x, test_y))

    cv_score = cross_val_score(krr, test_x, test_y, cv=10)
    print("Cross-Val Standard Deviation: ", np.std(cv_score))

    print("Scores:", krr.score(test_x, test_y))

    return krr.score(test_x, test_y)

def choose_knr_nn(train_x, test_x, train_y, test_y):
    nn = []
    for i in range(1, 41):
        nn.append(i)
    nn_scores = []
    best_nn_score = 0.0
    best_nn = ""

    for n in nn:
        knr = KNeighborsRegressor(n_neighbors=n)
        knr.fit(train_x, train_y)
        knr.predict(test_x)
        score = knr.score(test_x, test_y)
        if score > best_nn_score:
            best_nn_score = score
            best_nn = n
        nn_scores.append(score)
    
    print(nn_scores)
    print("Best Nearest Neighbor: " + str(best_nn))
    print("Score received: " + str(best_nn_score))

    plt.plot(nn, nn_scores)
    plt.xlabel('Nearest Neigbors Count')
    plt.ylabel('Score')
    plt.title('Tuning Nearest Neighbor Hyperparameter for KNR')
    plt.show()

def choose_knr_weight(train_x, test_x, train_y, test_y):
    weights = ["uniform", "distance"]
    weight_scores = []
    best_w_score = 0.0
    best_w = ""

    for w in weights:
        knr = KNeighborsRegressor(weights=w)
        knr.fit(train_x, train_y)
        knr.predict(test_x)
        score = knr.score(test_x, test_y)
        if score > best_w_score:
            best_w_score = score
            best_w = w
        weight_scores.append(score)
    
    print(weight_scores)
    print("Best Weight: " + str(best_w))
    print("Score received: " + str(best_w_score))

    plt.bar(weights, weight_scores)
    plt.xlabel('Weight')
    plt.ylabel('Score')
    plt.xticks(np.arange(len(weights)), weights)
    plt.title('Tuning Weight Hyperparameter for KNR')
    plt.show()

def choose_knr_power_param(train_x, test_x, train_y, test_y):
    pp = [1, 2]
    pp_scores = []
    best_p_score = 0.0
    best_p = ""

    for p in pp:
        knr = KNeighborsRegressor(p=p)
        knr.fit(train_x, train_y)
        knr.predict(test_x)
        score = knr.score(test_x, test_y)
        if score > best_p_score:
            best_p_score = score
            best_p = p
        pp_scores.append(score)
    
    print(pp_scores)
    print("Best Power parameter: " + str(best_p))
    print("Score received: " + str(best_p_score))

    plt.bar(pp, pp_scores)
    plt.xlabel('Power parameter')
    plt.ylabel('Score')
    plt.xticks(pp, pp)
    plt.title('Tuning Power Hyperparameter for KNR')
    plt.show()

def tune_knr_hyperparams(train_x, test_x, train_y, test_y):
    print("Starting KNR hyperparameters tuning")
    knr = KNeighborsRegressor()
    nn = []
    for i in range(1, 41):
        nn.append(i)
    w = ["uniform", "distance"]
    # best algorithm is already chosen with auto param
    p = [1, 2]
    params = {
        "n_neighbors": nn,
        "weights": w,
        "p": p
    }
    knr = GridSearchCV(knr, params, n_jobs=-1)
    knr.fit(train_x, train_y)
    print(knr.best_params_)
    return knr.best_params_

def knr_predict(knr_params, train_x, test_x, train_y, test_y):
    print("Starting KNR prediction")
    nn = knr_params["n_neighbors"]
    w = knr_params["weights"]
    p = knr_params["p"]
    knr = KNeighborsRegressor(n_neighbors=nn, weights=w, p=p)
    knr.fit(train_x, train_y)

    cv_score = cross_val_score(knr, test_x, test_y, cv=10)
    print("Cross-Val Standard Deviation: ", np.std(cv_score))

    print("Scores:", knr.score(test_x, test_y))

    return knr.score(test_x, test_y)

def choose_nn_hidden_layers(train_x, test_x, train_y, test_y):
    hidden_layer_sizes = [(5,5,5), (10, 10, 10), (30, 30, 30), (50,50,50), (50,100,50), (100, 100, 100)]
    hl_scores = []
    hl_labels = []
    best_hl_score = 0.0
    best_hl = ""

    for hl in hidden_layer_sizes:
        hl_labels.append(str(hl))
        mlp = MLPRegressor(hidden_layer_sizes=hl)
        mlp.fit(train_x, train_y)
        mlp.predict(test_x)
        score = mlp.score(test_x, test_y)
        if score > best_hl_score:
            best_hl_score = score
            best_hl = hl
        hl_scores.append(score)
    
    print(hl_scores)
    print("Best Hidden Layer Size: " + str(best_hl))
    print("Score received: " + str(best_hl_score))

    plt.bar(hl_labels, hl_scores)
    plt.xlabel('Hidden Layer Size')
    plt.ylabel('Score')
    plt.xticks(np.arange(len(hl_labels)), hidden_layer_sizes)
    plt.title('Tuning Hidden Layer Hyperparameter for NN')
    plt.show()

def choose_nn_alpha(train_x, test_x, train_y, test_y):
    alphas = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 1.0]
    alpha_scores = []
    a_labels = []
    best_a_score = 0.0
    best_a = ""

    for a in alphas:
        a_labels.append(str(a))
        mlp = MLPRegressor(hidden_layer_sizes=(50, 50, 50), alpha=a)
        mlp.fit(train_x, train_y)
        mlp.predict(test_x)
        score = mlp.score(test_x, test_y)
        if score > best_a_score:
            best_a_score = score
            best_a = a
        alpha_scores.append(score)
    
    print(alpha_scores)
    print("Best alphas: " + str(best_a))
    print("Score received: " + str(best_a_score))

    plt.plot(a_labels, alpha_scores)
    plt.xlabel('Alpha')
    plt.ylabel('Score')
    plt.xticks(np.arange(len(a_labels)), alphas)
    plt.title('Tuning Alpha Hyperparameter for NN')
    plt.show()

def tune_nn_hyperparams(train_x, test_x, train_y, test_y):
    print("Starting NN hyperparameters tuning")
    mlp = MLPRegressor()
    hidden_layer_sizes = [(5,5,5), (10, 10, 10), (30, 30, 30), (50,50,50), (50,100,50), (100, 100, 100)] 
    alphas = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 1.0]
    params = {
        "hidden_layer_sizes": hidden_layer_sizes,
        "alpha": alphas
    }
    mlp = GridSearchCV(mlp, params, n_jobs=-1)
    mlp.fit(train_x, train_y)
    print(mlp.best_params_)
    return mlp.best_params_

def nn_predict(mlp_params, train_x, test_x, train_y, test_y):
    print("Starting NN prediction")
    hls = mlp_params["hidden_layer_sizes"]
    a = mlp_params["alpha"]
    mlp = MLPRegressor(hidden_layer_sizes=hls, alpha=a)
    mlp.fit(train_x, train_y)

    cv_score = cross_val_score(mlp, test_x, test_y, cv=10)
    print("Cross-Val Standard Deviation: ", np.std(cv_score))

    print("Scores:", mlp.score(test_x, test_y))

    return mlp.score(test_x, test_y)

def compare_algos(krr_score, knr_score, nn_score):
    print("Comparing algorithms")
    score_list = [krr_score, knr_score, nn_score]
    plt.bar(np.arange(len(score_list)), score_list, label='Training Data')

    plt.xlabel('Regression Algorithm', fontweight='bold')
    plt.ylabel('Accuracy', fontweight='bold')

    plt.xticks([r for r in range(len(score_list))], ['Kernel Ridge', 'k-Neigbors', 'Neural Network'])
    plt.title('Accuracy of Various Regression Models')
    plt.show()

def main():
    labels, df = load_data()
    df_save = df

    df_list = list(df.columns)
    df = np.array(df)

    train_x, test_x, train_y, test_y = get_test_train(labels, df)

    reg = LinearRegression().fit(train_x, train_y)
    linreg_score = reg.score(train_x, train_y)
    print("Linear Regression Score:", linreg_score)

    # Kernal Ridge Regressor
    choose_krr_kernel(train_x, test_x, train_y, test_y)
    choose_krr_alpha(train_x, test_x, train_y, test_y)
    choose_krr_gamma(train_x, test_x, train_y, test_y)

    start_time = time.time()
    krr_params = tune_krr_hyperparams(train_x, test_x, train_y, test_y)
    krr_score = krr_predict(krr_params, train_x, test_x, train_y, test_y)
    end_time = time.time()

    print("KRR took", str(end_time - start_time), "seconds to train.")

    # k-Neighbors Regressor
    choose_knr_nn(train_x, test_x, train_y, test_y)
    choose_knr_weight(train_x, test_x, train_y, test_y)
    choose_knr_power_param(train_x, test_x, train_y, test_y)

    start_time = time.time()
    knr_params = tune_knr_hyperparams(train_x, test_x, train_y, test_y)
    knr_score = knr_predict(knr_params, train_x, test_x, train_y, test_y)
    end_time = time.time()

    print("KNR took", str(end_time - start_time), "seconds to train.")

    # Neural Network
    choose_nn_hidden_layers(train_x, test_x, train_y, test_y)
    choose_nn_alpha(train_x, test_x, train_y, test_y)

    start_time = time.time()
    nn_params = tune_nn_hyperparams(train_x, test_x, train_y, test_y)
    nn_score = nn_predict(nn_params, train_x, test_x, train_y, test_y)
    end_time = time.time()

    print("KNR took", str(end_time - start_time), "seconds to train.")

    # Return final scores
    print("KRR params used: ", krr_params)
    print("KRR Score: ", krr_score)
    print("KNR params used: ", knr_params)
    print("KNR Score: ", knr_score)
    print("NN params used: ", nn_params)
    print("NN Score: ", nn_score)

    compare_algos(krr_score, knr_score, nn_score)
    
    print("finished!")

    

if __name__ == '__main__':
    main()
