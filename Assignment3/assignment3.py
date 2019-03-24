#------------------Imports-----------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, FastICA
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.random_projection import GaussianRandomProjection
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, make_scorer, recall_score, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.neural_network import MLPClassifier
from collections import defaultdict
import time
from sklearn.metrics import silhouette_score
import warnings
warnings.simplefilter("ignore")

#------------------------------------------------------------------

#------------------Data--------------------------------------------------------------
df_australian = pd.read_csv("australian-credit-approval.csv", header=None)
df_german = pd.read_csv("german-credit-data.csv", header=None)
X_australian, y_australian = df_australian.iloc[:,1:-1], df_australian.iloc[:,-1]
X_german, y_german = df_german.iloc[:,1:-1], df_german.iloc[:,-1]
y_german[y_german == 1] = 0
y_german[y_german == 2] = 1

plt.scatter(X_german.loc[y_german == 1,:].iloc[:,1], X_german.loc[y_german == 1,:].iloc[:,2], c='red', label='Samples with label 1')
plt.scatter(X_german.loc[y_german == 0,:].iloc[:,1], X_german.loc[y_german == 0,:].iloc[:,2], c='green', label='Samples with label 0')
plt.title("German dataset without Dimensionality Reduction")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.legend()
plt.savefig("German-without-DR.png")
plt.close()

plt.scatter(X_australian.loc[y_australian == 1,:].iloc[:,1], X_australian.loc[y_australian == 1,:].iloc[:,2], c='red', label='Samples with label 1')
plt.scatter(X_australian.loc[y_australian == 0,:].iloc[:,1], X_australian.loc[y_australian == 0,:].iloc[:,2], c='green', label='Samples with label 0')
plt.title("Australian dataset without Dimensionality Reduction")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.legend()
plt.savefig("Australian-without-DR.png")
plt.close()

#------------------------------------------------------------------------------------

#----------------KMeans--------------------------------------------------------------
def kmeans(k, X, y, init='k-means++', n_init=10, max_iter=3000, tol=0.00001, precompute_distances=True, random_state=None, algorithm='auto', plot=1, comp1=0, comp2=1, dataset='german', action='none', part=0):
    kmeans_model = KMeans(n_clusters=k, init=init, n_init=n_init, max_iter=max_iter, tol=tol, precompute_distances=precompute_distances, random_state=random_state, algorithm=algorithm)
    kmeans_model.fit(X)
    c = ['red', 'green', 'blue', 'black', 'yellow', 'orange', 'pink', 'brown', 'violet', 'indigo']
    y_new = kmeans_model.predict(X)
    X = np.array(X)
    if plot:
        if dataset == 'german':
            if action == 'none':
                for i in range(0,k):
                    plt.scatter(X[y_new == i, comp1], X[y_new == i, comp2], c = c[i], label='Cluster {}'.format(i+1))
                plt.title("KMeans on German dataset with {} clusters without Dimensionality Reduction".format(k))
                plt.legend()
                plt.xlabel("Component 1")
                plt.ylabel("Component 2")
                plt.savefig("KMeans-german-without-DR.png")
                plt.close()

            elif action == 'pca':
                for i in range(0,k):
                    plt.scatter(X[y_new == i, comp1], X[y_new == i, comp2], c = c[i], label='Cluster {}'.format(i+1))
                plt.title("KMeans on German dataset with {} clusters after PCA".format(k))
                plt.legend()
                plt.xlabel("Component 1")
                plt.ylabel("Component 2")
                plt.savefig("KMeans-german-with-pca.png")
                plt.close()

            elif action == 'ica':
                for i in range(0,k):
                    plt.scatter(X[y_new == i, comp1], X[y_new == i, comp2], c = c[i], label='Cluster {}'.format(i+1))
                plt.title("KMeans on German dataset with {} clusters after ICA".format(k))
                plt.legend()
                plt.xlabel("Component 1")
                plt.ylabel("Component 2")
                plt.savefig("KMeans-german-with-ica.png")
                plt.close()

            elif action == 'rp':
                for i in range(0,k):
                    plt.scatter(X[y_new == i, comp1], X[y_new == i, comp2], c = c[i], label='Cluster {}'.format(i+1))
                plt.title("KMeans on German dataset with {} clusters after Randomized Projection".format(k))
                plt.legend()
                plt.xlabel("Component 1")
                plt.ylabel("Component 2")
                plt.savefig("KMeans-german-with-rp.png")
                plt.close()

            elif action == 'rfe':
                for i in range(0,k):
                    plt.scatter(X[y_new == i, comp1], X[y_new == i, comp2], c = c[i], label='Cluster {}'.format(i+1))
                plt.title("KMeans on German dataset with {} clusters after RFE".format(k))
                plt.legend()
                plt.xlabel("Component 1")
                plt.ylabel("Component 2")
                plt.savefig("KMeans-german-with-rfe.png")
                plt.close()

        elif dataset == 'australian':
            if action == 'none':
                for i in range(0, k):
                    plt.scatter(X[y_new == i, comp1], X[y_new == i, comp2], c=c[i], label='Cluster {}'.format(i+1))
                plt.title("KMeans on Australian dataset with {} clusters without Dimensionality Reduction".format(k))
                plt.legend()
                plt.xlabel("Component 1")
                plt.ylabel("Component 2")
                plt.savefig("KMeans-australian-without-DR.png")
                plt.close()

            elif action == 'pca':
                for i in range(0, k):
                    plt.scatter(X[y_new == i, comp1], X[y_new == i, comp2], c=c[i], label='Cluster {}'.format(i+1))
                plt.title("KMeans on Australian dataset with {} clusters after PCA".format(k))
                plt.legend()
                plt.xlabel("Component 1")
                plt.ylabel("Component 2")
                plt.savefig("KMeans-australian-with-pca.png")
                plt.close()

            elif action == 'ica':
                for i in range(0, k):
                    plt.scatter(X[y_new == i, comp1], X[y_new == i, comp2], c=c[i], label='Cluster {}'.format(i+1))
                plt.title("KMeans on Australian dataset with {} clusters after ICA".format(k))
                plt.legend()
                plt.xlabel("Component 1")
                plt.ylabel("Component 2")
                plt.savefig("KMeans-australian-with-ica.png")
                plt.close()

            elif action == 'rp':
                for i in range(0, k):
                    plt.scatter(X[y_new == i, comp1], X[y_new == i, comp2], c=c[i], label='Cluster {}'.format(i+1))
                plt.title("KMeans on Australian dataset with {} clusters after Randomized Projection".format(k))
                plt.legend()
                plt.xlabel("Component 1")
                plt.ylabel("Component 2")
                plt.savefig("KMeans-australian-with-rp.png")
                plt.close()

            elif action == 'rfe':
                for i in range(0, k):
                    plt.scatter(X[y_new == i, comp1], X[y_new == i, comp2], c=c[i], label='Cluster {}'.format(i+1))
                plt.title("KMeans on Australian dataset with {} clusters after RFE".format(k))
                plt.legend()
                plt.xlabel("Component 1")
                plt.ylabel("Component 2")
                plt.savefig("KMeans-australian-with-rfe.png")
                plt.close()

    if part == 5:
        return y_new
    else:
        return silhouette_score(X,y_new)

#-------------------------------------------------------------------------------------

#-------------EM----------------------------------------------------------------------
def em(n_components, X, y, covariance_type='full', tol=0.000001, reg_covar=0.000001, max_iter=3000, n_init=100, init_params='kmeans', random_state=None, warm_start=True, plot=1, comp1=0, comp2=1, dataset='german', action='none', part=0):
    em_model = GaussianMixture(n_components=n_components, covariance_type=covariance_type, tol=tol, reg_covar=reg_covar, max_iter=max_iter, n_init=n_init, init_params=init_params, random_state=random_state, warm_start=warm_start)
    em_model.fit(X)
    c = ['red', 'green', 'blue', 'black', 'yellow', 'orange', 'pink', 'brown', 'violet', 'indigo']
    y_new = em_model.predict(X)
    X = np.array(X)
    if plot:
        if dataset == 'german':
            if action == 'none':
                for i in range(0, n_components):
                    plt.scatter(X[y_new == i, comp1], X[y_new == i, comp2], c=c[i], label='Cluster {}'.format(i+1))
                plt.title("EM on German dataset with {} clusters without Dimensionality Reduction".format(n_components))
                plt.legend()
                plt.xlabel("Component 1")
                plt.ylabel("Component 2")
                plt.savefig("EM-german-without-DR.png")
                plt.close()

            elif action == 'pca':
                for i in range(0, n_components):
                    plt.scatter(X[y_new == i, comp1], X[y_new == i, comp2], c=c[i], label='Cluster {}'.format(i+1))
                plt.title("EM on German dataset with {} clusters after PCA".format(n_components))
                plt.legend()
                plt.xlabel("Component 1")
                plt.ylabel("Component 2")
                plt.savefig("EM-german-with-pca.png")
                plt.close()

            elif action == 'ica':
                for i in range(0, n_components):
                    plt.scatter(X[y_new == i, comp1], X[y_new == i, comp2], c=c[i], label='Cluster {}'.format(i+1))
                plt.title("EM on German dataset with {} clusters after ICA".format(n_components))
                plt.legend()
                plt.xlabel("Component 1")
                plt.ylabel("Component 2")
                plt.savefig("EM-german-with-ica.png")
                plt.close()

            elif action == 'rp':
                for i in range(0, n_components):
                    plt.scatter(X[y_new == i, comp1], X[y_new == i, comp2], c=c[i], label='Cluster {}'.format(i+1))
                plt.title("EM on German dataset with {} clusters after Randomized Projection".format(n_components))
                plt.legend()
                plt.xlabel("Component 1")
                plt.ylabel("Component 2")
                plt.savefig("EM-german-with-rp.png")
                plt.close()

            elif action == 'rfe':
                for i in range(0, n_components):
                    plt.scatter(X[y_new == i, comp1], X[y_new == i, comp2], c=c[i], label='Cluster {}'.format(i+1))
                plt.title("EM on German dataset with {} clusters after RFE".format(n_components))
                plt.legend()
                plt.xlabel("Component 1")
                plt.ylabel("Component 2")
                plt.savefig("EM-german-with-rfe.png")
                plt.close()

        elif dataset == 'australian':
            if action == 'none':
                for i in range(0, n_components):
                    plt.scatter(X[y_new == i, comp1], X[y_new == i, comp2], c=c[i], label='Cluster {}'.format(i+1))
                plt.title("EM on Australian dataset with {} clusters without Dimensionality Reduction".format(n_components))
                plt.legend()
                plt.xlabel("Component 1")
                plt.ylabel("Component 2")
                plt.savefig("EM-australian-without-DR.png")
                plt.close()

            elif action == 'pca':
                for i in range(0, n_components):
                    plt.scatter(X[y_new == i, comp1], X[y_new == i, comp2], c=c[i], label='Cluster {}'.format(i+1))
                plt.title("EM on Australian dataset with {} clusters after PCA".format(n_components))
                plt.legend()
                plt.xlabel("Component 1")
                plt.ylabel("Component 2")
                plt.savefig("EM-australian-with-pca.png")
                plt.close()

            elif action == 'ica':
                for i in range(0, n_components):
                    plt.scatter(X[y_new == i, comp1], X[y_new == i, comp2], c=c[i], label='Cluster {}'.format(i+1))
                plt.title("EM on Australian dataset with {} clusters after ICA".format(n_components))
                plt.legend()
                plt.xlabel("Component 1")
                plt.ylabel("Component 2")
                plt.savefig("EM-australian-with-ica.png")
                plt.close()

            elif action == 'rp':
                for i in range(0, n_components):
                    plt.scatter(X[y_new == i, comp1], X[y_new == i, comp2], c=c[i], label='Cluster {}'.format(i+1))
                plt.title("EM on Australian dataset with {} clusters after Randomized Projection".format(n_components))
                plt.legend()
                plt.xlabel("Component 1")
                plt.ylabel("Component 2")
                plt.savefig("EM-australian-with-rp.png")
                plt.close()

            elif action == 'rfe':
                for i in range(0, n_components):
                    plt.scatter(X[y_new == i, comp1], X[y_new == i, comp2], c=c[i], label='Cluster {}'.format(i+1))
                plt.title("EM on Australian dataset with {} clusters after RFE".format(n_components))
                plt.legend()
                plt.xlabel("Component 1")
                plt.ylabel("Component 2")
                plt.savefig("EM-australian-with-rfe.png")
                plt.close()

    if part == 5:
        return y_new
    else:
        return silhouette_score(X, y_new)
#------------------------------------------------------------------------------------

#---------------PCA------------------------------------------------------------------
def pca(n_components, X, y,copy=True, whiten=False, svd_solver='auto', tol=0.0, iterated_power='auto', random_state=None, plot=1, dataset='german'):
    pca_model = PCA(n_components=n_components, copy=copy, whiten=whiten, svd_solver=svd_solver, tol=tol, iterated_power=iterated_power, random_state=random_state)
    pca_model.fit(X)
    X_new = pca_model.transform(X)
    if plot:
        if dataset == 'german':
            plt.scatter(X_new[y == 1,1], X_new[y == 1,0], c='red', label='Samples with label 1')
            plt.scatter(X_new[y == 0, 1], X_new[y == 0, 0], c='green', label='Samples with label 0')
            plt.title("German dataset after PCA")
            plt.legend()
            plt.xlabel("Component 1")
            plt.ylabel("Component 2")
            plt.savefig("german-after-pca.png")
            plt.close()

        elif dataset == 'australian':
            plt.scatter(X_new[y == 1, 1], X_new[y == 1, 0], c='red', label='Samples with label 1')
            plt.scatter(X_new[y == 0, 1], X_new[y == 0, 0], c='green', label='Samples with label 0')
            plt.title("Australian dataset after PCA")
            plt.legend()
            plt.xlabel("Component 1")
            plt.ylabel("Component 2")
            plt.savefig("australian-after-pca.png")
            plt.close()
    return X_new

#-------------------------------------------------------------------------------------

#-------------ICA---------------------------------------------------------------------
def ica(n_components, X, y, algorithm='parallel', whiten=True, fun='logcosh', fun_args=None, max_iter=2000, tol=0.000001, w_init=None, random_state=3, plot=1, dataset='german'):
    ica_model = FastICA(n_components=n_components, algorithm=algorithm, whiten=whiten, fun=fun, fun_args=fun_args, max_iter=max_iter, tol=tol, w_init=w_init, random_state=random_state)
    ica_model.fit(X)
    X_new = ica_model.transform(X)
    if plot:
        if dataset == 'german':
            plt.scatter(X_new[y == 1, 0], X_new[y == 1, 1], c='red', label='Samples with label 1')
            plt.scatter(X_new[y == 0, 0], X_new[y == 0, 1], c='green', label='Samples with label 0')
            plt.title("German dataset after ICA")
            plt.legend()
            plt.xlabel("Component 1")
            plt.ylabel("Component 2")
            plt.savefig("german-after-ICA.png")
            plt.close()

        elif dataset == 'australian':
            plt.scatter(X_new[y == 1, 0], X_new[y == 1, 1], c='red', label='Samples with label 1')
            plt.scatter(X_new[y == 0, 0], X_new[y == 0, 1], c='green', label='Samples with label 0')
            plt.title("Australian dataset after ICA")
            plt.legend()
            plt.xlabel("Component 1")
            plt.ylabel("Component 2")
            plt.savefig("australian-after-ICA.png")
            plt.close()
    return X_new

#------------------------------------------------------------------------------------

#-----------Random Projection--------------------------------------------------------
def rp(X, y, n_components='auto', eps=0.1, random_state=None, plot=1, dataset='german'):
    rp_model = GaussianRandomProjection(n_components=n_components, eps=eps, random_state=random_state)
    rp_model.fit(X)
    X_new = rp_model.transform(X)
    if plot:
        if dataset == 'german':
            plt.scatter(X_new[y == 1, 0], X_new[y == 1, 1], c='red', label='Samples with label 1')
            plt.scatter(X_new[y == 0, 0], X_new[y == 0, 1], c='green', label='Samples with label 0')
            plt.title("German dataset after Randomized Projection")
            plt.legend()
            plt.xlabel("Component 1")
            plt.ylabel("Component 2")
            plt.savefig("german-after-Random-Projection.png")
            plt.close()

        elif dataset == 'australian':
            plt.scatter(X_new[y == 1, 0], X_new[y == 1, 1], c='red', label='Samples with label 1')
            plt.scatter(X_new[y == 0, 0], X_new[y == 0, 1], c='green', label='Samples with label 0')
            plt.title("Australian dataset after Randomized Projection")
            plt.legend()
            plt.xlabel("Component 1")
            plt.ylabel("Component 2")
            plt.savefig("australian-after-Random-Projection.png")
            plt.close()
    return X_new

#------------------------------------------------------------------------------------

#-----------RFE----------------------------------------------------------------------
def rfe(X, y, n_features_to_select=None, step=1, plot=1, dataset='german'):
    base_estimator = RandomForestClassifier()
    rfe_model = RFE(estimator=base_estimator, n_features_to_select=n_features_to_select, step=step)
    rfe_model.fit(X, y)
    X_new = rfe_model.transform(X)
    if plot:
        if dataset == 'german':
            plt.scatter(X_new[y == 1, 0], X_new[y == 1, 1], c='red', label='Samples with label 1')
            plt.scatter(X_new[y == 0, 0], X_new[y == 0, 1], c='green', label='Samples with label 0')
            plt.title("German dataset after RFE")
            plt.legend()
            plt.xlabel("Component 1")
            plt.ylabel("Component 2")
            plt.savefig("german-after-RFE.png")
            plt.close()

        elif dataset == 'australian':
            plt.scatter(X_new[y == 1, 0], X_new[y == 1, 1], c='red', label='Samples with label 1')
            plt.scatter(X_new[y == 0, 0], X_new[y == 0, 1], c='green', label='Samples with label 0')
            plt.title("Australian dataset after RFE")
            plt.legend()
            plt.xlabel("Component 1")
            plt.ylabel("Component 2")
            plt.savefig("australian-after-RFE.png")
            plt.close()
    return X_new
#------------------------------------------------------------------------------------

#--------Neural Network--------------------------------------------------------------
def nn(X_train, y_train, X_test, y_test,metric, n_splits, random_state, print_info, dataset):

    activation_functions = ['identity', 'logistic', 'tanh', 'relu']
    learning_rate = ['constant', 'invscaling', 'adaptive']
    hidden_layer_sizes = []
    for neurons in range(100, 250, 10):
        t = []
        val = neurons
        for size in range(0,1):
            t.append(val)
            val = val + neurons

        hidden_layer_sizes.append(tuple(t))

    param_grid_nn = {'activation': activation_functions, 'hidden_layer_sizes': hidden_layer_sizes, 'learning_rate': learning_rate}
    scorer_nn = make_scorer(metric)
    skf_nn = StratifiedKFold(n_splits=n_splits, random_state=random_state)
    nn_model = MLPClassifier(max_iter=10000)
    grid_search_model_nn = GridSearchCV(estimator=nn_model, param_grid=param_grid_nn, scoring=scorer_nn, cv=skf_nn)
    start_time = time.time()
    grid_search_model_nn.fit(X_train, y_train)
    end_time = time.time()
    y_pred_nn = grid_search_model_nn.predict(X_test)
    tn_nn, fp_nn, fn_nn, tp_nn = confusion_matrix(y_test, y_pred_nn).ravel()
    if print_info:
        print("Neural Network - Testing Data")
        print("Best paramaters = {}".format(grid_search_model_nn.best_params_))
        print("True negative = {}, False Positive = {}, False Negative = {}, True Positive = {}".format(tn_nn, fp_nn, fn_nn, tp_nn))
        print("Accuracy score = {}".format(accuracy_score(y_test, y_pred_nn)))
        print("Recall score = {}".format(recall_score(y_test, y_pred_nn)))
        print("AUC ROC score = {}".format(roc_auc_score(y_test, y_pred_nn)))
        print("F1 score = {}".format(f1_score(y_test, y_pred_nn)))
        print()
    if dataset == 1:
        return (recall_score(y_test, y_pred_nn)), end_time-start_time
    else:
        return (accuracy_score(y_test, y_pred_nn)), end_time-start_time
#------------------------------------------------------------------------------------

#---------Main Code------------------------------------------------------------------

#--------Part 1----------------------------------------------------------------------

#-------German dataset---------------------------------------------------------------
km = {}
em1 = {}
for i in range(2,21):
    km[i] = kmeans(i, X_german, y_german, plot=0)
    em1[i] = em(i, X_german, y_german, plot=0)

maxk_val = -1
maxk_key = -1
maxem_key = -1
maxem_val = -1

for key in km:
    if maxk_val < km[key]:
        maxk_key = key
        maxk_val = km[key]

for key in em1:
    if maxem_val < em1[key]:
        maxem_key = key
        maxem_val = em1[key]

kmeans(maxk_key, X_german, y_german, comp1=1, comp2=2, dataset='german', action='none')
em(maxem_key, X_german, y_german, comp1=1, comp2=2, dataset='german', action='none')
#---------------------------------------------------------------------------------------

#------Australian dataset---------------------------------------------------------------
km = {}
em1 = {}
for i in range(2,21):
    km[i] = kmeans(i, X_german, y_german, plot=0)
    em1[i] = em(i, X_german, y_german, plot=0)

maxk_val = -1
maxk_key = -1
maxem_key = -1
maxem_val = -1

for key in km:
    if maxk_val < km[key]:
        maxk_key = key
        maxk_val = km[key]

for key in em1:
    if maxem_val < em1[key]:
        maxem_key = key
        maxem_val = em1[key]

kmeans(maxk_key, X_australian, y_australian, comp1=1, comp2=2, dataset='australian', action='none')
em(maxem_key, X_australian, y_australian, comp1=1, comp2=2, dataset='australian', action='none')
#---------------------------------------------------------------------------------------

#----------End of Part 1----------------------------------------------------------------

#----------Part 2-----------------------------------------------------------------------

#---------German dataset----------------------------------------------------------------
pca(2, X_german, y_german, plot=1, dataset='german')
ica(2, X_german, y_german, plot=1, dataset='german')
rp(X_german, y_german, 2, plot=1, dataset='german')
rfe(X_german, y_german, 2, plot=1, dataset='german')
#----------------------------------------------------------------------------------------

#--------Australian dataset--------------------------------------------------------------
pca(2, X_australian, y_australian, plot=1, dataset='australian')
ica(2, X_australian, y_australian, plot=1, dataset='australian')
rp(X_australian, y_australian, 2, plot=1, dataset='australian')
rfe(X_australian, y_australian, 2, plot=1, dataset='australian')
#-----------------------------------------------------------------------------------------

#-------End of Part 2---------------------------------------------------------------------

#------Part 3-----------------------------------------------------------------------------

#--------German dataset-------------------------------------------------------------------
X_new_german_pca = pca(2, X_german, y_german, plot=0, dataset='german')
km = {}
em1 = {}
for i in range(2,21):
    km[i] = kmeans(i, X_new_german_pca, y_german, plot=0)
    em1[i] = em(i, X_new_german_pca, y_german, plot=0)

maxk_val = -1
maxk_key = -1
maxem_key = -1
maxem_val = -1

for key in km:
    if maxk_val < km[key]:
        maxk_key = key
        maxk_val = km[key]

for key in em1:
    if maxem_val < em1[key]:
        maxem_key = key
        maxem_val = em1[key]

kmeans(maxk_key, X_new_german_pca, y_german, comp1=0, comp2=1, dataset='german', action='pca')
em(maxem_key, X_new_german_pca, y_german, comp1=0, comp2=1, dataset='german', action='pca')


X_new_german_ica = ica(2, X_german, y_german, plot=0, dataset='german')
km = {}
em1 = {}
for i in range(2,21):
    km[i] = kmeans(i, X_new_german_ica, y_german, plot=0)
    em1[i] = em(i, X_new_german_ica, y_german, plot=0)

maxk_val = -1
maxk_key = -1
maxem_key = -1
maxem_val = -1

for key in km:
    if maxk_val < km[key]:
        maxk_key = key
        maxk_val = km[key]

for key in em1:
    if maxem_val < em1[key]:
        maxem_key = key
        maxem_val = em1[key]

kmeans(maxk_key, X_new_german_ica, y_german, comp1=0, comp2=1, dataset='german', action='ica')
em(maxem_key, X_new_german_ica, y_german, comp1=0, comp2=1, dataset='german', action='ica')


X_new_german_rp = rp(X_german, y_german, 2, plot=0, dataset='german')
km = {}
em1 = {}
for i in range(2,21):
    km[i] = kmeans(i, X_new_german_rp, y_german, plot=0)
    em1[i] = em(i, X_new_german_rp, y_german, plot=0)

maxk_val = -1
maxk_key = -1
maxem_key = -1
maxem_val = -1

for key in km:
    if maxk_val < km[key]:
        maxk_key = key
        maxk_val = km[key]

for key in em1:
    if maxem_val < em1[key]:
        maxem_key = key
        maxem_val = em1[key]

kmeans(maxk_key, X_new_german_rp, y_german, comp1=0, comp2=1, dataset='german', action='rp')
em(maxem_key, X_new_german_rp, y_german, comp1=0, comp2=1, dataset='german', action='rp')


X_new_german_rfe = rfe(X_german, y_german, 2, plot=0, dataset='german')
km = {}
em1 = {}
for i in range(2,21):
    km[i] = kmeans(i, X_new_german_rfe, y_german, plot=0)
    em1[i] = em(i, X_new_german_rfe, y_german, plot=0)

maxk_val = -1
maxk_key = -1
maxem_key = -1
maxem_val = -1

for key in km:
    if maxk_val < km[key]:
        maxk_key = key
        maxk_val = km[key]

for key in em1:
    if maxem_val < em1[key]:
        maxem_key = key
        maxem_val = em1[key]

kmeans(maxk_key, X_new_german_rfe, y_german, comp1=0, comp2=1, dataset='german', action='rfe')
em(maxem_key, X_new_german_rfe, y_german, comp1=0, comp2=1, dataset='german', action='rfe')
#--------------------------------------------------------------------------------------------------

#--------------Australian dataset------------------------------------------------------------------
X_new_australian_pca = pca(2, X_australian, y_australian, plot=0, dataset='australian')
km = {}
em1 = {}
for i in range(2,21):
    km[i] = kmeans(i, X_new_australian_pca, y_australian, plot=0)
    em1[i] = em(i, X_new_australian_pca, y_australian, plot=0)

maxk_val = -1
maxk_key = -1
maxem_key = -1
maxem_val = -1

for key in km:
    if maxk_val < km[key]:
        maxk_key = key
        maxk_val = km[key]

for key in em1:
    if maxem_val < em1[key]:
        maxem_key = key
        maxem_val = em1[key]

kmeans(maxk_key, X_new_australian_pca, y_australian, comp1=0, comp2=1, dataset='australian', action='pca')
em(maxem_key, X_new_australian_pca, y_australian, comp1=0, comp2=1, dataset='australian', action='pca')


X_new_australian_ica = ica(2, X_australian, y_australian, plot=0, dataset='australian')
km = {}
em1 = {}
for i in range(2,21):
    km[i] = kmeans(i, X_new_australian_ica, y_australian, plot=0)
    em1[i] = em(i, X_new_australian_ica, y_australian, plot=0)

maxk_val = -1
maxk_key = -1
maxem_key = -1
maxem_val = -1

for key in km:
    if maxk_val < km[key]:
        maxk_key = key
        maxk_val = km[key]

for key in em1:
    if maxem_val < em1[key]:
        maxem_key = key
        maxem_val = em1[key]

kmeans(maxk_key, X_new_australian_ica, y_australian, comp1=0, comp2=1, dataset='australian', action='ica')
em(maxem_key, X_new_australian_ica, y_australian, comp1=0, comp2=1, dataset='australian', action='ica')


X_new_australian_rp = rp(X_australian, y_australian, 2, plot=0, dataset='australian')
km = {}
em1 = {}
for i in range(2,21):
    km[i] = kmeans(i, X_new_australian_rp, y_australian, plot=0)
    em1[i] = em(i, X_new_australian_rp, y_australian, plot=0)

maxk_val = -1
maxk_key = -1
maxem_key = -1
maxem_val = -1

for key in km:
    if maxk_val < km[key]:
        maxk_key = key
        maxk_val = km[key]

for key in em1:
    if maxem_val < em1[key]:
        maxem_key = key
        maxem_val = em1[key]

kmeans(maxk_key, X_new_australian_rp, y_australian, comp1=0, comp2=1, dataset='australian', action='rp')
em(maxem_key, X_new_australian_rp, y_australian, comp1=0, comp2=1, dataset='australian', action='rp')


X_new_australian_rfe = rfe(X_australian, y_australian, 2, plot=0, dataset='australian')
km = {}
em1 = {}
for i in range(2,21):
    km[i] = kmeans(i, X_new_australian_rfe, y_australian, plot=0)
    em1[i] = em(i, X_new_australian_rfe, y_australian, plot=0)

maxk_val = -1
maxk_key = -1
maxem_key = -1
maxem_val = -1

for key in km:
    if maxk_val < km[key]:
        maxk_key = key
        maxk_val = km[key]

for key in em1:
    if maxem_val < em1[key]:
        maxem_key = key
        maxem_val = em1[key]

kmeans(maxk_key, X_new_australian_rfe, y_australian, comp1=0, comp2=1, dataset='australian', action='rfe')
em(maxem_key, X_new_australian_rfe, y_australian, comp1=0, comp2=1, dataset='australian', action='rfe')
#--------------------------------------------------------------------------------------------------

#------------End of Part 3-------------------------------------------------------------------------

#-----------Part 4---------------------------------------------------------------------------------

#--------Australian dataset----------------------------------------------------------------------------
c = []
acc = defaultdict(list)
for components in range(1,15):
    c.append(components)
    X_australian_after_pca = pca(components, X_australian, y_australian, plot=0)
    X_australian_after_pca_train, X_australian_after_pca_test, y_australian_train, y_australian_test = train_test_split(X_australian_after_pca, y_australian, test_size=0.33)
    acc_temp, time_temp = nn(X_australian_after_pca_train, y_australian_train, X_australian_after_pca_test, y_australian_test, accuracy_score, print_info=0, dataset=0, n_splits=5, random_state=None)
    acc['PCA'].append(acc_temp)

    X_australian_after_ica = ica(components, X_australian, y_australian, plot=0)
    X_australian_after_ica_train, X_australian_after_ica_test, y_australian_train, y_australian_test = train_test_split(X_australian_after_ica, y_australian, test_size=0.33)
    acc_temp, time_temp = nn(X_australian_after_ica_train, y_australian_train, X_australian_after_ica_test, y_australian_test, accuracy_score, print_info=0, dataset=0, n_splits=5, random_state=None)
    acc['ICA'].append(acc_temp)

    X_australian_after_rp = rp(n_components=components, X=X_australian, y=y_australian, plot=0)
    X_australian_after_rp_train, X_australian_after_rp_test, y_australian_train, y_australian_test = train_test_split(X_australian_after_rp, y_australian, test_size=0.33)
    acc_temp, time_temp = nn(X_australian_after_rp_train, y_australian_train, X_australian_after_rp_test, y_australian_test, accuracy_score, print_info=0, dataset=0, n_splits=5, random_state=None)
    acc['Randomized Projection'].append(acc_temp)

    X_australian_after_rfe = rfe(n_features_to_select=components, X=X_australian, y=y_australian, plot=0)
    X_australian_after_rfe_train, X_australian_after_rfe_test, y_australian_train, y_australian_test = train_test_split(X_australian_after_rfe, y_australian, test_size=0.33)
    acc_temp, time_temp = nn(X_australian_after_rfe_train, y_australian_train, X_australian_after_rfe_test, y_australian_test, accuracy_score, print_info=0, dataset=0, n_splits=5, random_state=None)
    acc['RFE'].append(acc_temp)

for algorithm in acc:
    plt.plot(c, acc[algorithm])
    plt.title("Australian Dataset - Number of components vs Accuracy for {}".format(algorithm))
    plt.xlabel("Number of components")
    plt.ylabel("Accuracy for Neural Network")
    plt.savefig("NN-accuracy-after-{}.png".format(algorithm))
    plt.close()

#--------------------------------------------------------------------------------------------------

#----------End of Part 4---------------------------------------------------------------------------

#----------Part 5----------------------------------------------------------------------------------

#---------Australian dataset-----------------------------------------------------------------------
X_australian_after_pca = pca(5, X_australian, y_australian, plot=0)
X_australian_after_ica = ica(14, X_australian, y_australian, plot=0)
X_australian_after_rp = rp(n_components=11, X=X_australian, y=y_australian, plot=0)
X_australian_after_rfe = rfe(n_features_to_select=4, X=X_australian, y=y_australian, plot=0)

km_dict = defaultdict(list)
em_dict = defaultdict(list)
km_em_dict = defaultdict(list)
clusters = []
for i in range(2,21):
    clusters.append(i)
    X_australian_after_pca_train, X_australian_after_pca_test, y_australian_train, y_australian_test = train_test_split(X_australian_after_pca, y_australian, test_size=0.33)
    y_new_train = kmeans(i, X_australian_after_pca_train, y_australian_train, plot=0, part=5)
    y_train_km_pca = y_new_train
    y_new_test = kmeans(i, X_australian_after_pca_test, y_australian_test, plot=0, part=5)
    y_test_km_pca = y_new_test
    acc_temp, time_temp = nn(y_new_train.reshape(-1,1), y_australian_train, y_new_test.reshape(-1,1), y_australian_test, accuracy_score, print_info=0, dataset=0, n_splits=5, random_state=None)
    km_dict['PCA'].append(acc_temp)

    y_new_train = em(n_components=i, X=X_australian_after_pca_train, y=y_australian_train, plot=0, part=5)
    y_train_em_pca = y_new_train
    y_new_test = em(n_components=i, X=X_australian_after_pca_test, y=y_australian_test, plot=0, part=5)
    y_test_em_pca = y_new_test
    acc_temp, time_temp = nn(y_new_train.reshape(-1,1), y_australian_train, y_new_test.reshape(-1,1), y_australian_test, accuracy_score, print_info=0, dataset=0, n_splits=5, random_state=None)
    em_dict['PCA'].append(acc_temp)

    y_train_km_em_pca = np.append(y_train_km_pca.reshape(-1,1), y_train_em_pca.reshape(-1,1), axis=1)
    y_test_km_em_pca = np.append(y_test_km_pca.reshape(-1,1), y_test_em_pca.reshape(-1,1), axis=1)
    acc_temp, time_temp = nn(y_train_km_em_pca, y_australian_train, y_test_km_em_pca, y_australian_test, accuracy_score, print_info=0, dataset=0, n_splits=5, random_state=None)
    km_em_dict['PCA'].append(acc_temp)

    X_australian_after_ica_train, X_australian_after_ica_test, y_australian_train, y_australian_test = train_test_split(
        X_australian_after_ica, y_australian, test_size=0.33)
    y_new_train = kmeans(i, X_australian_after_ica_train, y_australian_train, plot=0, part=5)
    y_train_km_ica = y_new_train
    y_new_test = kmeans(i, X_australian_after_ica_test, y_australian_test, plot=0, part=5)
    y_test_km_ica = y_new_test
    acc_temp, time_temp = nn(y_new_train.reshape(-1, 1), y_australian_train, y_new_test.reshape(-1, 1),
                             y_australian_test, accuracy_score, print_info=0, dataset=0, n_splits=5, random_state=None)
    km_dict['ICA'].append(acc_temp)

    y_new_train = em(n_components=i, X=X_australian_after_ica_train, y=y_australian_train, plot=0, part=5)
    y_train_em_ica = y_new_train
    y_new_test = em(n_components=i, X=X_australian_after_ica_test, y=y_australian_test, plot=0, part=5)
    y_test_em_ica = y_new_test
    acc_temp, time_temp = nn(y_new_train.reshape(-1, 1), y_australian_train, y_new_test.reshape(-1, 1),
                             y_australian_test, accuracy_score, print_info=0, dataset=0, n_splits=5, random_state=None)
    em_dict['ICA'].append(acc_temp)

    y_train_km_em_ica = np.append(y_train_km_ica.reshape(-1, 1), y_train_em_ica.reshape(-1, 1), axis=1)
    y_test_km_em_ica = np.append(y_test_km_ica.reshape(-1, 1), y_test_em_ica.reshape(-1, 1), axis=1)
    acc_temp, time_temp = nn(y_train_km_em_ica, y_australian_train, y_test_km_em_ica, y_australian_test, accuracy_score,
                             print_info=0, dataset=0, n_splits=5, random_state=None)
    km_em_dict['ICA'].append(acc_temp)

    X_australian_after_rp_train, X_australian_after_rp_test, y_australian_train, y_australian_test = train_test_split(
        X_australian_after_rp, y_australian, test_size=0.33)
    y_new_train = kmeans(i, X_australian_after_rp_train, y_australian_train, plot=0, part=5)
    y_train_km_rp = y_new_train
    y_new_test = kmeans(i, X_australian_after_rp_test, y_australian_test, plot=0, part=5)
    y_test_km_rp = y_new_test
    acc_temp, time_temp = nn(y_new_train.reshape(-1, 1), y_australian_train, y_new_test.reshape(-1, 1),
                             y_australian_test, accuracy_score, print_info=0, dataset=0, n_splits=5, random_state=None)
    km_dict['Randomized Projection'].append(acc_temp)

    y_new_train = em(n_components=i, X=X_australian_after_rp_train, y=y_australian_train, plot=0, part=5)
    y_train_em_rp = y_new_train
    y_new_test = em(n_components=i, X=X_australian_after_rp_test, y=y_australian_test, plot=0, part=5)
    y_test_em_rp = y_new_test
    acc_temp, time_temp = nn(y_new_train.reshape(-1, 1), y_australian_train, y_new_test.reshape(-1, 1),
                             y_australian_test, accuracy_score, print_info=0, dataset=0, n_splits=5, random_state=None)
    em_dict['Randomized Projection'].append(acc_temp)

    y_train_km_em_rp = np.append(y_train_km_rp.reshape(-1, 1), y_train_em_rp.reshape(-1, 1), axis=1)
    y_test_km_em_rp = np.append(y_test_km_rp.reshape(-1, 1), y_test_em_rp.reshape(-1, 1), axis=1)
    acc_temp, time_temp = nn(y_train_km_em_rp, y_australian_train, y_test_km_em_rp, y_australian_test, accuracy_score,
                             print_info=0, dataset=0, n_splits=5, random_state=None)
    km_em_dict['Randomized Projection'].append(acc_temp)

    X_australian_after_rfe_train, X_australian_after_rfe_test, y_australian_train, y_australian_test = train_test_split(
        X_australian_after_rfe, y_australian, test_size=0.33)
    y_new_train = kmeans(i, X_australian_after_rfe_train, y_australian_train, plot=0, part=5)
    y_train_km_rfe = y_new_train
    y_new_test = kmeans(i, X_australian_after_rfe_test, y_australian_test, plot=0, part=5)
    y_test_km_rfe = y_new_test
    acc_temp, time_temp = nn(y_new_train.reshape(-1, 1), y_australian_train, y_new_test.reshape(-1, 1),
                             y_australian_test, accuracy_score, print_info=0, dataset=0, n_splits=5, random_state=None)
    km_dict['RFE'].append(acc_temp)

    y_new_train = em(n_components=i, X=X_australian_after_rfe_train, y=y_australian_train, plot=0, part=5)
    y_train_em_rfe = y_new_train
    y_new_test = em(n_components=i, X=X_australian_after_rfe_test, y=y_australian_test, plot=0, part=5)
    y_test_em_rfe = y_new_test
    acc_temp, time_temp = nn(y_new_train.reshape(-1, 1), y_australian_train, y_new_test.reshape(-1, 1),
                             y_australian_test, accuracy_score, print_info=0, dataset=0, n_splits=5, random_state=None)
    em_dict['RFE'].append(acc_temp)

    y_train_km_em_rfe = np.append(y_train_km_rfe.reshape(-1, 1), y_train_em_rfe.reshape(-1, 1), axis=1)
    y_test_km_em_rfe = np.append(y_test_km_rfe.reshape(-1, 1), y_test_em_rfe.reshape(-1, 1), axis=1)
    acc_temp, time_temp = nn(y_train_km_em_rfe, y_australian_train, y_test_km_em_rfe, y_australian_test, accuracy_score,
                             print_info=0, dataset=0, n_splits=5, random_state=None)
    km_em_dict['RFE'].append(acc_temp)

for algorithm in km_dict:
    plt.plot(clusters, km_dict[algorithm])
    plt.title("NN accuracy using KMeans clusters as features with {}".format(algorithm))
    plt.xlabel("Number of Clusters")
    plt.ylabel("NN accuracy")
    plt.savefig("NN-accuarcy-kmeans-clusters-{}.png".format(algorithm))
    plt.close()

for algorithm in em_dict:
    plt.plot(clusters, em_dict[algorithm])
    plt.title("NN accuracy using EM clusters as features with {}".format(algorithm))
    plt.xlabel("Number of Clusters")
    plt.ylabel("NN accuracy")
    plt.savefig("NN-accuarcy-EM-clusters-{}.png".format(algorithm))
    plt.close()

for algorithm in km_em_dict:
    plt.plot(clusters, km_em_dict[algorithm])
    plt.title("NN accuracy using KMeans and EM clusters as features with {}".format(algorithm))
    plt.xlabel("Number of Clusters")
    plt.ylabel("NN accuracy")
    plt.savefig("NN-accuarcy-KMeans-EM-clusters-{}.png".format(algorithm))
    plt.close()

#--------------------------------------------------------------------------------------------------

#--------End of Part 5-----------------------------------------------------------------------------