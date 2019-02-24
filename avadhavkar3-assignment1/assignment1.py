#-------Imports---------------------------------------
import pandas as pd
import numpy as np
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, make_scorer, recall_score, f1_score
import matplotlib.pyplot as plt
#-------------------------------------------------------

#--------Data-------------------------------------------
print()
print("Data")
print("--------------------------------------------------------------")
df_german = pd.read_csv("german-credit-data.csv", header=None)
df1 = df_german.iloc[:,1:]
X_train_german, X_test_german, y_train_german, y_test_german = train_test_split(df1.iloc[:,0:-1], df1.iloc[:,-1], test_size=0.33, random_state=0)
y_train_german[y_train_german == 1] = 0
y_train_german[y_train_german == 2] = 1
y_test_german[y_test_german == 1] = 0
y_test_german[y_test_german == 2] = 1
print("German-Credit-Data : No. of positive samples = {}, No. of negative samples = {}".format(df1.loc[df1.iloc[:,-1] == 2,:].shape[0], df1.loc[df1.iloc[:,-1] == 1,:].shape[0]))

df_australian = pd.read_csv("australian-credit-approval.csv", header=None)
print("Australian-Credit-Approval : No. of positive samples = {}, No. of negative samples = {}".format(df_australian.loc[df_australian.iloc[:,-1] == 1,:].shape[0], df_australian.loc[df_australian.iloc[:,-1] == 0,:].shape[0]))
X_train_australian, X_test_australian, y_train_australian, y_test_australian = train_test_split(df_australian.iloc[:,0:-1], df_australian.iloc[:,-1], test_size=0.33, random_state=0)
#--------------------------------------------------------
print()
#----------Code for splitting training data---------------
def split_data(X, y, size, random_state):
    np.random.seed(random_state)
    number_of_instances = X.shape[0] * size
    x_values_selected = []
    y_values_selected = []
    already_selected = []
    count = 0
    while count < number_of_instances:
        index = np.random.randint(0,X.shape[0])
        if index not in already_selected:
            x_values_selected.append(X.iloc[index,:])
            y_values_selected.append(y.iloc[index])
            count = count + 1

    return pd.DataFrame(x_values_selected), pd.Series(y_values_selected)
#---------------------------------------------------------------------------

# ------ KNN -----------------------------------------
def KNN(X_train, y_train, X_test, y_test,metric, n_splits, random_state, num_features, print_info, dataset):
    neighbours = [i for i in range(3,37)]
    weights = ['uniform', 'distance']
    algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']
    param_grid_knn = {'n_neighbors': neighbours, 'weights': weights, 'algorithm': algorithm}
    scorer_knn = make_scorer(metric)
    skf_knn = StratifiedKFold(n_splits=n_splits, random_state=random_state)
    knn = KNeighborsClassifier()
    grid_search_model_knn = GridSearchCV(estimator=knn, param_grid=param_grid_knn, scoring=scorer_knn, cv=skf_knn)
    start_time = time.time()
    grid_search_model_knn.fit(X_train, y_train)
    end_time = time.time()
    y_pred_knn = grid_search_model_knn.predict(X_test)
    tn_knn, fp_knn, fn_knn, tp_knn = confusion_matrix(y_test, y_pred_knn).ravel()
    if print_info:
        print("KNN - Testing Data")
        print("Best paramaters = {}".format(grid_search_model_knn.best_params_))
        print("True negative = {}, False Positive = {}, False Negative = {}, True Positive = {}".format(tn_knn, fp_knn, fn_knn, tp_knn))
        print("Accuracy score = {}".format(accuracy_score(y_test, y_pred_knn)))
        print("Recall score = {}".format(recall_score(y_test, y_pred_knn)))
        print("AUC ROC score = {}".format(roc_auc_score(y_test, y_pred_knn)))
        print("F1 score = {}".format(f1_score(y_test, y_pred_knn)))
        print()
    if dataset == 1:
        return (1-recall_score(y_test, y_pred_knn)), end_time-start_time
    else:
        return (1-accuracy_score(y_test, y_pred_knn)), end_time-start_time
# ------------------------------------------------------

#------- Decision Tree----------------------------------
def dt(X_train, y_train, X_test, y_test,metric, n_splits, random_state, num_features, print_info, dataset):
    min_samples_leaf = [i for i in range(3,200)]
    max_depth = [i for i in range(5,100,5)]
    param_grid_dtl = {'min_samples_leaf': min_samples_leaf, 'max_depth': max_depth}
    scorer_dtl = make_scorer(metric)
    skf_dtl = StratifiedKFold(n_splits=n_splits, random_state=random_state)
    dtl = DecisionTreeClassifier()
    selector = SelectKBest(k=num_features)
    selector.fit(X_train, y_train)
    X_train = selector.transform(X_train)
    X_test = selector.transform(X_test)
    grid_search_model_dtl = GridSearchCV(estimator=dtl, param_grid=param_grid_dtl, scoring=scorer_dtl, cv=skf_dtl)
    start_time = time.time()
    grid_search_model_dtl.fit(X_train, y_train)
    end_time = time.time()
    y_pred_dtl = grid_search_model_dtl.predict(X_test)
    tn_dtl, fp_dtl, fn_dtl, tp_dtl = confusion_matrix(y_test, y_pred_dtl).ravel()
    if print_info:
        print("Decision Tree - Testing Data")
        print("Best paramaters = {}".format(grid_search_model_dtl.best_params_))
        print("True negative = {}, False Positive = {}, False Negative = {}, True Positive = {}".format(tn_dtl, fp_dtl, fn_dtl, tp_dtl))
        print("Accuracy score = {}".format(accuracy_score(y_test, y_pred_dtl)))
        print("Recall score = {}".format(recall_score(y_test, y_pred_dtl)))
        print("AUC ROC score = {}".format(roc_auc_score(y_test, y_pred_dtl)))
        print("F1 score = {}".format(f1_score(y_test, y_pred_dtl)))
        print()
    if dataset == 1:
        return (1-recall_score(y_test, y_pred_dtl)), end_time-start_time
    else:
        return (1-accuracy_score(y_test, y_pred_dtl)), end_time-start_time
#-------------------------------------------------------

#------ SVM---------------------------------------------
def svc(X_train, y_train, X_test, y_test,metric, n_splits, random_state, kernels, num_features, print_info, dataset):
    gamma = [0.1, 1.0, 10]
    param_grid_svm = {'kernel': kernels, 'gamma': gamma}
    scorer_svm = make_scorer(metric)
    skf_svm = StratifiedKFold(n_splits=n_splits, random_state=random_state)
    svm = SVC()
    grid_search_model_svm = GridSearchCV(estimator=svm, param_grid=param_grid_svm, scoring=scorer_svm, cv=skf_svm)
    start_time = time.time()
    grid_search_model_svm.fit(X_train, y_train)
    end_time = time.time()
    y_pred_svm = grid_search_model_svm.predict(X_test)
    tn_svm, fp_svm, fn_svm, tp_svm = confusion_matrix(y_test, y_pred_svm).ravel()
    if print_info:
        print("SVM - Testing Data")
        print("Best paramaters = {}".format(grid_search_model_svm.best_params_))
        print("True negative = {}, False Positive = {}, False Negative = {}, True Positive = {}".format(tn_svm, fp_svm, fn_svm, tp_svm))
        print("Accuracy score = {}".format(accuracy_score(y_test, y_pred_svm)))
        print("Recall score = {}".format(recall_score(y_test, y_pred_svm)))
        print("AUC ROC score = {}".format(roc_auc_score(y_test, y_pred_svm)))
        print("F1 score = {}".format(f1_score(y_test, y_pred_svm)))
        print()
    if dataset == 1:
        return (1-recall_score(y_test, y_pred_svm)), end_time-start_time
    else:
        return (1-accuracy_score(y_test, y_pred_svm)), end_time-start_time
#---------------------------------------------------------

#----- AdaBoostClassifier---------------------------------
def adaBoost(X_train, y_train, X_test, y_test,metric, n_splits, random_state, num_features, print_info, dataset):
    n_estimators = [i for i in range(90,300,10)]
    learning_rate = [0.5, 1.0, 1.5, 2.0]
    min_samples_leaf = [i for i in range(1,50)]
    max_depth = [i for i in range(5,55,10)]
    param_grid_ada = {'n_estimators': n_estimators,'base_estimator__max_depth': max_depth}
    scorer_ada = make_scorer(metric)
    skf_ada = StratifiedKFold(n_splits=n_splits, random_state=random_state)
    ada_dtl = DecisionTreeClassifier()
    ada_model = AdaBoostClassifier(base_estimator=ada_dtl)
    selector = SelectKBest(k=num_features)
    selector.fit(X_train, y_train)
    X_train = selector.transform(X_train)
    X_test = selector.transform(X_test)
    grid_search_model_ada = GridSearchCV(estimator=ada_model, param_grid=param_grid_ada, scoring=scorer_ada, cv=skf_ada)
    start_time = time.time()
    grid_search_model_ada.fit(X_train, y_train)
    end_time = time.time()
    y_pred_ada = grid_search_model_ada.predict(X_test)
    tn_ada, fp_ada, fn_ada, tp_ada = confusion_matrix(y_test, y_pred_ada).ravel()
    if print_info:
        print("AdaBoost - Testing Data")
        print("Best paramaters = {}".format(grid_search_model_ada.best_params_))
        print("True negative = {}, False Positive = {}, False Negative = {}, True Positive = {}".format(tn_ada, fp_ada, fn_ada, tp_ada))
        print("Accuracy score = {}".format(accuracy_score(y_test, y_pred_ada)))
        print("Recall score = {}".format(recall_score(y_test, y_pred_ada)))
        print("AUC ROC score = {}".format(roc_auc_score(y_test, y_pred_ada)))
        print("F1 score = {}".format(f1_score(y_test, y_pred_ada)))
        print()
    if dataset == 1:
        return (1-recall_score(y_test, y_pred_ada)), end_time-start_time
    else:
        return (1-accuracy_score(y_test, y_pred_ada)), end_time-start_time
#--------------------------------------------------------------

#----Neural Network--------------------------------------------
def nn(X_train, y_train, X_test, y_test,metric, n_splits, random_state, num_features, print_info, dataset):

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
        return (1-recall_score(y_test, y_pred_nn)), end_time-start_time
    else:
        return (1-accuracy_score(y_test, y_pred_nn)), end_time-start_time
#-----------------------------------------------------------------

#----------Main Code----------------------------------------------
print("German-Credit-Data")
print("------------------------------------------------------------")
training_sizes = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
algorithms = ['knn', 'dtl', 'svm', 'ada', 'nn']
time_taken = {}
for algorithm in algorithms:
    error_rate_training_data = []
    error_rate_testing_data = []
    time_taken_temp = []
    if algorithm == 'knn':
        for training_size in training_sizes:
            X_train_temp, y_train_temp = split_data(X_train_german, y_train_german, training_size, 6)
            if training_size == 1.0:
                train_error, temp_time1 = KNN(X_train_temp, y_train_temp, X_train_temp, y_train_temp, recall_score, 5, 1, 15, 0, 1)
                error_rate_training_data.append(train_error)
                test_error, temp_time2 = KNN(X_train_temp, y_train_temp, X_test_german, y_test_german, recall_score, 5, 1, 15, 1, 1)
                error_rate_testing_data.append(test_error)
                time_taken_temp.append(temp_time1)
            else:
                train_error, temp_time1 = KNN(X_train_temp, y_train_temp, X_train_temp, y_train_temp, recall_score, 5, 1, 15, 0, 1)
                error_rate_training_data.append(train_error)
                test_error, temp_time2 = KNN(X_train_temp, y_train_temp, X_test_german, y_test_german, recall_score, 5, 1, 15, 0, 1)
                error_rate_testing_data.append(test_error)
                time_taken_temp.append(temp_time1)

        time_taken['KNN'] = time_taken_temp
        plt.plot(training_sizes, error_rate_training_data, label='Training Error')
        plt.plot(training_sizes, error_rate_testing_data, label='Testing Error')
        plt.legend()
        plt.xlabel("Training Sizes as a fraction of the original training data")
        plt.ylabel("1 - recall score")
        plt.title("Training sizes vs (1 - recall score) for KNN")
        plt.savefig("German-Credit-Data-KNN.png")
        plt.close()

    elif algorithm == 'dtl':
        for training_size in training_sizes:
            X_train_temp, y_train_temp = split_data(X_train_german, y_train_german, training_size, 6)
            if training_size == 1.0:
                train_error, temp_time1 = dt(X_train_temp, y_train_temp, X_train_temp, y_train_temp, recall_score, 5, 0, 15, 0, 1)
                error_rate_training_data.append(train_error)
                test_error, temp_time2 = dt(X_train_temp, y_train_temp, X_test_german, y_test_german, recall_score, 5, 0, 15, 1, 1)
                error_rate_testing_data.append(test_error)
                time_taken_temp.append(temp_time1)
            else:
                train_error, temp_time1 = dt(X_train_temp, y_train_temp, X_train_temp, y_train_temp, recall_score, 5, 0, 15, 0, 1)
                error_rate_training_data.append(train_error)
                test_error, temp_time2 = dt(X_train_temp, y_train_temp, X_test_german, y_test_german, recall_score, 5, 0, 15, 0, 1)
                error_rate_testing_data.append(test_error)
                time_taken_temp.append(temp_time1)

        time_taken['Decision Tree'] = time_taken_temp
        plt.plot(training_sizes, error_rate_training_data, label='Training Error')
        plt.plot(training_sizes, error_rate_testing_data, label='Testing Error')
        plt.legend()
        plt.xlabel("Training Sizes as a fraction of the original training data")
        plt.ylabel("1 - recall score")
        plt.title("Training sizes vs (1 - recall score) for Decision Tree")
        plt.savefig("German-Credit-Data-DT.png")
        plt.close()

    elif algorithm == 'svm':
        for training_size in training_sizes:
            X_train_temp, y_train_temp = split_data(X_train_german, y_train_german, training_size, 6)
            if training_size == 1.0:
                train_error, temp_time1 = svc(X_train_temp, y_train_temp, X_train_temp, y_train_temp, recall_score, 5, 1, ['linear', 'poly', 'rbf', 'sigmoid'], 15, 0, 1)
                error_rate_training_data.append(train_error)
                test_error, temp_time2 = svc(X_train_temp, y_train_temp, X_test_german, y_test_german, recall_score, 5, 1, ['linear', 'poly', 'rbf', 'sigmoid'], 15, 1, 1)
                error_rate_testing_data.append(test_error)
                time_taken_temp.append(temp_time1)
            else:
                train_error, temp_time1 = svc(X_train_temp, y_train_temp, X_train_temp, y_train_temp, recall_score, 5, 1, ['linear', 'poly', 'rbf', 'sigmoid'], 15, 0, 1)
                error_rate_training_data.append(train_error)
                test_error, temp_time2 = svc(X_train_temp, y_train_temp, X_test_german, y_test_german, recall_score, 5, 1, ['linear', 'poly', 'rbf', 'sigmoid'], 15,0, 1)
                error_rate_testing_data.append(test_error)
                time_taken_temp.append(temp_time1)

        time_taken['SVM'] = time_taken_temp
        plt.plot(training_sizes, error_rate_training_data, label='Training Error')
        plt.plot(training_sizes, error_rate_testing_data, label='Testing Error')
        plt.legend()
        plt.xlabel("Training Sizes as a fraction of the original training data")
        plt.ylabel("1 - recall score")
        plt.title("Training sizes vs (1 - recall score) for SVM")
        plt.savefig("German-Credit-Data-SVM.png")
        plt.close()

    elif algorithm == 'ada':
        for training_size in training_sizes:
            X_train_temp, y_train_temp = split_data(X_train_german, y_train_german, training_size, 6)
            if training_size == 1.0:
                train_error, temp_time1 = adaBoost(X_train_temp, y_train_temp, X_train_temp, y_train_temp, recall_score, 5, 0, 15, 0, 1)
                error_rate_training_data.append(train_error)
                test_error, temp_time2 = adaBoost(X_train_temp, y_train_temp, X_test_german, y_test_german, recall_score, 5, 0, 15, 1, 1)
                error_rate_testing_data.append(test_error)
                time_taken_temp.append(temp_time1)
            else:
                train_error, temp_time1 = adaBoost(X_train_temp, y_train_temp, X_train_temp, y_train_temp, recall_score, 5, 0, 15, 0, 1)
                error_rate_training_data.append(train_error)
                test_error, temp_time2 = adaBoost(X_train_temp, y_train_temp, X_test_german, y_test_german, recall_score, 5, 0, 15, 0, 1)
                error_rate_testing_data.append(test_error)
                time_taken_temp.append(temp_time1)

        time_taken['Ada Boost'] = time_taken_temp
        plt.plot(training_sizes, error_rate_training_data, label='Training Error')
        plt.plot(training_sizes, error_rate_testing_data, label='Testing Error')
        plt.legend()
        plt.xlabel("Training Sizes as a fraction of the original training data")
        plt.ylabel("1 - recall score")
        plt.title("Training sizes vs (1 - recall score) for Ada Boost")
        plt.savefig("German-Credit-Data-ADA.png")
        plt.close()

    elif algorithm == 'nn':
        for training_size in training_sizes:
            X_train_temp, y_train_temp = split_data(X_train_german, y_train_german, training_size, 6)
            if training_size == 1.0:
                train_error, temp_time1 = nn(X_train_temp, y_train_temp, X_train_temp, y_train_temp, recall_score, 5, 0, 15, 0, 1)
                error_rate_training_data.append(train_error)
                test_error, temp_time2 = nn(X_train_temp, y_train_temp, X_test_german, y_test_german, recall_score, 5, 0, 15, 1, 1)
                error_rate_testing_data.append(test_error)
                time_taken_temp.append(temp_time1)
            else:
                train_error, temp_time1 = nn(X_train_temp, y_train_temp, X_train_temp, y_train_temp, recall_score, 5, 0, 15, 0, 1)
                error_rate_training_data.append(train_error)
                test_error, temp_time2 = nn(X_train_temp, y_train_temp, X_test_german, y_test_german, recall_score, 5, 0, 15, 0, 1)
                error_rate_testing_data.append(test_error)
                time_taken_temp.append(temp_time1)

        time_taken['Neural Network'] = time_taken_temp
        plt.plot(training_sizes, error_rate_training_data, label='Training Error')
        plt.plot(training_sizes, error_rate_testing_data, label='Testing Error')
        plt.legend()
        plt.xlabel("Training Sizes as a fraction of the original training data")
        plt.ylabel("1 - recall score")
        plt.title("Training sizes vs (1 - recall score) for Neural Network")
        plt.savefig("German-Credit-Data-NN.png")
        plt.close()

for algorithm in time_taken:
    plt.plot(training_sizes, time_taken[algorithm], label=algorithm)
plt.xlabel("Training Sizes as a fraction of the original training data")
plt.ylabel("Training time (seconds)")
plt.legend()
plt.title("Training sizes vs Time taken for training for German Credit Data")
plt.savefig("German-Credit-Data-Time.png")
plt.close()

print("------------------------------------------------------------")
print("------------------------------------------------------------")
print()
print("Australian-Credit-Approval")
print("-------------------------------------------------------------")
time_taken1 = {}
for algorithm in algorithms:
    error_rate_training_data = []
    error_rate_testing_data = []
    time_taken_temp = []
    if algorithm == 'knn':
        for training_size in training_sizes:
            X_train_temp, y_train_temp = split_data(X_train_australian, y_train_australian, training_size, 6)
            if training_size == 1.0:
                train_error, temp_time1 = KNN(X_train_temp, y_train_temp, X_train_temp, y_train_temp, accuracy_score, 5, 0, 14, 0, 2)
                error_rate_training_data.append(train_error)
                test_error, temp_time2 = KNN(X_train_temp, y_train_temp, X_test_australian, y_test_australian, accuracy_score, 5, 0, 14, 1, 2)
                error_rate_testing_data.append(test_error)
                time_taken_temp.append(temp_time1)
            else:
                train_error, temp_time1 = KNN(X_train_temp, y_train_temp, X_train_temp, y_train_temp, accuracy_score, 5, 0, 14, 0, 2)
                error_rate_training_data.append(train_error)
                test_error, temp_time2 = KNN(X_train_temp, y_train_temp, X_test_australian, y_test_australian, accuracy_score, 5, 0, 14, 0, 2)
                error_rate_testing_data.append(test_error)
                time_taken_temp.append(temp_time1)

        time_taken1['KNN'] = time_taken_temp
        plt.plot(training_sizes, error_rate_training_data, label='Training Error')
        plt.plot(training_sizes, error_rate_testing_data, label='Testing Error')
        plt.legend()
        plt.xlabel("Training Sizes as a fraction of the original training data")
        plt.ylabel("1 - accuracy score")
        plt.title("Training sizes vs (1 - accuracy score) for KNN")
        plt.savefig("Australian-Credit-Approval-KNN.png")
        plt.close()

    elif algorithm == 'dtl':
        for training_size in training_sizes:
            X_train_temp, y_train_temp = split_data(X_train_australian, y_train_australian, training_size, 6)
            if training_size == 1.0:
                train_error, temp_time1 = dt(X_train_temp, y_train_temp, X_train_temp, y_train_temp, accuracy_score, 5, 0, 14, 0, 2)
                error_rate_training_data.append(train_error)
                test_error, temp_time2 = dt(X_train_temp, y_train_temp, X_test_australian, y_test_australian, accuracy_score, 5, 0, 14, 1, 2)
                error_rate_testing_data.append(test_error)
                time_taken_temp.append(temp_time1)
            else:
                train_error, temp_time1 = dt(X_train_temp, y_train_temp, X_train_temp, y_train_temp, accuracy_score, 5, 0, 14, 0, 2)
                error_rate_training_data.append(train_error)
                test_error, temp_time2 = dt(X_train_temp, y_train_temp, X_test_australian, y_test_australian, accuracy_score, 5, 0, 14, 0, 2)
                error_rate_testing_data.append(test_error)
                time_taken_temp.append(temp_time1)

        time_taken1['Decision Tree'] = time_taken_temp
        plt.plot(training_sizes, error_rate_training_data, label='Training Error')
        plt.plot(training_sizes, error_rate_testing_data, label='Testing Error')
        plt.legend()
        plt.xlabel("Training Sizes as a fraction of the original training data")
        plt.ylabel("1 - accuracy score")
        plt.title("Training sizes vs (1 - accuracy score) for Decision Tree")
        plt.savefig("Australian-Credit-Approval-DT.png")
        plt.close()

    elif algorithm == 'svm':
        for training_size in training_sizes:
            X_train_temp, y_train_temp = split_data(X_train_australian, y_train_australian, training_size, 6)
            if training_size == 1.0:
                train_error, temp_time1 = svc(X_train_temp, y_train_temp, X_train_temp, y_train_temp, accuracy_score, 5, 0, ['linear'], 14, 0, 2)
                error_rate_training_data.append(train_error)
                test_error, temp_time2 = svc(X_train_temp, y_train_temp, X_test_australian, y_test_australian, accuracy_score, 5, 0, ['linear'], 14, 1, 2)
                error_rate_testing_data.append(test_error)
                time_taken_temp.append(temp_time1)
            else:
                train_error, temp_time1 = svc(X_train_temp, y_train_temp, X_train_temp, y_train_temp, accuracy_score, 5, 0, ['linear'], 14, 0, 2)
                error_rate_training_data.append(train_error)
                test_error, temp_time2 = svc(X_train_temp, y_train_temp, X_test_australian, y_test_australian, accuracy_score, 5, 0, ['linear'], 14, 0, 2)
                error_rate_testing_data.append(test_error)
                time_taken_temp.append(temp_time1)

        time_taken1['SVM'] = time_taken_temp
        plt.plot(training_sizes, error_rate_training_data, label='Training Error')
        plt.plot(training_sizes, error_rate_testing_data, label='Testing Error')
        plt.legend()
        plt.xlabel("Training Sizes as a fraction of the original training data")
        plt.ylabel("1 - accuracy score")
        plt.title("Training sizes vs (1 - accuracy score) for SVM")
        plt.savefig("Australian-Credit-Approval-SVM.png")
        plt.close()

    elif algorithm == 'ada':
        for training_size in training_sizes:
            X_train_temp, y_train_temp = split_data(X_train_australian, y_train_australian, training_size, 6)
            if training_size == 1.0:
                train_error, temp_time1 = adaBoost(X_train_temp, y_train_temp, X_train_temp, y_train_temp, accuracy_score, 5, 0, 14, 0, 2)
                error_rate_training_data.append(train_error)
                test_error, temp_time2 = adaBoost(X_train_temp, y_train_temp, X_test_australian, y_test_australian, accuracy_score, 5, 0, 14, 1, 2)
                error_rate_testing_data.append(test_error)
                time_taken_temp.append(temp_time1)
            else:
                train_error, temp_time1 = adaBoost(X_train_temp, y_train_temp, X_train_temp, y_train_temp, accuracy_score, 5, 0, 14, 0, 2)
                error_rate_training_data.append(train_error)
                test_error, temp_time2 = adaBoost(X_train_temp, y_train_temp, X_test_australian, y_test_australian, accuracy_score, 5, 0, 14, 0, 2)
                error_rate_testing_data.append(test_error)
                time_taken_temp.append(temp_time1)

        time_taken1['Ada Boost'] = time_taken_temp
        plt.plot(training_sizes, error_rate_training_data, label='Training Error')
        plt.plot(training_sizes, error_rate_testing_data, label='Testing Error')
        plt.legend()
        plt.xlabel("Training Sizes as a fraction of the original training data")
        plt.ylabel("1 - accuracy score")
        plt.title("Training sizes vs (1 - accuracy score) for Ada Boost")
        plt.savefig("Australian-Credit-Approval-ADA.png")
        plt.close()

    elif algorithm == 'nn':
        for training_size in training_sizes:
            X_train_temp, y_train_temp = split_data(X_train_australian, y_train_australian, training_size, 6)
            if training_size == 1.0:
                train_error, temp_time1 = nn(X_train_temp, y_train_temp, X_train_temp, y_train_temp, accuracy_score, 5, 0, 14, 0, 2)
                error_rate_training_data.append(train_error)
                test_error, temp_time2 = nn(X_train_temp, y_train_temp, X_test_australian, y_test_australian, accuracy_score, 5, 0, 14, 1, 2)
                error_rate_testing_data.append(test_error)
                time_taken_temp.append(temp_time1)
            else:
                train_error, temp_time1 = nn(X_train_temp, y_train_temp, X_train_temp, y_train_temp, accuracy_score, 5, 0, 14, 0, 2)
                error_rate_training_data.append(train_error)
                test_error, temp_time2 = nn(X_train_temp, y_train_temp, X_test_australian, y_test_australian, accuracy_score, 5, 0, 14, 0, 2)
                error_rate_testing_data.append(test_error)
                time_taken_temp.append(temp_time1)

        time_taken1['Neural Network'] = time_taken_temp
        plt.plot(training_sizes, error_rate_training_data, label='Training Error')
        plt.plot(training_sizes, error_rate_testing_data, label='Testing Error')
        plt.legend()
        plt.xlabel("Training Sizes as a fraction of the original training data")
        plt.ylabel("1 - accuracy score")
        plt.title("Training sizes vs (1 - accuracy score) for Neural Network")
        plt.savefig("Australian-Credit-Approval-NN.png")
        plt.close()

for algorithm in time_taken1:
    plt.plot(training_sizes, time_taken[algorithm], label=algorithm)
plt.xlabel("Training Sizes as a fraction of the original training data")
plt.ylabel("Training time (seconds)")
plt.legend()
plt.title("Training sizes vs Time taken for training for Australian Credit Approval")
plt.savefig("Australian-Credit-Approval-Time.png")
plt.close()