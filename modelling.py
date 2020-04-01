import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
from sklearn.metrics import roc_curve
from sklearn.naive_bayes import CategoricalNB, GaussianNB

def classification_metrics(y, y_hat):
    accuracy = round(accuracy_score(y, y_hat), 3)
    auc_score = round(roc_auc_score(y, y_hat), 3)
    precision = round(precision_score(y, y_hat), 3)
    recall = round(recall_score(y, y_hat), 3)
    f1_score = round(2 * (precision * recall) / (precision + recall), 3)
    print("accuracy = {}, auc_score = {}".format(accuracy, auc_score))
    print("")
    print("precision = {}, recall = {}, f1-score = {}".format(precision, recall, f1_score))
    return None

def plot_roc_curve(fprs, tprs, model_name):
    plt.plot([0,1], [0,1], 'k--', label = "Random")
    plt.plot(fprs, tprs, label = model_name)
    plt.title('ROC curve')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend(loc = "lower right")
    return plt.show();

def optimal_threshold(fprs, tprs, thresholds):
    distances = []
    for fpr, tpr in zip(fprs,tprs):
        distance = np.sqrt(fpr**2 + (1-tpr)**2)
        distances.append(distance)
    sorted_distances = sorted(enumerate(distances), key = lambda x: x[1])
    best_index = sorted_distances[0][0]
    best_threshold = thresholds[best_index]
    return best_threshold

def precision_recall_curve(y, y_proba, thresholds):
    precisions = []
    recalls = []
    for threshold in thresholds:
        y_pred = np.array(pd.Series(y_proba).map(lambda x: 1 if x > threshold else 0))
        precision, recall = round(precision_score(y, y_pred), 3), round(recall_score(y, y_pred), 3)
        precisions.append(precision)
        recalls.append(recall)
    return precisions, recalls, thresholds

def plot_precision_recall_curve(precisions, recalls, thresholds, best_threshold):
    plt.plot(thresholds, precisions, label = 'Precision')
    plt.plot(thresholds, recalls, label = 'Recall')
    plt.axvline(x = 0.5, ls = ':', color = 'red')
    plt.axvline(x = best_threshold, ls = ':', color = 'green')
    if best_threshold < 0.5:
      plt.text(0.52, 0.2, 'old threshold', color = 'red')
      plt.text(best_threshold - 0.25, 0.2, 'new threshold', color = 'green')
    else:
      plt.text(0.25, 0.2, 'old threshold', color = 'red')
      plt.text(best_threshold + 0.02, 0.2, 'new threshold', color = 'green')
    plt.title('Precision vs Recall')
    plt.xlabel('Threshold')
    plt.ylim(0,1)
    plt.legend()
    return plt.show();

def custom_confusion_matrix(y, y_hat):
    confusion_matrix = pd.crosstab(np.array(y), np.array(y_hat), rownames=['Actual'], colnames=['Predicted'])
    sns.heatmap(confusion_matrix, annot=True, cmap = ['red', 'green'], cbar = False, fmt = 'g')
    plt.title('Confusion Matrix')
    return plt.show();

def model_report(y, y_hat, y_proba, model_name):

    print("{} Model Report".format(model_name))
    print("")
    print("")

    print("- performance metrics (default threshold):")
    print("")
    classification_metrics(y, y_hat)
    print("")

    print("- ROC curve:")
    print("")
    fprs, tprs, thresholds = roc_curve(y, y_proba)
    plot_roc_curve(fprs, tprs, model_name)
    print("")

    best_threshold = optimal_threshold(fprs, tprs, thresholds)
    print("- optimal probability threshold:")
    print("")
    print(round(best_threshold, 3))
    print("")

    print("- precision recall tradeoff:")
    print("")
    precisions, recalls, thresholds = precision_recall_curve(y, y_proba, list(np.arange(0,1, 0.05)))
    plot_precision_recall_curve(precisions, recalls, thresholds, best_threshold)
    print("")

    print("- performance metrics (new threshold):")
    print("")
    y_hat_new = pd.Series(y_proba).map(lambda x: 1 if x > best_threshold else 0)
    classification_metrics(y, y_hat_new)
    print("")

    print("- confusion matrix:")
    print("")
    custom_confusion_matrix(y, y_hat_new)
    print("")
    
    return None

def mixed_NB(X, y):
    num_columns = ['age', 'blood_pressure_min', 'blood_pressure_max', 'bmi']
    X_num = X[num_columns]
    X_cat = X.drop(columns = num_columns)

    num_nb = GaussianNB()
    cat_nb = CategoricalNB()

    num_nb_model = num_nb.fit(X_num, y)
    cat_nb_model = cat_nb.fit(X_cat, y)

    y_num_proba = num_nb_model.predict_proba(X_num)[:,1].reshape(-1,1)
    y_cat_proba = cat_nb_model.predict_proba(X_cat)[:,1].reshape(-1,1)

    X_trans = np.hstack((y_cat_proba, y_num_proba))

    mixed_nb = GaussianNB()
    mixed_nb_model = mixed_nb.fit(X_trans, y)

    return num_nb, cat_nb, mixed_nb

def mixed_NB_predict(num_nb_model, cat_nb_model, mixed_nb_model, X):
    num_columns = ['age', 'blood_pressure_min', 'blood_pressure_max', 'bmi']
    X_num = X[num_columns]
    X_cat = X.drop(columns = num_columns)

    y_num_proba = num_nb_model.predict_proba(X_num)[:,1].reshape(-1,1)
    y_cat_proba = cat_nb_model.predict_proba(X_cat)[:,1].reshape(-1,1)

    X_new = np.hstack((y_cat_proba, y_num_proba))

    y_pred = mixed_nb_model.predict(X_new)
    y_proba = mixed_nb_model.predict_proba(X_new)[:,1]

    return y_pred, y_proba