import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
from sklearn.metrics import roc_curve

from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras import Sequential

from data_preprocessing import data_preprocessor
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

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

def build_nn(hidden_layers = 1, units = 10, activation = "relu", optimizer = "rmsprop"):

  network = Sequential()
  network.add(Dense(units, input_dim = 10, activation = activation))

  if hidden_layers > 1:
    for i in range(hidden_layers - 1):
      network.add(Dense(units, activation = activation))

  network.add(Dense(1, activation = "sigmoid"))

  network.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = ['accuracy'])

  return network

def concat_outputs(trained_models_tuple, X, output = "predict"):

  X_scaled = data_preprocessor(X)
  X_binned = data_preprocessor(X, bins = True)

  outputs = []
  for model in trained_models_tuple:
    if output == "predict":
      if model[1] == "scaled":
        y_hat = model[0].predict(X_scaled).reshape(-1,1)
      elif model[1] == "binned":
        y_hat = model[0].predict(X_binned).reshape(-1,1)
      outputs.append(y_hat)
    elif output == "proba":
      if model[1] == "scaled":
        y_proba = model[0].predict_proba(X_scaled)[:,1].reshape(-1,1)
      elif model[1] == "binned":
        y_proba = model[0].predict_proba(X_binned)[:,1].reshape(-1,1)
      outputs.append(y_proba)
  outputs_concat = np.hstack(outputs)
  return outputs_concat

def voting_classifier(trained_models_tuple, X, hard = True):
  if hard:
    X_concat = concat_outputs(trained_models_tuple, X)
    y_hat = stats.mode(X_concat, axis = 1)[0].reshape(-1)
    return y_hat
  else:
    X_concat = concat_outputs(trained_models_tuple, X, output = "proba")
    y_proba = np.mean(X_concat, axis = 1)
    y_hat = np.array(pd.Series(y_proba).map(lambda x: 1 if x > 0.5 else 0))
    return y_hat, y_proba

def stacked_model(models_tuple, X_train, X_test, y_train, set_2_split = 0.5):
  
  X_set_1, X_set_2, y_set_1, y_set_2 = train_test_split(X_train, y_train, test_size = set_2_split, random_state = 0, stratify = y_train)
  X_set_1_scaled = data_preprocessor(X_set_1)
  X_set_1_binned = data_preprocessor(X_set_1, bins = True)

  trained_models_tuple = []
  for model in models_tuple:
    if model[1] == "scaled":
      trained_model = model[0].fit(X_set_1_scaled, y_set_1)
    elif model[1] == "binned":
      trained_model = model[0].fit(X_set_1_binned, y_set_1)
    trained_models_tuple.append((trained_model, model[1]))
  
  X_set_2_concat = concat_outputs(trained_models_tuple, X_set_2, output = "proba")

  stacked = LogisticRegression()
  stacked_model = stacked.fit(X_set_2_concat, y_set_2)

  X_test_concat = concat_outputs(trained_models_tuple, X_test, output = "proba")
  y_test_hat = stacked_model.predict(X_test_concat)
  y_test_proba = stacked_model.predict_proba(X_test_concat)[:,1]

  return y_test_hat, y_test_proba