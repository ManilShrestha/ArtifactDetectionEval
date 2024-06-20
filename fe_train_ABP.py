import sys

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from lib.FE_ExtractFeatures import ExtractFeatures
from lib.Utilities import *

def train_and_eval_SVM(X_train, y_train, X_test, y_test):
    log_info("Training with SVM")
    svm_classifier = SVC(kernel='rbf')
    svm_classifier.fit(X_train, y_train)

    log_info("Evaluating the SVM classifier")
    y_pred_train = svm_classifier.predict(X_train)
    log_info(f"Train Accuracy: {accuracy_score(y_train, y_pred_train)}\n{classification_report(y_train, y_pred_train)}\n{confusion_matrix(y_train, y_pred_train)}")
    
    y_pred_test = svm_classifier.predict(X_test)
    log_info(f"Test Accuracy: {accuracy_score(y_test, y_pred_test)}\n{classification_report(y_test, y_pred_test)}\n{confusion_matrix(y_test, y_pred_test)}")

    log_info(f"Saving the trained SVM model")
    save_model(svm_classifier, 'models/svm_classifier_afib.pkl')


def train_and_eval_KNN(X_train, y_train, X_test, y_test, n_neighbors=5):
    log_info("Training with KNN")
    knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_classifier.fit(X_train, y_train)

    log_info("Evaluating the KNN classifier")
    y_pred_train = knn_classifier.predict(X_train)
    log_info(f"Train Accuracy: {accuracy_score(y_train, y_pred_train)}")
    log_info(f"{classification_report(y_train, y_pred_train)}")
    log_info(f"{confusion_matrix(y_train, y_pred_train)}")

    y_pred_test = knn_classifier.predict(X_test)
    log_info(f"Test Accuracy: {accuracy_score(y_test, y_pred_test)}")
    log_info(f"{classification_report(y_test, y_pred_test)}")
    log_info(f"{confusion_matrix(y_test, y_pred_test)}")

    log_info("Saving the trained KNN model")
    save_model(knn_classifier, 'models/knn_classifier_afib.pkl')


def train_and_eval_DT(X_train, y_train, X_test, y_test, max_depth=None, criterion='gini'):
    log_info("Training with Decision Tree")
    dt_classifier = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion)
    dt_classifier.fit(X_train, y_train)

    log_info("Evaluating the Decision Tree classifier")
    y_pred_train = dt_classifier.predict(X_train)
    log_info(f"Train Accuracy: {accuracy_score(y_train, y_pred_train)}")
    log_info(f"{classification_report(y_train, y_pred_train)}")
    log_info(f"{confusion_matrix(y_train, y_pred_train)}")

    y_pred_test = dt_classifier.predict(X_test)
    log_info(f"Test Accuracy: {accuracy_score(y_test, y_pred_test)}")
    log_info(f"{classification_report(y_test, y_pred_test)}")
    log_info(f"{confusion_matrix(y_test, y_pred_test)}")

    log_info("Saving the trained Decision Tree model")
    save_model(dt_classifier, 'models/dt_classifier_afib.pkl')



##############################################################################################
train_files = [
					'4_Patient_2022-02-05_08:59.h5'
					# , '34_Patient_2023-04-04_22:31.h5'
					# , '35_Patient_2023-04-03_19:51.h5'
					# , '50_Patient_2023-06-12_21:10.h5'
					# , '53_Patient_2023-06-25_21:39.h5'
					# , '90_Patient_2023-03-21_12:19.h5' 
                ]

# test_file = '85_Patient_2023-05-12_17:53.h5'
test_file = ['4_Patient_2022-02-05_08:59.h5']

features_csv_file = '/home/ms5267@drexel.edu/moberg-precicecap/ArtifactDetectionEval/data/FE_features_train.csv'
mode = ['ABP','ART']

df = pd.read_csv(features_csv_file, header=None)

train_features = df[df[0].isin(train_files)][df[1].isin(mode)]

test_features = df[df[0].isin(test_file)][df[1].isin(mode)]

# Training data
y_train = train_features.iloc[:, 2].to_numpy()
X_train = train_features.iloc[:, 3:].to_numpy()

# Test data
y_test = test_features.iloc[:, 2].to_numpy()
X_test = test_features.iloc[:, 3:].to_numpy()


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



train_and_eval_SVM(X_train, y_train, X_test, y_test)
train_and_eval_KNN(X_train, y_train, X_test, y_test)
train_and_eval_DT(X_train, y_train, X_test, y_test)



