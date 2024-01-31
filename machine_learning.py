# Step 1 import libraries
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm, tree, gaussian_process
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


# Step 2 read the csv files and create pandas dataframes
legitimate_df = pd.read_csv("structured_data_legitimate.csv")
phishing_df = pd.read_csv("structured_data_phishing.csv")

# Step 3 combine legitimate and phishing dataframes, and shuffle
df = pd.concat([legitimate_df, phishing_df], axis=0)
df = df.sample(frac=1)

# Step 4 remove 'URL' and remove duplicates, then create X and Y for the models, Supervised Learning
df = df.drop('URL', axis=1)
df = df.drop_duplicates()

X = df.drop('label', axis=1)
Y = df['label']

# Step 5 split data to train and test
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=10)

K = 5  # Define K here

# Step 6 create a ML model using sklearn
models = {
    'SVM': svm.LinearSVC(),
    'Random Forest': RandomForestClassifier(n_estimators=60),
    'Decision Tree': tree.DecisionTreeClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'Naive Bayes': GaussianNB(),
}

# Create instances of models for specific use in the loop
rf_model = models['Random Forest']
dt_model = models['Decision Tree']
ab_model = models['AdaBoost']
nb_model = models['Naive Bayes']

# Step 7 train the models and evaluate using StratifiedKFold
for model_name, model in models.items():
    accuracy_list, precision_list, recall_list = [], [], []
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)

    for train_index, test_index in skf.split(X, Y):
        x_train, x_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]

        model.fit(x_train, y_train)
        predictions = model.predict(x_test)

        # Calculate metrics
        report = classification_report(y_test, predictions, output_dict=True)
        accuracy_list.append(report['accuracy'])
        precision_list.append(report['1']['precision'])
        recall_list.append(report['1']['recall'])

    # Average metrics over K folds
    avg_accuracy = sum(accuracy_list) / K
    avg_precision = sum(precision_list) / K
    avg_recall = sum(recall_list) / K

    print(f"{model_name} - Accuracy: {avg_accuracy:.2f}, Precision: {avg_precision:.2f}, Recall: {avg_recall:.2f}")

# Step 8 make some predictions using test data
svm_model = svm.LinearSVC()
svm_model.fit(x_train, y_train)
predictions = svm_model.predict(x_test)

tn, fp, fn, tp = confusion_matrix(y_true=y_test, y_pred=predictions).ravel()
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)

accuracy_percentage = accuracy * 100
precision_percentage = precision * 100
recall_percentage = recall * 100

print("accuracy --> {:.2f}%".format(accuracy_percentage))
print("precision --> {:.2f}%".format(precision_percentage))
print("recall --> {:.2f}%".format(recall_percentage))

# Multiply accuracy, precision, and recall by 100 to convert them to percentages
accuracy_percentage = accuracy * 100
precision_percentage = precision * 100
recall_percentage = recall * 100

print("accuracy --> {:.2f}%".format(accuracy_percentage))
print("precision --> {:.2f}%".format(precision_percentage))
print("recall --> {:.2f}%".format(recall_percentage))

# K-fold cross validation, and K = 5
K = 5
total = X.shape[0]
index = int(total / K)

# 1
X_1_test = X.iloc[:index]
X_1_train = X.iloc[index:]
Y_1_test = Y.iloc[:index]
Y_1_train = Y.iloc[index:]

# 2
X_2_test = X.iloc[index:index*2]
X_2_train = X.iloc[np.r_[:index, index*2:]]
Y_2_test = Y.iloc[index:index*2]
Y_2_train = Y.iloc[np.r_[:index, index*2:]]

# 3
X_3_test = X.iloc[index*2:index*3]
X_3_train = X.iloc[np.r_[:index*2, index*3:]]
Y_3_test = Y.iloc[index*2:index*3]
Y_3_train = Y.iloc[np.r_[:index*2, index*3:]]

# 4
X_4_test = X.iloc[index*3:index*4]
X_4_train = X.iloc[np.r_[:index*3, index*4:]]
Y_4_test = Y.iloc[index*3:index*4]
Y_4_train = Y.iloc[np.r_[:index*3, index*4:]]

# 5
X_5_test = X.iloc[index*4:]
X_5_train = X.iloc[:index*4]
Y_5_test = Y.iloc[index*4:]
Y_5_train = Y.iloc[:index*4]


# X and Y train and test lists
X_train_list = [X_1_train, X_2_train, X_3_train, X_4_train, X_5_train]
X_test_list = [X_1_test, X_2_test, X_3_test, X_4_test, X_5_test]

Y_train_list = [Y_1_train, Y_2_train, Y_3_train, Y_4_train, Y_5_train]
Y_test_list = [Y_1_test, Y_2_test, Y_3_test, Y_4_test, Y_5_test]


def calculate_measures(TN, TP, FN, FP):
    model_accuracy = (TP + TN) / (TP + TN + FN + FP)

    # Check if the denominator (TP + FP) is zero
    if (TP + FP) != 0:
        model_precision = TP / (TP + FP)
    else:
        model_precision = 0

    # Check if the denominator (TP + FN) is zero
    if (TP + FN) != 0:
        model_recall = TP / (TP + FN)
    else:
        model_recall = 0

    return model_accuracy, model_precision, model_recall


rf_accuracy_list, rf_precision_list, rf_recall_list = [], [], []
dt_accuracy_list, dt_precision_list, dt_recall_list = [], [], []
ab_accuracy_list, ab_precision_list, ab_recall_list = [], [], []
svm_accuracy_list, svm_precision_list, svm_recall_list = [], [], []
nb_accuracy_list, nb_precision_list, nb_recall_list = [], [], []


for i in range(0, K):
    for i in range(0, K):
        # ----- RANDOM FOREST ----- #
        rf_model.fit(X_train_list[i], Y_train_list[i])
        rf_predictions = rf_model.predict(X_test_list[i])
        tn, fp, fn, tp = confusion_matrix(y_true=Y_test_list[i], y_pred=rf_predictions).ravel()
        rf_accuracy, rf_precision, rf_recall = calculate_measures(tn, tp, fn, fp)
        rf_accuracy_list.append(rf_accuracy)
        rf_precision_list.append(rf_precision)
        rf_recall_list.append(rf_recall)

        # ----- DECISION TREE ----- #
        dt_model.fit(X_train_list[i], Y_train_list[i])
        dt_predictions = dt_model.predict(X_test_list[i])
        tn, fp, fn, tp = confusion_matrix(y_true=Y_test_list[i], y_pred=dt_predictions).ravel()
        dt_accuracy, dt_precision, dt_recall = calculate_measures(tn, tp, fn, fp)
        dt_accuracy_list.append(dt_accuracy)
        dt_precision_list.append(dt_precision)
        dt_recall_list.append(dt_recall)

        # ----- SUPPORT VECTOR MACHINE ----- #
        svm_model.fit(X_train_list[i], Y_train_list[i])
        svm_predictions = svm_model.predict(X_test_list[i])
        tn, fp, fn, tp = confusion_matrix(y_true=Y_test_list[i], y_pred=svm_predictions).ravel()
        svm_accuracy, svm_precision, svm_recall = calculate_measures(tn, tp, fn, fp)
        svm_accuracy_list.append(svm_accuracy)
        svm_precision_list.append(svm_precision)
        svm_recall_list.append(svm_recall)

        # ----- ADABOOST ----- #
        ab_model.fit(X_train_list[i], Y_train_list[i])
        ab_predictions = ab_model.predict(X_test_list[i])
        tn, fp, fn, tp = confusion_matrix(y_true=Y_test_list[i], y_pred=ab_predictions).ravel()
        ab_accuracy, ab_precision, ab_recall = calculate_measures(tn, tp, fn, fp)
        ab_accuracy_list.append(ab_accuracy)
        ab_precision_list.append(ab_precision)
        ab_recall_list.append(ab_recall)

        # ----- GAUSSIAN NAIVE BAYES ----- #
        nb_model.fit(X_train_list[i], Y_train_list[i])
        nb_predictions = nb_model.predict(X_test_list[i])
        tn, fp, fn, tp = confusion_matrix(y_true=Y_test_list[i], y_pred=nb_predictions).ravel()
        nb_accuracy, nb_precision, nb_recall = calculate_measures(tn, tp, fn, fp)
        nb_accuracy_list.append(nb_accuracy)
        nb_precision_list.append(nb_precision)
        nb_recall_list.append(nb_recall)



RF_accuracy = sum(rf_accuracy_list) / len(rf_accuracy_list)
RF_precision = sum(rf_precision_list) / len(rf_precision_list)
RF_recall = sum(rf_recall_list) / len(rf_recall_list)

# Random Forest
print("Random Forest accuracy ==> {:.2f}%".format(RF_accuracy * 100))
print("Random Forest precision ==> {:.2f}%".format(RF_precision * 100))
print("Random Forest recall ==> {:.2f}%".format(RF_recall * 100))


DT_accuracy = sum(dt_accuracy_list) / len(dt_accuracy_list)
DT_precision = sum(dt_precision_list) / len(dt_precision_list)
DT_recall = sum(dt_recall_list) / len(dt_recall_list)

# Decision Tree
print("Decision Tree accuracy ==> {:.2f}%".format(DT_accuracy * 100))
print("Decision Tree precision ==> {:.2f}%".format(DT_precision * 100))
print("Decision Tree recall ==> {:.2f}%".format(DT_recall * 100))


AB_accuracy = sum(ab_accuracy_list) / len(ab_accuracy_list)
AB_precision = sum(ab_precision_list) / len(ab_precision_list)
AB_recall = sum(ab_recall_list) / len(ab_recall_list)

# AdaBoost
print("AdaBoost accuracy ==> {:.2f}%".format(AB_accuracy * 100))
print("AdaBoost precision ==> {:.2f}%".format(AB_precision * 100))
print("AdaBoost recall ==> {:.2f}%".format(AB_recall * 100))

SVM_accuracy = sum(svm_accuracy_list) / len(svm_accuracy_list)
SVM_precision = sum(svm_precision_list) / len(svm_precision_list)
SVM_recall = sum(svm_recall_list) / len(svm_recall_list)

# Support Vector Machine
print("Support Vector Machine accuracy ==> {:.2f}%".format(SVM_accuracy * 100))
print("Support Vector Machine precision ==> {:.2f}%".format(SVM_precision * 100))
print("Support Vector Machine recall ==> {:.2f}%".format(SVM_recall * 100))

NB_accuracy = sum(nb_accuracy_list) / len(nb_accuracy_list)
NB_precision = sum(nb_precision_list) / len(nb_precision_list)
NB_recall = sum(nb_recall_list) / len(nb_recall_list)

# Gaussian Process
print("Gaussian Naive Bayes accuracy ==> {:.2f}%".format(NB_accuracy * 100))
print("Gaussian Naive Bayes precision ==> {:.2f}%".format(NB_precision * 100))
print("Gaussian Naive Bayes recall ==> {:.2f}%".format(NB_recall * 100))


data = {'accuracy': [SVM_accuracy, DT_accuracy, RF_accuracy, AB_accuracy, NB_accuracy],
        'precision': [SVM_precision, DT_precision, RF_precision, AB_precision,NB_precision],
        'recall': [SVM_recall, DT_recall, RF_recall, AB_recall,NB_recall ]
        }

index = ['SVM', 'DT', 'RF', 'AB', 'NB']

df_results = pd.DataFrame(data=data, index=index)

# visualize the dataframe
ax = df_results.plot.bar(rot=0)
plt.show()