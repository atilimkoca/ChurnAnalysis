import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier # import KFold
#sonuçların yeniden üretilebilir olmasını amaçlıyoruz
np.random.seed(1)
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
columns = [
     'state',
     'account length', 
     'area code', 
     'phone number', 
     'international plan', 
     'voice mail plan', 
     'number vmail messages',
     'total day minutes',
     'total day calls',
     'total day charge',
     'total eve minutes',
     'total eve calls',
     'total eve charge',
     'total night minutes',
     'total night calls',
     'total night charge',
     'total intl minutes',
     'total intl calls',
     'total intl charge',
     'number customer service calls',
     'churn']
data = pd.read_csv('ChurnDataset.txt', names = columns)
#Datasetin orjinali hali
print("Dataset orjinal hali: " + str(data.shape))
# Preprocessing Adım 1: yes, no, true, false mapping
mapping = {'no': 0., 'yes':1., 'False.':0., 'True.':1., 'Female.':1., 'Male.':0.}
data.replace({'international plan' : mapping, 'voice mail plan' : mapping, 'churn' : mapping}, regex = True, inplace = True) 
#Preprocessing Adım 2: phone number, area code, state özniteliklerinin kaldırılması

data.drop('phone number', axis = 1, inplace = True)
data.drop('area code', axis = 1, inplace = True)
data.drop('state', axis = 1, inplace = True)
print("Dataset preprocessing sonrasi: " + str(data.shape))
correlation_matrix = data.corr()

# Set up the matplotlib figure
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)

# Adjust layout for better visibility
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.title('Feature Correlation Heatmap')

# Show the plot
#plt.show()
data1 = data[data['churn']==1]
print("Churn olanlar-data1:"+ str(data1.shape))
data2 = data[data['churn']==0]
print("Churn olmayanlar-data2:"+ str(data2.shape))
data = data1._append(data2[:483])
print("Son veriseti :"+ str(data.shape))
#Egitim  ve test verisini parcaliyoruz --> 80% / 20%
X = data.iloc[:, data.columns != 'churn']
Y = data['churn']
kf = KFold(n_splits=5)
kf.get_n_splits(X)
kf = KFold(n_splits=5, random_state=42, shuffle=True)
classifier = RandomForestClassifier() 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
#ölçeklendirme
scaler = preprocessing.MinMaxScaler((-1,1))
scaler.fit(X)
XX_train = scaler.transform(X_train.values)
XX_test  = scaler.transform(X_test.values)
YY_train = Y_train.values 
YY_test  = Y_test.values
models = []
models.append(('Logistic Regression', LogisticRegression()))
models.append(('Naive Bayes', GaussianNB()))
models.append(('Decision Tree (CART)',DecisionTreeClassifier())) 
models.append(('K-NN', KNeighborsClassifier()))
models.append(('SVM', SVC(probability=True)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('AdaBoostClassifier', AdaBoostClassifier()))
models.append(('BaggingClassifier', BaggingClassifier()))
models.append(('RandomForestClassifier', RandomForestClassifier()))
models.append(('XGBoost', XGBClassifier()))
models.append(('Extra Decision Tree', DecisionTreeClassifier(max_depth=10))) # Example with different hyperparameter
# Modelleri test edelim
for name, model in models:
    model = model.fit(X_train, Y_train)
    cv_scores = cross_val_score(model, X_train, Y_train, cv=kf)
    Y_pred = model.predict(X_test)
    from sklearn import metrics
    print("Model -> %s -> ACC: %%%.2f" % (name,metrics.accuracy_score(Y_test, Y_pred)*100))
    Y_proba = model.predict_proba(X_test)[:, 1]  # Probabilities for ROC curve

    # Accuracy, Recall (Sensitivity), and F1 Score
    accuracy = accuracy_score(Y_test, Y_pred)
    recall = recall_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred)

    # Specificity
    tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred).ravel()
    specificity = tn / (tn + fp)

    # ROC Curve and AUC
    fpr, tpr, _ = roc_curve(Y_test, Y_proba)
    roc_auc = auc(fpr, tpr)

    print(f"Model -> {name}")
    print(f"CV Accuracy: {cv_scores.mean() * 100:.2f}%")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Sensitivity: {recall:.2f}")
    print(f"Specificity: {specificity:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"AUC: {roc_auc:.2f}\n")

    # Plot ROC Curve
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {name}')
    plt.legend(loc="lower right")
    plt.show()