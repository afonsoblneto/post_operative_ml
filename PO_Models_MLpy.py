import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

from copy import deepcopy

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir(""))

import warnings
warnings.filterwarnings("ignore")

# Any results you write to the current directory are saved as output.

df = pd.read_csv("PO_Generated_Data.data")

del df['key_PO']
df.head(5)

df.columns

def selecionar_variaveis_df(dados, se_dados_teste=False):
    if se_dados_teste == False:
        decision = dados['decision_C']
        return decision, dados[['L-CORE_C', 'L-SURF_C', 'L-O2_C', 'L-BP_C', 'SURF-STBL_C', 'CORE-STBL_C', 'BP-STBL_C']]
    
    return dados[['L-CORE_C', 'L-SURF_C', 'L-O2_C', 'L-BP_C', 'SURF-STBL_C', 'CORE-STBL_C', 'BP-STBL_C']]

decision, dados = selecionar_variaveis_df(df)

modelos = [
    ('Decision Tree', DecisionTreeClassifier(max_depth=5)),
    ('Random Forest', RandomForestClassifier(max_depth=5, n_estimators=250, max_features=1)),
    ('AdaBoost', AdaBoostClassifier()),
    ('KNN', KNeighborsClassifier(n_neighbors=3)),
    ('MLP - neural network', MLPClassifier()),
    ('Naive Bayes', GaussianNB()),
    ('Logistic Regression', LogisticRegression()),
    ('Support Vector Machines', SVC(gamma=2, C=1)),
    ('Linear SVC', LinearSVC()),
    ('Stochastic Gradient Descent (SGD)', SGDClassifier())    
]

resultados = []
nomes = []
acuracia = []

print("######## Estimators Comparision ##########")
print('\n')
print("Total rows: ", len(df['L-CORE_C']))
print("Number of itens per decision", df.groupby('decision_C').size())
df.groupby('decision_C').size()
print("0:A (patient sent to general hospital floor)")
print("1:I (patient sent to Intensive Care Unit)")
print("2:S (patient prepared to go home)")

print('\n')
i=1
for nome, modelo in modelos:
    kfold = KFold(n_splits=5, random_state=100)
    #cv = cross_val_score(estimator=modelo,X=dados,y=decision,cv=kfold,scoring='r2')
    cv = cross_val_score(estimator=modelo,X=dados,y=decision,cv=kfold,scoring='accuracy')
    resultados.append(cv)
    nomes.append(nome)
    print(str(i)+' - '+nome+': Accuracy= ' + str(round(cv.mean()*100, 1)) + ' %'+': std= ' + str(cv.std()*100))
    acuracia.append([nome,round(cv.mean()*100)])
    i += 1    

print('\n')    
import seaborn as sns
sns.set(style="whitegrid")
df2 = pd.DataFrame(data=acuracia, columns='Estimator Accuracy'.split())
df2 = df2.sort_values(['Accuracy'], ascending=[False])
print(df2)

print('\n')
ax = sns.barplot(x="Estimator", y="Accuracy", data=df2)
ax.set_xticklabels(df2['Estimator'], rotation=90)

