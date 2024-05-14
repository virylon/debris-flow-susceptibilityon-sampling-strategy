import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix,cohen_kappa_score

from sklearn.neural_network import MLPClassifier #MLP
from sklearn.linear_model import LogisticRegression #LGR

from sklearn.metrics import roc_curve, auc


sample = pd.read_csv(r'csv\STRUNOUTPOLYGON.csv')

factors = ["aspect", "curvature", "dem", "disfault","disriver", "lithology", "pga", "rain", "slope", "twi"]

x = sample[factors]
y = sample.label

# scaler = MinMaxScaler()
# x=scaler.fit_transform(x)

LGR = LogisticRegression(penalty='l2', solver='lbfgs')

MLP = MLPClassifier(hidden_layer_sizes=(100, 3), batch_size=8,
                    activation='logistic', max_iter=10)

names = ['LGR',  'MLP']  
models = [LGR, MLP]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=0) 

x_train = x_train.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
x_test = x_test.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))

for m, n in zip(models, names):
    m.fit(x_train, y_train)
    joblib.dump(m, "records/"+n+".pkl") 
    # m = joblib.load('records/'+n+'.pkl')

    
    train_pred = m.predict_proba(x_train)[:, 1]
    train_FPR, train_TPR, thresholds = roc_curve(y_train, train_pred)
    train_roc_auc = auc(train_FPR, train_TPR)
    
    train_fpr = pd.DataFrame(train_FPR, columns=[['FPR']])
    train_tpr = pd.DataFrame(train_TPR, columns=[['TPR']])
    roc_record = pd.concat([train_fpr, train_tpr], axis=1)
    roc_record.to_csv("records/"+n+"-train.csv")

    test_pred = m.predict_proba(x_test)[:, 1]
    test_FPR, test_TPR, thresholds = roc_curve(y_test, test_pred)
    test_roc_auc = auc(test_FPR, test_TPR)
    
    test_fpr = pd.DataFrame(test_FPR, columns=[['FPR']])
    test_tpr = pd.DataFrame(test_TPR, columns=[['TPR']])
    roc_record = pd.concat([test_fpr, test_tpr], axis=1)
    roc_record.to_csv("records/"+n+"-test.csv")

    print(n)
    # print(train_roc_auc)
    # print(test_roc_auc)
    test_pred = np.where(test_pred>0.5,1,0)
    cm=confusion_matrix(y_test,test_pred)
    TN, FP, FN, TP = cm.ravel()
    OA=(TN+TP)/(TN+FP+FN+TP)
    precision =TP/(TP+FP)
    recall=TP/(TP+FN) 
    sensitivity=TP/(TP+FN)
    specificity=TN/(FP+TN)
    f1=f1_score(y_test, test_pred)
    kappa=cohen_kappa_score(y_test,test_pred)
    # print(TP,TN, FP, FN)
    # print(test_roc_auc,OA,precision,recall,sensitivity,specificity,f1,kappa)
    print(test_roc_auc)

    # print(accuracy_score(y_test, test_pred))
    # print(precision_score(y_test, test_pred))
    # print(recall_score(y_test, test_pred))
    # print(f1_score(y_test, test_pred))
    # cf=confusion_matrix(y_,y_pr)
    # cfs=cf.reshape(1,4)
    # print(cfs)
    # print(cohen_kappa_score(y_test,test_pred))

# print(RF.feature_importances_)
