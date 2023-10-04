import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from scipy.spatial import distance

n_samples = 0
filedir = "C:/ReIDDataset_PR/"
X = []
y = []

textfile = open(filedir + "/dataset.txt", "rb")
#X = np.loadtxt(filedir + "/dataset.txt", max_rows=2)
#print(X.shape[1])
#textfile.seek(115202, 1)
#textfile.seek(115202, 1)
#print(textfile.tell())
#y = [int(x) for x in next(textfile).split()]

#line = len(textfile.readlines())
for i in range(2000):
    n = np.loadtxt(filedir + "/dataset.txt", max_rows=2, skiprows=i*3)
    euclidean_dis = np.zeros(1)
    euclidean_dis[0] = distance.euclidean(n[0], n[1])
    #print(euclidean_dis)
    #print(euclidean_dis.shape)
    n = n.reshape(-1)
    textfile.seek(115202, 1)
    textfile.seek(115202, 1)
    c = [int(x) for x in next(textfile).split()]

    if(c[0] == 1):
        X.append(n)
        X[n_samples] = np.concatenate((X[n_samples], euclidean_dis))
        #print(X)
        y.append(c[0])
        n_samples = n_samples + 1
    else:
        if i%8 == 0:
            X.append(n)
            y.append(c[0])
            X[n_samples] = np.concatenate((X[n_samples], euclidean_dis))
            #print(X)
            n_samples = n_samples + 1
    print(i)



X = np.array(X)
y = np.array(y)

X = X.reshape(n_samples, -1)
print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# SVM RBF SCALE
clf_svm = make_pipeline(StandardScaler(), SVC(gamma='scale'))
clf_svm.fit(X_train, y_train)

y_pred = clf_svm.predict(X_test)
print("-------------------SVM RBF SCALE-----------------")
print("Y_pred :", y_pred)
print("Y_Test :", y_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy SVM:", accuracy)

# SVM RBF AUTO
clf_svm = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf_svm.fit(X_train, y_train)

y_pred = clf_svm.predict(X_test)
print("-------------------SVM RBF AUTO-----------------")
print("Y_pred :", y_pred)
print("Y_Test :", y_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy SVM:", accuracy)

# SVM SIGMOID SCALE
clf_svm = make_pipeline(StandardScaler(), SVC(kernel= 'sigmoid', gamma='scale'))
clf_svm.fit(X_train, y_train)

y_pred = clf_svm.predict(X_test)
print("-------------------SVM SIGMOID SCALE-----------------")
print("Y_pred :", y_pred)
print("Y_Test :", y_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy SVM:", accuracy)

# SVM SIGMOID AUTO
clf_svm = make_pipeline(StandardScaler(), SVC(kernel= 'sigmoid', gamma='auto'))
clf_svm.fit(X_train, y_train)

y_pred = clf_svm.predict(X_test)
print("-------------------SVM SIGMOID AUTO-----------------")
print("Y_pred :", y_pred)
print("Y_Test :", y_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy SVM:", accuracy)

# LR
#clf_lr = make_pipeline(StandardScaler(), LogisticRegression(random_state=0))
#clf_lr.fit(X_train, y_train)

#y_pred = clf_lr.predict(X_test)
#print("-----------------------LR-----------------")
#print("Y_pred: ", y_pred)
#print("Y_test: ", y_test)
#accuracy = accuracy_score(y_test, y_pred)
#print("Accuracy LR: ", accuracy)

textfile.close()