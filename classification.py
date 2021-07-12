from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import pickle as p
#from numpy._distributor_init import NUMPY_MKL
from sklearn import svm
from sklearn.naive_bayes import GaussianNB as nb
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.linear_model import LogisticRegression as lr
import numpy as np
import sklearn.metrics as sm

train = pd.read_csv("train75.csv",error_bad_lines=False)
train=train.dropna()
print(train.head())
y = np.array(train.index)
x = np.array(train)/255.

print("here1")
#train = dataset.iloc[:,1:].values
test = pd.read_csv("train25.csv",error_bad_lines=False)
test=test.dropna()
label_test=np.array(test.index)
x_ = np.array(test)/255.
print("here2")
test1= pd.read_csv("test.csv",error_bad_lines=False)
test1=test1.dropna()
x1= np.array(test1)/255.
print("here3")

#printing the results
def calc_accuracy(method,label_test,pred):
	print("accuracy score for ",method,sm.accuracy_score(label_test,pred))
	print("precision_score for ",method,sm.precision_score(label_test,pred,average='micro'))
	print("f1 score for ",method,sm.f1_score(label_test,pred,average='micro'))
	print("recall score for ",method,sm.recall_score(label_test,pred,average='micro'))


def run_svm():
    clf=svm.SVC(decision_function_shape='ovo')
    print("svm started")
    clf.fit(x,y)
    filename='f.sav'
    p.dump(clf,open(filename,'wb'))
    #print clf.n_layers_
    pred=clf.predict(x_)
    np.savetxt('submission_svm.csv', np.c_[range(1,len(test)+1),pred,label_test], delimiter=',', header = 'ImageId,Label,TrueLabel', comments = '', fmt='%d')
    calc_accuracy("SVM",label_test,pred)
    

def run_lr():
	clf = lr()
	print("lr started")
	clf.fit(x,y)
	#print clf.n_layers_
	pred=clf.predict(x_)
	#print(pred)
	np.savetxt('submission_lr.csv', np.c_[range(1,len(test)+1),pred,label_test], delimiter=',', header = 'ImageId,Label,TrueLabel', comments = '', fmt='%d')
	calc_accuracy("Logistic regression",label_test,pred)


def run_nb():
	clf = nb()
	print("nb started")
	clf.fit(x,y)
	#print(clf.classes_)
	#print clf.n_layers_
	pred=clf.predict(x_)
	#print(pred)
	np.savetxt('submission_nb.csv', np.c_[range(1,len(test)+1),pred,label_test], delimiter=',', header = 'ImageId,Label,TrueLabel', comments = '', fmt='%d')
	calc_accuracy("Naive Bayes",label_test,pred)


def run_knn():
	clf=knn(n_neighbors=3)
	print("knn started")
	clf.fit(x,y)
	#print(clf.classes_)
	#print clf.n_layers_
	pred=clf.predict(x_)
	#print(pred)
	np.savetxt('submission_knn.csv', np.c_[range(1,len(test)+1),pred,label_test], delimiter=',', header = 'ImageId,Label,TrueLabel', comments = '', fmt='%d')
	calc_accuracy("K nearest neighbours",label_test,pred)
def pred1():
    clf=svm.SVC(decision_function_shape='ovo')
    clf.fit(x,y)
    pred12=clf.predict(x1)
    print(pred12[0])
    
    a=['1','2','3','4','5','6','7','8','9','a','b','c','d','e','f','g','h','i','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y']
    for i in pred12:
        
      
      print(a[i])
    
    

run_svm()
run_knn()
run_nb()
run_lr()
pred1()


