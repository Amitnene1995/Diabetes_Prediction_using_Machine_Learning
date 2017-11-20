# -*- coding: utf-8 -*-

from sklearn import svm    			# To fit the svm classifier
import numpy
import matplotlib.pyplot as plt
import csv
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

filename = 'diabetes.csv'
raw_data = open(filename, 'rt')
reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
x = list(reader)
data = numpy.array(x).astype('float')

	
print "Dataset Length:: ", len(data)
print "Dataset Shape:: ", data.shape

X = data[:,0:7]
Y = data[:,8]

print data

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.05, random_state = 30)


plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('Sepal Width & Length')
###plt.show()

svm_model_linear = SVC(kernel = 'linear', C=1).fit(X_train,Y_train)
Y_pred = svm_model_linear.predict(X_test)
print Y_pred

print "Accuracy is",svm_model_linear.score(X_test,Y_test)*100
