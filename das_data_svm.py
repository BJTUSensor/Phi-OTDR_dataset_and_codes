#coding = UTF-8
import sys
class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
sys.stdout = Logger('svm_result.log', sys.stdout)
import datetime
from sklearn import svm, preprocessing
from get_das_data import get_das_data
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
import seaborn as sns
rootpath = '/das_data'
train_rootpath = rootpath+'/train'
train_labelpath = rootpath+'/train/label.txt'
test_rootpath = rootpath+'/test'
test_labelpath = rootpath+'/test/label.txt'
start_train = datetime.datetime.now()
X_train, y_train = get_das_data(train_rootpath, train_labelpath)
X_test, y_test = get_das_data(test_rootpath, test_labelpath)

pre_y_test = y_test[:, np.newaxis]

minMaxScaler = preprocessing.MinMaxScaler()
trainingData = minMaxScaler.fit_transform(X_train)
testData = minMaxScaler.fit_transform(X_test)

feature_data = np.concatenate((testData, pre_y_test), axis=1)
np.savetxt('5km_10km_svm_feature_data.csv', feature_data, delimiter=',')

clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovo', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
clf.fit(trainingData, y_train)
end_train = datetime.datetime.now()

train_result = clf.predict(trainingData)

start_test = datetime.datetime.now()
test_result = clf.predict(testData)
end_test = datetime.datetime.now()

train_matrix = confusion_matrix(y_train, train_result)
test_matrix = confusion_matrix(y_test, test_result)
print('train_matrix:\n', train_matrix)
print('test_matrix:\n', test_matrix)
print('train time is ', end_train - start_train)
print('test time is ', end_test - start_test)
C = test_matrix
fig = plt.figure()
ax = fig.add_subplot(111)
df = pd.DataFrame(C)
f1 = fm.FontProperties('Times New Roman', size=15)
sns.heatmap(df, fmt='g', annot=True,
                annot_kws={'size': 10},
                xticklabels=['1', '2', '3', '4', '5', '6'],
                yticklabels=['1', '2', '3', '4', '5', '6'],
                cmap='Blues')
ax.set_xlabel('Predicted label', FontProperties=f1)  # x轴
ax.set_ylabel('True label', FontProperties=f1)  # y轴
plt.savefig('./5km_10km_svm_confusion_matrix.jpg')
plt.show()
Acc = (C[0][0] + C[1][1] + C[2][2] + C[3][3] + C[4][4] + C[5][5]) / sum(C[0] + C[1] + C[2] + C[3] + C[4] + C[5])
print('acc: %.3f' % Acc)
lie_he = sum(C, 1) - 1
for i in range(1, 7):
    Precision = C[i - 1][i - 1] / lie_he[i - 1]
    NAR = (sum(C[i - 1]) - C[i - 1][i - 1]) / sum(C[i - 1])
    F1_score = 2 * C[i - 1][i - 1] / (lie_he[i - 1] + sum(C[i - 1]))
    print('precision_%d: %.3f' % (i, Precision))
    print('NAR_%d: %.3f' % (i, NAR))
    print('F1_score_%d: %.3f' % (i, F1_score))

