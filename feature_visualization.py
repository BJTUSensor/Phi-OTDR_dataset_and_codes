import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd
datapath = 'E:/DAS_小论文/das_project/5km_10km_result_labpc/5km_10km_svm_feature_data.csv'
dataset = pd.read_csv(datapath, header=None)
dataset = np.array(dataset)  # 160，385   4分类测试    2260,385 六分类结果  150,401
print(dataset.shape)
X = dataset[:, :384]   # svm
# X = dataset[:, :400]  # cnn
y = dataset[:, -1]
lda = LinearDiscriminantAnalysis(n_components=3)
lda.fit(X, y)  # 拟合数据
X = lda.transform(X)  # 降维
figure = plt.subplot(projection="3d")
for i in range(len(y)):
    if y[i] == 0:
        s1 = plt.scatter(X[i, 0], X[i, 1], s=20, color='red', marker=".")
    elif y[i] == 1:
        s2 = plt.scatter(X[i, 0], X[i, 1], s=20, color='blue', marker=".")
    elif y[i] == 2:
        s3 = plt.scatter(X[i, 0], X[i, 1], s=20, color='darkorange', marker=".")
    elif y[i] == 3:
        s4 = plt.scatter(X[i, 0], X[i, 1], s=20, color='green', marker=".")
    elif y[i] == 4:
        s5 = plt.scatter(X[i, 0], X[i, 1], s=20, color='m', marker=".")  # 紫色
    elif y[i] == 5:
        s6 = plt.scatter(X[i, 0], X[i, 1], s=20, color='black', marker=".")

figure.view_init(elev=30, azim=45)
ax = plt.legend((s1, s2, s3, s4, s5, s6), ('1', '2', '3', '4', '5', '6'),
           fontsize='small', edgecolor='black',
            bbox_to_anchor=(1, 0.4), loc='lower left')
# plt.xlim(-6, 5.5)  # svm
# plt.ylim(-5, 5)  # svm
plt.xlim(-6, 5)
plt.ylim(-7.5, 6)
plt.savefig('./svm_feature_data.jpg')
plt.show()







