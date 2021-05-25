from sklearn import svm

import scipy.io
import matplotlib.pyplot as plt
import numpy as np


def read_data():
    X = scipy.io.loadmat('data/PaviaU.mat')['paviaU']
    y = scipy.io.loadmat('data/PaviaU_gt.mat')['paviaU_gt']
    return X, y

def group_by_class(X, y):
    height, width, bant_count = X.shape
    height_gt, width_gt = y.shape

    # print(height, width, bant_count)
    # print(height_gt, width_gt)

    classes = np.unique(y.flatten())

    data = {}

    for c in classes:
        data[c] = []

    for y_pos in range(height_gt):
        for x_pos in range(width_gt):
            class_gt = y[y_pos, x_pos]
            pixel_data = X[y_pos, x_pos, :]
            data[class_gt].append(pixel_data)
    
    # for c in classes:
    #     data[c] = np.array(data[c])

    return data

def combine(data):
    X = []
    y = []
    for key in data.keys():    
        for line in data[key]:
            X.append(line)
            y.append(key)
    return X, y

X, y = read_data()

data = group_by_class(X, y)

X, y = combine(data)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.95, )


print(len(X_train))
print(len(X_test))



from sklearn.svm import SVC
svclassifier = SVC(kernel='rbf', verbose=False, C=1, )
svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))






exit()



class_labels = {
    0: "",
    1: "Asphalt",
    2: "Meadows",
    3: "Gravel",
    4: "Trees",
    5: "Painted metal sheets",
    6: "Bare Soil",
    7: "Bitumen",
    8: "Self-Blocking Bricks",
    9: "Shadows"
}

# print(paviau)
# print(paviau.shape)









