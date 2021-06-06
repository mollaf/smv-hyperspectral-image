from numpy.random import normal
from sklearn import svm

import scipy.io
import matplotlib.pyplot as plt
import numpy as np


def read_data():
    HS = scipy.io.loadmat('data/PaviaU.mat')['paviaU']
    gt = scipy.io.loadmat('data/PaviaU_gt.mat')['paviaU_gt']
    return HS, gt


def bit_vectors(HS, gt):
    X = []
    y = []

    height_gt, width_gt = gt.shape
    
    for j in range(height_gt):
        for i in range(width_gt):
            X.append(HS[j,i,:])
            y.append(gt[j,i])

    return np.array(X), np.array(y)



class_labels = {
    0: "",
    1: "Asphalt - Asfalt",
    2: "Meadows - Çayırlar",
    3: "Gravel - Çakıl",
    4: "Trees - Ağaçlar",
    5: "Painted metal sheets \nBoyalı saclar",
    6: "Bare Soil - Çıplak Toprak",
    7: "Bitumen - Zift",
    8: "Self-Blocking Bricks \nKendinden Blokaj Tuğlalar",
    9: "Shadows - Gölgeler"
}




from matplotlib import colors


cmap = colors.ListedColormap(['black', 
    colors.to_rgba('#c4cccd', alpha=None),
    colors.to_rgba('#02ff00', alpha=None),
    colors.to_rgba('#20fec9', alpha=None),
    colors.to_rgba('#00b800', alpha=None),
    colors.to_rgba('#e333fe', alpha=None),
    colors.to_rgba('#be3000', alpha=None),
    colors.to_rgba('#8500e5', alpha=None),
    colors.to_rgba('#fe051c', alpha=None),
    colors.to_rgba('#edfe00', alpha=None), ], N=10)

ticks=[0,1,2,3,4,5,6,7,8,9] 
bounds=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
norm = colors.BoundaryNorm(bounds, cmap.N)





HS, gt = read_data()
HS = HS/8192.0


X, y = bit_vectors(HS, gt)

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler(feature_range=(0, 1))
# X = scaler.fit_transform(X)










from sklearn.model_selection import train_test_split

non_zeros = np.where(y != 0)
# y_train = np.delete(y_train, zeros)
# X_train = np.delete(X_train, zeros)
y = y[non_zeros]
X = X[non_zeros]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.95, shuffle=True)









from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score

c_options = [0.0001, 0.001, 0.01, 0.1, 1, 100, 1000]
gamma_options = [0.0001, 0.001, 0.005, 0.1, 1, 3, 5, 10, 100, 1000]

results = []

for c in c_options: 
    for gamma in gamma_options:
        
        scores = []

        cv = KFold(n_splits=10, shuffle=False)
        
        i = 1
        for train_index, test_index in cv.split(X_train):

            X_train_part, X_test_part, y_train_part, y_test_part = X[train_index], X[test_index], y[train_index], y[test_index]

            svn = SVC(kernel='rbf', C=c, gamma=gamma)
            
            clf = svn.fit(X_train_part, y_train_part)
            
            y_pred_part = clf.predict(X_test_part)
            # scores.append(svn.score(X_test_part, y_test_part))
            current_score = accuracy_score(y_test_part, y_pred_part)
            scores.append(current_score)

            # print("Fold:", i, "Train Data:",  X_train_part.shape, "Train Labels:", y_train_part.shape, "Test Data:", X_test_part.shape, "Test Labels:", y_test_part.shape, "Score:", current_score)
            i = i+1

        print(f'c = {c}, gamma = {gamma}, avg. score = {np.array(scores).mean()}')
        results.append({ 'c': c, 'gamma': gamma, 'score': np.array(scores).mean() })
        # print('---------------------------------------------------------------')


result_sorted = sorted(results, key=lambda x: x['score'], reverse=True)
first_result = result_sorted[0]



c = first_result['c']
gamma = first_result['gamma']


# c = 1000
# gamma = 3

print(f'c: {c}, gamma: {gamma}')


from sklearn.svm import SVC
svclassifier = SVC(kernel='rbf', C=c, gamma=gamma)
svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score, cohen_kappa_score

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

print(f'overall accuracy: {accuracy_score(y_test, y_pred)}')
print(f'average accuracy: {balanced_accuracy_score(y_test, y_pred)}')
print(f'kappa: {cohen_kappa_score(y_test, y_pred)}')



image_data = []

height, width, _ = HS.shape

for j in range(height):
    # print(j)
    # print(HS[j].shape)
    line = svclassifier.predict(HS[j])
    image_data.append(line)


labeled_data = (gt != 0) * image_data



fig, (ax1, ax2, ax3)= plt.subplots(ncols=3)

ax1.imshow(gt, cmap=cmap, interpolation='none', norm=norm )
ax2.imshow(labeled_data, cmap=cmap, interpolation='none', norm=norm )
ax3.imshow(image_data, cmap=cmap, interpolation='none', norm=norm )


plt.show()






import seaborn as sns
import pandas as pd

df = pd.DataFrame(confusion_matrix(y_test, y_pred), columns= list(range(1,10)),  index= list(range(1,10)))


plt.figure()
p = sns.heatmap(df,
                fmt="d", 
                annot=True,
                annot_kws={'size':8},
                cbar=False,
                square=True)

plt.show()


