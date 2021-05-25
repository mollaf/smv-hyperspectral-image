

#%%

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









#%%

# Labels

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













#%%

# Color Map

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











#%%

HS, gt = read_data()
X, y = bit_vectors(HS, gt)











#%% 

from sklearn.model_selection import train_test_split

non_zeros = np.where(y != 0)
# y_train = np.delete(y_train, zeros)
# X_train = np.delete(X_train, zeros)
y = y[non_zeros]
X = X[non_zeros]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.95, random_state=42 )












#%%

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)



print(HS.shape, gt.shape)
print(X.shape, y.shape)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# print(zeros)

total_histo, val1 = np.histogram(y, range(0,10))
partial_histo, val2 = np.histogram(y_train, range(0,10))

print(np.array([total_histo, partial_histo]))
print(np.array(partial_histo/total_histo).round(3))



# from sklearn.model_selection import KFold
# kf = KFold(n_splits=10)

# for train, test in kf.split(X_train, y_train):

#     train_X = X_train[train]
#     train_y = y_train[train]
#     test_y = y_train[test]

#     histo_train_y, value_train_y = np.histogram(train_y, range(0,10))
#     histo_test_y, value_test_y = np.histogram(test_y, range(0,10))
#     print([train.shape, test.shape], (histo_train_y, value_train_y.tolist()), (histo_test_y, value_test_y.tolist()))












#%% 


from sklearn.svm import SVC
svclassifier = SVC(kernel='rbf')
svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)













#%% 

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

print(accuracy_score(y_test, y_pred))
print(balanced_accuracy_score(y_test, y_pred))

# print(accuracy_score(y_test, y_pred, normalize=False))









#%%

cfm = confusion_matrix(y_test,y_pred)







#%% 


results = []

height, width, _ = HS.shape

for j in range(height):
    print(j)
    line = svclassifier.predict(HS[j])
    results.append(line)










#%%

fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(ncols=2, nrows=3)

ax1.imshow(results, cmap=cmap, interpolation='none', norm=norm )
ax2.imshow(gt, cmap=cmap, interpolation='none', norm=norm )


true_positions_labeled = (results == gt) * (gt != 0) * results
false_positions_labeled = (results != gt) * (gt != 0) * results

ax3.imshow(true_positions_labeled, cmap=cmap, interpolation='none', norm=norm )
ax4.imshow(false_positions_labeled, cmap=cmap, interpolation='none', norm=norm )


true_positions = (results == gt) * results
false_positions = (results != gt) * results

ax5.imshow(true_positions, cmap=cmap, interpolation='none', norm=norm )
ax6.imshow(false_positions, cmap=cmap, interpolation='none', norm=norm )

# ax1.set_axis_off()

plt.tight_layout()

plt.show()


#%%

plt.imshow(HS[:,:,(60,30,27)]/4096)
# plt.set_axis_off()


plt.show()













#%%

img = plt.imshow(gt, cmap=cmap, interpolation='none', norm=norm )
plt.colorbar(img, cmap=cmap, boundaries=bounds, ticks=ticks, norm=norm)
plt.show()
# np.savetxt(f'all.csv', gt, delimiter=",", fmt='%d')

for i in range(1,10):
    gt1 = np.copy(gt)
    idx = np.where(gt1 != i)
    gt1[idx] = 0
    plt.title(class_labels[i])
    img = plt.imshow(gt1, cmap=cmap, interpolation='none', norm=norm)
    plt.colorbar(img, cmap=cmap, boundaries=bounds, ticks=ticks, norm=norm)
    plt.show()
    # np.savetxt(f"{i}.csv", gt1, delimiter=",")
    














#%%

def group_by_class(HS, gt):
    X = []
    y = []

    height, width, bant_count = HS.shape
    height_gt, width_gt = gt.shape

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


















# %%

# Custom Colormap

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

# np.random.seed(101)
# zvals = np.random.rand(100, 100) * 10
zvals = np.concatenate([  
    np.zeros((100, 10)), 
    np.ones((100, 10)),
    np.ones((100, 10)) * 2,
    np.ones((100, 10)) * 3,
    np.ones((100, 10)) * 4,
    # np.ones((100, 10)) * 5,
    # np.ones((100, 10)) * 6,
    # np.ones((100, 10)) * 7,
    # np.ones((100, 10)) * 8,
    # np.ones((100, 10)) * 9,
], axis=1)

# make a color map of fixed colors
# cmap = colors.ListedColormap(['white', 'red'])

cmap = colors.ListedColormap(['black', 
    colors.to_rgba('#c4cccd', alpha=None),
    colors.to_rgba('#02ff00', alpha=None), 
    colors.to_rgba('#20fec9', alpha=None),
    colors.to_rgba('#00b800', alpha=None),
    colors.to_rgba('#e333fe', alpha=None),
    colors.to_rgba('#be3000', alpha=None),
    colors.to_rgba('#8500e5', alpha=None),
    colors.to_rgba('#fe051c', alpha=None),
    colors.to_rgba('#edfe00', alpha=None), ])

ticks=[0,1,2,3,4,5,6,7,8,9] 
bounds=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
norm = colors.BoundaryNorm(bounds, cmap.N)

# tell imshow about color map so that only set colors are used
img = plt.imshow(zvals, cmap=cmap, interpolation='none', norm=norm )
# img = plt.imshow(gt1, interpolation='nearest', origin='lower', cmap=cmap)

# make a color bar
plt.colorbar(img, cmap=cmap, boundaries=bounds, ticks=ticks, norm=norm)

plt.show()












# %%

# Colormaps
# https://matplotlib.org/stable/gallery/color/colormap_reference.html

import numpy as np
import matplotlib.pyplot as plt


cmaps = [('Perceptually Uniform Sequential', ['viridis', 'plasma', 'inferno', 'magma', 'cividis']),
         ('Sequential', ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']),
         ('Sequential (2)', ['binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink', 'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper']),
         ('Diverging', ['PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']),
         ('Cyclic', ['twilight', 'twilight_shifted', 'hsv']),
         ('Qualitative', ['Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2', 'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b', 'tab20c']),
         ('Miscellaneous', ['flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern', 'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'gist_rainbow', 'rainbow', 'jet', 'turbo', 'nipy_spectral', 'gist_ncar'])]


gradient = np.linspace(0, 1, 256)
gradient = np.vstack((gradient, gradient))


def plot_color_gradients(cmap_category, cmap_list):
    # Create figure and adjust figure height to number of colormaps
    nrows = len(cmap_list)
    figh = 0.35 + 0.15 + (nrows + (nrows-1)*0.1)*0.22
    fig, axs = plt.subplots(nrows=nrows, figsize=(6.4, figh))
    fig.subplots_adjust(top=1-.35/figh, bottom=.15/figh, left=0.2, right=0.99)

    axs[0].set_title(cmap_category + ' colormaps', fontsize=14)

    for ax, name in zip(axs, cmap_list):
        ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap(name))
        ax.text(-.01, .5, name, va='center', ha='right', fontsize=10, transform=ax.transAxes)

    # Turn off *all* ticks & spines, not just the ones with colormaps.
    for ax in axs:
        ax.set_axis_off()


for cmap_category, cmap_list in cmaps:
    plot_color_gradients(cmap_category, cmap_list)

plt.show()








#%%

