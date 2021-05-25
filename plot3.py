import scipy.io
import matplotlib.pyplot as plt
import numpy as np


mat = scipy.io.loadmat('data/PaviaU.mat')
paviau = mat['paviaU']

mat_gt = scipy.io.loadmat('data/PaviaU_gt.mat')
paviau_gt = mat_gt['paviaU_gt']


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

height, width, bant_count = paviau.shape
print(height, width, bant_count)


height_gt, width_gt = paviau_gt.shape
print(height_gt, width_gt)

classes = np.unique(paviau_gt.flatten())

data = {}

for c in classes:
    data[c] = []

for y in range(height_gt):
    for x in range(width_gt):
        class_gt = paviau_gt[y, x]
        pixel_data = paviau[y,x,:]
        data[class_gt].append(pixel_data)



def median_line(data1):
    # data1 = data[1]
    median = np.median(data1, axis=0)
    all = np.array([])

    for med in median:
        result = np.where(data1[:,0] == int(med))
        all = np.append(all, result[0])
        # print(med, result)

    all = np.array(all).astype(int)
    # print(all)
    # print(all.shape)
    # print(np.bincount(all))
    return np.bincount(all).argmax()


averages = {}
maxs = {}
mins = {}
median_lines = {}

for c in classes:
    data[c] = np.array(data[c])
    averages[c] = np.average(data[c], axis=0)
    maxs[c] = data[c].max(axis=0)
    mins[c] = data[c].min(axis=0)
    median_lines[c] = data[c][median_line(data[c])]
    # print(median_lines[c])
    # print(data[c][median_line(data[c])])


nrows = 4
ncols = 3

fig, axs = plt.subplots(nrows=nrows, ncols=ncols)

for i in data.keys():
    x = int(i/ncols)
    y = i % ncols

    x_line = list(range(len(averages[i])))
    

    axs[x,y].set_title(f"{i} - {class_labels[i]}")
    # axs[x,y].errorbar(x_line, averages[i], yerr=[mins[i], maxs[i]], ecolor='g')
    axs[x,y].plot(averages[i], color='green')
    axs[x,y].plot(mins[i], color='red')
    axs[x,y].plot(maxs[i], color='red')
    axs[x,y].fill_between(x_line, mins[i], maxs[i], alpha=0.2)
    axs[x,y].plot(median_lines[i], color='blue')

plt.tight_layout(True)

plt.show()


# https://matplotlib.org/1.2.1/examples/pylab_examples/errorbar_demo.html
