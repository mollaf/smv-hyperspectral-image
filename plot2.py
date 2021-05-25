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
        
print(list(map(lambda x: (x, len(data[x])), data.keys())))



nrows = 4
ncols = 3

fig, axs = plt.subplots(nrows= nrows, ncols=ncols)

for i in data.keys():
    x = int(i/ncols)
    y = i % ncols
    print(i)

    axs[x, y].set_title(f"{i} - {class_labels[i]}")
    # axs[x, y].plot(data[i][0])
    for line in data[i]:
        axs[x, y].plot(line)


plt.tight_layout(True)

plt.show()



# fig, axs = plt.subplots(nrows= nrows, ncols=ncols)

# for i in range(bant_count):
#     x = int(i/ncols)
#     y = i % ncols
#     axs[x, y].imshow(paviau[:,:,i], cmap='gray', vmin=0, vmax=4096)


# axs[nrows-1, ncols-1].imshow(paviau_gt)

# plt.tight_layout()

# plt.show()

