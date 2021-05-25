import scipy.io
import matplotlib.pyplot as plt

mat = scipy.io.loadmat('data/PaviaU.mat')
paviau = mat['paviaU']

mat_gt = scipy.io.loadmat('data/PaviaU_gt.mat')
paviau_gt = mat_gt['paviaU_gt']


print(paviau)
print(paviau.shape)

height, width, bant_count = paviau.shape
print(bant_count)

nrows = 7
ncols = 15

fig, axs = plt.subplots(nrows=nrows, ncols=ncols)

for i in range(bant_count):
    x = int(i/ncols)
    y = i % ncols
    axs[x, y].imshow(paviau[:,:,i], cmap='gray', vmin=0, vmax=4096)


axs[nrows-1, ncols-1].imshow(paviau_gt)

plt.tight_layout(True)

plt.show()

