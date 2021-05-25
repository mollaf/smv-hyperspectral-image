# https://www.researchgate.net/publication/335782175/figure/fig3/AS:802477687459844@1568336998248/Figure-A4-Ground-truth-of-the-University-of-Pavia-scene-with-nine-classes.png

import scipy.io
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np



def read_data():
    HS = scipy.io.loadmat('data/PaviaU.mat')['paviaU']
    gt = scipy.io.loadmat('data/PaviaU_gt.mat')['paviaU_gt']
    return HS, gt



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


HS, gt = read_data()







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


fig, ax = plt.subplots(ncols=5, nrows=2)


img = ax[0,0].imshow(gt, cmap=cmap, interpolation='none', norm=norm )
fig.colorbar(img, cmap=cmap, boundaries=bounds, ticks=ticks, norm=norm, ax=ax[0,0], shrink=0.7)
# np.savetxt(f'all.csv', gt, delimiter=",", fmt='%d')

for i in range(1,9):
    print(i, class_labels[i])
    gt1 = np.copy(gt)
    idx = np.where(gt1 != i)
    gt1[idx] = 0

    ax_current = ax[int(i/5), i%5]
    ax_current.set_title(class_labels[i])
    img = ax_current.imshow(gt1, cmap=cmap, interpolation='none', norm=norm)
    fig.colorbar(img, cmap=cmap, boundaries=bounds, ticks=ticks, norm=norm, ax=ax_current, shrink=0.7)
    # plt.show()
    # np.savetxt(f"{i}.csv", gt1, delimiter=",")
    
ax[-1,-1].axis('off')

# plt.imshow(gt, cmap=cmap)

plt.tight_layout(True)
plt.show()

