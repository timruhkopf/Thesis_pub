import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.tri as mtri  # for trisurface with irregular grid

def triangulate_remove_artifacts(x, y, xl=0.1, xu=9.9, yl=0.1, yu=9.9, plot=True):
    # remove artifacts from triangulation method with some boundary
    triang = mtri.Triangulation(x, y)
    isBad = np.where((x < xl) | (x > xu) | (y < yl) | (y > yu), True, False)

    mask = np.any(isBad[triang.triangles], axis=1)
    triang.set_mask(mask)

    if plot:
        # look at the triangulation result
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        ax.triplot(triang, c="#D3D3D3", marker='.',
                   markerfacecolor="#DC143C", markeredgecolor="black",
                   markersize=10)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.show()

    return triang
