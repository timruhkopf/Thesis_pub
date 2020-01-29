# Deprec: Now only data generation not prediction evaluation
# adding a heatmap contour of the prediction error below it
# if pred_error is not None:
#
#     ax1.contourf(X, Y, pred_error, zdir='z', offset=-1.5, cmap='magma_r', alpha=0.6)
#     #ax1.scatter(X, Y, -9, marker='o')
#
#
#     positions = np.vstack([X.ravel(), Y.ravel()])
#
#     from scipy import stats
#     kernel = stats.gaussian_kde(positions)
#     Z = np.reshape(kernel(positions).T, X.shape)
#
#     ax1.contourf(X, Y, Z, zdir='z', offset=-2, cmap='magma_r')

# Deprec
# idea (1) ---------
# ax2 = fig.add_subplot(224, projection='3d')
# ax2.plot_surface(X, Y, pred_error[80:120, 80:120], rstride=1, cstride=1,
#                  linewidth=0, antialiased=False)
#
# ax3 = fig.add_subplot(223)
#
# CS = ax3.contour(X, Y, pred_error[80:120, 80:120]/800)
# ax3.clabel(CS, inline=1, fontsize=10)
# ax3.set_title('Simplest default with labels')

# idea (2) ----------
# ax1.contour(X, Y, pred_error[80:120, 80:120], zdir='z', offset=0, cmap='magma_r')

# idea (3) ----------
# saving to png
# grid_x = np.linspace(0, 129, num=129)
# grid_y = np.linspace(0, 129, num=129)
#
# # evaluate at grid points
# xdisplace = xd.__call__(grid_x, grid_y, grid=True)
# ydisplace = yd.__call__(grid_x, grid_y, grid=True)
#
# # save output using skimage
# io.imsave("xdimgs.png", xdisplace.astype('uint8'))
# io.imsave("ydimgs.png", ydisplace.astype('uint8'))


# TODO plot error heatmap underneath the surface
# _, _, pred_error = axes3d.get_test_data(0.05)
# pred_error = pred_error[80:120, 80:120] / 800,