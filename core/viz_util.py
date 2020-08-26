"""Visualization utilities"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def plot_2ax_image_and_mask_superimposed(image, mask, mask_labels=[]):
    """Plot image and mask on the image in 2ax format

    Assumes 'mask' is composed of integer labels and is not
    a continuous probability map
    """
    # remove channel dimension for plotting
    image = image[..., 0]
    mask = mask[..., 0]

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].set_title('Raw Image')
    axs[0].imshow(image, vmin=0, vmax=1, cmap='gray')

    num_foreground_classes = len(np.unique(mask)) - 1
    mask_no_bg = np.ma.masked_where(mask == 0, mask)
    axs[1].set_title('Positive Class Mask(s) on Image')
    axs[1].imshow(image, vmin=0, vmax=1, cmap='gray')
    cmap = plt.get_cmap('plasma', num_foreground_classes)
    axs[1].imshow(mask_no_bg, cmap=cmap)

    if num_foreground_classes > 0:
        # use a colorbar as the plot legend
        norm = matplotlib.colors.BoundaryNorm(
            boundaries=np.arange(0, num_foreground_classes + 1) - 0.5,
            ncolors=num_foreground_classes
        )
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cax = fig.add_axes([axs[1].get_position().x1 + 0.01,
                            axs[1].get_position().y0,
                            0.02,
                            axs[1].get_position().height])
        cbar = plt.colorbar(sm, ticks=np.arange(0, num_foreground_classes),
                            cax=cax)
        if any(mask_labels):
            cbar.set_ticklabels(mask_labels)

    for ax in axs:
        ax.axis('off')
    plt.show()
