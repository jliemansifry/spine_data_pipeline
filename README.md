# Data Loader Pipeline

This repo contains a basic data loading class `DataLoader` designed to read images and masks from a directory structure and hand them off to a neural network in batches. It also contains some example data as well as visualization tools and some basic unit tests.

What, if anything, did you do to verify that the segmentation masks and images were correctly aligned in the data loader?

Considering the small dataset size, it was reasonable to create some plotting code and visually verify that the masks aligned with the images for all twenty examples. To my (relatively untrained in this domain) eye, it looks like the image/mask pairs are correctly aligned in the data loader. However, image 16 may be missing a disc mask for the lowest disc in the image, and image 13 may be missing a disc mask for the highest disc.

I continued to to verify that image/mask pairs were correctly paired when using the `get_batch` method instead of the `get_image` and `get_mask` methods directly. It's always good to check that the data are coherent and as expected as close to the handoff to the model architecture as possible.

What assumptions did you make about the data or model training during this process?

For this exercise, I designed the general `DataLoader` to work only with images and masks that are stored in a directory structure on disk, and made some assumptions about the directory format (that are confirmed). A more general framework could allow different data stores. Furthermore, I assumed that the filenames of the files in 'images' and 'masks' are identical. All files in a given directory must be of the same file type.

With regards to model training, I designed the system such that if the number of examples is not divisible by the batch_size, the necessary number of surplus examples will be randomly selected from the rest of the available examples to fill out the batch. Furthermore, the batch size must be smaller than the number of training examples.

For further discussion on these assumptions, see the `DataLoader` class docstring.

With regards to model training, I used a simple UNet architecture as specified in the `segmentation_models` github (https://github.com/qubvel/segmentation_models) for Keras. I considered the problem as a multi-class segmentation problem and utilized softmax activation. More complex loss functions and mask processing could be designed in the case of missing labels on only some images while other labels are present; to my eye, this was not the case in these data (and would have been overkill for this exercise). I also assume that all images beyond the twenty provided for this example would be of the same shape and format and that masks would be provided in the same format as well.
