import numpy as np

def blob_to_tile(blob, padsize=1, padval=0):
    """
    take an array of shape (n, channels, height, width)
    and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
    """
    assert(blob.ndim == 4)
    blob = blob.transpose(0, 2, 3, 1)
    if blob.shape[3] != 3:
        blob = blob.transpose(0, 3, 1, 2).reshape(
            -1, 1, blob.shape[1], blob.shape[2]
            ).transpose(0, 2, 3, 1)
    blob -= blob.min()
    blob /= blob.max()
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(blob.shape[0])))
    padding = ((0, n ** 2 - blob.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (blob.ndim - 3)
    blob = np.pad(blob, padding, mode='constant', constant_values=(padval, padval))
    
    # tile the filters into an image
    blob = blob.reshape((n, n) + blob.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, blob.ndim + 1)))
    blob = blob.reshape((n * blob.shape[1], n * blob.shape[3]) + blob.shape[4:])
    return blob
