def corrupt_binary_data(data, avg_bits_corrupted=None, bits_corrupted=None):
    ndim = data.ndim
    data = np.atleast_2d(data)
    size, n = data.shape

    if avg_bits_corrupted is not None:
        p = 1.0*avg_bits_corrupted/n  # prob bit flipped
        which_to_flip = np.random.rand(size,n) < p
    elif bits_corrupted is not None:
        which_to_flip = np.random.rand(size, n).argsort(1) < bits_corrupted
    else:
        raise ValueError('either bits_corrupted or avg_bits_corrupted option has to be given')

    out = data.copy()
    out[which_to_flip] *= -1
    out[which_to_flip] += 1

    if ndim == 1: return out[0]
    return out
