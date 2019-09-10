import numpy as np


def batch_to_ndarray(iterable, batch_size, item_shape, dtype=np.int8):
    def make_output_array():
        return np.ndarray(shape=(batch_size, *item_shape), dtype=dtype)

    output_array = make_output_array()

    # Make sure count is defined for when we reference it after for loop
    count = 0

    for count, item in enumerate(iterable):
        index = count % batch_size
        output_array[index] = item
        if index == batch_size - 1:
            yield output_array
            output_array = make_output_array()

    # Handle last output array
    if count > 0 and index >= 0 and index != batch_size - 1:
        yield output_array[:index + 1]