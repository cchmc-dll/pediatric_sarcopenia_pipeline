import numpy as np

from util.iterable import batch_to_ndarray


def test_batch_to_ndarray_should_chunk_1D_array():
    input_array = [1, 2, 3]
    batch_size = 1

    result = [
        x
        for x
        in batch_to_ndarray(
            input_array,
            batch_size=batch_size,
            item_shape=(1,)
        )
    ]

    assert len(result) == 3


def test_batch_to_ndarray_should_handle_2D_items():
    input_array = [
        np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]),
        np.array([
            [10, 11, 12],
            [13, 14, 15],
            [16, 17, 18]
        ])
    ]
    batch_size = 1

    result = [
        x
        for x
        in batch_to_ndarray(
            input_array,
            batch_size=batch_size,
            item_shape=(3, 3)
        )
    ]

    assert len(result) == 2

    for item in result:
        assert item.shape == (batch_size, 3, 3)


def test_batch_to_ndarray_should_handle_shorter_batch_well():
    input_array = [
        np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]),
        np.array([
            [10, 11, 12],
            [13, 14, 15],
            [16, 17, 18]
        ]),
        np.array([
            [20, 21, 22],
            [23, 24, 25],
            [26, 27, 28]
        ])
    ]
    batch_size = 2

    result = [
        x
        for x
        in batch_to_ndarray(
            input_array,
            batch_size=batch_size,
            item_shape=(3, 3)
        )
    ]

    assert len(result) == 2

    assert result[0].shape == (batch_size, 3, 3)
    assert result[1].shape == (1, 3, 3)


def test_batch_to_ndarray_should_handle_empty_array():
    input_array = []
    batch_size = 16

    result = [
        x
        for x
        in batch_to_ndarray(
            input_array,
            batch_size=batch_size,
            item_shape=(1,)
        )
    ]

    assert len(result) == 0
