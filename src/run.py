from typing import List


def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """
    Check if in given list of numbers, are any two numbers closer to each other than given threshold.

    Args:
    numbers (List[float]): A list of float numbers.
    threshold (float): The minimum distance between two numbers to be considered close.

    Returns:
    bool: True if any two numbers in the list are closer than the threshold, False otherwise.
    """

    # Sort the list of numbers
    numbers = sorted(numbers)

    # Iterate over the list, comparing each pair of adjacent numbers
    for i in range(1, len(numbers)):
        # Calculate the distance between the current and the previous number
        distance = numbers[i] - numbers[i - 1]

        # If the distance is less than or equal to the threshold, return True
        if distance <= threshold:
            return True

    # If no pairs of adjacent numbers are closer than the threshold, return False
    return False


def test_no_close_elements():
    # Test case where no two numbers are closer than the threshold
    return has_close_elements([1.0, 2.0, 3.0], 0.5)


# def test_no_close_elements():
#     # Test case where no two numbers are closer than the threshold
#     assert not has_close_elements([1.0, 2.0, 3.0], 0.5)


def test_close_elements():
    # Test case where two numbers are closer than the threshold
    assert has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)


def test_empty_list():
    # Test case with an empty list
    assert not has_close_elements([], 0.5)


def test_single_element_list():
    # Test case with a list containing a single element
    assert not has_close_elements([1.0], 0.5)


def test_threshold_equal_to_difference():
    # Test case where two numbers are exactly threshold distance apart
    assert not has_close_elements([1.0, 2.0], 1.0)


def test_negative_threshold():
    # Test case with a negative threshold
    assert not has_close_elements([1.0, 2.0, 3.0], -0.5)


def test_threshold_zero():
    # Test case with a threshold of zero
    assert has_close_elements([1.0, 2.0], 0.0)


def test_duplicate_elements():
    # Test case with duplicate elements in the list
    assert has_close_elements([1.0, 1.0, 2.0, 3.0], 0.5)


# suite = unittest.TestLoader().loadTestsFromTestCase(TestCloseElements)
# runner = unittest.TextTestRunner(stream=output, verbosity=2)
# result = runner.run(suite)
# locals_dict["result"] = result
