def format_prompt_informative_1shot(prompt):
    prompt_template_informative = """
Please refer to the given examples to generate test cases for my problem.
Examples are listed as follows:

>>> Problem
def how_many_times(string: str, substring: str) -> int:
    \"\"\" 
    Find how many times a given substring can be found in the original string. Count overlaping cases.
    \"\"\"
>>> Test Cases
assert how_many_times("a bb ccc dddd", "dd") == 3
assert how_many_times("aaaa", "aa") == 3
assert how_many_times('bbaaa', 'aa') == 2
assert how_many_times('AAAA', 'AA') == 3
assert how_many_times("abcxabc", "abc") == 2

Here is my problem:

>>> Problem:
Write a function with the following description:
{}
>>> Test Cases:
""".format(prompt)
    return prompt_template_informative

def format_prompt_informative_3shot(prompt):
    prompt_template_informative = """
Please refer to the given examples to generate test cases for my problem.
Examples are listed as follows:

>>> Problem:
def longest(strings: List[str]) -> Optional[str]:
    \"\"\" 
    Out of list of strings, return the longest one. 
    Return the first one in case of multiple strings of the same length. 
    Return None in case the input list is empty.
    \"\"\" 
>>> Test Cases:
assert longest(['a', 'abc', 'def']) == 'abc'
assert longest(['hi', 'hello', 'howdy', 'hi']) == 'hello'

>>> Problem:
def is_prime(num):
    \"\"\" 
    Returns true if a given number is prime, and false otherwise.
    \"\"\" 
>>> Test Cases:
assert(is_prime(1) == False)
assert is_prime(2) == True
assert is_prime(7) == True

>>> Problem
def how_many_times(string: str, substring: str) -> int:
    \"\"\" 
    Find how many times a given substring can be found in the original string. Count overlaping cases.
    \"\"\"
>>> Test Cases
assert how_many_times("a bb ccc dddd", "dd") == 3
assert how_many_times("aaaa", "aa") == 3
assert how_many_times('bbaaa', 'aa') == 2
assert how_many_times('AAAA', 'AA') == 3
assert how_many_times("abcxabc", "abc") == 2

Here is my problem:

>>> Problem:
Write a function with the following description:
{}
>>> Test Cases:
""".format(prompt)
    return prompt_template_informative


def format_prompt_informative_5shot(prompt):
    prompt_template_informative = """
Please refer to the given examples to generate test cases for my problem.
Examples are listed as follows:

>>> Problem:
def fix_spaces(text):
    \"\"\" 
    Given a string text, replace all spaces in it with underscores, 
    and if a string has more than 2 consecutive spaces, 
    then replace all consecutive spaces with - 
     \"\"\"  
>>> Test Cases
assert fix_spaces("This   is    a     test     case") == "This-is-a-test-case"
assert fix_spaces("Greetings     from      Earth") == "Greetings-from-Earth"
assert(fix_spaces("Hi, my name is     John") == "Hi,_my_name_is-John")
assert fix_spaces("hello    there") == "hello-there"
assert fix_spaces('this is a string with    space') == 'this_is_a_string_with-space'

>>> Problem
def below_threshold(l: list, t: int):
    \"\"\"
    Return True if all numbers in the list l are below threshold t.
    \"\"\"
>>> Test Cases
assert below_threshold([1, 2, 3], 3) == False
assert below_threshold([1, 2, 3, 4], 4) == False
assert below_threshold([2, 5, 8], 8) == False
assert below_threshold([1, 2, 3, 4, 5], 5) == False
assert below_threshold([2, 3], 3) == False

>>> Problem:
def longest(strings: List[str]) -> Optional[str]:
    \"\"\" 
    Out of list of strings, return the longest one. 
    Return the first one in case of multiple strings of the same length. 
    Return None in case the input list is empty.
    \"\"\" 
>>> Test Cases:
assert longest(['a', 'abc', 'def']) == 'abc'
assert longest(['hi', 'hello', 'howdy', 'hi']) == 'hello'

>>> Problem:
def is_prime(num):
    \"\"\" 
    Returns true if a given number is prime, and false otherwise.
    \"\"\" 
>>> Test Cases:
assert(is_prime(1) == False)
assert is_prime(2) == True
assert is_prime(7) == True

>>> Problem
def how_many_times(string: str, substring: str) -> int:
    \"\"\" 
    Find how many times a given substring can be found in the original string. Count overlaping cases.
    \"\"\"
>>> Test Cases
assert how_many_times("a bb ccc dddd", "dd") == 3
assert how_many_times("aaaa", "aa") == 3
assert how_many_times('bbaaa', 'aa') == 2
assert how_many_times('AAAA', 'AA') == 3
assert how_many_times("abcxabc", "abc") == 2

Here is my problem:

>>> Problem:
Write a function with the following description:
{}
>>> Test Cases:
""".format(prompt)
    return prompt_template_informative

def format_mbpp_prompt_informative_5shot(prompt):
    prompt_template_random = """
Please refer to the given examples to generate test cases for my function.

Examples are listed as follows:

>>> Problem:
def digit_distance_nums(n1, n2):
    \"\"\"
    Write a python function to find the digit distance between two integers.
    \"\"\"
>>> Test Cases:
assert(digit_distance_nums(12345, 1234) == 5)
assert(digit_distance_nums(23, 43) == 2)
assert(digit_distance_nums(12,34) == 4)
assert(digit_distance_nums(100, 120) == 2)
assert(digit_distance_nums(1234567, 1234587) == 2)

>>> Problem:
import sys
def next_smallest_palindrome(num):
    \"\"\"
    Write a function to find the next smallest palindrome of a specified number.
    \"\"\"
>>> Test Cases
assert(next_smallest_palindrome(9) == 11)
assert(next_smallest_palindrome(1) == 2)
assert(next_smallest_palindrome(999999) == 1000001)
assert(next_smallest_palindrome(101) == 111)
assert(next_smallest_palindrome(5) == 6)

>>> Problem:
def count_occurance(s):
    \"\"\"
    Write a function to find the occurence of characters 'std' in the given string
    \"\"\"
>>> Test Cases
assert(count_occurance("stdhare code std")== 2)
assert(count_occurance("std std std std") == 4)
assert(count_occurance("std, std") == 2)
assert(count_occurance('stdstd') == 2)
assert(count_occurance("std") == 1)

>>> Problem:
def max_subarray_product(arr):
    \"\"\"
    Write a function to find the maximum product subarray of the given array.
    \"\"\"
>>> Test Cases
assert(max_subarray_product([-1, -2, -3]) == 6)
assert(max_subarray_product([-1,-2,1,-3]) == 6)
assert(max_subarray_product([]) == 0)
assert(max_subarray_product([-1,2,3,4]) == 24)
assert(max_subarray_product([1, -2, 3, -4]) == 24)

>>> Problem
def replace_char(str1,ch,newch):
    \"\"\"
    Write a function to replace characters in a string.
    \"\"\"
>>> Test Cases
assert(replace_char("","","") == "")
assert(replace_char("hello","l","k")=="hekko")
assert(replace_char("hello","l","-") == "he--o")
assert(replace_char('aaaaa','a','b')=='bbbbb')
assert(replace_char('hello','l','!')=='he!!o')

Here is my problem:

>>> Problem:
{}
>>> Test Cases:
""".format(prompt)
    return prompt_template_random

def format_prompt_informative_10shot(prompt):
    prompt_template_informative = """
Please refer to the given examples to generate test cases for my problem.
Examples are listed as follows:

>>> Problem:
def find_max(words):
    \"\"\"
    Write a function that accepts a list of strings.
    The list contains different words. Return the word with maximum number
    of unique characters. If multiple strings have maximum number of unique
    characters, return the one which comes first in lexicographical order.
    \"\"\"
>>> Test Cases:
assert find_max(["e", "a", "o", "k", "a", "f", "m", "a", "d", "r", "a"]) == "a"
assert find_max(['dog', 'cat']) == 'cat'
assert find_max(['flower', 'flow', 'flight', 'fli']) == 'flight'
assert find_max(['cat', 'car', 'dog', 'tac', 'bass']) == 'bass'
assert find_max(['apple', 'apple', 'apple', 'banana', 'apple']) == 'apple'

>>> Problem:
def simplify(x, n):
    \"\"\"
    Your task is to implement a function that will simplify the expression
    x * n. The function returns True if x * n evaluates to a whole number and False
    otherwise. Both x and n, are string representation of a fraction, and have the following format,
    <numerator>/<denominator> where both numerator and denominator are positive whole numbers.

    You can assume that x, and n are valid fractions, and do not have zero as denominator.
    \"\"\"
>>> Test Cases
assert(simplify("1/2", "2/4") == False)
assert simplify('1/2', '2/1') == True
assert simplify("2/4", "3/6") == False
assert simplify('2/3', '3/3') == False
assert simplify("2/1", "1/1") == True

>>> Problem:
def triangle_area(a, b, c):
    \"\"\"
    Given the lengths of the three sides of a triangle. Return the area of
    the triangle rounded to 2 decimal points if the three sides form a valid triangle. 
    Otherwise return -1
    Three sides make a valid triangle when the sum of any two sides is greater 
    than the third side.
    \"\"\"
>>> Test Cases
assert triangle_area(1, 1, 1) == 0.43
assert triangle_area(5, 3, 4) == 6
assert triangle_area(3, 4, 5) == 6
assert triangle_area(1, 2, 1) == -1

>>> Problem:
def same_chars(s0: str, s1: str):
    \"\"\"
    Check if two words have the same characters.
    \"\"\"
>>> Test Cases:
assert same_chars('hello', 'lllohe') == True
assert same_chars("", "1") == False
assert same_chars("monkey", "monkey2") == False
assert same_chars("", "hi") == False
assert same_chars("hello", "olleh") == True

>>> Problem:
def fix_spaces(text):
    \"\"\" 
    Given a string text, replace all spaces in it with underscores, 
    and if a string has more than 2 consecutive spaces, 
    then replace all consecutive spaces with - 
     \"\"\"  
>>> Test Cases
assert fix_spaces("This   is    a     test     case") == "This-is-a-test-case"
assert fix_spaces("Greetings     from      Earth") == "Greetings-from-Earth"
assert(fix_spaces("Hi, my name is     John") == "Hi,_my_name_is-John")
assert fix_spaces("hello    there") == "hello-there"
assert fix_spaces('this is a string with    space') == 'this_is_a_string_with-space'

>>> Problem
def below_threshold(l: list, t: int):
    \"\"\"
    Return True if all numbers in the list l are below threshold t.
    \"\"\"
>>> Test Cases
assert below_threshold([1, 2, 3], 3) == False
assert below_threshold([1, 2, 3, 4], 4) == False
assert below_threshold([2, 5, 8], 8) == False
assert below_threshold([1, 2, 3, 4, 5], 5) == False
assert below_threshold([2, 3], 3) == False

>>> Problem:
def longest(strings: List[str]) -> Optional[str]:
    \"\"\" 
    Out of list of strings, return the longest one. 
    Return the first one in case of multiple strings of the same length. 
    Return None in case the input list is empty.
    \"\"\" 
>>> Test Cases:
assert longest(['a', 'abc', 'def']) == 'abc'
assert longest(['hi', 'hello', 'howdy', 'hi']) == 'hello'

>>> Problem:
def is_prime(num):
    \"\"\" 
    Returns true if a given number is prime, and false otherwise.
    \"\"\" 
>>> Test Cases:
assert(is_prime(1) == False)
assert is_prime(2) == True
assert is_prime(7) == True

>>> Problem
def how_many_times(string: str, substring: str) -> int:
    \"\"\" 
    Find how many times a given substring can be found in the original string. Count overlaping cases.
    \"\"\"
>>> Test Cases
assert how_many_times("a bb ccc dddd", "dd") == 3
assert how_many_times("aaaa", "aa") == 3
assert how_many_times('bbaaa', 'aa') == 2
assert how_many_times('AAAA', 'AA') == 3
assert how_many_times("abcxabc", "abc") == 2

>>> Problem
def fruit_distribution(s,n):
    \"\"\"
    In this task, you will be given a string that represents a number of apples and oranges 
    that are distributed in a basket of fruit this basket contains 
    apples, oranges, and mango fruits. Given the string that represents the total number of 
    the oranges and apples and an integer that represent the total number of the fruits 
    in the basket return the number of the mango fruits in the basket.
    \"\"\"
>>> Test Cases
assert fruit_distribution('2 apples, 4 oranges and 1 mango', 10) == 3
assert fruit_distribution("1, 2, 3", 3) == 0
assert fruit_distribution('1 1 1', 3) == 0
assert fruit_distribution("8 2",12) == 2
assert fruit_distribution("1 2 0 0",7) == 4

Here is my problem:

>>> Problem:
Write a function with the following description:
{}
>>> Test Cases:
""".format(prompt)
    return prompt_template_informative

def format_prompt_random_1shot(prompt):
    prompt_template_random = """
Please refer to the given examples to generate test cases for my function.

Examples are listed as follows:

>>> Problem
def how_many_times(string: str, substring: str) -> int:
    \"\"\" 
    Find how many times a given substring can be found in the original string. Count overlaping cases.
    \"\"\"
>>> Test Cases
assert how_many_times('mississippi', 'i') == 4
assert how_many_times("hello", "lo") == 1
assert how_many_times("aa", "a") == 2
assert how_many_times("abcdef", "xyz") == 0
assert how_many_times("abcabc", "abc") == 2

Here is my problem:

>>> Problem:
{}
>>> Test Cases:
""".format(prompt)
    return prompt_template_random

def format_prompt_random_3shot(prompt):
    prompt_template_random = """
Please refer to the given examples to generate test cases for my function.

Examples are listed as follows:

>>> Problem:
def longest(strings: List[str]) -> Optional[str]:
    \"\"\" 
    Out of list of strings, return the longest one. 
    Return the first one in case of multiple strings of the same length. 
    Return None in case the input list is empty.
    \"\"\" 
>>> Test Cases
assert longest(['a', 'ab', 'abc']) == 'abc'
assert longest(['a', 'bb', 'c']) == 'bb'

>>> Problem:
def is_prime(num):
    \"\"\" 
    Returns true if a given number is prime, and false otherwise.
    \"\"\" 
>>> Test Cases:
assert(is_prime(10) == False)
assert is_prime(12) == False
assert is_prime(6) == False

>>> Problem
def how_many_times(string: str, substring: str) -> int:
    \"\"\" 
    Find how many times a given substring can be found in the original string. Count overlaping cases.
    \"\"\"
>>> Test Cases
assert how_many_times('mississippi', 'i') == 4
assert how_many_times("hello", "lo") == 1
assert how_many_times("aa", "a") == 2
assert how_many_times("abcdef", "xyz") == 0
assert how_many_times("abcabc", "abc") == 2

Here is my problem:

>>> Problem:
{}
>>> Test Cases:
""".format(prompt)
    return prompt_template_random

def format_prompt_random_5shot(prompt):
    prompt_template_random = """
Please refer to the given examples to generate test cases for my function.

Examples are listed as follows:

>>> Problem:
def fix_spaces(text):
    \"\"\" 
    Given a string text, replace all spaces in it with underscores, 
    and if a string has more than 2 consecutive spaces, 
    then replace all consecutive spaces with - 
     \"\"\" 
>>> Test Cases
assert fix_spaces('hello world ') == 'hello_world_'
assert fix_spaces('  ') == '_'
assert fix_spaces("Greetings     from      Earth") == "Greetings-from-Earth"
assert fix_spaces("hello world from coding interview") == "hello_world_from_coding_interview"
assert fix_spaces("hello    world") == "hello-world"

>>> Problem
def below_threshold(l: list, t: int):
    \"\"\"
    Return True if all numbers in the list l are below threshold t.
    \"\"\"
>>> Test Cases
assert below_threshold([-1, -2, -3], -2) == False
assert below_threshold([10, 11], 10) == False
assert below_threshold([5, 6, 7], 4) == False
assert below_threshold([20, 30, 40], 25) == False
assert below_threshold([1, 2, 3], 3) == False

>>> Problem:
def longest(strings: List[str]) -> Optional[str]:
    \"\"\" 
    Out of list of strings, return the longest one. 
    Return the first one in case of multiple strings of the same length. 
    Return None in case the input list is empty.
    \"\"\" 
>>> Test Cases
assert longest(['a', 'ab', 'abc']) == 'abc'
assert longest(['a', 'bb', 'c']) == 'bb'

>>> Problem:
def is_prime(num):
    \"\"\" 
    Returns true if a given number is prime, and false otherwise.
    \"\"\" 
>>> Test Cases:
assert(is_prime(10) == False)
assert is_prime(12) == False
assert is_prime(6) == False

>>> Problem
def how_many_times(string: str, substring: str) -> int:
    \"\"\" 
    Find how many times a given substring can be found in the original string. Count overlaping cases.
    \"\"\"
>>> Test Cases
assert how_many_times('mississippi', 'i') == 4
assert how_many_times("hello", "lo") == 1
assert how_many_times("aa", "a") == 2
assert how_many_times("abcdef", "xyz") == 0
assert how_many_times("abcabc", "abc") == 2

Here is my problem:

>>> Problem:
{}
>>> Test Cases:
""".format(prompt)
    return prompt_template_random

def format_mbpp_prompt_random_5shot(prompt):
    prompt_template_random = """
Please refer to the given examples to generate test cases for my function.

Examples are listed as follows:

>>> Problem:
def digit_distance_nums(n1, n2):
    \"\"\"
    Write a python function to find the digit distance between two integers.
    \"\"\"
>>> Test Cases:
assert(digit_distance_nums(5, 5) == 0)
assert(digit_distance_nums(55, 56) == 1)
assert(digit_distance_nums(3, 3) == 0)
assert(digit_distance_nums(123, 123) == 0)
assert(digit_distance_nums(1234567, 1234587) == 2)

>>> Problem:
import sys
def next_smallest_palindrome(num):
    \"\"\"
    Write a function to find the next smallest palindrome of a specified number.
    \"\"\"
>>> Test Cases
assert(next_smallest_palindrome(145) == 151)
assert(next_smallest_palindrome(8) == 9)
assert(next_smallest_palindrome(100)==101)
assert(next_smallest_palindrome(4321) == 4334)
assert(next_smallest_palindrome(100) == 101)

>>> Problem:
def count_occurance(s):
    \"\"\"
    Write a function to find the occurence of characters 'std' in the given string
    \"\"\"
>>> Test Cases
assert(count_occurance("std, std") == 2)
assert(count_occurance("string")==0)
assert(count_occurance("stdabcstdxyzstdijk") == 3)
assert(count_occurance("dstd") == 1)
assert(count_occurance("stdc++") == 1)

>>> Problem:
def max_subarray_product(arr):
    \"\"\"
    Write a function to find the maximum product subarray of the given array.
    \"\"\"
>>> Test Cases
assert(max_subarray_product([1,2,3,4]) == 24)
assert(max_subarray_product([1, -2, 3, -4]) == 24)
assert(max_subarray_product([1, 2, 3, 4, 5]) == 120)
assert(max_subarray_product([-1,-2,1,-3]) == 6)
assert(max_subarray_product([10, -10, 1, -1, 100]) == 10000)

>>> Problem
def replace_char(str1,ch,newch):
    \"\"\"
    Write a function to replace characters in a string.
    \"\"\"
>>> Test Cases
assert(replace_char("hello", "l", "w") == "hewwo")
assert(replace_char("abcd","a","A")=="Abcd")
assert(replace_char("hello","l","L") == "heLLo")
assert(replace_char('abcd','c','d') == 'abdd')
assert(replace_char('hello','h','j') == 'jello')

Here is my problem:

>>> Problem:
{}
>>> Test Cases:
""".format(prompt)
    return prompt_template_random

def format_prompt_random_10shot(prompt):
    prompt_template_random = """
Please refer to the given examples to generate test cases for my function.

Examples are listed as follows:

>>> Problem:
def find_max(words):
    \"\"\"
    Write a function that accepts a list of strings.
    The list contains different words. Return the word with maximum number
    of unique characters. If multiple strings have maximum number of unique
    characters, return the one which comes first in lexicographical order.
    \"\"\"
>>> Test Cases:
assert find_max(['a', 'a', 'a', 'a']) == 'a'
assert find_max(['two', 'score', 'and', 'seven']) == 'score'
assert find_max(['hello', 'world']) == 'world'
assert find_max(['code', 'test']) == 'code'
assert find_max(['abcd','abc','xyz','123456']) == '123456'

>>> Problem:
def simplify(x, n):
    \"\"\"
    Your task is to implement a function that will simplify the expression
    x * n. The function returns True if x * n evaluates to a whole number and False
    otherwise. Both x and n, are string representation of a fraction, and have the following format,
    <numerator>/<denominator> where both numerator and denominator are positive whole numbers.

    You can assume that x, and n are valid fractions, and do not have zero as denominator.
    \"\"\"
>>> Test Cases
assert simplify("1/3", "2/4") == False
assert simplify("2/1", "1/4") == False
assert(simplify("1/3", "1/4") == False)
assert simplify('1/2', '4/3') == False
assert simplify("1/2", "5/8") == False

>>> Problem:
def triangle_area(a, b, c):
    \"\"\"
    Given the lengths of the three sides of a triangle. Return the area of
    the triangle rounded to 2 decimal points if the three sides form a valid triangle. 
    Otherwise return -1
    Three sides make a valid triangle when the sum of any two sides is greater 
    than the third side.
    \"\"\"
>>> Test Cases
assert triangle_area(1, 1, 3) == -1
assert triangle_area(1, 2, 4) == -1
assert triangle_area(1, 2, 3) == -1
assert triangle_area(3, 4, 7) == -1
assert triangle_area(1, 2, 1) == -1

>>> Problem:
def same_chars(s0: str, s1: str):
    \"\"\"
    Check if two words have the same characters.
    \"\"\"
>>> Test Cases:
assert same_chars("", "1") == False
assert same_chars("abcd", "bad") == False
assert same_chars('is love', 'love is') == True
assert same_chars('sponge', 'sponge') == True
assert same_chars('ab', 'cbb') == False

>>> Problem:
def fix_spaces(text):
    \"\"\" 
    Given a string text, replace all spaces in it with underscores, 
    and if a string has more than 2 consecutive spaces, 
    then replace all consecutive spaces with - 
     \"\"\" 
>>> Test Cases:
assert fix_spaces('hello world ') == 'hello_world_'
assert fix_spaces('  ') == '_'
assert fix_spaces("Greetings     from      Earth") == "Greetings-from-Earth"
assert fix_spaces("hello world from coding interview") == "hello_world_from_coding_interview"
assert fix_spaces("hello    world") == "hello-world"

>>> Problem:
def below_threshold(l: list, t: int):
    \"\"\"
    Return True if all numbers in the list l are below threshold t.
    \"\"\"
>>> Test Cases:
assert below_threshold([-1, -2, -3], -2) == False
assert below_threshold([10, 11], 10) == False
assert below_threshold([5, 6, 7], 4) == False
assert below_threshold([20, 30, 40], 25) == False
assert below_threshold([1, 2, 3], 3) == False

>>> Problem:
def longest(strings: List[str]) -> Optional[str]:
    \"\"\" 
    Out of list of strings, return the longest one. 
    Return the first one in case of multiple strings of the same length. 
    Return None in case the input list is empty.
    \"\"\" 
>>> Test Cases:
assert longest(['a', 'ab', 'abc']) == 'abc'
assert longest(['a', 'bb', 'c']) == 'bb'

>>> Problem:
def is_prime(num):
    \"\"\" 
    Returns true if a given number is prime, and false otherwise.
    \"\"\" 
>>> Test Cases:
assert(is_prime(10) == False)
assert is_prime(12) == False
assert is_prime(6) == False

>>> Problem:
def how_many_times(string: str, substring: str) -> int:
    \"\"\" 
    Find how many times a given substring can be found in the original string. Count overlaping cases.
    \"\"\"
>>> Test Cases:
assert how_many_times('mississippi', 'i') == 4
assert how_many_times("hello", "lo") == 1
assert how_many_times("aa", "a") == 2
assert how_many_times("abcdef", "xyz") == 0
assert how_many_times("abcabc", "abc") == 2

>>> Problem:
def fruit_distribution(s,n):
    \"\"\"
    In this task, you will be given a string that represents a number of apples and oranges 
    that are distributed in a basket of fruit this basket contains 
    apples, oranges, and mango fruits. Given the string that represents the total number of 
    the oranges and apples and an integer that represent the total number of the fruits 
    in the basket return the number of the mango fruits in the basket.
    \"\"\"
>>> Test Cases:
assert fruit_distribution("1 2 0 0",7) == 4
assert fruit_distribution("200 200", 400) == 0
assert fruit_distribution("1 1", 2) == 0
assert fruit_distribution('1 1 1', 3) == 0
assert fruit_distribution("1, 2, 3", 3) == 0

Here is my problem:

>>> Problem:
{}
>>> Test Cases:
""".format(prompt)
    return prompt_template_random