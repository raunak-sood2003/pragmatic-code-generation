import signal
import re
import random
import numpy as np
import time
import os
import sys

"""
Timer for executed programs. Waits a specified number of seconds 
after execution before singaling a TimeOut error.
"""
class timeout:
    def __init__(self, seconds=0.1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    
    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)
    
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        # signal.alarm(self.seconds)
        signal.setitimer(signal.ITIMER_REAL, self.seconds)
    
    def __exit__(self, type, value, traceback):
        # signal.alarm(0)
        signal.setitimer(signal.ITIMER_REAL, 0)


def extract_function_name(code_string):
    '''
    Extracts the name of a function from a string.

    code_string: (string) String that contains code for a function
    returns: the name of the function
    '''
    pattern = r'def\s+(\w+)\s*\('
    match = re.search(pattern, code_string)
    if match:
        return match.group(1)  # The captured function name
    else:
        return None


def evaluate_function(func_string, *args, **kwargs):
    '''
    Evaluates a function on the specified parameters.

    func_string: a string that contains code for a function
    *args/*kwargs: contains the arguments to the function
    returns: the function evaluated on the arguments passed in
    '''
    namespace = {}
    
    exec(func_string, globals(), namespace)

    func_name = extract_function_name(func_string)
    func = namespace.get(func_name)
    
    res = None
    if func is not None and callable(func):
        try:
            res = func(*args, **kwargs)
        except:
            res = None
    
    return res

def extract_function(string):
    defs = [m.start() for m in re.finditer('def ', string)]
    if len(defs) > 1:
        return string[:defs[1]].strip()
    return string.strip()

def extract_testcase(text):
    # Regular expression pattern to match 'assert x == y' or 'assert(x == y)'
    
    pattern1 = r'assert\s+\(?[^\n]*\s*==\s*[^\n]*\)?'
    pattern2 = r'assert\s*\(.*?=\s*.*?\)'
    
    # Find all matches in the text using the pattern
    matches1 = re.findall(pattern1, text)
    matches2 = re.findall(pattern2, text)

    if len(matches1) > 0:
        return matches1[0]
    elif len(matches2) > 0:
        return matches2[0]
    else:
        return ""

def valid_program_testcase_pair(x, y):
    '''
    x: a sample program hypothetical inlier
    y: a sample test case hypothetical inlier
    returns: True if the sample program passed the test case, False if not
    '''
    
    if len(x) == 0 or len(y) == 0:
        return False
    
    executable = x + "\n" + y
    
    old_stdout = sys.stdout
    try:
        sys.stdout = open(os.devnull, "w")
        with timeout():
            exec(executable)
        sys.stdout = old_stdout
        return True

    except:
        sys.stdout = old_stdout
        return False

def subsample_matrix(matrix, n, m):
    '''
    Randomly samples a submatrix of size (n, m).
    '''
    assert(n <= len(matrix))
    assert(m <= len(matrix[0]))
    
    row = random.randint(0, len(matrix) - n)
    col = random.randint(0, len(matrix[0]) - m)
    return matrix[row : row + n, col : col + m], row, col

def verify_const_matrix(programs, tests, const_matrix):
    '''
    Checks the correctness of the consistency matrix. 
    programs[j] is consistent with tests[i] <=> const_matrix[i][j] == 1
    '''
    for i in range(len(tests)):
        for j in range(len(programs)):
            is_const = valid_program_testcase_pair(programs[j], tests[i])
            is_const_mat = const_matrix[i][j] == 1
            if is_const != is_const_mat:
                print("****FAILED**** (i, j) = (%d, %d)" % (i, j))
                print("Program:")
                print(programs[i])
                print("Test:")
                print(tests[i])
                print("True: %d" % is_const)
                print("Const mat: %d" % is_const_mat)
                return False
    return True