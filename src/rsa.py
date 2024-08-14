import numpy as np
from .utils import valid_program_testcase_pair
import random

class RSA:
    """
    Implementation of Rational Speech Acts paradigm for pragmatic program selection 
    based on these papers: https://www.science.org/doi/10.1126/science.1218633, 
    https://arxiv.org/abs/2311.05740. We assume programs-test pairs can be checked 
    by executing the concatentation of the two strings (i.e exec(x + "\n" + y) runs).
    """
    def __init__(self, X, Y, top_k):
        """
        X: (List[string]) list of sampled programs
        Y: (List[string]) list of sampled test cases
        top_k: (int) number of programs to output (<= |Y|)
        """

        self.X = X
        self.Y = Y
        self.top_k = top_k

        self.const_matrix = self.create_consistency_matrix()
        
        # Literal listener
        self.l0 = self.normalize_rows(self.const_matrix)
        
        # Pragmatic speaker
        self.s1 = self.normalize_cols(self.l0)
                
        self.tests = self.select_tests()

        # DEBUG
        # print("Consistency Matrix:")
        # print(self.const_matrix)
        # print("L0:")
        # print(self.l0)
        # print("S1:")
        # print(self.s1)
        # print("Selected tests:")
        # print(self.tests)

    def create_consistency_matrix(self):
        """
        Creates the consistency matrix for the programs and test cases.
        Returns a matrix M such that M[i][j] = 1 if the ith test case 
        in Y matches the jth program in X.
        returns: (np.ndarray) consistency matrix
        """
        n, m = len(self.X), len(self.Y)
        res = np.zeros([m, n])

        for i in range(m):
            for j in range(n):
                x, y = self.X[j], self.Y[i]
                if valid_program_testcase_pair(x, y):
                    res[i][j] = 1
        return res

    @staticmethod
    def normalize_rows(matrix):
        """
        Normalizes the rows in the matrix.
        matrix: (np.ndarray) consistency matrix
        returns: (np.ndarray) literal listener distribution
        """
        row_sum = np.sum(matrix, axis = 1, keepdims = True)
        row_norm = matrix / row_sum
        return np.nan_to_num(row_norm)

    @staticmethod
    def normalize_cols(matrix):
        """
        Normalizes the columns in the matrix.
        matrix: (np.ndarray) literal listener distribution
        returns: pragmatic speaker distribution
        """
        col_sum = np.sum(matrix, axis = 0, keepdims = True)
        col_norm = matrix / col_sum
        return np.nan_to_num(col_norm)

    def select_tests(self):
        """
        Selects the top_k best programs from the pragmatic 
        speaker distribution.
        matrix: (np.ndarray) literal listener distribution
        returns: top_k most probable programs
        """
       
        true_tests = self.s1[:, -1]
        true_tests = np.argsort(true_tests)[::-1]
        true_tests = true_tests.tolist()

        res = []
        for idx in true_tests:
            res.append(self.Y[idx])
        
        return res[:self.top_k]



        
