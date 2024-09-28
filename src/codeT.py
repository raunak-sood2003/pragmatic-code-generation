import random
from .utils import valid_program_testcase_pair

class CodeT:
    """
    CodeT implementation based on this paper: https://arxiv.org/abs/2207.10397. 
    Assumes that programs and tests are executable when concatenated (i.e x + '\n' y).
    Uses the RANSAC algorithm to randomly sample and re-rank programs based on 
    dual execution agreement with test cases.
    """
    def __init__(self, X, Y, n, k, const_matrix = None):
        """
        X: (List[string]) list of sampled programs
        Y: (List[string]) list of sampled test cases
        n: (int) how many samples to select
        k: top k of these generated programs (k <= n)
        const_matrix: (np.ndarray optional) Program/Test consistency matrix
        """
        self.save_clusters = []
        try:            
            if const_matrix is not None:
                assert(len(Y) == const_matrix.shape[0])
                assert(len(X) == const_matrix.shape[1])
                self.programs = self.dual_execution_agreement_from_const_matrix(X, Y, n, k, const_matrix)
                # print("Did CodeT from Consistency Matrix")
            else:
                self.programs = self.dual_execution_agreement(X, Y, n, k)
                # print("Did CodeT From Scratch")
        except:
            self.programs = X[:k]
            # print("Did Not Do CodeT")

    
    def dual_execution_agreement_from_const_matrix(self, X, Y, n, k, const_matrix):
        '''
        X: (List[string]) list of programs
        Y: (List[tuple]) test cases
        k: (int) the number of times we repeat the selection process
        returns: (S, score) where S is the consensus set with the highest 
                score based on random sampling over k iterations
        const_matrix: (np.ndarray) consistency matrix between programs in X and tests in Y
        '''
        
        patience = 2 * n
        D = [(x, y) for y in Y for x in X]
        ctr = 0
        num_iters = 0
        clusters = []
        save_clusters = []

        while ctr < n and num_iters < patience:
            idx = random.randint(0, len(D) - 1)
            x, y = D[idx]
            x_idx, y_idx = X.index(x), Y.index(y)
            if const_matrix[y_idx][x_idx]:
                S_x, S_y = self.get_groups_from_const_matrix(X, Y, x, const_matrix)
                score = len(S_x) * len(S_y)
                clusters.append((S_x, score))
                save_clusters.append((S_x, S_y, score))
                ctr += 1
            num_iters += 1
        
        if len(clusters) < 1:
            raise Exception("No matching programs and test \
                            cases were found with given patience.")
        
        clusters.sort(key = lambda x : x[1], reverse = True)
        save_clusters.sort(key = lambda x : x[2], reverse = True)
        self.save_clusters = save_clusters
    
        ctr = 0
        n_clusters = len(clusters)
        res = []
        while ctr < k:
            cluster_idx = ctr % n_clusters
            S_x = clusters[cluster_idx][0]
            program_idx = random.randint(0, len(S_x) - 1)
            res.append(S_x[program_idx])
            ctr += 1
        
        return res
    
    def dual_execution_agreement(self, X, Y, n, k):
        '''
        X: (List[string]) list of programs
        Y: (List[tuple]) test cases
        k: (int) the number of times we repeat the selection process
        returns: (S, score) where S is the consensus set with the highest 
                score based on random sampling over k iterations
        '''
        
        patience = 2 * n
        D = [(x, y) for y in Y for x in X]
        ctr = 0
        num_iters = 0
        clusters = []

        while ctr < n and num_iters < patience:
            idx = random.randint(0, len(D) - 1)
            x, y = D[idx]
            if valid_program_testcase_pair(x, y):
                S_x, S_y = self.get_groups(X, Y, x)
                score = len(S_x) * len(S_y)
                clusters.append((S_x, score))
                ctr += 1
            num_iters += 1
        
        if len(clusters) < 1:
            raise Exception("No matching programs and test \
                            cases were found with given patience.")
        
        clusters.sort(key = lambda x : x[1], reverse = True)
    
        ctr = 0
        n_clusters = len(clusters)
        res = []
        while ctr < k:
            cluster_idx = ctr % n_clusters
            S_x = clusters[cluster_idx][0]
            program_idx = random.randint(0, len(S_x) - 1)
            res.append(S_x[program_idx])
            ctr += 1
        
        return res

    
    def get_groups(self, X, Y, x):
        '''
        X: (List[String]) set of programs
        Y: (List[tuple]) set of test cases
        x: a sample program hypothetical inlier
        returns: (S, score) where S is the consensus set of the program x 
                and score is the score given to that consensus set
        '''
        
        S_y = []
        for y in Y:
            if valid_program_testcase_pair(x, y):
                S_y.append(y)
        
        S_x = []
        for program in X:
            matching_program = True
            for y in S_y:
                if (not valid_program_testcase_pair(program, y)):
                    matching_program = False
                    break
            if matching_program:
                S_x.append(program)
        
        return S_x, S_y

    def get_groups_from_const_matrix(self, X, Y, x, const_matrix):
        '''
        X: (List[String]) set of programs
        Y: (List[tuple]) set of test cases
        x: a sample program hypothetical inlier
        returns: (S, score) where S is the consensus set of the program x 
                and score is the score given to that consensus set
        '''
        
        S_y = []
        x_idx = X.index(x)
        for y in Y:
            y_idx = Y.index(y)
            if const_matrix[y_idx][x_idx]:
                S_y.append(y)
        
        S_x = []
        for program in X:
            program_idx = X.index(program)
            matching_program = True
            for y in S_y:
                y_idx = Y.index(y)
                if (not const_matrix[y_idx][program_idx]):
                    matching_program = False
                    break
            if matching_program:
                S_x.append(program)
        
        return S_x, S_y