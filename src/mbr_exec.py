from .utils import execute_testcase

class MBRExec:
    def __init__(self, programs, tests):
        self.reranked_programs = self.select_programs(programs, tests)
       
    def select_programs(self, programs, tests):
        mbr_scores = [0 for _ in range(len(programs))]
        value_matrix = [[None for _ in range(len(programs))] for _ in range(len(tests))]
        for i in range(len(tests)):
            for j in range(len(programs)):
                exec_res = execute_testcase(programs[j], tests[i])
                value_matrix[i][j] = exec_res
        
        for i in range(len(programs)):
            for j in range(len(programs)):
                if i != j:
                    for t in range(len(tests)):
                        exec_i = value_matrix[t][i]
                        exec_j = value_matrix[t][j]
                        if exec_i != exec_j:
                            mbr_scores[i] += 1
                            break
        
        program_scores = list(zip(programs, mbr_scores))
        program_scores.sort(key = lambda x : x[1])
        self.program_scores = program_scores
        return [program_score[0] for program_score in program_scores]


if __name__ == '__main__':
    p1 = "def count(s): return len([c for c in s if c.islower()])"
    p2 = """
def count(string):
    cnt = 0
    for ch in string:
        if ch.islower():
            cnt += 1
    return cnt
"""
    p3 = "def count(s): return len(s)"

    t = "assert count(\"abc1\") == 3"

    programs = [p1, p2, p3]
    tests = [t]


    mbr = MBRExec(programs, tests)
    print(mbr.program_scores)
