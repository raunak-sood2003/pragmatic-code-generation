import numpy as np
import json
from ..src.rsa import RSA
from ..src.utils import write_jsonl
from typing import Callable
from const_matrix import create_const_matrix
from agent import Speaker, Listener

class InferenceSpeakerListener:
    def __init__(self, 
                 speaker : Speaker, 
                 listener : Listener, 
                 num_generations : int,
                 program_prompt_template : str = None,
                 test_prompt_template : str = None,
                 extract_program_fn : Callable = None, 
                 extract_testcase_fn : Callable = None
                 ):
        self.speaker = speaker
        self.listener = listener
        self.num_generations = num_generations
        
        if extract_program_fn is None:
            self.extract_program_fn = lambda x : x
        else:
            self.extract_program_fn = extract_program_fn
        
        if extract_testcase_fn is None:
            self.extract_testcase_fn = lambda x : x
        else:
            self.extract_testcase_fn = extract_testcase_fn

        if program_prompt_template is None:
            self.program_prompt_template = "{}"
        else:
            self.program_prompt_template = program_prompt_template
        
        if test_prompt_template is None:
            self.test_prompt_template = "{}"
        else:
            self.test_prompt_template = test_prompt_template
    
    def generate_const_matrix(self, program_ctx, testcase_ctx, true_program, pragmatic_testcases = None):
        program_prompt = self.program_prompt_template.format(program_ctx)
        test_prompt = self.test_prompt_template.format(testcase_ctx)
        
        listener_outputs = self.listener.generate(program_prompt, self.num_generations)        
        speaker_outputs = self.speaker.generate(test_prompt, self.num_generations)
        
        programs = [self.extract_program_fn(output) for output in listener_outputs]
        tests = [self.extract_testcase_fn(output) for output in speaker_outputs]

        gen_programs, gen_tests = programs.copy(), tests.copy()
        
        programs.append(true_program)
        if pragmatic_testcases is not None:
            tests.extend(pragmatic_testcases)
        return gen_programs, gen_tests, create_const_matrix(programs, tests)

    def generate_next_pragmatic_testcase(self, program_ctx, testcase_ctx, true_program, pragmatic_testcases):
        gen_programs, gen_tests, const_matrix = self.generate_const_matrix(program_ctx, testcase_ctx, true_program, pragmatic_testcases)
        len_prag_tests = len(pragmatic_testcases)
        if len_prag_tests > 0:
            # Auto-regressive step
            prag_tests_const = const_matrix[-len_prag_tests:, :] == 1
            all_prag_tests_const = np.logical_and.reduce(prag_tests_const)
            const_matrix = const_matrix[:, all_prag_tests_const]
        if const_matrix.size == 0:
            return None
        
        P_L0 = RSA.normalize_rows(const_matrix)
        P_S1 = RSA.normalize_cols(P_L0)
        if len_prag_tests > 0:
            # Don't include already chosen tests
            P_S1_truth = P_S1[:-len_prag_tests, -1]
        else:
            P_S1_truth = P_S1[:, -1]
        
        if P_S1_truth.sum() == 0: 
            # If there are no consistent tests
            return None
        rsa_testcase_idx = np.argmax(P_S1_truth)
        return gen_tests[rsa_testcase_idx]
    
    def generate_pragmatic_testcases(self, program_ctx, testcase_ctx, true_program, num_tescases):
        pragmatic_testcases = []
        for _ in range(num_tescases):
            next_testcase = self.generate_next_pragmatic_testcase(program_ctx, testcase_ctx, true_program, pragmatic_testcases)
            if next_testcase is not None:
                pragmatic_testcases.append(next_testcase)
        return pragmatic_testcases
    
    def create_informative_dataset(self, num_testcases, train_json_dir, save_dir):
        with open(train_json_dir) as f:
            train_json = [json.loads(line) for line in f]
        
        train_res = []
        for example in train_json:
            program_ctx, testcase_ctx, true_program = example['program_ctx'], example['test_ctx'], example['program']
            pragmatic_testcases = self.generate_pragmatic_testcases(program_ctx, testcase_ctx, true_program, num_testcases)
            train_example = {
                'program_ctx' : program_ctx,
                'testcase_ctx' : testcase_ctx,
                'program' : true_program,
                'test' : "\n".join(pragmatic_testcases)
            }
            train_res.append(train_example)
        write_jsonl(save_dir, train_res)
        




