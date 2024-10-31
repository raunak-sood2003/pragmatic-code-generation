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
    
    def generate_const_matrix(self, program_ctx, testcase_ctx, true_program):
        program_prompt = self.program_prompt_template.format(program_ctx)
        test_prompt = self.test_prompt_template.format(testcase_ctx)
        
        listener_outputs = self.listener.generate(program_prompt, self.num_generations)        
        speaker_outputs = self.speaker.generate(test_prompt, self.num_generations)
        
        self.programs = [self.extract_program_fn(output) for output in listener_outputs]
        self.programs.append(true_program)
        self.tests = [self.extract_testcase_fn(output) for output in speaker_outputs]
        return create_const_matrix(self.programs, self.tests)

    def generate_pragmatic_testcase(self, program_ctx, testcase_ctx, true_program):
        const_matrix = self.generate_const_matrix(program_ctx, testcase_ctx, true_program)
        P_L0 = RSA.normalize_rows(const_matrix)
        P_S1 = RSA.normalize_cols(P_L0)
        rsa_testcase_idx = np.argmax(P_S1[:, -1]) # DOES THIS HAVE TO BE A CONSISTENT TEST CASE? WHAT TO DO IF THERE AREN'T ANY?
        return self.tests[rsa_testcase_idx]
    
    def create_informative_dataset(self, train_json_dir, save_dir):
        with open(train_json_dir) as f:
            train_json = [json.loads(line) for line in f]

        train_res = []
        for example in train_json:
            program_ctx, testcase_ctx, true_program = example['program_ctx'], example['test_ctx'], example['program']
            pragmatic_testcase = self.generate_pragmatic_testcase(program_ctx, testcase_ctx, true_program)
            train_example = {
                'program_ctx' : program_ctx,
                'testcase_ctx' : testcase_ctx,
                'program' : true_program,
                'test' : pragmatic_testcase
            }
            train_res.append(train_example)
        write_jsonl(save_dir, train_res)
        




