import numpy as np
from ..src.rsa import RSA
from typing import Callable
from const_matrix import create_const_matrix
from agent import Speaker, Listener

class InferenceSpeakerListener:
    def __init__(self, 
                 speaker : Speaker, 
                 listener : Listener, 
                 num_generations : int,
                 extract_program_fn : Callable, 
                 extract_testcase_fn : Callable
                 ):
        self.speaker = speaker
        self.listener = listener
        self.num_generations = num_generations
        self.extract_program_fn = extract_program_fn
        self.extract_testcase_fn = extract_testcase_fn
    
    def generate_const_matrix(self, program_ctx, testcase_ctx, true_program):
        listener_outputs = self.listener.generate(program_ctx, self.num_generations)        
        self.programs = [self.extract_program_fn(output) for output in listener_outputs]
        self.programs.append(true_program)
        speaker_outputs = self.speaker.generate(testcase_ctx, self.num_generations)
        self.tests = [self.extract_testcase_fn(output) for output in speaker_outputs]
        return create_const_matrix(self.programs, self.tests)

    def generate_pragmatic_testcase(self, program_ctx, testcase_ctx, true_program):
        const_matrix = self.generate_const_matrix(program_ctx, testcase_ctx, true_program)
        P_L0 = RSA.normalize_rows(const_matrix)
        P_S1 = RSA.normalize_cols(P_L0)
        rsa_testcase_idx = np.argmax(P_S1[:, -1])
        return self.tests[rsa_testcase_idx]