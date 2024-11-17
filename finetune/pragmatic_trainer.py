import numpy as np
import json
from ..src.rsa import RSA
from ..src.utils import write_jsonl, extract_function, extract_testcase
from typing import Callable, List
from ..src.utils import create_const_matrix
from .agent import Speaker, Listener
import fire
import torch
import os

class InferenceSpeakerListener:
    def __init__(self, 
                 speaker : Speaker, 
                 listener : Listener, 
                 num_generations : int = 1,
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
    
    def generate_const_matrix(self, program_ctx, testcase_ctx, true_program, pragmatic_testcases = []):
        program_prompt = self.program_prompt_template.format(program_ctx)
        test_prompt = self.test_prompt_template.format(testcase_ctx)
        
        listener_outputs = self.listener.generate(program_prompt, self.num_generations)        
        speaker_outputs = self.speaker.generate(test_prompt, self.num_generations)
        
        programs = [self.extract_program_fn(output) for output in listener_outputs]
        tests = [self.extract_testcase_fn(output) for output in speaker_outputs]
        
        programs.append(true_program)
        if len(pragmatic_testcases) > 0:
            tests.extend(pragmatic_testcases)
        return programs, tests, create_const_matrix(programs, tests)

    def generate_next_pragmatic_testcase(self, program_ctx, testcase_ctx, true_program, pragmatic_testcases):
        gen_programs, gen_tests, const_matrix = self.generate_const_matrix(program_ctx, testcase_ctx, true_program, pragmatic_testcases)
        # De-duplcating generated programs and tests
        new_programs = gen_programs[:-1]
        new_tests = gen_tests[:-len(pragmatic_testcases)] if len(pragmatic_testcases) > 0 else gen_tests
        unique_programs, unique_tests = set(), set()
        duplicate_program_idxs, duplicate_test_idxs = [], []
        for j, program in enumerate(new_programs):
            if program in unique_programs:
                duplicate_program_idxs.append(j)
            else:
                unique_programs.add(program)
        for j, test in enumerate(new_tests):
            if test in unique_tests:
                duplicate_test_idxs.append(j)
            else:
                unique_tests.add(test)
        
        # Update gen_programs, gen_tests and const_matrix
        program_mask = np.ones([len(gen_programs)], dtype = bool)
        program_mask[np.array(duplicate_program_idxs, dtype = int)] = False
        gen_programs = np.array(gen_programs)[program_mask].tolist()

        test_mask = np.ones([len(gen_tests)], dtype = bool)
        test_mask[np.array(duplicate_test_idxs, dtype = int)] = False
        gen_tests = np.array(gen_tests)[test_mask].tolist()

        const_matrix = const_matrix[test_mask, :][:, program_mask]
        
        # Auto-regressively update const matrix given existing pragmatic test cases
        len_prag_tests = len(pragmatic_testcases)
        if len_prag_tests > 0:
            prag_tests_const = const_matrix[-len_prag_tests:, :] == 1
            all_prag_tests_const = np.logical_and.reduce(prag_tests_const)
            const_matrix = const_matrix[:, all_prag_tests_const]
        if const_matrix.size == 0:
            return None
        
        # RSA on updated const matrix
        P_L0 = RSA.normalize_rows(const_matrix)
        P_S1 = RSA.normalize_cols(P_L0)
        if len_prag_tests > 0:
            # Don't include already chosen tests
            P_S1_truth = P_S1[:-len_prag_tests, -1]
        else:
            P_S1_truth = P_S1[:, -1]
        
        if P_S1_truth.size == 0 or P_S1_truth.sum() == 0: 
            # If there are no new consistent tests
            return None
        rsa_testcase_idx = np.argmax(P_S1_truth)
        return gen_tests[rsa_testcase_idx]
    
    def generate_pragmatic_testcases_from_cache(self, gen_programs, gen_tests, const_matrix, num_testcases):
        # De-deuplicating programs and tests
        unique_programs = {}
        unique_tests = {}
        for j, program in enumerate(gen_programs):
            unique_programs[program] = j
        for j, test in enumerate(gen_tests):
            unique_tests[test] = j
        
        program_idxs = np.array(list(unique_programs.values()))
        test_idxs = np.array(list(unique_tests.values()))
        
        gen_programs = np.array(gen_programs)[program_idxs].tolist()
        gen_tests = np.array(gen_tests)[test_idxs].tolist()
        const_matrix = np.concatenate((const_matrix[test_idxs, :][:, program_idxs], const_matrix[test_idxs, -1].reshape(-1, 1)), axis = 1)

        # Auto-regressively selecting pragmatic tests
        pragmatic_testcases = []
        const_matrix_update = np.copy(const_matrix)
        for _ in range(num_testcases):
            if const_matrix_update.size == 0:
                break
            P_L0 = RSA.normalize_rows(const_matrix_update)
            P_S1 = RSA.normalize_cols(P_L0)
            P_S1_truth = P_S1[:, -1]
            if P_S1_truth.sum() != 0: 
                rsa_testcase_idx = np.argmax(P_S1_truth)
                pragmatic_testcases.append(gen_tests[rsa_testcase_idx])
                # Auto-regressively update the const matrix
                const_matrix_update = const_matrix_update[:, const_matrix_update[rsa_testcase_idx, :] == 1]
                exclude = np.ones(const_matrix_update.shape[0], dtype = bool)
                exclude[rsa_testcase_idx] = 0
                const_matrix_update = const_matrix_update[exclude, :]
                gen_tests = np.array(gen_tests)[exclude].tolist()
                
        return pragmatic_testcases
    
    def generate_pragmatic_testcases(self, program_ctx, testcase_ctx, true_program, num_testcases, regen = False):
        pragmatic_testcases = []
        gen_programs, gen_tests, const_matrix = None, None, None # If we re-generate, we can't pick a single set of generations
        if regen:
            for _ in range(num_testcases):
                next_testcase = self.generate_next_pragmatic_testcase(program_ctx, testcase_ctx, true_program, pragmatic_testcases)
                if next_testcase is not None:
                    pragmatic_testcases.append(next_testcase)
        else:
            gen_programs, gen_tests, const_matrix = self.generate_const_matrix(program_ctx, testcase_ctx, true_program)
            pragmatic_testcases = self.generate_pragmatic_testcases_from_cache(gen_programs, gen_tests, const_matrix, num_testcases)
        return gen_programs, gen_tests, const_matrix, pragmatic_testcases

    
    def create_informative_dataset(self, train_json_dir, save_dir, num_testcases = 1, regen = False, 
                                   cached_gen_programs_dir = None, cached_gen_tests_dir = None, cached_const_matrices_dir = None):
        with open(train_json_dir) as f:
            train_json = [json.loads(line) for line in f]
        
        is_cached = (cached_gen_programs_dir is not None) and (cached_gen_tests_dir is not None) and (cached_const_matrices_dir is not None)
        if is_cached:
            with open(cached_gen_programs_dir) as f:
                cached_gen_programs = [json.loads(line) for line in f]
            with open(cached_gen_tests_dir) as f:
                cached_gen_tests = [json.loads(line) for line in f]
            cached_const_matrices = np.load(cached_const_matrices_dir)

        res_train_dir = os.path.join(save_dir, 'dataset.jsonl')
        res_programs_dir = os.path.join(save_dir, 'gen_programs.jsonl')
        res_tests_dir = os.path.join(save_dir, 'gen_tests.jsonl')
        res_const_matrices_dir = os.path.join(save_dir, 'const_matrices.npy')
        res_const_matrices = np.zeros([len(train_json), self.num_generations, self.num_generations + 1])
        
        for i, example in enumerate(train_json):
            task_id, program_ctx, testcase_ctx, true_program = example['task_id'], example['program_ctx'], example['test_ctx'], example['program']
            if is_cached:
                gen_programs = [example['completion'] for example in cached_gen_programs[i * self.num_generations : (i + 1) * self.num_generations]]
                gen_tests = [example['completion'] for example in cached_gen_tests[i * self.num_generations : (i + 1) * self.num_generations]]
                cached_const_matrix = cached_const_matrices[i]
                pragmatic_testcases = self.generate_pragmatic_testcases_from_cache(gen_programs, gen_tests, cached_const_matrix, num_testcases)
            else:
                gen_programs, gen_tests, const_matrix, pragmatic_testcases = self.generate_pragmatic_testcases(program_ctx, testcase_ctx, true_program, num_testcases, regen)
                if not regen:
                    write_jsonl(res_programs_dir, [{'task_id' : task_id, 'completion' : program} for program in gen_programs], True)
                    write_jsonl(res_tests_dir, [{'task_id' : task_id, 'completion' : test} for test in gen_tests], True)
                    res_const_matrices[i] = const_matrix

            train_example = {
                'task_id' : task_id,
                'program_ctx' : program_ctx,
                'test_ctx' : testcase_ctx,
                'program' : true_program,
                'test' : "\n".join(pragmatic_testcases),
                'debug_tests' : pragmatic_testcases
            }
            write_jsonl(res_train_dir, [train_example], True)
        
        np.save(res_const_matrices_dir, res_const_matrices)

def main(model_name_or_path, temperature, max_new_tokens, num_generations, num_testcases, regen, train_json_dir, save_dir):
    device = "cuda" #if torch.cuda.is_available() else "cpu"

    gen_config = {
        'temperature' : temperature,
        'max_new_tokens' : max_new_tokens,
        'do_sample' : True
    }

    speaker = Speaker(
        model_name_or_path,
        gen_config,
        device
    )

    listener = Listener(
        model_name_or_path,
        gen_config,
        device
    )

    program_prompt_template = "# Complete the following function.\n{}"
    test_prompt_template = "# Write test cases for the following function.\n{}    pass\n\nassert"

    inference = InferenceSpeakerListener(speaker, listener, num_generations, program_prompt_template, test_prompt_template, extract_function, extract_testcase)
    inference.create_informative_dataset(train_json_dir, save_dir, num_testcases, regen)

if __name__ == '__main__':
    fire.Fire(main)
    # model_name_or_path = 'Salesforce/codegen-350M-mono'
    # # model_name_or_path = 'codellama/CodeLlama-13b-hf'
    # gen_config = {
    #     'temperature' : 0.8,
    #     'max_new_tokens' : 128,
    #     'do_sample' : True
    # }
    
    # #os.environ["CUDA_VISIBLE_DEVICES"] = '1' # IDK why but u need this
    
    # device = "cuda"

    # speaker = Speaker(
    #     model_name_or_path,
    #     gen_config,
    #     device
    # )

    # listener = Listener(
    #     model_name_or_path,
    #     gen_config,
    #     device
    # )

    # # speaker, listener = None, None

    # num_generations = 100
    # program_prompt_template = "# Complete the following function.\n{}"
    # test_prompt_template = "# Write test cases for the following function.\n{}    pass\n\nassert"

    # inference = InferenceSpeakerListener(speaker, listener, num_generations, program_prompt_template, test_prompt_template, extract_function, extract_testcase)

    # train_json_dir = '/home/rrsood/CodeGen/test_training/humaneval_test_dataset.jsonl'
    # save_dir = '/home/rrsood/CodeGen/test_training/'
    # num_testcases = 5
    # regen = True
    # # cached_gen_programs_dir = '/home/rrsood/CodeGen/data/humaneval/codellama-13b/generations/codellama_humaneval_programs_k100.jsonl'
    # # cached_gen_tests_dir = '/home/rrsood/CodeGen/data/humaneval/codellama-13b/generations/codellama_humaneval_tests_k100_temp0.8.jsonl'
    # # cached_const_matrices_dir = '/home/rrsood/CodeGen/data/humaneval/codellama-13b/const-matrices/codellama_humaneval_const_matrix.npy'

    # inference.create_informative_dataset(train_json_dir, save_dir, num_testcases, regen) # cached_gen_programs_dir, cached_gen_tests_dir, cached_const_matrices_dir)


    
        




