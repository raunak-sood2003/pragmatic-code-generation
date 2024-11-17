N_PROGRAMS=100 # Number of programs sampled from model
N_TESTS=100 # Number of tests sampled from model
PROGRAMS_DIR="<path to repo>/pragmatic-code-generation/data/codellama_humaneval_programs_k100.jsonl"
TESTS_DIR="<path to repo>/pragmatic-code-generation/data/codellama_humaneval_tests_k100_temp0.8.jsonl"
CANONICAL_PROGRAMS_DIR="<path to repo>/pragmatic-code-generation/data/humaneval_canonical_solutions.jsonl"
CONST_MATRIX_PATH="<where consistency matrix should be saved as .npy file>"

python3 -m pragmatic-code-generation.scripts.gen_consistency_matrix \
        --num_programs $N_PROGRAMS \
        --num_tests $N_TESTS \
        --programs_dir $PROGRAMS_DIR \
        --tests_dir $TESTS_DIR \
        --canonical_programs_dir $CANONICAL_PROGRAMS_DIR \
        --save_dir $CONST_MATRIX_PATH


N_OUT_TESTS=5 # Number of tests to output per problem
SAVE_DIR="<folder where json file should be saved>"

python3 -m pragmatic-code-generation.inference.gen_reflexion_tests \
        --num_programs $N_PROGRAMS \
        --num_tests $N_TESTS \
        --num_out_tests $N_OUT_TESTS \
        --programs_path $PROGRAMS_DIR \
        --tests_path $TESTS_DIR \
        --canonical_programs_path $CANONICAL_PROGRAMS_DIR \
        --save_dir $SAVE_DIR \
        --const_matrix_path $CONST_MATRIX_PATH