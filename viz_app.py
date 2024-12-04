import streamlit as st
import numpy as np
import json
import re

def load_data():
    pass_matrix = np.load("pass_matrix.npy")
    with open("outputs/solutions.json", "r") as f:
        solutions = json.load(f)
    with open("outputs/tests.json", "r") as f:
        tests = json.load(f)
    with open("outputs/num_tests.json", "r") as f:
        num_tests = json.load(f)
    with open("outputs/reports.json", "r") as f:
        reports = json.load(f)
    return pass_matrix, solutions, tests, num_tests, reports

def extract_test_functions(test_code):
    """Extract individual test functions from a test suite"""
    pattern = r"def\s+(test[^(]*)\([^)]*\):\s*\n\s*([^def]*)"
    matches = re.finditer(pattern, test_code, re.MULTILINE)
    return [(m.group(1), m.group(2).strip()) for m in matches]

def main():
    st.title("Test Results Visualization")
    
    pass_matrix, solutions, tests, num_tests, reports = load_data()
    
    st.header("Pass Matrix Visualization")
    st.dataframe(
        pass_matrix,
        use_container_width=True,
        hide_index=False,
        column_config={
            str(i): st.column_config.NumberColumn(
                f"Test {i}",
                format="%.0f",
            )
            for i in range(pass_matrix.shape[1])
        }
    )
    
    st.header("Solutions and Tests")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Solutions")
        solution_idx = st.selectbox("Select Solution", range(len(solutions)), key="sol")
        st.code(solutions[solution_idx], language="python")
        
    with col2:
        st.subheader("Tests")
        # Calculate cumulative sum for test suite boundaries
        test_boundaries = np.cumsum([0] + num_tests)
        
        # Select test suite
        test_suite_idx = st.selectbox("Select Test Suite", range(len(tests)), key="suite")
        
        # Show individual tests from the selected suite
        test_functions = extract_test_functions(tests[test_suite_idx])
        st.write(f"Test Suite {test_suite_idx} (contains {len(test_functions)} tests)")
        
        for i, (name, body) in enumerate(test_functions):
            with st.expander(f"Test {test_boundaries[test_suite_idx] + i}: {name}"):
                st.code(body, language="python")
                # Show pass/fail for selected solution
                passed = pass_matrix[solution_idx, test_boundaries[test_suite_idx] + i]
                st.write("Status:", "✅ Passed" if passed else "❌ Failed")

if __name__ == "__main__":
    main()
