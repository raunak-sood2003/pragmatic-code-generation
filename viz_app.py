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
    pattern = r"def\s+(test[^(]*)\([^)]*\):\s*\n\s*(.*?)(?=\n\s*def|\Z)"
    matches = re.finditer(pattern, test_code, re.MULTILINE | re.DOTALL)
    return [(m.group(1), m.group(2).strip()) for m in matches]

def main():
    st.set_page_config(layout="wide")
    st.title("Test Results Visualization")
    
    pass_matrix, solutions, tests, num_tests, reports = load_data()
    
    st.header("Pass Matrix Visualization")
    
    # Display num_tests as a row vector
    st.write("Number of tests per test suite:")
    st.dataframe(
        np.array([num_tests]),
        use_container_width=True,
        hide_index=True,
        column_config={
            str(i): st.column_config.NumberColumn(
                f"Suite {i}",
                format="%.0f",
            )
            for i in range(len(num_tests))
        }
    )

    st.write("Pass matrix: (1 pass 0 fail)")
    st.write("The first N belong to suite 1, and so on")
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
        solution_idx = st.number_input("Select Solution", min_value=0, max_value=len(solutions)-1, value=0, step=1, key="sol")
        st.code(solutions[solution_idx], language="python")
        
    with col2:
        st.subheader("Tests")
        # Calculate cumulative sum for test suite boundaries
        test_boundaries = np.cumsum([0] + num_tests)
        
        # Select test suite
        test_suite_idx = st.number_input("Select Test Suite", min_value=0, max_value=len(tests)-1, value=0, step=1, key="suite")
        
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
