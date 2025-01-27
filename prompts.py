CODE_PROMPT_BASE_TEMPLATE = '{before}{function_signature}\n\t"""{docstring}"""'
TEST_PROMPT_BASE_TEMPLATE = '{before}{function_signature}\n\t"""{docstring}"""{function_body}\n# test_{entry_point} function with a single assert statement to test the {entry_point} function\n'
CODE_PROMPT_INSTRUCT_TEMPLATE = """Write a Python function implementation for the following prompt:

{instruction}

The function should pass the following test case:
```
{test_case}
```

Return only the implementation code, no explanations. Be sure to include the relevant import statements:
```python
code
```"""
TEST_PROMPT_INSTRUCT_TEMPLATE = '''Write a test case for the following function:
```
{before}{function_signature}\n\t"""{docstring}"""{function_body}
```

Return only the test cases in Python code format, wrapped like.
```python
def test_{entry_point}_1():
    assert ...

def test_{entry_point}_2():
    assert ...
```

Do not repeat the original function.'''
