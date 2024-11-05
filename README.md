# Pragmatic Neural Program/Test Case Synthesis

## Overview
Recently, LLMs have shown remarkable success on code generation tasks such as natural language -> program/test case tasks. However, it often takes several attempts for an LLM to produce the user's intended program. Accordingly, this project aims to tackle the problem of program synthesis by using informatively chosen test cases to filter out spuriously generated programs.


## Best-of-n with self-generated tests
```
uv run python prompt.py
```
