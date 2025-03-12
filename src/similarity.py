import ast
import copy


def normalize_variable_names(node, mapping=None):
    """
    Traverse AST and normalize all variable names to a standard format.
    Returns the modified node and the updated mapping.
    """
    if mapping is None:
        mapping = {}

    # First collect all variable names to assign indices deterministically
    def collect_names(node, names=None):
        if names is None:
            names = set()

        if isinstance(node, (ast.Name, ast.arg)):
            name = node.id if isinstance(node, ast.Name) else node.arg
            names.add(name)

        for child in ast.iter_child_nodes(node):
            collect_names(child, names)

        return names

    # Get sorted list of all names to ensure deterministic numbering
    all_names = sorted(collect_names(node))

    # Create mapping if not already present
    for i, name in enumerate(all_names):
        if name not in mapping:
            mapping[name] = f"var_{i}"

    # Apply mapping to node
    if isinstance(node, ast.Name):
        node.id = mapping[node.id]
    elif isinstance(node, ast.arg):
        node.arg = mapping[node.arg]

    # Recursively normalize all child nodes
    for child in ast.iter_child_nodes(node):
        normalize_variable_names(child, mapping)

    return node


def are_structurally_identical(code1, code2):
    """
    Check if two pieces of Python code are structurally identical,
    ignoring differences in variable names.

    Args:
        code1 (str): First piece of Python code
        code2 (str): Second piece of Python code

    Returns:
        bool: True if the code pieces are structurally identical
        dict: Mapping between variable names in code1 and code2
    """
    try:
        # Parse both code strings into ASTs
        tree1 = ast.parse(code1)
        tree2 = ast.parse(code2)

        # Create copies to avoid modifying original ASTs
        tree1_copy = copy.deepcopy(tree1)
        tree2_copy = copy.deepcopy(tree2)

        # Normalize variable names in both ASTs
        mapping1 = {}
        mapping2 = {}
        normalize_variable_names(tree1_copy, mapping1)
        normalize_variable_names(tree2_copy, mapping2)

        # Compare the normalized ASTs
        are_identical = ast.dump(tree1_copy) == ast.dump(tree2_copy)

        # Create mapping between original variable names
        var_mapping = {}
        for var1, normalized in mapping1.items():
            for var2, normalized2 in mapping2.items():
                if normalized == normalized2:
                    var_mapping[var1] = var2

        return are_identical, var_mapping

    except SyntaxError:
        return False, {}


# Example usage:
if __name__ == "__main__":
    code1 = """
    def calculate(x, y):
        result = x + y
        return result
    """

    code2 = """
    def calculate(a, b):
        sum_val = a + b
        return sum_val
    """

    identical, mapping = are_structurally_identical(code1, code2)
    print(f"Are the code pieces structurally identical? {identical}")
    if identical:
        print("Variable name mapping:")
        for var1, var2 in mapping.items():
            print(f"{var1} -> {var2}")
