"""
Functions for printing dictionary structures in a tree format.
"""

def print_dict_tree(d, indent=0, prefix=""):
    """
    Print a dictionary's keys and values in a tree-like structure.
    
    Args:
        d (dict): The dictionary to display
        indent (int): Current indentation level
        prefix (str): Prefix for the current line
        
    Example:
        >>> data = {'a': {'b': 1, 'c': {'d': 2}}, 'e': 3}
        >>> print_dict_tree(data)
        ├── a
        │   ├── b: 1
        │   └── c
        │       └── d: 2
        └── e: 3
    """
    for i, (key, value) in enumerate(d.items()):
        is_last = i == len(d) - 1
        current_prefix = "└── " if is_last else "├── "
        next_prefix = "    " if is_last else "│   "
        
        if isinstance(value, dict):
            print(" " * indent + prefix + current_prefix + str(key))
            print_dict_tree(value, indent + 8, prefix + next_prefix)
        else:
            print(" " * indent + prefix + current_prefix + f"{key}: {value}")