"""
Functions for printing dictionary structures in a tree format.
"""

def print_dict_tree(d, indent=0, prefix=""):
    """
    Print a dictionary's keys in a tree-like structure.
    
    Args:
        d (dict): The dictionary to display
        indent (int): Current indentation level
        prefix (str): Prefix for the current line
        
    Example:
        >>> data = {'a': {'b': 1, 'c': {'d': 2}}, 'e': 3}
        >>> print_dict_tree(data)
        ├── a
        │   ├── b
        │   └── c
        │       └── d
        └── e
    """
    for i, (key, value) in enumerate(d.items()):
        is_last = i == len(d) - 1
        current_prefix = "└── " if is_last else "├── "
        print(" " * indent + prefix + current_prefix + str(key))
        
        if isinstance(value, dict):
            next_prefix = "    " if is_last else "│   "
            print_dict_tree(value, indent + 4, prefix + next_prefix)
