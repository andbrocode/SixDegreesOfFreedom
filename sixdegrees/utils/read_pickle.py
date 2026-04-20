"""
Functions for reading pickle files.
"""
import pickle

def read_pickle(path, filename):
    """
    Read a pickle file from the specified path.
    
    Args:
        path (str): Path to the directory containing the pickle file
        filename (str): Name of the pickle file
        
    Returns:
        object: Contents of the pickle file
    """
    with open(path + filename, 'rb') as f:
        data = pickle.load(f)
    return data
