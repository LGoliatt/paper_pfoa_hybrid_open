import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.complexfloating):
            return {'real': obj.real, 'imag': obj.imag}
        if isinstance(obj, (pd.DataFrame, pd.Series)):
            return obj.to_dict()
        if isinstance(obj, set):
            return list(obj)
        if pd.isna(obj):
            return None
        return super().default(obj)

def pkl_to_json(pkl_file, json_file=None, indent=2):
    """
    Read a pickle file and convert it to JSON format.
    
    Parameters:
    -----------
    pkl_file : str
        Path to the pickle file
    json_file : str, optional
        Path to save JSON file. If None, uses same name as pkl_file with .json extension
    indent : int
        JSON indentation level for readability
    
    Returns:
    --------
    dict : The loaded data
    """
    
    # Load pickle file
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    
    # Convert to JSON-serializable format
    json_data = convert_to_serializable(data)
    
    # Determine output filename
    if json_file is None:
        json_file = Path(pkl_file).with_suffix('.json')
    
    # Save as JSON with custom encoder
    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=indent, cls=NumpyEncoder)
    
    print(f"Successfully converted {pkl_file} to {json_file}")
    return json_data


def convert_to_serializable(obj):
    """
    Recursively convert objects to JSON-serializable format.
    Handles numpy arrays, pandas DataFrames, and nested structures.
    """
    
    # Handle None first
    if obj is None:
        return None
    
    # Handle pandas NA/NaN - check with try/except to avoid ambiguous truth value
    try:
        if pd.isna(obj):
            return None
    except (ValueError, TypeError):
        pass
    
    # Handle pandas DataFrame - MUST come before array check
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    
    # Handle pandas Series - MUST come before array check
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    
    # Handle numpy arrays
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    
    # Handle all numpy integer types
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    
    # Handle all numpy floating types
    if isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    
    # Handle numpy bool
    if isinstance(obj, np.bool_):
        return bool(obj)
    
    # Handle dictionaries recursively
    if isinstance(obj, dict):
        return {str(key): convert_to_serializable(value) for key, value in obj.items()}
    
    # Handle lists and tuples recursively
    if isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    
    # Handle sets
    if isinstance(obj, set):
        return [convert_to_serializable(item) for item in obj]
    
    # Handle basic types (str, int, float, bool)
    if isinstance(obj, (str, int, float, bool)):
        return obj
    
    # For other objects, try to convert to string
    try:
        return str(obj)
    except:
        return f"<non-serializable: {type(obj).__name__}>"


def batch_pkl_to_json(input_dir, output_dir=None, pattern='*.pkl'):
    """
    Convert all pickle files in a directory to JSON.
    
    Parameters:
    -----------
    input_dir : str
        Directory containing pickle files
    output_dir : str, optional
        Directory to save JSON files. If None, saves in same directory
    pattern : str
        File pattern to match (default: '*.pkl')
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir) if output_dir else input_path
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all pickle files
    pkl_files = list(input_path.glob(pattern))
    
    if not pkl_files:
        print(f"No pickle files found in {input_dir} matching pattern '{pattern}'")
        return
    
    print(f"Found {len(pkl_files)} pickle file(s)")
    
    # Convert each file
    for pkl_file in pkl_files:
        json_file = output_path / pkl_file.with_suffix('.json').name
        try:
            pkl_to_json(pkl_file, json_file)
        except Exception as e:
            print(f"Error converting {pkl_file}: {e}")
    
    print(f"\nConversion complete! JSON files saved to {output_path}")


# Example usage functions
def example_single_file():
    """Convert a single pickle file"""
    pkl_to_json('data.pkl', 'data.json')


def example_batch_conversion():
    """Convert all pickle files in a directory"""
    batch_pkl_to_json('./pkl_files', './json_files')


def example_with_custom_handling():
    """Example with custom data inspection"""
    # Load pickle file
    with open('data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    # Inspect the data structure
    print("Data type:", type(data))
    if isinstance(data, dict):
        print("Keys:", data.keys())
    
    # Convert and save
    pkl_to_json('data.pkl', 'data.json')


if __name__ == "__main__":
    # Example 1: Convert single file
    # pkl_to_json('your_file.pkl')
    
    # Example 2: Batch conversion
    # batch_pkl_to_json('./data', './output')
    
    # Example 3: Specific conversion
    # data = pkl_to_json('model_results.pkl', 'model_results.json')
    # print(data)
    
    batch_pkl_to_json('./pkl_pfoa___removal_efficiency/', './output')