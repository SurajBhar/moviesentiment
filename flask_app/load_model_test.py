import pickle
import os

# Define the path to the .pkl file
pkl_file_path = '../models/vectorizer.pkl'

# Check if the file exists
if os.path.exists(pkl_file_path):
    print(f"File found: {pkl_file_path}")
    try:
        # Attempt to load the file
        vectorizer = pickle.load(open(pkl_file_path, 'rb'))
        print("File loaded successfully.")
    except Exception as e:
        # Handle exceptions during file loading
        print(f"Error loading the .pkl file: {e}")
else:
    print(f"File not found: {pkl_file_path}. Please check the path and try again.")
