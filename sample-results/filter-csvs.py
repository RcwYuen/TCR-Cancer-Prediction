import argparse
import os
import shutil
from pathlib import Path

def copy_csv_files(source_folder, destination_folder):
    """
    Copies all CSV files from the source folder and its subfolders to the destination folder,
    preserving the directory structure.
    """
    # Count the number of files copied
    files_copied = 0

    for subdir, dirs, files in os.walk(source_folder):
        for filename in files:
            if filename.endswith('.csv'):
                # Build the relative path to the source file
                rel_path = os.path.relpath(subdir, source_folder)
                # Build the source and destination file paths
                source_file = os.path.join(subdir, filename)
                destination_subdir = os.path.join(destination_folder, rel_path)
                destination_file = os.path.join(destination_subdir, filename)
                
                # Ensure the destination subdirectory exists
                if not os.path.exists(destination_subdir):
                    os.makedirs(destination_subdir)
                
                # Copy the file
                shutil.copy2(source_file, destination_file)
                files_copied += 1
                print(f"Copied {filename} to {destination_subdir}")

    print(f"Total {files_copied} CSV files copied to {destination_folder}.")

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description='Copy CSV files from a source folder and its subfolders to a destination folder, preserving the directory structure.')
    
    # Add the arguments
    parser.add_argument('source_folder', type=str, help='The path to the source folder containing CSV files')
    
    # Parse the arguments
    args = parser.parse_args()

    # Set the destination folder path
    destination_folder = Path.cwd() / 'sample-results' / args.source_folder
    
    # Copy the CSV files
    copy_csv_files(args.source_folder, destination_folder)
