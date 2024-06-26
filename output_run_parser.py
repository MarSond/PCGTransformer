import os
import shutil
from datetime import datetime
from collections import defaultdict
from MLHelper.constants import *

folder = "./runs/"

blacklist = ["run1"]

def list_sub_folders(folder_path):
    return next(os.walk(folder_path))[1]

def update_stats(sub_folder_path, stats):
    files = os.listdir(sub_folder_path)
    stats['sub_folders']['total'] += 1
    if FILENAME_RUN_CONFIG_VALUE in files:
        stats['files'][FILENAME_RUN_CONFIG_VALUE] += 1
    if FILENAME_LOG_OUTPUT_VALUE in files:
        stats['files'][FILENAME_LOG_OUTPUT_VALUE] += 1
    if FILENAME_METRICS_VALUE in files:
        stats['files'][FILENAME_METRICS_VALUE] += 1
    
    # Count .pt files in the models subfolder
    models_path = os.path.join(sub_folder_path, MODELS_FOLDER_NAME)
    if os.path.exists(models_path):
        pt_files = [f for f in os.listdir(models_path) if f.endswith(MODEL_FILE_EXTENSION)]
        stats['models'][len(pt_files)] += 1

def delete_folder(folder_path):
    shutil.rmtree(folder_path)
    print(f"Deleted folder: {folder_path}")

def folder_date(folder_name):
    try:
        return datetime.strptime(folder_name.split('_')[0], "%Y-%d-%m").date()
    except ValueError:
        return None

def confirm_and_delete_folders(folders_to_delete):
    if not folders_to_delete:
        print("No folders to delete.")
        return
    
    print("These folders will be deleted:")
    for folder in folders_to_delete:
        print(folder)
    
    confirmation = input("Continue? (Y/N): ")
    if confirmation.lower() == 'y':
        for folder in folders_to_delete:
            delete_folder(folder)
    else:
        print("Deletion cancelled.")

def query_and_delete_folders(sub_folders, base_folder, stats):
    today = datetime.now().date()
    folders_to_delete = []

    print("Options:")
    print("1: Delete all folders except today's")
    print("2: Delete folders where no metrics file is found")
    print("3: Delete folders with more than 10 model files")
    print("4: Exit")
    choice = input("Enter your choice: ")

    if choice == '1':
        for sub_folder in sub_folders:
            sub_folder_path = os.path.join(base_folder, sub_folder)
            date = folder_date(sub_folder)
            if date and date != today:
                folders_to_delete.append(sub_folder_path)
    elif choice == '2':
        for sub_folder in sub_folders:
            sub_folder_path = os.path.join(base_folder, sub_folder)
            if FILENAME_METRICS_VALUE not in os.listdir(sub_folder_path):
                folders_to_delete.append(sub_folder_path)
    elif choice == '3':
        for sub_folder in sub_folders:
            sub_folder_path = os.path.join(base_folder, MODELS_FOLDER_NAME, sub_folder)
            if os.path.exists(sub_folder_path):
                pt_files = [f for f in os.listdir(sub_folder_path) if f.endswith(MODEL_FILE_EXTENSION)]
                if len(pt_files) > 10:
                    folders_to_delete.append(sub_folder_path)
    elif choice == '4':
        print("Exiting.")
        return
    else:
        print("Invalid choice.")
        return
    folders_to_delete = [f for f in folders_to_delete if not any([b in f for b in blacklist])]
    confirm_and_delete_folders(folders_to_delete)

# Statistics dictionary initialized
stats = defaultdict(lambda: defaultdict(int))

# List the sub-folders in the runs folder
sub_folders = list_sub_folders(folder)

# Update stats for each sub-folder
for sub_folder in sub_folders:
    sub_folder_path = os.path.join(folder, sub_folder)
    update_stats(sub_folder_path, stats)

# List statistics
print(f"Total sub-folders: {stats['sub_folders']['total']}")
for file_type, count in stats['files'].items():
    print(f"{file_type}: {count}")
for model_count, folders_with_count in stats['models'].items():
    print(f"Folders with {model_count} model files: {folders_with_count}")

# Perform folder deletion based on user input
query_and_delete_folders(sub_folders, folder, stats)