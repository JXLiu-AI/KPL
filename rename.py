import os

def get_folder_names(directory):
    # Get all items in the specified directory
    items = os.listdir(directory)
    # Filter out folders
    folders = [item for item in items if os.path.isdir(os.path.join(directory, item))]
    return folders

# Set the folder path to read
directory_path = '/path/to/Place365/val'

# Get list of folder names
folder_names = get_folder_names(directory_path)

# Format output
formatted_output = 'imagenet_classes = ' + str(folder_names)
print(formatted_output)

# import os

# def rename_folders(directory):
#     # Iterate through all items in the specified directory
#     for folder_name in os.listdir(directory):
#         # Build the full folder path
#         folder_path = os.path.join(directory, folder_name)
        
#         # Ensure it is a folder, not a file
#         if os.path.isdir(folder_path):
#             # Find the position of the dot "." in the folder name
#             dot_index = folder_name.find('.')
            
#             # If a dot is found and there is content after the dot
#             if dot_index != -1 and dot_index + 1 < len(folder_name):
#                 # The new folder name is the name after removing the dot and everything before it
#                 new_folder_name = folder_name[dot_index + 1:]
#                 new_folder_path = os.path.join(directory, new_folder_name)
                
#                 # Rename the folder
#                 os.rename(folder_path, new_folder_path)
#                 print(f'Renamed "{folder_name}" to "{new_folder_name}"')

# # Usage example, replace with your folder path
# directory_path = '/path/to/CUB/val'
# rename_folders(directory_path)

# import json

# def modify_json_categories(file_path):
#     # Open and read JSON file
#     with open(file_path, 'r') as file:
#         data = json.load(file)
    
#     # Get cub_Features section
#     cub_features = data['cub_Features']
#     new_cub_features = {}

#     # Iterate through keys and modify
#     for key in cub_features:
#         # Find dot and split
#         new_key = key.split('.', 1)[-1] if '.' in key else key
#         new_cub_features[new_key] = cub_features[key]

#     # Update cub_Features
#     data['cub_Features'] = new_cub_features

#     # Write back to file
#     with open(file_path, 'w') as file:
#         json.dump(data, file, indent=4)

# # Usage example
# file_path = '/path/to/feature_50.json'
# modify_json_categories(file_path)


