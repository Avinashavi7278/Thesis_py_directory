import json
import os

def save_data_to_directory(directory, id_number, data):
    # Ensure the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Create a file path with the ID number inside the specified directory
    file_path = os.path.join(directory, f"{id_number}.json")
    
    # Serialize the data to a JSON formatted string and write to the file
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)
    print(f"Data saved to {file_path}")

# Directory where all JSON files will be stored
directory_name = 'students_data'

# Example student data
students_info = [
    {'id_number': 'S001', 'name': 'John Doe', 'age': 20, 'class': '12A'},
    {'id_number': 'S002', 'name': 'Jane Smith', 'age': 21, 'class': '12B'},
    {'id_number': 'S003', 'name': 'Jim Beam', 'age': 22, 'class': '12C'},
    # ... more students
]

# Save each student's data to a separate JSON file in the directory
for student in students_info:
    save_data_to_directory(directory_name, student['id_number'], student)
