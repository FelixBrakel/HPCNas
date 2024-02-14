import os
import subprocess
import prettytable
import traceback

def execute_files_in_directory(directory_path):
    # Get a list of all files in the specified directory
    files = [file for file in os.listdir(directory_path) if file.endswith(".py")]

    # Create a table to display the results
    result_table = prettytable.PrettyTable(["File", "Status"])

    # Iterate through each Python file and execute it with FlexFlow interpreter
    for file in files:
        file_path = os.path.join(directory_path, file)
        command = f"flexflow_python {file_path} -ll:gpu 1 -ll:fsize 5000 -ll:zsize 5000"

        try:
            # Execute the command using subprocess
            process = subprocess.run(command, shell=True, check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            result_table.add_row([file, "Success"])
            print(f"{file}: Success")
        except subprocess.CalledProcessError as e:
            error_filename = f"./errors/{file}_error.txt"
            with open(error_filename, "wb") as error_file:
                error_file.write(e.stderr)

            result_table.add_row([file, f"Exception. Stack trace saved to {error_filename}"])
            print(f"{file}: Exception. Stack trace saved to {error_filename}")
        # print(result_table[-1])

    # Print the result table
    print(result_table)


if __name__ == "__main__":
    # Specify the directory path where Python files are located
    directory_path = "./op_test"

    # Execute Python files in the specified directory and display results
    execute_files_in_directory(directory_path)
