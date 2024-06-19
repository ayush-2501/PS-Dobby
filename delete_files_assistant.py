from openai import OpenAI
import datetime
from dotenv import load_dotenv
load_dotenv()

client = OpenAI()

'''def upload_file(p):
    filename = input("Enter the filename to upload: ")
    try:
        with open(filename, "rb") as file:
            response = client.files.create(file=file, purpose=p)
            print(response)
            print(f"File uploaded successfully: {response.filename} [{response.id}]")
    except FileNotFoundError:
        print("File not found. Please make sure the filename and path are correct.")'''

def list_files(p):
    response = client.files.list(purpose=p)
    if len(response.data) == 0:
        print("No files found.")
        return
    for file in response.data:
        created_date = datetime.datetime.utcfromtimestamp(file.created_at).strftime('%Y-%m-%d')
        print(f"{file.filename} [{file.id}], Created: {created_date}")

def list_assistants():
    response = client.beta.assistants.list()
    if len(response.data) == 0:
        print("No files found.")
        return
    for assistant in response.data:
        created_date = datetime.datetime.utcfromtimestamp(assistant.created_at).strftime('%Y-%m-%d')
        print(f"{assistant}")

def list_and_delete_file(p):
    while True:
        response = client.files.list(purpose=p)
        files = list(response.data)
        if len(files) == 0:
            print("No files found.")
            return
        for i, file in enumerate(files, start=1):
            created_date = datetime.datetime.utcfromtimestamp(file.created_at).strftime('%Y-%m-%d')
            print(f"[{i}] {file.filename} [{file.id}], Created: {created_date}")
        choice = input("Enter a file number to delete, or any other input to return to menu: ")
        if not choice.isdigit() or int(choice) < 1 or int(choice) > len(files):
            return
        selected_file = files[int(choice) - 1]
        client.files.delete(selected_file.id)
        print(f"File deleted: {selected_file.filename}")

def delete_all_files(p):
    confirmation = input("This will delete all OpenAI files with purpose 'assistants'.\n Type 'YES' to confirm: ")
    if confirmation == "YES":
        response = client.files.list(purpose=p)
        for file in response.data:
            client.files.delete(file.id)
        print("All files with purpose 'assistants' have been deleted.")
    else:
        print("Operation cancelled.")

def delete_all_assistants():
    confirmation = input("This will delete all OpenAI files with purpose 'assistants'.\n Type 'YES' to confirm: ")
    if confirmation == "YES":
        response = client.beta.assistants.list()
        for assistant in response.data:
            client.beta.assistants.delete(assistant.id)
        print("All files with purpose 'assistants' have been deleted.")
    else:
        print("Operation cancelled.")

def main():
    print("\n== What do you want to delete Assistants or Vision? ==")
    print("[1] Assistants")
    print("[2] Vision")
    choice = input("Enter your choice: ")

    if choice == "1":
        p = "assistants"
    elif choice == "2":
        p = "vision"
    else:
        print("Invalid choice. Please try again.")
    while True:
        print("\n== Assistants file utility ==")
        #print("[1] Upload file")
        print("[1] List all files")
        print("[2] List all and delete one of your choice")
        print("[3] List all Assistants")
        print("[4] Delete all Assistants")
        print("[5] Delete all assistant files (confirmation required)")
        print("[9] Exit")
        choice = input("Enter your choice: ")

        if choice == "1":
            list_files(p)
        elif choice == "2":
            list_and_delete_file(p)
        elif choice == "3":
            list_assistants()
        elif choice == "4":
            delete_all_assistants()
        elif choice == "5":
            delete_all_files(p)
        elif choice == "9":
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()