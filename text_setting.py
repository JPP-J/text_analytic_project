import nltk
import os

# Download relative nltk only 1 time
def nltk_download():
    # Get the project's root directory dynamically
    project_root = os.path.dirname(os.path.abspath(__file__))
    print(project_root)

    # Construct the path to nltk_data within the project
    nltk_data_path = os.path.join(project_root, ".venv", "Lib", "nltk_data")

    # Append the constructed path to nltk's data path
    nltk.data.path.append(nltk_data_path)

    # Check if the path was added successfully
    print(nltk.data.path)

    nltk.download('punkt', download_dir=nltk_data_path)
    nltk.download('stopwords',download_dir=nltk_data_path)
    nltk.download('punkt_tab', download_dir=nltk_data_path)

if __name__ == "__main__":
    nltk_download()





