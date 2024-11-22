import os
import requests
import shutil
import zipfile

# Get the root directory of the project
root_dir = os.path.abspath(os.path.join(os.getcwd()))

# Directory to create data directory
data_dir = os.path.join(root_dir, 'data')
audio_dir = os.path.join(data_dir, 'audio')
spectrograms_dir = os.path.join(data_dir, 'Spectrograms')

# URLs for downloading the zip files
metadata_url = 'https://os.unil.cloud.switch.ch/fma/fma_metadata.zip'
small_url = 'https://os.unil.cloud.switch.ch/fma/fma_small.zip'

def create_directories():
    # Create data directory
    if not os.path.isdir(data_dir):
        print(f"Creating data directory at {data_dir}")
        os.makedirs(data_dir)
        print("Created data directory")

    # Create audio directory
    if not os.path.isdir(audio_dir):
        print(f"Creating audio directory at {audio_dir}")
        os.makedirs(audio_dir)
        print("Created audio directory")

    # Create Spectrograms directory
    if not os.path.isdir(spectrograms_dir):
        print(f"Creating Spectrograms directory at {spectrograms_dir}")
        os.makedirs(spectrograms_dir)
        print("Created Spectrograms directory")

def download_file(url, dest):
    print(f"Downloading {url} to {dest}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(dest, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    print(f"Downloaded {dest}")

def extract_file(zip_file, extract_to):
    print(f"Extracting {zip_file} to {extract_to}")
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    os.remove(zip_file)
    print(f"Extracted and removed {zip_file}")

def move_files(source, destination):
    for item in os.listdir(source):
        source_path = os.path.join(source, item)
        destination_path = os.path.join(destination, item)
        shutil.move(source_path, destination_path)
    # Remove the empty source directory
    os.rmdir(source)

def main():
    # Check if the data directory exists
    if not os.path.exists(data_dir):
        try:
            # Create necessary directories
            create_directories()

            # Download and extract fma_metadata.zip
            metadata_zip = os.path.join(data_dir, 'fma_metadata.zip')
            download_file(metadata_url, metadata_zip)
            # Extract without creating an additional folder
            extract_file(metadata_zip, data_dir)
            # Move the extracted files to the parent directory
            move_files(os.path.join(data_dir, 'fma_metadata'), data_dir)

            for unwanted in ['README.txt', 'checksums', 'not_found.pickle']:
                unwanted_path = os.path.join(data_dir, unwanted)
                if os.path.exists(unwanted_path):
                    if os.path.isfile(unwanted_path):
                        os.remove(unwanted_path)
                        print(f"Removed file {unwanted}")
                    elif os.path.isdir(unwanted_path):
                        shutil.rmtree(unwanted_path)
                        print(f"Removed directory {unwanted}")

            # Download and extract fma_small.zip
            small_zip = os.path.join(data_dir, 'fma_small.zip')
            download_file(small_url, small_zip)
            extract_file(small_zip, audio_dir)

            # Remove unwanted files
            for unwanted in ['README.txt', 'checksums']:
                unwanted_path = os.path.join(os.path.join(audio_dir, 'fma_small'), unwanted)
                if os.path.exists(unwanted_path):
                    if os.path.isfile(unwanted_path):
                        os.remove(unwanted_path)
                        print(f"Removed file {unwanted}")
                    elif os.path.isdir(unwanted_path):
                        shutil.rmtree(unwanted_path)
                        print(f"Removed directory {unwanted}")

            # Move the extracted files to the parent directory
            move_files(os.path.join(audio_dir, 'fma_small'), audio_dir)

            # Move each audio file to audio
            for item in os.listdir(audio_dir):
                move_files(os.path.join(audio_dir,item),audio_dir)

            print("Setup completed successfully.")
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            print("Deleting data directory.")
            try:
                os.chdir(root_dir)  # Change to a safe directory before attempting to delete
                if os.path.isdir(data_dir):
                    shutil.rmtree(data_dir)
                print("Data directory deleted.")
            except Exception as cleanup_error:
                print(f"An error occurred during cleanup: {str(cleanup_error)}")
    else:
        print("Data directory already exists. Setup skipped.")
