import os
import subprocess

import requests


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def replace_in_file(filename, text_to_search, replacement_text):
    import fileinput

    with fileinput.FileInput(filename, inplace=True, backup='.bak') as file:
        for line in file:
            print(line.replace(text_to_search, replacement_text), end='')


if __name__ == '__main__':
    os.makedirs('.env/models', exist_ok=True)
    print("Downloading ResNet model trained on raw data (Experiment I)...")
    download_file_from_google_drive('1-qILYi4tw8PxgKIvks0oKT_a8SZQ3WKY', '.env/models/retinanet_raw_data.h5')
    print("Downloading ResNet model trained on augmented data  (Experiment II)...")
    download_file_from_google_drive('191JRKHiuyhMRm6WyoYN8RfN9ROYLgEKk', '.env/models/retinanet_augmented_data.h5')

    download_file_from_google_drive('1pAqLYcVNM9gW473b21aL9zPzh1iv7AiW', '.env/models/retinanet_validation.txt')

    replace_in_file('.env/models/retinanet_validation.txt', '../datasets/', '.env/datasets')

    print("Downloading counting (regression) model trained on augmented data  (Experiment IV)...")
    download_file_from_google_drive('1-HY9YeCsZo6dZYRLeFHcF0GHZCtWga4r', '.env/models/counter.h5')
    print("Downloading counting (regression) model trained on augmented balanced datasets  (Experiment V)...")
    download_file_from_google_drive('1-1GSVl4Yk6acVjZB8s4Jb9Df1vpzlYey', '.env/models/counter_balanced.h5')

    print("Downloading datasets...")
    download_file_from_google_drive('0BwSzgS8Mm48Ud2h2dW40Wko3a1E', '.env/dataset.zip')
    subprocess.run(["unzip", "-P", "hsieh_iccv17", ".env/dataset.zip", "-d", ".env/"])