import os
from zipfile import ZipFile

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


if __name__ == '__main__':
    os.makedirs('.env/models', exist_ok=True)
    print("Downloading ResNet model trained on raw data (Experiment I)...")
    download_file_from_google_drive('10KhNPIH80cSkfUMghcqLtw1NZoc9cAIS', '.env/models/retinanet_raw_data.h5')
    print("Downloading ResNet model trained on augmented data  (Experiment II)...")
    download_file_from_google_drive('10KhNPIH80cSkfUMghcqLtw1NZoc9cAIS', '.env/models/retinanet_augmented_data.h5')
    print("Downloading counting (regression) model trained on augmented data  (Experiment IV)...")
    download_file_from_google_drive('10KhNPIH80cSkfUMghcqLtw1NZoc9cAIS', '.env/models/counter.h5')
    print("Downloading counting (regression) model trained on balanced datasets  (Experiment V)...")
    download_file_from_google_drive('10KhNPIH80cSkfUMghcqLtw1NZoc9cAIS', '.env/models/counter_balanced.h5')

    download_file_from_google_drive('0BwSzgS8Mm48Ud2h2dW40Wko3a1E', '../dataset.zip')
    with ZipFile('../dataset.zip') as zf:
        zf.extractall(path='../', pwd='hsieh_iccv17')




