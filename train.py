import tensorflow as tf

import stow
import tarfile
from tqdm import tqdm
from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile

def download_and_unzip(url, extract_to='Datasets'):
    response = urlopen(url)

    with open('IAM_Words.zip', 'wb') as file:
        while True:
            chunk = response.read(1024 * 1024)

            if not chunk:
                break

            file.write(chunk)

    with ZipFile('IAM_Words.zip') as zipfile:
        zipfile.extractall(path=extract_to)
    
dataset_path = stow.join('Datasets', 'IAM_Words')
if not stow.exists(dataset_path):
    download_and_unzip('https://git.io/J0fjL', extract_to='Datasets')

    file = tarfile.open(stow.join(dataset_path, "words.tgz"))
    file.extractall(stow.join(dataset_path, "words"))