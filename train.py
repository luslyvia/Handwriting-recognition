import tensorflow as tf

from configs import ModelConfigs
from mltu.dataProvider import DataProvider
from mltu.preprocessors import ImageReader
from mltu.transformers import ImageResizer, LabelIndexer, LabelPadding, ImageShowCV2
from mltu.annotations.images import Image

import stow
import tarfile
from tqdm import tqdm
from urllib.request import urlopen
from zipfile import ZipFile

#Download and unzip datasets
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

dataset, vocab, max_len = [], set(), 0

# Preprocess the dataset by the specific IAM_Words dataset file structure
words = open(stow.join(dataset_path, "words.txt"), "r").readlines()
for line in tqdm(words):

    # If the line start with #, skip the line
    if line.startswith("#"):
        continue

    # Split the line by " ", if the second element is "err", skip the line
    line_split = line.split(" ")
    if line_split[1] == "err":
        continue

    # Extracts the first 3 and 8 chars of the file name and the label
    folder1 = line_split[0][:3] # Get the id 
    folder2 = line_split[0][:8]
    file_name = line_split[0] + ".png"
    label = line_split[-1].rstrip('\n') # Get the actual label

    # Constructs the file path
    rel_path = stow.join(dataset_path, "words", folder1, folder2, file_name)
    if not stow.exists(rel_path):
        continue

    # Add file path and label to the datasets
    dataset.append([rel_path, label]) # Dictionary of path, label
    vocab.update(list(label)) # All the char in the label
    max_len = max(max_len, len(label)) # Longest words

# Create a ModelConfigs object to store model configurations
configs = ModelConfigs()

# Save vocab and maximum text length to configs
configs.vocab = "".join(vocab)
configs.max_text_length = max_len
configs.save()

# Create a provider for the dataset
data_provider = DataProvider(
    dataset=dataset,
    skip_validation=True,
    batch_size=configs.batch_size,
    data_preprocessors=[ImageReader(image_class=Image)],
    transformers=[
        ImageResizer(configs.width, configs.height, keep_aspect_ratio=False),
        LabelIndexer(configs.vocab),
        LabelPadding(max_word_length=configs.max_text_length, padding_value=len(configs.vocab)),
    ],
)

