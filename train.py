import tensorflow as tf

try: [tf.config.experimental.set_memory_growth(gpu, True) for gpu in tf.config.experimental.list_physical_devices('GPU')]
except: pass
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau

from configs import ModelConfigs
from model import train_model

from mltu.dataProvider import DataProvider
from mltu.preprocessors import ImageReader
from mltu.transformers import ImageResizer, LabelIndexer, LabelPadding, ImageShowCV2
from mltu.annotations.images import CVImage
from mltu.augmentors import RandomBrightness, RandomRotate, RandomErodeDilate, RandomSharpen
from mltu.tensorflow.losses import CTCloss
from mltu.tensorflow.callbacks import Model2onnx, TrainLogger
from mltu.tensorflow.metrics import CWERMetric

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
    data_preprocessors=[ImageReader(image_class=CVImage)],
    transformers=[
        ImageResizer(configs.width, configs.height, keep_aspect_ratio=False),
        LabelIndexer(configs.vocab), # Transform string to numerical type
        LabelPadding(max_word_length=configs.max_text_length, padding_value=len(configs.vocab)), # All label should be the same size
    ],
)

# Split datasets into training and validation sets
train_data_provider, val_data_provider = data_provider.split(split=0.9)

# Augment training data with random brightness, rotation and erode/dilate
train_data_provider.augmentors = [
    RandomBrightness(),
    RandomErodeDilate(),
    RandomSharpen(),
    RandomRotate(angle=10),
]

# Creating TensorFlow model architecture
model = train_model(
    input_dim = (configs.height, configs.width, 3),
    output_dim = len(configs.vocab),
)

# Compile the model and print summary
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=configs.learning_rate), 
    loss=CTCloss(), 
    metrics=[CWERMetric(padding_token=len(configs.vocab))],
    run_eagerly=False
)
model.summary(line_length=110)

# Define callbacks
earlystopper = EarlyStopping(monitor='val_CER', patience=20, verbose=1, mode='min')
checkpoint = ModelCheckpoint(f"{configs.model_path}/model.h5", monitor='val_CER', verbose=1, save_best_only=True, mode='min')
trainLogger = TrainLogger(configs.model_path)
tb_callback = TensorBoard(f'{configs.model_path}/logs', update_freq=1)
reduceLROnPlat = ReduceLROnPlateau(monitor='val_CER', factor=0.9, min_delta=1e-10, patience=10, verbose=1, mode='min')
model2onnx = Model2onnx(f"{configs.model_path}/model.h5")

def mltu_generator(data_provider):
    while True:
        for batch in data_provider:
            yield batch

# Train the model
model.fit(
    mltu_generator(train_data_provider),
    steps_per_epoch=len(train_data_provider),
    validation_data=mltu_generator(val_data_provider),
    validation_steps=len(val_data_provider),
    epochs=configs.train_epochs,
    callbacks=[
        earlystopper,
        checkpoint,
        trainLogger,
        reduceLROnPlat,
        tb_callback,
        model2onnx
    ]
)

# Save training and validation datasets as csv files
train_data_provider.to_csv(stow.join(configs.model_path, 'train.csv'))
val_data_provider.to_csv(stow.join(configs.model_path, 'val.csv'))



