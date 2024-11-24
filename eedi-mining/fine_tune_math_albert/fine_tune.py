import logging 
logging.basicConfig(level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
logging.info("starting script")

from transformers import AlbertTokenizer
import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from datasets import Dataset as ds
# Load the tokenizer
tokenizer = AlbertTokenizer.from_pretrained("AnReu/math_albert")

# manually download the dataset
"""
logging.info("manually download the dataset")
import requests
import tarfile
from datasets import load_from_disk

# URL to the dataset
url = "https://storage.googleapis.com/mathematics-dataset/mathematics_dataset-v1.0.tar.gz"

# Download the dataset
response = requests.get(url)
with open("mathematics_dataset-v1.0.tar.gz", "wb") as file:
    file.write(response.content)

# Extract the dataset
with tarfile.open("mathematics_dataset-v1.0.tar.gz", "r:gz") as tar:
    tar.extractall()

# Load the dataset from the extracted files
dataset = load_from_disk("mathematics_dataset-v1.0")

# Display some information about the dataset
print(dataset)
"""
# Define a custom dataset class
class MathMisconceptionsDataset(Dataset) :
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, truncation=True, padding=True)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

# Load the dataset
logging.info("loadding dataset")
model_dir = 'c:/ai_ml/kaggle/eedi-mining/model'
model_dataset=f"{model_dir}/math_misconceptions_dataset.pt"
# dataset = load_dataset(path="math_dataset", name='algebra__linear_1d', trust_remote_code=True)  # Replace with your actual dataset
dataset = torch.load(model_dataset) 

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True,  return_tensors="pt")

tokenized_datasets = dataset.map(tokenize_function, batched=True)

from transformers import AlbertForSequenceClassification, TrainingArguments, Trainer

# Load the model
model = AlbertForSequenceClassification.from_pretrained("AnReu/math_albert", num_labels=2)
outputs = model(**tokenized_datasets)
loss = outputs.loss
logging.info('loss',loss)
# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch"
)

# Initialize the Trainer
logging.info("initialize trainer")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer
)

# Train the model
logging.info('training the model')
trainer.train()

logging.info('evaluate model')
results = trainer.evaluate()
print(results)

