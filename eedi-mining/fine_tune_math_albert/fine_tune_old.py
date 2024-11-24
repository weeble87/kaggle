import logging 
logging.basicConfig(level=logging.DEBUG)
logging.info("starting script")

from transformers import Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset

# Load the tokenizer and model
# Load model directly
from transformers import AutoTokenizer, AutoModelForPreTraining, AlbertForPreTraining, AlbertTokenizer
from datetime import datetime


def print_log (message):
    # get timestamp as string
    # Get the current date and time
    now = datetime.now()
    # Format the date and time as a string
    date_time_string = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f"{date_time_string} {message}")



logging.info("setting up tokenizer")
tokenizer = AlbertTokenizer.from_pretrained("AnReu/math_pretrained_bert")
model = AlbertForPreTraining.from_pretrained("AnReu/math_pretrained_bert")
logging.info('loading model')
# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, truncation=True, padding=True)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
# get data
logging.info('getting data')
model_dir = 'c:/ai_ml/kaggle/eedi-mining/model'
#           'C:\ai_ml\kaggle\eedi-mining\model\flattened_misconceptions.csv'
model_data=f"{model_dir}/flattened_misconceptions.csv"
out_dir = model_dir

# Prepare your dataset (replace with your own data)
texts = ["sample text 1", "sample text 2"]
labels = [0, 1]
dataset = CustomDataset(texts, labels)

 
# Sample input data 
inputs = tokenizer("This is a sample input", return_tensors="pt") 
inputs["labels"] = inputs["input_ids"] # Set labels for loss calculation
logging.debug(inputs)
# Set up training arguments and trainer
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=inputs
)

# Train the model
logging.info("training the model")
trainer.train()

# Save the fine-tuned model
model.save_pretrained('./fine_tuned_math_albert')
