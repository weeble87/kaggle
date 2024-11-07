from transformers import Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset

# Load the tokenizer and model
# Load model directly
from transformers import AutoTokenizer, AutoModelForPreTraining

tokenizer = AutoTokenizer.from_pretrained("AnReu/math_pretrained_bert")
model = AutoModelForPreTraining.from_pretrained("AnReu/math_pretrained_bert")

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
model_dir = 'c:/ai_ml/kaggle/eedi-mining/model'
#           'C:\ai_ml\kaggle\eedi-mining\model\flattened_misconceptions.csv'
model_data=f"{model_dir}/flattened_misconceptions.csv"
out_dir = model_dir

# Prepare your dataset (replace with your own data)
texts = ["sample text 1", "sample text 2"]
labels = [0, 1]
dataset = CustomDataset(texts, labels)

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
    train_dataset=dataset,
    eval_dataset=dataset,
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained('./fine_tuned_math_albert')
