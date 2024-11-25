from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import torch

# Step 1: Load the saved model and tokenizer
model_name = "./pretrained/clonedetection"  # Directory where you saved the fine-tuned model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Step 2: Load the dataset from your CSV file
def load_dataset(file_path):
    df = pd.read_csv(file_path)
    # Convert the dataframe to a Hugging Face dataset
    dataset = Dataset.from_pandas(df)
    return dataset

# Step 3: Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['code1'], examples['code2'], padding="max_length", truncation=True)

# Load your fine-tuning dataset
dataset = load_dataset("dataset/train.csv")

# Apply tokenization to the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Step 4: Split the dataset into train and validation sets
train_dataset, eval_dataset = train_test_split(tokenized_datasets, test_size=0.1)

# Make sure the datasets are in the proper format
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Step 5: Define training arguments
training_args = TrainingArguments(
    output_dir='./results',          # Output directory to save models
    evaluation_strategy="epoch",     # Evaluate at the end of each epoch
    learning_rate=2e-5,              # Learning rate
    per_device_train_batch_size=16,  # Batch size for training
    per_device_eval_batch_size=16,   # Batch size for evaluation
    num_train_epochs=3,              # Number of epochs
    weight_decay=0.01,               # Weight decay
    save_steps=10_000,               # Save checkpoint every 10,000 steps
    save_total_limit=2,              # Limit the total number of checkpoints
)

# Step 6: Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Step 7: Start the fine-tuning process
trainer.train()


# Step 8: Save the fine-tuned model
model.save_pretrained("./our_model")
tokenizer.save_pretrained("./our_model")

print("Fine-tuned model saved to ./our_model")
