import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:250'
import torch
print(torch.__version__)

from transformers import GPTNeoConfig, GPTNeoForCausalLM, TextDataset, DataCollatorForLanguageModeling, AutoTokenizer
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
import logging
import pickle

CHECKPOINT = None # 'results/checkpoint-684000'

logging.basicConfig(level=logging.INFO)

# This maatchs the TinyStories-1M configuration
configuration = GPTNeoConfig(
    vocab_size=50257,
    max_position_embeddings=2048,
    hidden_size=64,
    num_layers=8,
    num_heads=16,
    activation_function="gelu_new",
    attention_types=[[["global", "local"], 4]],
    attention_layers=['global', 'local'] * 4,
    attention_dropout=0,
    embed_dropout=0,
    resid_dropout=0,
    window_size=256
)

# Create a new model
model = GPTNeoForCausalLM(configuration)

print(model.num_parameters())

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M", eos_token='<|endoftext|>', bos_token='<|endoftext|>')
tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

from datasets import load_from_disk

data = load_from_disk("train_dataset")
train_dataset = data["train"]
test_dataset = data["test"]

training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=4,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    eval_steps=6000,
    weight_decay=0.01,
    max_grad_norm=1.0,
    evaluation_strategy='steps',
    # save_strategy='epoch',  # The model and tokenizer will be saved at the end of each epoch
    save_steps=6000,
    warmup_steps=500,
    fp16 = True
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
)


try:
    trainer.train(resume_from_checkpoint=CHECKPOINT)
    torch.cuda.empty_cache()  # Clear GPU cache after training
finally:
    evaluation_results = trainer.evaluate()

    print("Evaluation Loss: ", evaluation_results["eval_loss"])
    torch.cuda.empty_cache()  # Clear GPU cache after training

    model.save_pretrained("./results")
    tokenizer.save_pretrained("./results")
