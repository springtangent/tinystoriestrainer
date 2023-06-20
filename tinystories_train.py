import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:500'
os.environ["WANDB_DISABLED"] = "true"
import torch
from transformers import GPTNeoConfig, GPTNeoForCausalLM, TextDataset, DataCollatorForLanguageModeling, AutoTokenizer
from transformers import Trainer, TrainingArguments, TrainerCallback
from datasets import load_dataset, load_from_disk
from settings import MAX_LENGTH, END_OF_TEXT
import gc
import pynvml
from optimum.bettertransformer import BetterTransformer

def print_gpu_utilization():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()



CHECKPOINT = None # 'results/checkpoint-684000'

# This matchs the TinyStories-1M configuration
configuration = GPTNeoConfig(
    vocab_size=50257,
    max_position_embeddings=MAX_LENGTH,
    hidden_size=768,
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

print_gpu_utilization()

# Create a new model
model = GPTNeoForCausalLM(configuration)
model.to('cuda')
model = BetterTransformer.transform(model)

print_gpu_utilization()

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M", eos_token=END_OF_TEXT, bos_token=END_OF_TEXT)
tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

data = load_from_disk("train_dataset")

print(f"Number of parameters in model {model.num_parameters()}")
print(f"Size of training dataset: {data['train'].num_rows} examples")
print(f"Size of validation dataset: {data['valid'].num_rows} examples")
print_gpu_utilization()

train_dataset = data["train"]
valid_dataset = data["valid"]


training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=1,
    gradient_accumulation_steps=2,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    eval_steps=10000,
    weight_decay=0.01,
    max_grad_norm=1.0,
    evaluation_strategy='steps',
    # save_strategy='epoch',  # The model and tokenizer will be saved at the end of each epoch
    save_steps=10000,
    warmup_steps=500,
    # fp16 = True,
    # gradient_checkpointing=True
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=data_collator
)


def empty_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Clear GPU cache before training


try:
    empty_cache()
    print_gpu_utilization()
    trainer.train(resume_from_checkpoint=CHECKPOINT)
finally:
    empty_cache()
    print_gpu_utilization()
    evaluation_results = trainer.evaluate()

    print("Evaluation Loss: ", evaluation_results["eval_loss"])
    torch.cuda.empty_cache()  # Clear GPU cache after training

    model.save_pretrained("./results")
    tokenizer.save_pretrained("./results")
