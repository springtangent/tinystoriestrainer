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
from tqdm import tqdm

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

device = 'cuda'

# Create a new model
if not CHECKPOINT:
    model = GPTNeoForCausalLM(configuration)
else:
    model = GPTNeoForCausalLM.from_pretrained(f'./results/{CHECKPOINT}')
model.to(device)
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
    eval_steps=40000,
    weight_decay=0.01,
    max_grad_norm=1.0,
    evaluation_strategy='steps',
    save_steps=10000,
    warmup_steps=500,
    fp16=True
)


def empty_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Clear GPU cache before training



import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.optim import AdamW

# Initialize the AdamW optimizer with weight decay
optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=training_args.weight_decay)

def evaluate(model, eval_loader):
    model.eval()
    eval_loss = 0
    with torch.no_grad():
        for eval_batch in tqdm(eval_loader, desc="Eval"):
            eval_batch = {k: v.to(device) for k, v in eval_batch.items()} 
            eval_outputs = model(**eval_batch)
            eval_loss += eval_outputs.loss.item()

    eval_loss /= len(eval_loader)
    return eval_loss


def train(model, train_dataset, eval_dataset, training_args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=training_args.per_device_train_batch_size, shuffle=True, collate_fn=data_collator)
    eval_loader = DataLoader(eval_dataset, batch_size=training_args.per_device_eval_batch_size, collate_fn=data_collator)

    optimizer = AdamW(model.parameters(), weight_decay=training_args.weight_decay)
    total_steps = len(train_loader) * training_args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=training_args.warmup_steps, num_training_steps=total_steps)

    global_step = 0
    start_epoch = 0

    if CHECKPOINT:
        start_epoch, best_val_loss, global_step = load_most_recent_checkpoint(model, optimizer, scheduler)
        
    model.zero_grad()

    for epoch in range(start_epoch, training_args.num_train_epochs):
        for batch in tqdm(train_loader, desc="Training"):
            print('train')
            model.train()
            batch = {k: v.to(device) for k, v in batch.items()}  
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            if global_step != 0 and global_step % training_args.gradient_accumulation_steps == 0:
                print('gradient accum')
                torch.nn.utils.clip_grad_norm_(model.parameters(), training_args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            if global_step != 0 and global_step % training_args.eval_steps == 0:
                print('eval')
                eval_loss = evaluate(model, eval_loader)
                print(f"Step: {global_step}, Eval Loss: {eval_loss}")

            if global_step != 0 and global_step % training_args.save_steps == 0:
                checkpoint_dir = f"{training_args.output_dir}/checkpoint-{global_step}"
                # torch.save(model.state_dict(), f"{training_args.output_dir}/checkpoint-{global_step}.pt")
                m = BetterTransformer.reverse(model)
                m.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)
            global_step += 1


try:
    empty_cache()
    print_gpu_utilization()
    # trainer.train(resume_from_checkpoint=CHECKPOINT)
    train(model, train_dataset, valid_dataset, training_args)
finally:
    empty_cache()
    print_gpu_utilization()
    # evaluation_results = trainer.evaluate()
    eval_loader = DataLoader(valid_dataset, batch_size=training_args.per_device_eval_batch_size)
    evaluation_results = evaluate(model, eval_loader)
    print("Evaluation Loss: ", evaluation_results["eval_loss"])
    torch.cuda.empty_cache()  # Clear GPU cache after training

    model = BetterTransformer.reverse(model)
    model.save_pretrained("./results")
    tokenizer.save_pretrained("./results")
