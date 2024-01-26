from transformers import MixtralForCausalLM, MixtralConfig, LlamaTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_from_disk, DatasetDict, Features, Value, Sequence
from torch.utils.data import DataLoader
import torch
import pynvml
import json
from tqdm import tqdm
import settings
import logging

logging.basicConfig(level=logging.DEBUG)


pynvml.nvmlInit()


CHECKPOINT = None # 'checkpoint-50000' # 'checkpoint-70000'
results_directory = 'mixtral_results'
DEVICE = 'cuda'
tokenizer = settings.tokenizer
tokenizer.pad_token = tokenizer.eos_token
dataset_name = 'prepared_tinystories2'
train_dataset_name = 'train'
validation_dataset_name = 'valid'
features = Features({
    'input_ids': Sequence(feature=Value(dtype='int32')),
    'attention_mask': Sequence(feature=Value(dtype='int32')),
    # Define other features if you have them
})



def print_gpu_utilization():
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()


def empty_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Clear GPU cache before training


def dataset_metrics(dataset):
	print(f"Size of training dataset: {dataset[train_dataset_name].num_rows} examples")
	if validation_dataset_name in dataset:
		print(f"Size of validation dataset: {dataset[validation_dataset_name].num_rows} examples")


configuration_9m = MixtralConfig(
	vocab_size = tokenizer.vocab_size,
	hidden_size = 128,
	intermediate_size = 1024,
	num_hidden_layers = 2,
	num_attention_heads = 2,
	num_key_value_heads = 2,
	hidden_act = 'silu',
	max_position_embeddings = settings.MAX_LENGTH,
	initializer_range = 0.02,
	rms_norm_eps = 1e-06,
	use_cache = True,
	pad_token_id = None,
	bos_token_id = 1,
	eos_token_id = 2,
	pretraining_tp = 1,
	tie_word_embeddings = False,
	rope_theta = 10000.0,
	rope_scaling = None,
	attention_bias = False
)

configuration = configuration_9m

# Initializing a model from the llama-7b style configuration, or from pretrained if CHECKPOINT.
checkpoint_dir = f'{results_directory}/{CHECKPOINT}' if CHECKPOINT is not None else None
if checkpoint_dir is not None:
	print('loading checkpoint')
	model = MixtralForCausalLM.from_pretrained(checkpoint_dir)
else:
	model = MixtralForCausalLM(configuration)
    
print(f"Number of parameters in model {model.num_parameters()}")
print_gpu_utilization()

model.to(DEVICE)
print_gpu_utilization()

dataset = load_from_disk(dataset_name)
dataset_metrics(dataset)

training_args = TrainingArguments(
    output_dir=results_directory,
    overwrite_output_dir=True,
    num_train_epochs=1,
    gradient_accumulation_steps=4,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    eval_steps=100000,
    weight_decay=0.01,
    max_grad_norm=1.0,
    evaluation_strategy='steps',
    save_steps=10000,
    warmup_steps=500
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset[train_dataset_name],
    eval_dataset=dataset[validation_dataset_name],
    data_collator=data_collator
)

results = None
try:
	results = trainer.train(resume_from_checkpoint=checkpoint_dir)
except:
	raise
finally:
	model.save_pretrained("./saved_model")
	tokenizer.save_pretrained("./saved_model")
	# training_args.save_to_json("./saved_model/training_args.json")
	if results:
		with open('./saved_model/training_results.json', 'w') as result_file:
		    json.dump(results, result_file)
