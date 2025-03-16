# This script was converted from a Jupyter notebook
# Original had pip install commands that we've moved to requirements.txt
# Skip restarting message in Colab
import sys; modules = list(sys.modules.keys())
for x in modules: sys.modules.pop(x) if "PIL" in x or "google" in x else None

from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
max_seq_length = 512 # Can increase for longer reasoning traces
lora_rank = 16 # Larger rank = smarter, but slower

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Phi-4",
    max_seq_length = max_seq_length,
    load_in_4bit = True, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.7, # Reduce if out of memory
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["gate_proj", "up_proj", "down_proj",],
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth", # Enable long context finetuning
    random_state = 3407,
)

import re
from datasets import load_dataset, Dataset

# Load and prep dataset
SYSTEM_PROMPT = """
You are a financial assistant that categorizes transactions.
For each transaction, you will be given the description, amount, and five possible categories.
Choose the most appropriate category from the given options.

Respond in the following format:
<reasoning>
Think step by step about the transaction details and determine the appropriate category from the provided options.
</reasoning>
<answer>
[SELECTED CATEGORY]
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

# uncomment middle messages for 1-shot prompting
def get_gsm8k_questions(split = "train") -> Dataset:
    data = load_dataset('openai/gsm8k', 'main')[split] # type: ignore
    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    }) # type: ignore
    return data # type: ignore

# Replace GSM8K with transaction data
def get_transaction_dataset(split = "train") -> Dataset:
    # Sample transaction data - in a real scenario, you would load from a file or API
    transactions = [
        {"description": "STARBUCKS COFFEE #123", "amount": 5.75, "category": "Food & Dining"},
        {"description": "AMAZON.COM AMZN.COM/BI", "amount": 29.99, "category": "Shopping"},
        {"description": "UBER TRIP 12345", "amount": 18.50, "category": "Transportation"},
        {"description": "NETFLIX.COM", "amount": 13.99, "category": "Entertainment"},
        {"description": "CVS PHARMACY #1234", "amount": 32.47, "category": "Health & Medical"},
        {"description": "WALMART GROCERY", "amount": 87.65, "category": "Groceries"},
        {"description": "GEICO AUTO INSURANCE", "amount": 112.00, "category": "Insurance"},
        {"description": "AT&T WIRELESS", "amount": 85.99, "category": "Bills & Utilities"},
        {"description": "SHELL OIL 12345", "amount": 45.23, "category": "Auto & Transport"},
        {"description": "MARRIOTT HOTELS", "amount": 189.99, "category": "Travel"},
    ]
    
    import random
    
    # Define a function to get 4 random categories different from the correct one
    def get_category_options(correct_category):
        all_categories = [
            "Food & Dining", "Shopping", "Transportation", "Entertainment", 
            "Health & Medical", "Groceries", "Insurance", "Bills & Utilities",
            "Auto & Transport", "Travel", "Income", "Transfer", "Education",
            "Personal Care", "Gifts & Donations", "Fees & Charges"
        ]
        # Remove the correct category from options
        other_categories = [cat for cat in all_categories if cat != correct_category]
        # Select 4 random categories
        random_categories = random.sample(other_categories, 4)
        # Add back the correct category and shuffle
        all_options = random_categories + [correct_category]
        random.shuffle(all_options)
        return all_options
    
    # Create a Dataset object
    data = Dataset.from_dict({
        "description": [t["description"] for t in transactions],
        "amount": [t["amount"] for t in transactions],
        "category": [t["category"] for t in transactions]
    })
    
    # Map function that correctly handles category options
    def create_example(example):
        # Generate options just once per example
        options = get_category_options(example['category'])
        
        # Format the options as a bulleted list
        options_text = "- " + "\n- ".join(options)
        
        return {
            'category_options': options,
            'prompt': [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': f"Categorize this transaction: Description: {example['description']}, Amount: ${example['amount']}\n\nPossible categories:\n{options_text}"}
            ],
            'answer': example['category']
        }
    
    # Apply the mapping function
    data = data.map(create_example)
    
    return data

dataset = get_transaction_dataset()

# Update reward functions for transaction categorization
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    
    # Format the printed output to clearly show transaction, options, and results
    print('-'*40)
    print(f"Transaction:\n{q.split('Possible categories:')[0].strip()}")
    print(f"\nCategory Options:\n{q.split('Possible categories:')[1].strip()}")
    print(f"\nExpected Category:\n{answer[0]}")
    print(f"\nModel Response:\n{responses[0]}")
    print(f"\nExtracted Category:\n{extracted_responses[0]}")
    print('-'*40)
    
    # Reward 2.0 for correct answer, 0.0 for incorrect
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

# Remove the int_reward_func as it's no longer relevant for categorization
# def int_reward_func(completions, **kwargs) -> list[float]:
#     responses = [completion[0]['content'] for completion in completions]
#     extracted_responses = [extract_xml_answer(r) for r in responses]
#     return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

# Add a category validation reward function
def category_validation_reward_func(completions, prompts, **kwargs) -> list[float]:
    """Reward function that checks if the completion has selected one of the provided categories."""
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    
    # Extract the category options from the prompts
    rewards = []
    for i, prompt in enumerate(prompts):
        # Get the user message which contains the category options
        user_message = prompt[-1]['content']
        
        # Extract the categories from the message
        categories_section = user_message.split("Possible categories:\n")[1] if "Possible categories:\n" in user_message else ""
        available_categories = [cat.strip() for cat in categories_section.split("- ") if cat.strip()]
        
        # Check if the response is one of the available categories
        if i < len(extracted_responses):
            rewards.append(0.5 if extracted_responses[i] in available_categories else 0.0)
        else:
            rewards.append(0.0)
            
    return rewards

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

from trl import GRPOConfig, GRPOTrainer
training_args = GRPOConfig(
    use_vllm = True, # use vLLM for fast inference!
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "paged_adamw_8bit",
    logging_steps = 1,
    bf16 = is_bfloat16_supported(),
    fp16 = not is_bfloat16_supported(),
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 1, # Increase to 4 for smoother training
    num_generations = 6, # Decrease if out of memory
    max_prompt_length = 256,
    max_completion_length = 200,
    # num_train_epochs = 1, # Set to 1 for a full training run
    max_steps = 100,
    save_steps = 250,
    max_grad_norm = 0.1,
    report_to = "none", # Can use Weights & Biases
    output_dir = "outputs",
)

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        category_validation_reward_func,  # Replace int_reward_func with category validation
        correctness_reward_func,
    ],
    args = training_args,
    train_dataset = dataset,
)
trainer.train()

# Get random category options for the test transaction
import random
test_options = random.sample([
    "Food & Dining", "Shopping", "Transportation", "Entertainment", 
    "Health & Medical", "Groceries", "Insurance", "Bills & Utilities",
    "Auto & Transport", "Travel"
], 5)

# Update the test prompt to be transaction-related with options
text = tokenizer.apply_chat_template([
    {"role" : "system", "content" : SYSTEM_PROMPT},
    {"role" : "user", "content" : f"Categorize this transaction: Description: CHIPOTLE MEXICAN GRILL, Amount: $12.49\n\nPossible categories:\n- {test_options[0]}\n- {test_options[1]}\n- {test_options[2]}\n- {test_options[3]}\n- {test_options[4]}"},
], tokenize = False, add_generation_prompt = True)

from vllm import SamplingParams
sampling_params = SamplingParams(
    temperature = 0.8,
    top_p = 0.95,
    max_tokens = 1024,
)
output = model.fast_generate(
    [text],
    sampling_params = sampling_params,
    lora_request = None,
)[0].outputs[0].text

output