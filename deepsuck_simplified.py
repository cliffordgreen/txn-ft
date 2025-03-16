"""
Simplified version of the fine-tuning script with enhanced transaction categorization
featuring multi-choice categorization
"""
import random
import re
from datasets import Dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer
from transformers import TrainingArguments

# Define system prompt
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

# Define the extraction functions
def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

# Define all valid categories
ALL_CATEGORIES = [
    "Food & Dining", "Shopping", "Transportation", "Entertainment", 
    "Health & Medical", "Groceries", "Insurance", "Bills & Utilities",
    "Auto & Transport", "Travel", "Income", "Transfer", "Education",
    "Personal Care", "Gifts & Donations", "Fees & Charges"
]

# Define function to get random category options
def get_category_options(correct_category):
    # Remove the correct category from options
    other_categories = [cat for cat in ALL_CATEGORIES if cat != correct_category]
    # Select 4 random categories
    random_categories = random.sample(other_categories, 4)
    # Add back the correct category and shuffle
    all_options = random_categories + [correct_category]
    random.shuffle(all_options)
    return all_options

# Sample transaction data
def get_transaction_dataset() -> Dataset:
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
        
        # Create the user prompt
        user_prompt = f"Categorize this transaction: Description: {example['description']}, Amount: ${example['amount']}\n\nPossible categories:\n{options_text}"
        
        # Create the expected response
        expected_response = f"<reasoning>\nThis transaction is from {example['description']} for ${example['amount']}. Based on the merchant name and amount, this is clearly a {example['category']} transaction.\n</reasoning>\n<answer>\n{example['category']}\n</answer>"
        
        return {
            'category_options': options,
            'prompt': user_prompt,
            'completion': expected_response
        }
    
    # Apply the mapping function
    data = data.map(create_example)
    return data

# Create and display dataset with categories
dataset = get_transaction_dataset()

# Print a few examples to demonstrate the format
for i, example in enumerate(dataset):
    print(f"\n--- Example {i+1} ---")
    print(f"Transaction: {example['description']}, Amount: ${example['amount']}")
    print(f"Correct category: {example['category']}")
    print("Provided category options:")
    for option in example['category_options']:
        print(f"- {option}")
    print("\nPrompt:")
    print(example['prompt'])
    print("\nExpected completion:")
    print(example['completion'])
    
    if i >= 2:  # Just show the first 3 examples
        break

# Load a small model for demonstration (not actual training)
print("\nLoading model for demonstration (not doing actual training)...")
try:
    model_name = "microsoft/phi-2"  # Using a small model for demonstration
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create a test input
    test_input = f"{SYSTEM_PROMPT}\n\nUser: Categorize this transaction: Description: CHIPOTLE MEXICAN GRILL, Amount: $12.49\n\nPossible categories:\n- Auto & Transport\n- Food & Dining\n- Entertainment\n- Shopping\n- Travel"
    
    # Print tokenized input
    print(f"\nExample input for model:")
    print(test_input)
    
    print("\nTraining would use standard SFT or RLHF techniques")
    print("Full training would require more computational resources and specialized libraries like unsloth/vllm")
    
except Exception as e:
    print(f"Error loading model: {e}")
    print("That's okay - this is just a demonstration of the data format")