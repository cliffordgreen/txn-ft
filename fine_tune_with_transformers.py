"""
Fine-tuning a model for transaction categorization using transformers.
This script uses the Hugging Face Transformers library to fine-tune a model
for selecting the appropriate category from a set of options.
"""
import random
import json
import os
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling,
    Trainer
)
from trl import SFTTrainer

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
        
        # Create the input prompt
        input_text = f"{SYSTEM_PROMPT}\n\nUser: Categorize this transaction: Description: {example['description']}, Amount: ${example['amount']}\n\nPossible categories:\n{options_text}\n\nAssistant:"
        
        # Create the expected response
        response_text = f" <reasoning>\nThis transaction is from {example['description']} for ${example['amount']}. Based on the merchant name and amount, this is clearly a {example['category']} transaction.\n</reasoning>\n<answer>\n{example['category']}\n</answer>"
        
        return {
            "input_text": input_text,
            "response_text": response_text,
            "text": input_text + response_text
        }
    
    # Apply the mapping function
    data = data.map(create_example)
    return data

def train_model(save_dir="finetuned-model"):
    """Train a model on the transaction categorization dataset."""
    # Prepare dataset
    dataset = get_transaction_dataset()
    print(f"Dataset size: {len(dataset)} examples")
    
    # Display a sample
    print("\n--- Example ---")
    print(dataset[0]["text"])
    
    # Load a small model (for demonstration)
    model_name = "microsoft/phi-2"  # You can replace with a larger model
    
    # Load tokenizer and model
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        print(f"Loading model: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Prepare for training
        output_dir = save_dir
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            save_steps=10,
            logging_steps=10,
            learning_rate=2e-5,
            weight_decay=0.01,
            warmup_steps=5,
            optim="adamw_torch",
            bf16=False,  # Set to True if hardware supports it
            save_total_limit=3,
        )
        
        # Set up the SFT trainer
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
            max_seq_length=1024,
            dataset_text_field="text",
        )
        
        print("Starting training...")
        # For demonstration, we don't actually run the training
        # trainer.train()
        
        # Save the model
        # model.save_pretrained(output_dir)
        # tokenizer.save_pretrained(output_dir)
        
        print(f"Training would save the model to {output_dir}")
        
    except Exception as e:
        print(f"Error in training: {e}")
        print("This is expected - we're just demonstrating the setup")

def test_with_example():
    """Test the model with an example transaction."""
    print("\n--- Testing with an example transaction ---")
    example = "CHIPOTLE MEXICAN GRILL"
    amount = 12.49
    
    # Generate random categories including "Food & Dining"
    options = get_category_options("Food & Dining")
    options_text = "- " + "\n- ".join(options)
    
    prompt = f"{SYSTEM_PROMPT}\n\nUser: Categorize this transaction: Description: {example}, Amount: ${amount}\n\nPossible categories:\n{options_text}\n\nAssistant:"
    
    print(prompt)
    print("\nExpected response would be something like:")
    print(" <reasoning>\nThis transaction is from CHIPOTLE MEXICAN GRILL for $12.49. Chipotle is a restaurant chain serving Mexican food. Based on the merchant name and amount, this is clearly a Food & Dining transaction.\n</reasoning>\n<answer>\nFood & Dining\n</answer>")

if __name__ == "__main__":
    print("Transaction Categorization Fine-Tuning Demo")
    train_model()
    test_with_example()
    print("\nNote: This script demonstrates the setup but doesn't actually run the training.")
    print("To run actual training, uncomment the trainer.train() line and provide a suitable model.")