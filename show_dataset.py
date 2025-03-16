"""
Demonstrates the dataset format for transaction categorization fine-tuning.
"""
import random
from datasets import Dataset

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
            "text": input_text + response_text,
            "category_options": options,
            "correct_category": example['category']
        }
    
    # Apply the mapping function
    data = data.map(create_example)
    return data

def test_with_example():
    """Test with an example transaction."""
    print("\n--- Sample Test Example ---")
    example = "CHIPOTLE MEXICAN GRILL"
    amount = 12.49
    
    # Generate random categories including "Food & Dining"
    options = get_category_options("Food & Dining")
    options_text = "- " + "\n- ".join(options)
    
    prompt = f"{SYSTEM_PROMPT}\n\nUser: Categorize this transaction: Description: {example}, Amount: ${amount}\n\nPossible categories:\n{options_text}\n\nAssistant:"
    
    print(prompt)
    print("\nExpected response:")
    print(" <reasoning>\nThis transaction is from CHIPOTLE MEXICAN GRILL for $12.49. Chipotle is a restaurant chain serving Mexican food. Based on the merchant name and amount, this is clearly a Food & Dining transaction.\n</reasoning>\n<answer>\nFood & Dining\n</answer>")

if __name__ == "__main__":
    print("Transaction Categorization Dataset Demo")
    
    # Get and display the dataset
    dataset = get_transaction_dataset()
    print(f"Dataset size: {len(dataset)} examples")
    
    # Display the first 3 examples
    for i in range(min(3, len(dataset))):
        print(f"\n--- Example {i+1} ---")
        print(f"Transaction: {dataset[i]['description']}, Amount: ${dataset[i]['amount']}")
        print(f"Correct category: {dataset[i]['correct_category']}")
        print("Provided category options:")
        for option in dataset[i]['category_options']:
            print(f"- {option}")
        
        print("\nFull prompt + response:")
        print(dataset[i]["text"])
    
    # Show a test example
    test_with_example()
    
    print("\nThis dataset format is compatible with both transformers and unsloth/vllm fine-tuning.")
    print("The improvements from the original script include:")
    print("1. Presenting each transaction with 5 possible categories")
    print("2. Including the correct category among the options")
    print("3. Using a structured format with reasoning and answer sections")
    print("4. Providing clear instructions in the system prompt")