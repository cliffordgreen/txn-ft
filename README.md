# Transaction Categorization Fine-Tuning

This repository contains scripts for fine-tuning language models to categorize financial transactions using multiple-choice options.

## Project Overview

Financial transaction categorization is a common task in personal finance apps, where the goal is to assign the appropriate category to a transaction based on its description and amount. This project improves upon standard categorization by:

1. Presenting the model with 5 potential categories (including the correct one)
2. Using a structured format for responses with reasoning and answers
3. Providing a clear system prompt with instructions

## Repository Contents

- `deepsuck.py` - Original fine-tuning script using unsloth and vllm
- `deepsuck_simplified.py` - Simplified version demonstrating the improved approach
- `fine_tune_with_transformers.py` - Implementation using HuggingFace Transformers
- `show_dataset.py` - Script to demonstrate the dataset format
- `simple_test.py` - Simple script to test the dataset preparation
- `requirements.txt` - Project dependencies
- `setup.sh` - Setup script for creating a virtual environment

## Getting Started

1. Clone this repository
2. Create a virtual environment and install dependencies:
   ```bash
   ./setup.sh
   source venv/bin/activate
   ```
3. Run the demonstration script:
   ```bash
   python show_dataset.py
   ```

## Implementation Details

The fine-tuning approach presents transactions with a structured format:

```
User: Categorize this transaction: Description: STARBUCKS COFFEE #123, Amount: $5.75

Possible categories:
- Personal Care
- Fees & Charges
- Auto & Transport
- Food & Dining
- Insurance