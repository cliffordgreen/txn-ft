#!/bin/bash
# Demo script to run the transaction categorization examples

# Ensure we're in a virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Check if required packages are installed
if ! python -c "import datasets" 2>/dev/null; then
    echo "Installing required packages..."
    pip install -r requirements.txt
fi

# Run the simplified dataset demonstration
echo "=== Running Transaction Categorization Dataset Demo ==="
python show_dataset.py

echo ""
echo "=== Demo Options ==="
echo "1. To see a more detailed implementation with transformers:"
echo "   python fine_tune_with_transformers.py"
echo ""
echo "2. To run the original implementation (requires unsloth and vllm):"
echo "   python deepsuck.py"
echo ""
echo "The improved version provides multiple-choice categories, making the task more focused and realistic."