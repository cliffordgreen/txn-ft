# CLAUDE.md - Project Guidelines

## Project Overview
Financial transaction categorization using LLM fine-tuning with Python, Unsloth, and vLLM.

## Key Commands
- **Run model**: `python deepsuck.py`
- **Install dependencies**: `pip install unsloth vllm pillow datasets torch`
- **Tests**: No formal test structure detected

## Code Style Guidelines
- **Imports**: Group standard library, then third-party packages
- **Type Hints**: Use Python type hints (str, list, etc.) for function parameters and returns
- **Error Handling**: Use structured error handling with clear messages
- **Naming**: 
  - Functions: snake_case (e.g., extract_xml_answer)
  - Variables: snake_case (e.g., max_seq_length)
  - Constants: UPPER_SNAKE_CASE (e.g., SYSTEM_PROMPT)
- **String Formatting**: f-strings preferred over .format()
- **Regex**: Prefer compiled patterns for performance with re.compile()
- **Model Parameters**: Keep hyperparameters like batch size, learning rate in a central config

## Notes
- Project uses XML structure for response formatting with <reasoning> and <answer> tags
- Core functionality is transaction categorization to predefined categories