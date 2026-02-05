# BADM500: Fine-tuning Gemma 270M for Tool Calling with TRL


<p align="center">
  <img src="assets/overview.png" alt="Description" width="400"/>
</p>


This example demonstrates how to fine-tune Google's Gemma 270m model for tool calling using the TRL (Transformer Reinforcement Learning) library form Hugginface

## Overview 

This example of training pipeline includes:
1. **Synthetic Dataset Creation**: 10 examples covering weather queries, calculations, web searches, and regular conversation
2. **4-bit Quantization**: Memory-efficient training using bitsandbytes
3. **Completion-Only Training**: Only compute loss on assistant responses

## Installation
You can use all commond environment managers: `uv`, `pip`, or `conda` 

Example for pip (prepend `uv` for using uv): 
```bash
pip install -r requirements.txt
```

## Dataset Format
The training data uses a simple format with special tokens:

```
<|user|>
What's the weather in Paris?
<|assistant|>
<tool_call>get_weather(location='Paris', unit='celsius')</tool_call>
```

For regular conversations without tool calls:
```
<|user|>
Hello, how are you?
<|assistant|>
I'm doing well, thank you! How can I help you today?
```

## Available Tools in Dataset

1. **get_weather**: Get current weather for a location
2. **calculate**: Perform mathematical calculations
3. **search_web**: Search the web for information

## Running the Training
The training is rooted in `train.py`. 
```bash
python train.py
```
To see how to parse arguments use:
```bash
python train.py --help
```

### Training Configuration
Example configuration:
- **Model**: google/gemma-3-270m-it (instruction-tuned)
- **Batch Size**: 2 (with gradient accumulation of 4)
- **Learning Rate**: 2e-4
- **Epochs**: 30
- **Optimizer**: adamw
We encourage students to explore different settings and new parameters! E.g. look into other optimizers, LR-Schedulars, even batch-size can help performance.

## Key Components Explained

## Expected Output

After training, the model should be able to:
- Recognize when to call tools vs respond normally
- Extract correct parameters from user queries
- Format tool calls properly
- Handle multiple tool calls in one response

## Memory Requirements

- **With 4-bit quantization**: ~8-10 GB VRAM
- **Without quantization**: ~20-24 GB VRAM

## Output Files

After training, you'll find:
- `.gemma-270m-tool-calling`: Checkpoints during training
- `.gemma-270m-tool-calling`: Final trained model with 

## Using the Trained Model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("gemma-270m-tool-calling")
tokenizer = AutoTokenizer.from_pretrained("gemma-270m-tool-calling")

# Inference
prompt = "<|user|>\nWhat's the weather in Tokyo?\n<|assistant|>\n"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

## Tips for Better Results

1. **Increase dataset size**: 100-1000 examples work much better
2. **Add variety**: Include edge cases, ambiguous queries, multi-turn conversations
3. **Balance dataset**: Equal distribution of tool calls vs regular responses
4. **Train longer**: More epochs or larger datasets need more training
5. **Tune hyperparameters**: Experiment with learning rate, batch size

## Alternative Formats

Instead of the simple format used here, you could also:
- Use JSON format for tool calls
- Follow OpenAI's function calling format
- Use XML-style tags
- Implement chain-of-thought reasoning before tool calls

### Tools that your project should support 
````Python
    tools = [
        {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "location": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            }
        },
        {
            "name": "correct_grammar",
            "description": "Correct grammatical errors in text using GEC API",
            "parameters": {
                "text": {"type": "string", "description": "Text to check and correct"}
            }
        },
        {
            "name": "generate_image",
            "description": "Generate an image from a text description",
            "parameters": {
                "prompt": {"type": "string", "description": "Description of the image to generate"},
                "style": {"type": "string", "description": "Art style (optional)"}
            }
        },
        {
            "name": "text_to_speech",
            "description": "Convert text to speech audio",
            "parameters": {
                "text": {"type": "string", "description": "Text to convert to speech"},
                "voice": {"type": "string", "description": "Voice type (male/female/neutral)"}
            }
        },
        {
            "name": "search_web",
            "description": "Search the web for information",
            "parameters": {
                "query": {"type": "string", "description": "Search query"}
            }
        }
    ]
    
```



## Common Issues

**Out of Memory**: Reduce batch size or increase gradient accumulation steps
**Poor performance**: Add more diverse training examples
**Model not calling tools**: Ensure balanced dataset with both tool and non-tool examples

## Resources

- [TRL Documentation](https://huggingface.co/docs/trl)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [Gemma Model Card](https://huggingface.co/google/gemma-2-270m-it)

