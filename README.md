# Julia - Personalized AI Assistant
Julia is a personalized AI assistant built using Small Language Models (SLMs) for natural language understanding, task-oriented interactions, and command generation. The project investigates fine-tuning, quantization, and deployment strategies for efficient AI assistants capable of running on consumer hardware and edge computing environments.

## Overview
The goal of Julia is to explore how compact language models can be adapted to perform assistant-like tasks while maintaining low computational requirements.

The assistant is capable of:
* Understanding natural language requests
* Recognizing user intents
* Generating task-specific responses
* Producing executable Python actions from user commands
* Integrating with external tools and automation workflows
* Operating efficiently on resource-constrained hardware

## Architecture
Julia follows a dual-output architecture where the model generates:

1. Natural language responses.
2. Executable Python actions derived from user requests.

This approach enables both conversational interaction and task automation.

## Project Evolution
### Version 1
**Model:** Phi-3 Mini 128K Instruct

**Implementation:** Python

The first version of Julia focused on validating the feasibility of using Small Language Models as personal assistants.

Main objectives:
* Instruction-following evaluation
* Command generation
* Latency analysis
* Resource consumption analysis
* LoRA fine-tuning experimentation

**Dataset:** treino_julia.jsonl

---

### Version 2

**Models:** Qwen 2.5 Series

**Implementation:** Jupyter Notebook

The second version introduced an improved training dataset and a broader model evaluation process.

Main objectives:

* Improve intent recognition
* Improve command generation
* Improve response quality
* Evaluate different model sizes
* Analyze performance versus resource consumption
* Investigate quantization strategies

**Dataset:** treino_julia_2.jsonl

## Evaluated Models

The following models were tested throughout the project:

| Model                    | Parameters |
| ------------------------ | ---------- |
| Qwen 2.5 Instruct        | 0.5B       |
| Qwen 2.5 Instruct        | 1.5B       |
| Qwen 2.5 Instruct        | 3B         |
| Phi-3 Mini 128K Instruct | 3.8B       |

The maximum model size evaluated was constrained by the available GPU memory (GTX 1650 4GB).

## Model Benchmarking

The models were evaluated according to:

* Response quality
* Instruction-following capability
* Reasoning performance
* Inference latency
* GPU memory consumption
* Fine-tuning efficiency
* Deployment feasibility

### Results

#### Qwen 2.5 0.5B Instruct

**Pros**

* Very low memory consumption
* Fast inference
* Suitable for highly constrained environments

**Cons**

* Reduced reasoning capability
* Lower instruction-following performance

---

#### Qwen 2.5 3B Instruct

**Pros**

* Highest overall performance
* Better reasoning ability
* More accurate responses

**Cons**

* Higher memory requirements
* Increased inference latency

---

#### Qwen 2.5 1.5B Instruct

**Pros**

* Strong instruction-following performance
* Good reasoning capability
* Moderate memory requirements
* Efficient inference speed

**Conclusion**

Among the evaluated models, Qwen 2.5 1.5B Instruct provided the best balance between response quality, computational requirements, and deployment feasibility, making it the preferred model for the Julia assistant.

## Fine-Tuning

The project explores parameter-efficient fine-tuning techniques including:

* LoRA (Low-Rank Adaptation)
* Instruction tuning
* Custom conversational datasets

The fine-tuning process focused on:

* Intent recognition
* Command interpretation
* Action generation
* Assistant behavior consistency

## Optimization

Several optimization techniques were investigated to improve deployment efficiency.

### Quantization

The evaluated models were quantized using:

```python
quantization_method = "q8_0"
```

The Q8_0 quantization format was selected because it provided a good balance between memory efficiency and model quality.

Benefits observed:

* Reduced VRAM usage
* Faster inference
* Lower hardware requirements
* Minimal degradation in response quality

This approach enabled experimentation with larger models while maintaining compatibility with consumer-grade GPUs.

## Technologies

### Artificial Intelligence

* Natural Language Processing (NLP)
* Small Language Models (SLMs)
* Transformer Architectures
* LoRA (Low-Rank Adaptation)
* Quantization (Q8_0)
* Instruction Tuning
* Parameter-Efficient Fine-Tuning (PEFT)

### Frameworks & Libraries

* PyTorch
* Hugging Face Transformers
* PEFT
* Datasets
* TRL
* BitsAndBytes
* Unsloth
* vLLM
* Accelerate

### Model Optimization

* 8-bit Quantization
* Memory-Efficient Fine-Tuning
* Fast Inference Optimization
* Consumer GPU Deployment
* Edge AI Deployment Strategies

### Development Environment

* Python
* Jupyter Notebook
* Google Colab
* Git

## Training Pipeline

The training workflow consisted of:

1. Dataset preparation and formatting.
2. Model loading using quantized weights.
3. LoRA configuration and parameter-efficient fine-tuning.
4. Training using Hugging Face Transformers and TRL.
5. Model evaluation and benchmarking.
6. Quantized inference using Q8_0.
7. Deployment experiments on consumer-grade hardware.

## Fine-Tuning Stack

The project leverages the following ecosystem:

* Hugging Face Transformers for model loading and training.
* PEFT for efficient LoRA fine-tuning.
* BitsAndBytes for memory-efficient quantization.
* Unsloth for accelerated training and inference.
* vLLM for optimized model serving and inference.
* PyTorch as the underlying deep learning framework.


## Research Objectives

This project investigates:

* Personalized AI assistants
* Efficient language model deployment
* Edge AI applications
* Resource-constrained inference
* Fine-tuning strategies
* Quantization techniques
* Trade-offs between model size and performance

## Future Work

* Tool calling and AI agents
* Multi-model orchestration
* Edge deployment optimization
* Real-time task automation

## Author

Victor Lima

M.Sc. Student in Computer Science
Federal University of Alagoas (UFAL)

Research Area:
Artificial Intelligence for Edge Computing

LinkedIn:
https://linkedin.com/in/victor-lima-dev/
