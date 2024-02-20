# Text-Generation
This repository contains is implementation of text Generation use case.

# README

## Finetuning NousResearch/Llama-2-7b-chat-hf for Email Spam Classification using LoRA

This repository contains code for finetuning the NousResearch/Llama-2-7b-chat-hf model on a conversation dataset
### Task
- **Text Generation**: The task involves Generating the text.

### Model
- **NousResearch/Llama-2-7b-chat-hf**: Pretrained NousResearch/Llama-2-7b-chat-hf model is utilized for the task.

### Dataset
- **Dataset**: The dataset used is [mlabonne/guanaco-llama2-1k](mlabonne/guanaco-llama2-1k). It consists of 1000 conversation samples formatted in llama2 prompt.

### Libraries Used
- `transformers`: For utilizing and fine-tuning the NousResearch/Llama-2-7b-chat-hf model.
- `huggingface-hub`: For accessing the model and tokenizer from the Hugging Face model hub.
- `peft`: For training and evaluation of the model.
- `datasets`: For handling and processing the dataset.
- `numpy`: For numerical computations.
- `torch`: For building and training neural networks.
- `trl`: For Supervised finetuning

### Training Details

### Hyperparameters
- **Pretrained Model**: NousResearch/Llama-2-7b-chat-hf.
- **Weight Decay**: 0.01
- **Learning Rate**: 0.0002
- **train_batch_size**: 8
- **eval_Batch Size**: 4
- **Number of Epochs**: 10

### Usage
- Clone the repository.
- Install the required libraries listed in `requirements.txt`.
- Run the training script with appropriate configurations.