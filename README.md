# [HCMUS - Math for AI] [Final Project] Implement Transformer From Scratch 

This project implements the "Attention is All You Need" paper using PyTorch to build a Transformer model for English-Vietnamese translation.

## Requirements

Set up the environment with **conda**:

```bash
conda create -n attention python=3.9
conda activate attention
pip install -r requirements.txt
```

The code will automatically download dataset from Kaggle Hub,  
or you can manually download from [English-Vietnamese Translation Datset]([https://huggingface.co/Qwen](https://www.kaggle.com/datasets/hungnm/englishvietnamese-translation)).

## Project Structure

| File                         | Description                                        |
|-------------------------------|----------------------------------------------------|
| `tokenizer_input.json`        | Trained Tokenizer for English                      |
| `tokenizer_output.json`       | Trained Tokenizer for Vietnamese |
| `utils.py`                   | Transformer Implementations        |
| `train1.py`                   | Config and train script for model1.       |
| `train2.py`                   | Config and train script for model2     |
| `infer.py`       | Script for inference          |
| `eval.py`| Script for evaluate all dataset         |
| `/train1`                     | Output directory for model1                    |
| `/train2`                     | Output directory for model2                  |
| `guide.txt`                     | Guide in Vietnamese to run this project                     |
| `requirements.txt`                     | Requirements for enviroment                     |



## Training Instructions

We trained the first model for 5 epochs in 6 hours and the second model for 10 epochs in 12 hours on an RTX 2080 Ti.
You can find the trained models in the `train1` and `train2` folders for inference.

### First Model
Output directory: `/train1`

Run:

```bash
CUDA_VISIBLE_DEVICES=0 python train1.py
```

### Second Model
Output directory: `/train2`

Run:

```bash
CUDA_VISIBLE_DEVICES=0 python train2.py
```

## Inference
After training, change the config for the chosen model (default is model2 with 10 epochs, `train2/model_epoch_10.pt`) and run the inference script:

```bash
CUDA_VISIBLE_DEVICES=0 python infer.py
```

## Evaluate

After training, change the config for the chosen model (default is model2 with 10 epochs, `train2/model_epoch_10.pt`) and run the eval script:

```bash
CUDA_VISIBLE_DEVICES=0 python eval.py
```

Select the correct model and config for evaluate.
## Notes
- The dataset is automatically downloaded from Kaggle Hub. If you encounter any issues, you can manually download it from the link provided above.
- Ensure that the proper environment and dependencies are installed as outlined in the `requirements.txt` file.
- Modify the configuration to switch between the two models for inference and evaluation.

## Credits
This project is based on the "Attention is All You Need" paper by Vaswani et al. (2017), which introduces the Transformer model.  
The implementation is also based on building the Transformer model from scratch using PyTorch.

Special thanks to [Arun Mohan](https://www.kaggle.com/code/arunmohan003/transformer-from-scratch-using-pytorch) for the inspiration and code shared in the Kaggle notebook: [Transformer From Scratch Using PyTorch](https://www.kaggle.com/code/arunmohan003/transformer-from-scratch-using-pytorch).
