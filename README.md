# SwinVED-SCST — Official Code Repository

This repository contains the **official implementation** of the first model baseline of the paper:

📄 **Paper:** *MIMIC-CXR-VQA: A Medical Visual Question Answering Dataset Constructed with LLaMA-based Annotations*


🔗 **Link:** [Comming Soon!!]

-----

## Model Architecture

Below is the architecture diagram used in the baseline:

![Model](model.png)

-----

## Dataset

This work is built upon the **MIMIC-CXR-VQA** dataset:

🔗 **Link:** [Comming Soon!!]

## 📂 Directory Structure and Contents

To ensure the training scripts function correctly, you must download the dataset and set up the following structure.

### 1. Dataset Download and Placement

You need to download the CSV files for the **MIMIC-CXR-VQA** dataset and place them inside a folder named `MIMIC-CXR-VQA` within your main data directory (e.g., `/data`).

### 2. Vocabulary Creation

Use the provided script to generate the vocabulary file:

```bash
python createvocab.py
```
This will create the necessary vocabulary file: `vocab-mimic-cxr-vqa.tgt`.

### 3. Configure File Paths

Before running the experiments, verify the settings in the `paths.py` file.  
The only variable you are required to set correctly is:

- **`IMAGES_MIMIC_PATH`**: This must point to the root directory containing the actual MIMIC-CXR images.

Ensure your directory structure resembles the following (assuming you use a `/data` folder):

/data/
├── MIMIC-CXR-VQA/
│ ├── train.csv
│ ├── val.csv
│ └── test.csv
├── [Directory containing all MIMIC-CXR Images] <-- This path is set by IMAGES_MIMIC
-----

## 1\. Environment Setup 🐍

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate swinbed
```

-----


## 2\. Running Experiments 🚀


### Base Training Pipeline

To execute the full training pipeline, follow these sequential steps:

1.  **Change Directory:** First, navigate to the training directory:
    ```bash
    cd train
    ```

2.  **Stage 1: NLL Training (Frozen Encoder)**
    Execute the initial Negative Log-Likelihood (NLL) training run with the model's encoder frozen:
    ```bash
    ./nll_train_freeze_econder.sh
    ```

3.  **Stage 2: NLL Training (Unfrozen Encoder)**
    Follow up with the NLL training using an unfrozen encoder for full model fine-tuning:
    ```bash
    ./nll_train_unfreeze_econder.sh
    ```

4.  **Stage 3: Reinforcement Learning (RL) Training**
    Complete the process by executing the Reinforcement Learning (RL) training phase:
    ```bash
    ./rl_train.sh
    ```