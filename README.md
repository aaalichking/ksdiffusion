KSDiffusion: Kinase-Specific Phosphorylation Site Prediction Framework
KSDiffusion is a deep learning framework designed for kinase-specific phosphorylation site prediction and data augmentation. By integrating a large protein language model (ESM-2) with a Conditional Diffusion Transformer (DiT), this project specifically addresses the challenge of insufficient sample sizes for rare kinases. Through Meta-Learning and Adversarial Fine-Tuning, KSDiffusion generates high-quality, biologically plausible peptide sequences, significantly improving the predictive performance of downstream classification models.

🌟 Key Features
Dual-Model Architecture: Combines ESM-2 for extracting deep sequence features with a Conditional DiT for generating target kinase peptide sequences.

Meta-Learning Initialization: Utilizes a small support set of samples to quickly adapt the conditional mapping of the diffusion model, fitting the feature distribution of rare kinases.

Adversarial Fine-Tuning: Introduces a pre-trained classifier as a discriminator and incorporates biophysical loss constraints (length, hydrophobicity, charge balance) to guide the diffusion model in generating highly biologically valid samples.

Smart Sample Selection: Integrates KMeans clustering, diversity scoring, and classifier quality scoring to dynamically filter and select the optimal generated sequences.

Automated Data Augmentation Pipeline: Automatically executes multi-gradient data augmentation (10%, 20%, 30%) and balanced resampling, outputting comprehensive cross-validation comparison reports with detailed metrics (AUC, ACC, F1, PR AUC, etc.).

Project Structure
ksdiffusion/
├── main.py                   # Main entry point; loops through datasets and aggregates reports
├── experiment.py             # Core experiment pipeline for a single dataset
├── config.py                 # Global configurations (hyperparameters, paths, hardware device)
├── dataset.py                # Dataset definitions (Dataset) and batch processing logic
├── models.py                 # Core network components (ESM Classifier, DiTBlock, Conditional DiT)
├── diffusion.py              # Diffusion processes (noise addition, denoising, and cosine scheduling)
├── diffuser_wrapper.py       # High-level wrapper for the diffusion model, handling ESM encoding & decoding
├── trainer.py                # Classifier training module (includes early stopping and metric calculations)
├── meta_adv_finetune.py      # Meta-learning and adversarial fine-tuning modules for the diffusion model
└── sample_selector.py        # Evaluation and smart filtering of augmented samples (clustering & scoring)

🛠️ Requirements
This project is built on PyTorch and the Hugging Face transformers library. Please ensure your environment has the following dependencies installed:
torch>=2.0.0
transformers>=4.30.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.2
matplotlib>=3.5.0
tqdm>=4.64.0

Quick Start
1. Data Preparation
Ensure your data is placed in the corresponding directories. By default, the program reads data from the following paths:

Rare kinase positive samples: dataset/kinase_low/*.csv

Hard negative samples: dataset/kinase_group_all/*.csv

Easy negative samples: dataset/pretrain/*.csv

2. Model Preparation
Configure the local path for the ESM-2 model weights. Modify self.esm_model_path in config.py to point to your local ESM model directory (or keep the default huggingface string to download it automatically over the internet).

3. Run the Experiment
Run the main script directly from the project root directory:
python main.py

4. View Results
All training logs, model weights, generated augmented samples, and ROC curve plots will be saved in the configured output directory (default is the result_low6/ folder).
After the program finishes, an augmentation_report.csv and a detailed summary comparison log will be generated in the root directory.

⚙️ Configuration
You can easily modify the core experimental parameters in config.py:

self.augmentation_factors = [0.1, 0.2, 0.3]: Modify the data augmentation multipliers.

self.classifier_epochs = 200: The maximum number of training epochs for the classifier.

self.meta_learning_steps = 1000: The number of steps for meta-learning fine-tuning.

self.device = torch.device("cuda:1"): Change the corresponding GPU ID based on your hardware setup.
