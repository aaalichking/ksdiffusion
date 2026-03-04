KSDiffusion: Kinase-Specific Phosphorylation Site Prediction Framework
KSDiffusion is a deep learning framework designed for kinase-specific phosphorylation site prediction and data augmentation. This project integrates a large protein language model (ESM-2) with a Conditional Diffusion Transformer (DiT), specifically addressing the challenge of insufficient sample sizes for rare kinases.

Through a two-stage pre-training pipeline (ESM-2 feature fine-tuning and diffusion model pre-training), Meta-Learning, and Adversarial Fine-Tuning, KSDiffusion is capable of generating high-quality, biologically plausible peptide sequences. This significantly improves the predictive performance of downstream classification models.

🌟 Key Features
Two-Stage Pre-training Pipeline:

ESM-2 Semantic Enhancement: Combines Masked Language Modeling (MLM) and Supervised Contrastive Learning (SCL) to pre-train ESM-2, effectively extracting and distinguishing deep sequence features.

ProteinDiT Diffusion Pre-training: Based on an unconditional diffusion process, it applies noise addition and denoising training on large-scale protein sequences, enabling the model to learn the underlying data distribution of the sequences.

Dual-Model Architecture: Combines ESM-2 for feature extraction with a Conditional DiT for generating target kinase peptide sequences.

Meta-Learning Initialization: Utilizes a small support set of samples to quickly adapt the conditional mapping of the diffusion model, fitting the feature distribution of rare kinases.

Adversarial Fine-Tuning: Introduces a pre-trained classifier as a discriminator and incorporates biophysical property losses (length, hydrophobicity, charge balance) to guide the diffusion model in generating samples with extremely high biological validity.

Automated Data Augmentation Pipeline: Automatically executes multi-gradient data augmentation and balanced resampling, dynamically screens and selects optimal generated sequences (via clustering and quality scoring), and automatically outputs comprehensive comparison reports with detailed metrics (AUC, ACC, F1, PR AUC, etc.).

Project Structure

ksdiffusion/

ksdiffusion/

├── pre_esm2/                 # ESM-2 pre-training module (MLM + Supervised Contrastive Learning)

│   ├── train.py

│   └── config.py

├── pre_diffusion/            # Diffusion model pre-training module (ProteinDiT)

│   ├── train.py

│   └── config.py

├── main.py                   # Main experiment entry point; loops datasets and aggregates reports

├── experiment.py             # Core experiment pipeline for a single dataset

├── config.py                 # Global experimental configurations

├── dataset.py                # Dataset definitions and batch processing logic

├── models.py                 # Core network components (ESM Classifier, DiTBlock, Conditional DiT)

├── diffusion.py              # Diffusion processes (noise addition, denoising, and cosine scheduling

├── diffuser_wrapper.py       # High-level wrapper for the diffusion model, handling ESM encoding & decoding

├── trainer.py                # Classifier training module (includes early stopping and evaluation metrics)

├── meta_adv_finetune.py      # Meta-learning and adversarial fine-tuning modules for the diffusion model

└── sample_selector.py        # Evaluation and smart filtering of augmented samples (clustering & quality scoring)

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
The complete workflow is divided into two phases: Pre-training and the Main Experiment.

Phase 1: Model Pre-training
Before running the main data augmentation experiment, you can initialize the feature extractor and the diffusion model through the pre-training modules.

1. ESM-2 Model Pre-training (MLM + SCL)
This step uses formatted positive and negative samples to train ESM-2 by computing MLM loss and SCL loss:
python -m pre_esm2.train

2. Diffusion Model Pre-training (ProteinDiT)
This step uses the pre-trained ESM model to generate embeddings and trains the DiT diffusion model based on them:
python -m pre_diffusion.train --data_csv path/to/your/peptides.csv --batch_size 16 --epochs 50 --save_dir dit_diffusion_model

Phase 2: Main Experiment Pipeline
1. Data and Model Preparation

Ensure the pre-trained model weights are located in the configured paths (e.g., local_esm_model/ and diffusion_model_checkpoints/best_model.pth).

Place your datasets in the corresponding directories:

Rare kinase positive samples: dataset/kinase_low/*.csv

Hard negative samples: dataset/kinase_group_all/*.csv

Easy negative samples: dataset/pretrain/*.csv

2. Run the Experiment
Run the main script directly from the project root directory. The program will automatically perform meta-learning, adversarial fine-tuning, sample generation and screening, and multi-gradient performance evaluation:
python main.py

3. View Results
All training logs, model weights, generated augmented samples, and ROC curve plots will be saved in the result_low6/ folder. After the experiment finishes, an augmentation_report.csv and a printed log containing detailed comparison metrics will be generated in the root directory.

⚙️ Configuration
You can easily modify hyperparameters via the config.py files in each module:

Pre-training Phase (pre_esm2/config.py & pre_diffusion/config.py):

ESM-2 includes a contrastive learning temperature parameter (TEMPERATURE = 0.1) and loss weights (SCL_WEIGHT = 0.5, MLM_WEIGHT = 0.5).

The diffusion model supports quick modification of core parameters via command-line arguments, such as --embed_dim 1280, --diffusion_steps 500, and --grad_accum_steps 1.

Main Experiment Phase (config.py):

self.augmentation_factors = [0.1, 0.2, 0.3]: Controls the generation multipliers for data augmentation.

self.meta_learning_steps = 1000: The number of training steps for meta-learning fine-tuning.

self.generation_factor = 10: The amplification factor for generating candidate samples (to facilitate selecting the highest quality samples later).
