import os
import shutil
import warnings
import glob
import numpy as np
import pandas as pd
import torch

from transformers import EsmTokenizer, EsmModel
from sklearn.model_selection import train_test_split

from .config import Config
from .data_processing import KinaseDataProcessor
from .diffuser_wrapper import KinaseDiffuser
from .meta_adv_finetune import MetaDiffusionFineTuner, AdversarialDiffusionFineTuner
from .models import KinaseClassifier
from .trainer import simple_train_test, KinaseTrainer
from .sample_selector import SampleSelector

warnings.filterwarnings("ignore", message="The `max_length` parameter has no effect after initialization")


def run_experiment(rare_kinase_csv):
    config = Config(rare_kinase_csv)
    try:
        print(f"\n{'=' * 80}")
        print(f"Starting experiment for: {os.path.basename(rare_kinase_csv)}")
        print(f"Output directory: {config.output_dir}")
        print(f"{'=' * 80}\n")

        print("Step 1: Preparing datasets...")
        processor = KinaseDataProcessor(config)
        datasets = processor.prepare_datasets()
        train_pos = datasets['train'][datasets['train']['label'] == 1]
        rare_kinase_sequences = train_pos['PEPTIDE'].tolist()

        test_df = datasets['test']

        datasets['train'].to_csv(os.path.join(config.output_dir, 'train_dataset.csv'), index=False)
        datasets['val'].to_csv(os.path.join(config.output_dir, 'val_dataset.csv'), index=False)
        datasets['test'].to_csv(os.path.join(config.output_dir, 'test_dataset.csv'), index=False)

        def calculate_pos_weight(labels):
            pos_count = (labels == 1).sum()
            neg_count = len(labels) - pos_count
            return torch.tensor([neg_count / max(1, pos_count)], device=config.device)

        train_val_df = pd.concat([datasets['train'], datasets['val']], ignore_index=True)
        base_labels = train_val_df['label'].values
        base_pos_weight = calculate_pos_weight(base_labels)

        print("\nStep 2: Training baseline model with simple train/val split...")
        tokenizer = EsmTokenizer.from_pretrained(config.esm_model_path)
        esm_model = EsmModel.from_pretrained(config.esm_model_path).to(config.device)

        train_df, val_df = train_test_split(
            train_val_df,
            test_size=0.2,
            random_state=config.seed,
            stratify=train_val_df['label'] if 'label' in train_val_df.columns else None
        )

        baseline_classifier = KinaseClassifier(
            esm_model,
            tokenizer,
            hidden_dim=256,
            freeze_esm=config.freeze_esm
        ).to(config.device)

        baseline_results = simple_train_test(
            config,
            baseline_classifier,
            tokenizer,
            train_df,
            val_df,
            test_df,
            base_pos_weight
        )

        baseline_auc = baseline_results['test_results']['auc']
        print("\nBaseline Model Results:")
        print(f"Validation AUC: {baseline_results['val_auc']:.4f}")
        print(f"Test AUC: {baseline_auc:.4f}")
        print(f"F1 Score: {baseline_results['test_results']['f1']:.4f}")
        print(f"PR AUC: {baseline_results['test_results']['pr_auc']:.4f}")
        print(f"ACC: {baseline_results['test_results']['acc']:.4f}")
        print(f"FPR: {baseline_results['test_results']['fpr']:.4f}")

        print("\nStep 3: Meta-learning initialization for diffusion model...")
        diffuser = KinaseDiffuser(config)
        meta_finetuner = MetaDiffusionFineTuner(diffuser, config)
        meta_finetuner.meta_adapt(
            support_seqs=rare_kinase_sequences,
            inner_lr=1e-4,
            meta_lr=1e-5,
            adapt_steps=5,
            meta_steps=config.meta_learning_steps
        )

        print("\nSkipping base classifier pre-training, using baseline classifier directly...")
        base_classifier = baseline_classifier

        print("\nStep 5: Adversarial fine-tuning of diffusion model...")
        adv_finetuner = AdversarialDiffusionFineTuner(
            diffuser,
            base_classifier,
            tokenizer,
            config
        )
        adv_finetuner.adversarial_fine_tune(
            sequences=rare_kinase_sequences,
            epochs=config.adv_finetune_epochs,
            batch_size=config.adv_batch_size
        )

        print("\nStep 6: Generating augmented samples with fine-tuned model...")
        num_train_pos = len(train_pos)
        augmentation_results = {}

        for multiplier in config.augmentation_factors:
            print(f"\n=== Processing augmentation multiplier: {multiplier * 100}% ===")
            multiplier_dir = os.path.join(config.output_dir, f"multiplier_{multiplier}")
            os.makedirs(multiplier_dir, exist_ok=True)

            augmented_target = int(num_train_pos * (1 + multiplier))
            num_needed = max(0, augmented_target - num_train_pos)
            num_to_generate = num_needed * config.generation_factor

            augmentation_stats = {
                'multiplier': multiplier,
                'original_samples': num_train_pos,
                'augmented_target': augmented_target,
                'needed_samples': num_needed,
                'generated_samples': num_to_generate
            }

            if num_needed > 0:
                print(f"Generating {num_to_generate} samples for {multiplier * 100}% augmentation")
                batch_size = min(50, max(10, num_to_generate // 10))

                raw_augmented = diffuser.generate_samples(
                    condition_sequences=rare_kinase_sequences,
                    num_samples=num_to_generate,
                    batch_size=batch_size
                )
                print(f"Generated {len(raw_augmented)} raw augmented samples for multiplier {multiplier * 100}%")

                selector = SampleSelector(
                    classifier=base_classifier,
                    tokenizer=tokenizer,
                    config=config,
                    diffuser=diffuser
                )
                raw_avg_quality = selector.calculate_average_quality(raw_augmented)
                print(f"Raw generated samples average quality: {raw_avg_quality:.4f}")

                print(f"\nStep 7: Dynamically selecting best samples for multiplier {multiplier * 100}%...")

                selected_samples, num_selected = selector.select_samples(
                    raw_augmented,
                    rare_kinase_sequences,
                    target_count=num_needed,
                    quality_threshold=0.6,
                    diversity_threshold=0.5,
                    max_attempts=config.max_regeneration_attempts
                )

                selected_avg_quality = selector.calculate_average_quality(selected_samples)
                print(f"Selected samples average quality: {selected_avg_quality:.4f}")

                augmentation_stats['selected_samples'] = num_selected
                augmentation_stats['augmentation_factor_used'] = num_selected / num_needed if num_needed > 0 else 0
                augmentation_stats['raw_avg_quality'] = raw_avg_quality
                augmentation_stats['selected_avg_quality'] = selected_avg_quality

                kinase_family = 'RARE_KINASE'
                if not train_pos.empty and 'KINASE_FAMILY' in train_pos.columns:
                    kinase_family = train_pos['KINASE_FAMILY'].iloc[0]

                augmented_df = pd.DataFrame({
                    'PEPTIDE': selected_samples,
                    'KINASE_FAMILY': kinase_family,
                    'label': 1.0,
                    'source': f'adv_augmented_{multiplier * 100}%'
                })
                augmented_df.to_csv(
                    os.path.join(multiplier_dir, f'augmented_{multiplier * 100}%.csv'),
                    index=False
                )

                full_train = pd.concat([
                    datasets['train'],
                    augmented_df
                ], ignore_index=True)
            else:
                print(
                    f"No augmentation needed for multiplier {multiplier * 100}%. "
                    f"Current positive samples ({num_train_pos}) meet target ({augmented_target})."
                )
                full_train = datasets['train']
                augmentation_stats['selected_samples'] = 0
                augmentation_stats['augmentation_factor_used'] = 0
                augmentation_stats['raw_avg_quality'] = 0.0
                augmentation_stats['selected_avg_quality'] = 0.0

            full_train['label'] = pd.to_numeric(
                full_train['label'], errors='coerce'
            ).fillna(0).astype(float)

            print(f"\nStep 8: Balancing the dataset after {multiplier * 100}% augmentation...")
            target_neg_count = len(full_train[full_train['label'] == 1])
            balanced_train = processor.resample_negatives(
                full_train,
                target_neg_count
            )

            balanced_train_path = os.path.join(multiplier_dir, f'balanced_train_{multiplier * 100}%.csv')
            balanced_train.to_csv(balanced_train_path, index=False)
            print(f"Balanced dataset saved to {balanced_train_path}")

            print(
                f"\nStep 9: Training initial model with {multiplier * 100}% augmented data "
                f"using baseline model as starting point..."
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            initial_classifier = base_classifier.__class__(
                esm_model,
                tokenizer,
                hidden_dim=256,
                freeze_esm=config.freeze_esm
            ).to(config.device)
            initial_classifier.load_state_dict(base_classifier.state_dict())

            if config.freeze_esm and hasattr(initial_classifier, 'classifier'):
                for param in initial_classifier.classifier.parameters():
                    param.requires_grad = True
                print("Ensured classifier head is trainable after deepcopy")

            balanced_labels = balanced_train['label'].values
            initial_pos_weight = calculate_pos_weight(balanced_labels)

            from torch.utils.data import DataLoader
            from .dataset import KinaseDataset

            # 正确的初始化代码
            initial_train_loader = DataLoader(
                KinaseDataset(balanced_train, tokenizer, config.max_length),
                batch_size=config.classifier_batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
                drop_last=True
            )

            initial_val_loader = DataLoader(
                KinaseDataset(datasets['val'], tokenizer, config.max_length),
                batch_size=config.classifier_batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
                drop_last=False
            )

            trainer = KinaseTrainer(config, initial_classifier, tokenizer, initial_pos_weight)
            initial_classifier, val_auc = trainer.train(initial_train_loader, initial_val_loader)

            test_loader = trainer.create_dataloader(test_df, shuffle=False)
            test_results = trainer.evaluate(test_loader, f'initial_{multiplier * 100}%_roc_curve.png')

            initial_auc = test_results['auc']
            print(f"\n{multiplier * 100}% Initial Model Results:")
            print(f"Test AUC: {initial_auc:.4f}")
            print(f"F1 Score: {test_results['f1']:.4f}")
            print(f"PR AUC: {test_results['pr_auc']:.4f}")
            print(f"ACC: {test_results['acc']:.4f}")
            print(f"FPR: {test_results['fpr']:.4f}")

            results_path = os.path.join(multiplier_dir, f'results_{multiplier * 100}%.txt')
            with open(results_path, 'w') as f:
                f.write("=== Experiment Settings ===\n")
                f.write(f"Dataset: {os.path.basename(rare_kinase_csv)}\n")
                f.write(f"Augmentation Multiplier: {multiplier * 100}%\n")
                f.write(f"Original Samples: {augmentation_stats['original_samples']}\n")
                f.write(f"Augmented Target: {augmentation_stats['augmented_target']}\n")
                f.write(f"Needed Samples: {augmentation_stats['needed_samples']}\n")
                f.write(f"Generated Samples: {augmentation_stats['generated_samples']}\n")
                f.write(f"Selected Samples: {augmentation_stats['selected_samples']}\n")
                f.write(
                    f"Effective Augmentation Factor: "
                    f"{augmentation_stats['augmentation_factor_used']:.2f}\n"
                )
                f.write(
                    f"Raw Generated Samples Average Quality: "
                    f"{augmentation_stats.get('raw_avg_quality', 0):.4f}\n"
                )
                f.write(
                    f"Selected Samples Average Quality: "
                    f"{augmentation_stats.get('selected_avg_quality', 0):.4f}\n\n"
                )

                f.write("=== Baseline Model Results ===\n")
                f.write(f"Test AUC: {baseline_auc:.4f}\n")
                f.write(f"F1 Score: {baseline_results['test_results']['f1']:.4f}\n")
                f.write(f"PR AUC: {baseline_results['test_results']['pr_auc']:.4f}\n")
                f.write(f"ACC: {baseline_results['test_results']['acc']:.4f}\n")
                f.write(f"FPR: {baseline_results['test_results']['fpr']:.4f}\n\n")

                f.write(f"=== {multiplier * 100}% Initial Model Results ===\n")
                f.write(f"Test AUC: {initial_auc:.4f}\n")
                f.write(f"F1 Score: {test_results['f1']:.4f}\n")
                f.write(f"PR AUC: {test_results['pr_auc']:.4f}\n")
                f.write(f"ACC: {test_results['acc']:.4f}\n")
                f.write(f"FPR: {test_results['fpr']:.4f}\n\n")

                f.write("=== Improvements ===\n")
                f.write(f"Initial vs Baseline AUC: {initial_auc - baseline_auc:.4f}\n")
                f.write(
                    f"Initial vs Baseline ACC: "
                    f"{test_results['acc'] - baseline_results['test_results']['acc']:.4f}\n"
                )

            augmentation_results[multiplier] = {
                'baseline': baseline_results,
                'initial': {
                    'test_results': test_results,
                    'auc': initial_auc
                },
                'augmentation_stats': augmentation_stats
            }

        summary_path = os.path.join(config.output_dir, 'summary_results.txt')
        with open(summary_path, 'w') as f:
            f.write("=== Summary of All Multipliers ===\n\n")
            for multiplier, results in augmentation_results.items():
                f.write(f"=== Multiplier: {multiplier * 100}% ===\n")
                f.write(f"Baseline AUC: {results['baseline']['test_results']['auc']:.4f}\n")
                f.write(f"Initial AUC: {results['initial']['auc']:.4f}\n")
                f.write(
                    f"Improvement (Initial vs Baseline AUC): "
                    f"{results['initial']['auc'] - results['baseline']['test_results']['auc']:.4f}\n"
                )
                f.write(f"Baseline ACC: {results['baseline']['test_results']['acc']:.4f}\n")
                f.write(f"Initial ACC: {results['initial']['test_results']['acc']:.4f}\n")
                f.write(
                    f"Improvement (Initial vs Baseline ACC): "
                    f"{results['initial']['test_results']['acc'] - results['baseline']['test_results']['acc']:.4f}\n"
                )
                f.write(f"Selected Samples: {results['augmentation_stats']['selected_samples']}\n")
                f.write(
                    f"Raw Generated Samples Average Quality: "
                    f"{results['augmentation_stats'].get('raw_avg_quality', 0):.4f}\n"
                )
                f.write(
                    f"Selected Samples Average Quality: "
                    f"{results['augmentation_stats'].get('selected_avg_quality', 0):.4f}\n\n"
                )

        print("\nExperiment completed! Results saved to", config.output_dir)
        return augmentation_results

    finally:
        print("\nCleaning up temporary files...")
        if os.path.exists(config.temp_dir):
            shutil.rmtree(config.temp_dir)
            print(f"Temporary directory {config.temp_dir} removed")
