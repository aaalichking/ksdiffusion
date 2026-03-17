import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
    accuracy_score,
    confusion_matrix
)
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import KinaseDataset
import copy


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, val_auc):
        score = -val_loss + val_auc  # 综合损失和AUC

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


class KinaseTrainer:
    def __init__(self, config, model, tokenizer, pos_weight=None):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.device = config.device

        if pos_weight is not None:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.criterion = nn.BCEWithLogitsLoss()

        # 获取可训练参数
        trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))

        if not trainable_params and getattr(config, 'freeze_esm', False):
            print("No trainable parameters found. Unfreezing classifier head...")
            if hasattr(model, 'classifier'):
                for param in model.classifier.parameters():
                    param.requires_grad = True
                print("Classifier head parameters unfrozen.")
            trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))

        if not trainable_params:
            print("Still no trainable parameters found. Unfreezing all model parameters...")
            for param in model.parameters():
                param.requires_grad = True
            trainable_params = list(model.parameters())

        print(f"Number of trainable parameter tensors: {len(trainable_params)}")
        print(f"Total parameters: {sum(p.numel() for p in trainable_params)}")

        self.optimizer = optim.AdamW(
            trainable_params,
            lr=config.classifier_lr,
            weight_decay=config.weight_decay
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True,
            min_lr=1e-6
        )

        self.early_stopping = EarlyStopping(patience=10, min_delta=0.001)

    def create_dataloader(self, df, shuffle=True, batch_size=None, weighted_sampling=False):
        if batch_size is None:
            batch_size = self.config.classifier_batch_size

        df = df.copy()
        df['label'] = df['label'].astype(float)

        dataset = KinaseDataset(df, self.tokenizer, self.config.max_length)
        drop_last = shuffle

        if not weighted_sampling:
            return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=4,
                pin_memory=True,
                drop_last=drop_last
            )
        else:
            sources = df.get('source', ['original'] * len(df))
            weights = []
            for source in sources:
                if isinstance(source, str) and 'augmented' in source:
                    weights.append(2.0)
                else:
                    weights.append(1.0)

            sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
            return DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=4,
                pin_memory=True,
                drop_last=drop_last
            )

    def train(self, train_loader, val_loader, epochs=None):
        if epochs is None:
            epochs = self.config.classifier_epochs

        best_score = 0
        best_model_path = os.path.join(self.config.temp_dir, 'best_classifier.pth')
        best_val_auc = 0
        best_epoch = 0

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            all_preds = []
            all_labels = []

            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                train_loss += loss.item()
                preds = torch.sigmoid(logits).detach().cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

            train_loss /= len(train_loader)
            train_auc = roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.5
            train_f1 = f1_score(
                all_labels,
                (np.array(all_preds) > 0.5).astype(int)
            ) if len(np.unique(all_labels)) > 1 else 0.0

            val_metrics = self.evaluate(val_loader)
            val_auc = val_metrics['auc']
            val_f1 = val_metrics['f1']
            val_loss = val_metrics['loss']

            print(
                f"Epoch {epoch + 1}: "
                f"Train Loss: {train_loss:.4f}, AUC: {train_auc:.4f}, F1: {train_f1:.4f} | "
                f"Val Loss: {val_loss:.4f}, AUC: {val_auc:.4f}, F1: {val_f1:.4f}"
            )

            self.scheduler.step(val_auc)

            self.early_stopping(val_loss, val_auc)
            if self.early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

            if val_auc > best_score:
                best_score = val_auc
                best_val_auc = val_auc
                best_epoch = epoch + 1
                torch.save(self.model.state_dict(), best_model_path)
                print(f"Saved new best model with AUC: {val_auc:.4f}")

        if os.path.exists(best_model_path):
            print(
                f"Loading best model from {best_model_path} "
                f"with AUC: {best_val_auc:.4f} (epoch {best_epoch})"
            )
            self.model.load_state_dict(
                torch.load(best_model_path, map_location=self.device)
            )
        return self.model, best_val_auc

    def evaluate(self, test_loader, roc_filename=None):
        self.model.eval()
        all_preds = []
        all_labels = []
        all_logits = []
        total_loss = 0

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()

                outputs = torch.sigmoid(logits)
                all_logits.extend(logits.cpu().numpy())
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        preds = np.array(all_preds)
        labels = np.array(all_labels)
        logits = np.array(all_logits)
        avg_loss = total_loss / len(test_loader)

        if len(np.unique(labels)) < 2:
            auc_score = 0.5
            f1 = 0.0
            pr_auc = 0.5
            best_threshold = 0.5
            acc = 0.5
            fpr = 0.0
        else:
            auc_score = roc_auc_score(labels, preds)

            best_threshold = 0.5
            best_f1 = 0
            for threshold in np.arange(0.1, 0.9, 0.05):
                binary_preds = (preds > threshold).astype(int)
                f1_val = f1_score(labels, binary_preds)
                if f1_val > best_f1:
                    best_f1 = f1_val
                    best_threshold = threshold

            f1 = best_f1

            precision, recall, _ = precision_recall_curve(labels, preds)
            pr_auc = auc(recall, precision)

            binary_preds = (preds > best_threshold).astype(int)
            acc = accuracy_score(labels, binary_preds)

            tn, fp, fn, tp = confusion_matrix(labels, binary_preds).ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        results = {
            'loss': avg_loss,
            'auc': auc_score,
            'f1': f1,
            'pr_auc': pr_auc,
            'acc': acc,
            'fpr': fpr,
            'preds': preds,
            'labels': labels,
            'best_threshold': best_threshold
        }

        if roc_filename:
            roc_path = os.path.join(self.config.output_dir, roc_filename)
            self.plot_roc_curve(labels, preds, roc_path)

        return results

    def plot_roc_curve(self, labels, preds, save_path):
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(labels, preds)

        plt.figure()
        plt.plot(
            fpr,
            tpr,
            color='darkorange',
            lw=2,
            label=f'ROC curve (AUC = {roc_auc_score(labels, preds):.2f})'
        )
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(save_path)
        plt.close()


def cross_validate_train(config, model, tokenizer, train_val_df, test_df, pos_weight, n_folds=5):
    from dataset import KinaseDataset

    dataset = KinaseDataset(train_val_df, tokenizer, config.max_length)
    labels = train_val_df['label'].values

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.seed)

    best_test_auc = 0
    best_test_results = None
    best_fold_index = 0
    all_fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"\n{'=' * 50}")
        print(f"Fold {fold + 1}/{n_folds}")
        print(f"{'=' * 50}")

        train_subsampler = Subset(dataset, train_idx)
        val_subsampler = Subset(dataset, val_idx)

        train_loader = DataLoader(
            train_subsampler,
            batch_size=config.classifier_batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )

        val_loader = DataLoader(
            val_subsampler,
            batch_size=config.classifier_batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=False
        )

        model_copy = copy.deepcopy(model).to(config.device)
        trainer = KinaseTrainer(config, model_copy, tokenizer, pos_weight=pos_weight)
        model_copy, fold_val_auc = trainer.train(train_loader, val_loader)

        test_loader = trainer.create_dataloader(test_df, shuffle=False)
        test_results = trainer.evaluate(test_loader, f'fold_{fold + 1}_roc_curve.png')

        fold_result = {
            'fold': fold + 1,
            'val_auc': fold_val_auc,
            'test_results': test_results
        }
        all_fold_results.append(fold_result)

        print(f"\nFold {fold + 1} Results:")
        print(f"Validation AUC: {fold_val_auc:.4f}")
        print(
            f"Test AUC: {test_results['auc']:.4f}, F1: {test_results['f1']:.4f}, "
            f"PR AUC: {test_results['pr_auc']:.4f}, ACC: {test_results['acc']:.4f}, "
            f"FPR: {test_results['fpr']:.4f}"
        )

        if test_results['auc'] > best_test_auc:
            best_test_auc = test_results['auc']
            best_test_results = test_results
            best_fold_index = fold + 1
            print(f"New best fold: Fold {best_fold_index} with Test AUC: {best_test_auc:.4f}")

    print(f"\nSelected best fold: Fold {best_fold_index} with Test AUC: {best_test_auc:.4f}")
    print(
        f"Best Test Results: AUC: {best_test_results['auc']:.4f}, "
        f"F1: {best_test_results['f1']:.4f}, PR AUC: {best_test_results['pr_auc']:.4f}, "
        f"ACC: {best_test_results['acc']:.4f}, FPR: {best_test_results['fpr']:.4f}"
    )

    return {
        'best_fold': best_fold_index,
        'best_test_auc': best_test_auc,
        'best_test_results': best_test_results,
        'all_fold_results': all_fold_results
    }


def simple_train_test(config, model, tokenizer, train_df, val_df, test_df, pos_weight, weighted_sampling=False):
    from dataset import KinaseDataset

    if weighted_sampling:
        train_df = train_df.copy()
        train_df['source'] = train_df['source'].fillna('original').astype(str)

        weights = []
        for src in train_df['source']:
            if isinstance(src, str) and 'augmented' in src:
                weights.append(2.0)
            else:
                weights.append(1.0)

        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    else:
        sampler = None

    train_loader = DataLoader(
        KinaseDataset(train_df, tokenizer, config.max_length),
        batch_size=config.classifier_batch_size,
        shuffle=not weighted_sampling,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        KinaseDataset(val_df, tokenizer, config.max_length),
        batch_size=config.classifier_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )

    trainer = KinaseTrainer(config, model, tokenizer, pos_weight=pos_weight)
    model, val_auc = trainer.train(train_loader, val_loader)

    test_loader = trainer.create_dataloader(test_df, shuffle=False)
    test_results = trainer.evaluate(test_loader, 'test_roc_curve.png')

    print("\nTest Results:")
    print(
        f"AUC: {test_results['auc']:.4f}, F1: {test_results['f1']:.4f}, "
        f"PR AUC: {test_results['pr_auc']:.4f}, ACC: {test_results['acc']:.4f}, "
        f"FPR: {test_results['fpr']:.4f}"
    )

    return {
        'val_auc': val_auc,
        'test_results': test_results
    }
