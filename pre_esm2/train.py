import os
from tqdm import tqdm
import torch
from transformers import AdamW

from .config import (
    DEVICE,
    EPOCHS,
    LR,
    MLM_WEIGHT,
    SCL_WEIGHT,
    LOCAL_MODEL_PATH,
    CHECKPOINT_PATH,
)
from .losses import SupConLoss
from .dataset import create_dataloaders
from .model_utils import load_model, save_checkpoint, load_checkpoint


def train_model(model, tokenizer, train_loader, val_loader):
    model.to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR)
    con_loss_fn = SupConLoss()

    # 加载检查点（如果存在）
    start_epoch, best_val_loss, train_loss_history, val_loss_history = load_checkpoint(
        model, optimizer, CHECKPOINT_PATH
    )

    # 如果已经完成所有 epoch
    if start_epoch >= EPOCHS:
        print("训练已完成!")
        return

    for epoch in range(start_epoch, EPOCHS):
        # 训练阶段
        model.train()
        total_train_loss, mlm_train_loss, scl_train_loss = 0.0, 0.0, 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} - Training")

        for batch_idx, batch in enumerate(train_bar):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            mlm_labels = batch["mlm_labels"].to(DEVICE)
            scl_labels = batch["scl_labels"].to(DEVICE)

            # 前向传播 - 获取 MLM 损失和隐藏状态
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=mlm_labels,  # MLM 标签
                output_hidden_states=True,
            )

            # 获取 [CLS] 隐藏状态用于监督对比学习
            cls_hidden = outputs.hidden_states[-1][:, 0, :]

            # 计算损失
            mlm_loss = outputs.loss
            scl_loss = con_loss_fn(cls_hidden, scl_labels)
            total_loss = MLM_WEIGHT * mlm_loss + SCL_WEIGHT * scl_loss

            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # 记录损失
            total_train_loss += total_loss.item()
            mlm_train_loss += mlm_loss.item()
            scl_train_loss += scl_loss.item()

            # 更新进度条
            train_bar.set_postfix(
                {
                    "loss": f"{total_loss.item():.4f}",
                    "mlm": f"{mlm_loss.item():.4f}",
                    "scl": f"{scl_loss.item():.4f}",
                }
            )

        # 计算平均训练损失
        avg_total_train = total_train_loss / len(train_loader)
        avg_mlm_train = mlm_train_loss / len(train_loader)
        avg_scl_train = scl_train_loss / len(train_loader)
        train_loss_history.append(avg_total_train)

        # 验证阶段
        model.eval()
        total_val_loss, mlm_val_loss, scl_val_loss = 0.0, 0.0, 0.0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} - Validation")

        with torch.no_grad():
            for batch in val_bar:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                mlm_labels = batch["mlm_labels"].to(DEVICE)
                scl_labels = batch["scl_labels"].to(DEVICE)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=mlm_labels,
                    output_hidden_states=True,
                )

                cls_hidden = outputs.hidden_states[-1][:, 0, :]
                mlm_loss = outputs.loss
                scl_loss = con_loss_fn(cls_hidden, scl_labels)
                total_loss = MLM_WEIGHT * mlm_loss + SCL_WEIGHT * scl_loss

                # 记录损失
                total_val_loss += total_loss.item()
                mlm_val_loss += mlm_loss.item()
                scl_val_loss += scl_loss.item()

                # 更新进度条
                val_bar.set_postfix(
                    {
                        "val_loss": f"{total_loss.item():.4f}",
                        "mlm": f"{mlm_loss.item():.4f}",
                        "scl": f"{scl_loss.item():.4f}",
                    }
                )

        # 计算平均验证损失
        avg_total_val = total_val_loss / len(val_loader)
        avg_mlm_val = mlm_val_loss / len(val_loader)
        avg_scl_val = scl_val_loss / len(val_loader)
        val_loss_history.append(avg_total_val)

        # 打印训练统计信息
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print(
            f"训练集 - 总损失: {avg_total_train:.4f} | MLM损失: {avg_mlm_train:.4f} | SCL损失: {avg_scl_train:.4f}"
        )
        print(
            f"验证集 - 总损失: {avg_total_val:.4f} | MLM损失: {avg_mlm_val:.4f} | SCL损失: {avg_scl_val:.4f}"
        )

        # 保存最佳模型
        if avg_total_val < best_val_loss:
            best_val_loss = avg_total_val
            model.save_pretrained(LOCAL_MODEL_PATH)
            tokenizer.save_pretrained(LOCAL_MODEL_PATH)
            print(f"保存最佳模型，验证损失: {avg_total_val:.4f}")

        # 保存检查点（每个 epoch 结束后保存）
        save_checkpoint(
            epoch,
            model,
            optimizer,
            best_val_loss,
            train_loss_history,
            val_loss_history,
            CHECKPOINT_PATH,
        )

    # 训练完成后删除检查点
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
        print("训练完成，检查点已删除")


def main():
    # 加载模型和 tokenizer
    model, tokenizer = load_model()

    # 创建数据加载器
    train_loader, val_loader = create_dataloaders(tokenizer)

    # 训练模型
    train_model(model, tokenizer, train_loader, val_loader)

    print("\n训练完成!")