import os
import torch
from transformers import EsmForMaskedLM, EsmTokenizer

from .config import LOCAL_MODEL_PATH, DEVICE, CHECKPOINT_PATH


def load_model():
    """从本地目录加载模型和 tokenizer"""
    if not os.path.exists(LOCAL_MODEL_PATH):
        raise FileNotFoundError(f"本地模型路径未找到: {LOCAL_MODEL_PATH}")

    print(f"从本地目录加载模型和 tokenizer: {LOCAL_MODEL_PATH}")
    tokenizer = EsmTokenizer.from_pretrained(LOCAL_MODEL_PATH)
    model = EsmForMaskedLM.from_pretrained(LOCAL_MODEL_PATH)

    return model, tokenizer


def save_checkpoint(
    epoch,
    model,
    optimizer,
    best_val_loss,
    train_loss_history,
    val_loss_history,
    filename: str = CHECKPOINT_PATH,
):
    """保存训练检查点"""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "train_loss_history": train_loss_history,
        "val_loss_history": val_loss_history,
    }
    torch.save(checkpoint, filename)
    print(f"检查点已保存到 {filename}")


def load_checkpoint(model, optimizer, filename: str = CHECKPOINT_PATH):
    """加载训练检查点"""
    if os.path.exists(filename):
        checkpoint = torch.load(filename, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1  # 从下一个 epoch 开始
        best_val_loss = checkpoint["best_val_loss"]
        train_loss_history = checkpoint.get("train_loss_history", [])
        val_loss_history = checkpoint.get("val_loss_history", [])

        print(
            f"加载检查点: 从第 {start_epoch} 个 epoch 开始, 最佳验证损失 = {best_val_loss:.4f}"
        )
        return start_epoch, best_val_loss, train_loss_history, val_loss_history

    return 0, float("inf"), [], []  # 如果没有检查点，从头开始