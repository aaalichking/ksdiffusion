import os
import time
import warnings

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import EsmTokenizer, EsmModel

from .config import get_config
from .diffusion import DiffusionProcess
from .dataset import ProteinSequenceDataset
from .model import ProteinDiT

# 忽略特定警告
warnings.filterwarnings("ignore", message="The `max_length` parameter has no effect after initialization")


def train():
    args = get_config()

    # 初始化扩散过程
    diffusion = DiffusionProcess(steps=args.diffusion_steps, device=args.device)

    # 加载数据集
    dataset = ProteinSequenceDataset(
        args.data_csv,
        max_length=args.max_length
    )

    # 数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=lambda x: x  # 返回原始序列列表
    )

    # 初始化ESM模型
    tokenizer = EsmTokenizer.from_pretrained(args.esm_model_path)
    esm_model = EsmModel.from_pretrained(args.esm_model_path).to(args.device)
    esm_model.eval()

    # 初始化DiT模型
    model = ProteinDiT(
        input_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        depth=8,
        heads=8
    ).to(args.device)

    # 优化器 & 学习率调度
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    best_loss = float('inf')
    grad_accum_steps = args.grad_accum_steps

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        batch_count = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        # 清空CUDA缓存
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

        optimizer.zero_grad()

        for batch_idx, sequences in enumerate(progress_bar):
            start_time = time.time()

            # 动态生成嵌入
            with torch.no_grad():
                inputs = tokenizer(
                    sequences,
                    return_tensors='pt',
                    padding='max_length',
                    truncation=True,
                    max_length=args.max_length,
                    return_token_type_ids=False
                ).to(args.device)

                outputs = esm_model(**inputs)
                embeddings = outputs.last_hidden_state

                # 确保长度为 max_length
                if embeddings.size(1) > args.max_length:
                    embeddings = embeddings[:, :args.max_length, :]
                elif embeddings.size(1) < args.max_length:
                    padding = torch.zeros(
                        embeddings.size(0),
                        args.max_length - embeddings.size(1),
                        embeddings.size(2),
                        device=args.device
                    )
                    embeddings = torch.cat([embeddings, padding], dim=1)

            # 随机采样时间步
            t = torch.randint(0, args.diffusion_steps, (embeddings.shape[0],), device=args.device)

            # 添加噪声
            noisy_embeddings, noise = diffusion.add_noise(embeddings, t)

            # 预测噪声
            predicted_noise = model(noisy_embeddings, t)

            # 计算损失
            loss_weights = diffusion.get_loss_weights(t)
            loss = torch.mean(loss_weights * (predicted_noise - noise) ** 2)

            # 梯度累积
            loss = loss / grad_accum_steps
            loss.backward()

            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * grad_accum_steps
            batch_count += 1

            batch_time = time.time() - start_time
            seqs_per_sec = args.batch_size / batch_time if batch_time > 0 else 0.0

            progress_bar.set_postfix(
                loss=loss.item() * grad_accum_steps,
                seqs_per_sec=f"{seqs_per_sec:.1f}"
            )

            # 定期清理缓存
            if args.device.startswith("cuda") and batch_idx % 50 == 0:
                torch.cuda.empty_cache()

        avg_loss = epoch_loss / batch_count
        print(f"Epoch {epoch + 1} - Avg Loss: {avg_loss:.4f}")

        # 更新学习率
        scheduler.step(avg_loss)

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, os.path.join(args.save_dir, 'best_model.pth'))
            print(f"Saved best model with loss: {avg_loss:.4f}")

        # 定期保存检查点
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, os.path.join(args.save_dir, f'checkpoint_epoch_{epoch + 1}.pth'))

        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    print("Training completed!")


if __name__ == "__main__":
    train()