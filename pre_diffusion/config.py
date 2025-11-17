import argparse
import os
import torch


def get_config():
    parser = argparse.ArgumentParser(description='ProteinDiT Diffusion Model Training')
    parser.add_argument('--data_csv', type=str, required=True, help='Path to peptide sequences CSV file')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--embed_dim', type=int, default=1280, help='ESM embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=512, help='DiT hidden dimension')
    parser.add_argument('--diffusion_steps', type=int, default=500, help='Number of diffusion steps')
    parser.add_argument('--save_dir', type=str, default='dit_diffusion_model', help='Directory to save model')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--esm_model_path', type=str, default='local_esm_model', help='Path to local ESM model')
    parser.add_argument('--max_length', type=int, default=61, help='Max sequence length')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of data loader workers')
    parser.add_argument('--grad_accum_steps', type=int, default=1, help='Gradient accumulation steps')

    args = parser.parse_args()

    # 确保保存目录存在
    os.makedirs(args.save_dir, exist_ok=True)

    return args