import os
import torch
from transformers import EsmTokenizer, EsmModel

from .diffusion import DiffusionProcess
from .models import ConditionalProteinDiT


class KinaseDiffuser:
    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.tokenizer = EsmTokenizer.from_pretrained(config.esm_model_path)

        self.esm_model = EsmModel.from_pretrained(
            config.esm_model_path,
            add_pooling_layer=False
        ).to(self.device)

        self.esm_model.eval()

        self.diffusion = DiffusionProcess(
            steps=config.diffusion_steps,
            device=self.device
        )
        self.model = ConditionalProteinDiT(
            input_dim=config.embed_dim,
            hidden_dim=config.hidden_dim
        ).to(self.device)

        if os.path.exists(config.diffusion_model_path):
            try:
                checkpoint = torch.load(
                    config.diffusion_model_path,
                    map_location=self.device
                )

                model_state_dict = self.model.state_dict()
                pretrained_dict = {}

                for key, value in checkpoint['model_state_dict'].items():
                    if key in model_state_dict and value.size() == model_state_dict[key].size():
                        pretrained_dict[key] = value

                model_state_dict.update(pretrained_dict)
                self.model.load_state_dict(model_state_dict, strict=False)

                print(
                    f"Loaded {len(pretrained_dict)}/{len(checkpoint['model_state_dict'])} "
                    f"weights from {config.diffusion_model_path}"
                )
            except Exception as e:
                print(f"Warning: Failed to load pre-trained weights. Error: {str(e)}")
        else:
            print(f"Warning: Diffusion model not found at {config.diffusion_model_path}")

    def encode_sequences(self, sequences):
        if not sequences:
            return torch.tensor([])

        embeddings = []
        batch_size = 32
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i + batch_size]

            inputs = self.tokenizer(
                batch_seqs,
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=self.config.max_length,
                return_token_type_ids=False
            ).to(self.device)

            with torch.no_grad():
                outputs = self.esm_model(**inputs)
                batch_emb = outputs.last_hidden_state

                if batch_emb.size(1) > self.config.max_length:
                    batch_emb = batch_emb[:, :self.config.max_length, :]
                elif batch_emb.size(1) < self.config.max_length:
                    padding = torch.zeros(
                        batch_emb.size(0),
                        self.config.max_length - batch_emb.size(1),
                        batch_emb.size(2),
                        device=self.device
                    )
                    batch_emb = torch.cat([batch_emb, padding], dim=1)

                embeddings.append(batch_emb.cpu())

        return torch.cat(embeddings, dim=0) if embeddings else torch.tensor([])

    def generate_samples(self, condition_sequences, num_samples, batch_size=100):
        if num_samples <= 0 or not condition_sequences:
            print("No samples to generate. Skipping augmentation.")
            return []

        from tqdm import tqdm

        print(f"Generating {num_samples} augmented samples in batches of {batch_size}...")
        condition_embs = self.encode_sequences(condition_sequences).to(self.device)
        if condition_embs.numel() == 0:
            print("Error: Condition embeddings are empty. Cannot generate samples.")
            return []

        mean_condition = condition_embs.mean(dim=0, keepdim=True)
        all_sequences = []
        num_batches = (num_samples + batch_size - 1) // batch_size

        self.model.eval()
        with torch.no_grad():
            for i in tqdm(range(num_batches), desc="Generating batches", total=num_batches):
                current_batch_size = min(batch_size, num_samples - i * batch_size)
                condition_embs_batch = mean_condition.repeat(current_batch_size, 1, 1)

                x = torch.randn(
                    current_batch_size,
                    self.config.max_length,
                    self.config.embed_dim,
                    device=self.device
                )

                for step in tqdm(
                    range(self.config.diffusion_steps - 1, -1, -1),
                    total=self.config.diffusion_steps,
                    desc=f"Batch {i + 1}/{num_batches}",
                    leave=False
                ):
                    t = torch.full(
                        (current_batch_size,),
                        step,
                        device=self.device,
                        dtype=torch.long
                    )
                    predicted_noise = self.model(x, t, condition_embs_batch)

                    alpha_t = self.diffusion.alphas[t][:, None, None]
                    alpha_t_cumprod = self.diffusion.alphas_cumprod[t][:, None, None]
                    beta_t = self.diffusion.betas[t][:, None, None]

                    if step > 0:
                        noise = torch.randn_like(x)
                    else:
                        noise = torch.zeros_like(x)

                    x = (1 / torch.sqrt(alpha_t)) * (
                        x - ((1 - alpha_t) / torch.sqrt(1 - alpha_t_cumprod)) * predicted_noise
                    ) + torch.sqrt(beta_t) * noise

                batch_sequences = self.decode_embeddings(x)
                all_sequences.extend(batch_sequences)

                del x, predicted_noise, noise, condition_embs_batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return all_sequences

    def decode_embeddings(self, embeddings):
        if embeddings.numel() == 0:
            return []

        sequences = []
        embedding_matrix = self.esm_model.embeddings.word_embeddings.weight.data

        batch_size = 32
        for i in range(0, embeddings.size(0), batch_size):
            batch_emb = embeddings[i:i + batch_size]
            batch_emb = batch_emb.view(-1, batch_emb.size(-1))
            similarities = torch.matmul(batch_emb, embedding_matrix.t())
            token_ids = torch.argmax(similarities, dim=-1)
            token_ids = token_ids.view(-1, self.config.max_length)

            for ids in token_ids:
                seq = self.tokenizer.decode(
                    ids.cpu().numpy(),
                    skip_special_tokens=True
                )
                sequences.append(seq.replace(" ", ""))

        return sequences