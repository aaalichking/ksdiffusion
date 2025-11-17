import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm


class MetaDiffusionFineTuner:
    def __init__(self, diffuser, config):
        self.diffuser = diffuser
        self.config = config
        self.device = config.device

    def meta_adapt(self, support_seqs, inner_lr=1e-4, meta_lr=1e-5, adapt_steps=5, meta_steps=1000):
        self.diffuser.esm_model.requires_grad_(False)
        support_emb = self.diffuser.encode_sequences(support_seqs).to(self.device)
        if support_emb.numel() == 0:
            print("Error: Support embeddings are empty. Skipping meta-learning.")
            return

        condition_params = list(self.diffuser.model.condition_proj.parameters()) + \
                           list(self.diffuser.model.fusion_layer.parameters()) + \
                           list(self.diffuser.model.adaLN_modulation.parameters()) + \
                           list(self.diffuser.model.positional_embedding.parameters())

        meta_optimizer = optim.Adam(condition_params, lr=meta_lr)

        print(f"Meta-learning with {len(support_seqs)} support samples...")

        for step in range(meta_steps):
            fast_weights = {name: param.clone() for name, param in self.diffuser.model.named_parameters()}

            for _ in range(adapt_steps):
                idx = torch.randint(0, len(support_emb), (4,))
                batch_emb = support_emb[idx]
                t = torch.randint(0, self.diffuser.diffusion.steps, (4,), device=self.device)
                noise = torch.randn_like(batch_emb)
                noisy_emb = self.diffuser.diffusion.add_noise(batch_emb, t)[0]

                predicted_noise = self.diffuser.model(noisy_emb, t, batch_emb)
                loss = F.mse_loss(predicted_noise, noise)

                grads = torch.autograd.grad(
                    loss,
                    list(fast_weights.values()),
                    create_graph=True,
                    allow_unused=True
                )

                new_fast_weights = {}
                for (name, param), grad in zip(fast_weights.items(), grads):
                    if grad is None:
                        new_fast_weights[name] = param
                    else:
                        new_fast_weights[name] = param - inner_lr * grad
                fast_weights = new_fast_weights

            meta_optimizer.zero_grad()
            idx = torch.randint(0, len(support_emb), (4,))
            batch_emb = support_emb[idx]
            t = torch.randint(0, self.diffuser.diffusion.steps, (4,), device=self.device)
            noise = torch.randn_like(batch_emb)
            noisy_emb = self.diffuser.diffusion.add_noise(batch_emb, t)[0]

            original_params = {}
            for name, param in self.diffuser.model.named_parameters():
                original_params[name] = param.data.clone()
                param.data = fast_weights[name].data

            predicted_noise = self.diffuser.model(noisy_emb, t, batch_emb)

            for name, param in self.diffuser.model.named_parameters():
                param.data = original_params[name]

            meta_loss = F.mse_loss(predicted_noise, noise)
            meta_loss.backward()

            torch.nn.utils.clip_grad_norm_(condition_params, max_norm=1.0)

            meta_optimizer.step()

            if step % 100 == 0:
                print(f"Meta-step {step}/{meta_steps} | Loss: {meta_loss.item():.4f}")

        print("Meta-learning completed!")


class AdversarialDiffusionFineTuner:
    def __init__(self, diffuser, classifier, tokenizer, config):
        self.diffuser = diffuser
        self.classifier = classifier
        self.tokenizer = tokenizer
        self.config = config
        self.device = config.device

    def adversarial_fine_tune(self, sequences, epochs=200, batch_size=4):
        self.classifier.eval()
        for param in self.classifier.parameters():
            param.requires_grad = False

        embeddings = self.diffuser.encode_sequences(sequences).to(self.device)
        if embeddings.numel() == 0:
            print("Error: Embeddings are empty. Skipping adversarial fine-tuning.")
            return

        dataset = torch.utils.data.TensorDataset(embeddings)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        trainable_params = []
        if hasattr(self.diffuser.model, 'condition_proj'):
            trainable_params.extend(list(self.diffuser.model.condition_proj.parameters()))
        if hasattr(self.diffuser.model, 'fusion_layer'):
            trainable_params.extend(list(self.diffuser.model.fusion_layer.parameters()))
        if hasattr(self.diffuser.model, 'adaLN_modulation'):
            trainable_params.extend(list(self.diffuser.model.adaLN_modulation.parameters()))
        if hasattr(self.diffuser.model, 'positional_embedding'):
            trainable_params.extend(list(self.diffuser.model.positional_embedding.parameters()))

        if not trainable_params:
            trainable_params = list(self.diffuser.model.parameters())

        optimizer = optim.AdamW(
            trainable_params,
            lr=1e-6,
            weight_decay=0.1
        )

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs * len(loader),
            eta_min=1e-8
        )

        accumulation_steps = max(1, 16 // batch_size)

        for epoch in range(epochs):
            total_loss = 0
            optimizer.zero_grad()

            for i, batch in enumerate(tqdm(loader, desc=f"Adversarial Epoch {epoch + 1}")):
                batch_emb = batch[0].to(self.device)
                batch_size_current = batch_emb.size(0)

                t = torch.randint(0, self.diffuser.diffusion.steps, (batch_size_current,), device=self.device).long()
                noise = torch.randn_like(batch_emb)

                if hasattr(self.diffuser.diffusion, 'add_noise'):
                    noisy_emb, _ = self.diffuser.diffusion.add_noise(batch_emb, t)
                else:
                    noisy_emb = batch_emb + noise

                predicted_noise = self.diffuser.model(noisy_emb, t, batch_emb)
                recon_loss = F.mse_loss(predicted_noise, noise)

                if (hasattr(self.diffuser.diffusion, 'sqrt_recip_alphas') and
                        hasattr(self.diffuser.diffusion, 'sqrt_one_minus_alphas_cumprod')):
                    denoised_emb = (
                        self.diffuser.diffusion.sqrt_recip_alphas[t][:, None, None] *
                        (noisy_emb - self.diffuser.diffusion.sqrt_one_minus_alphas_cumprod[t][:, None, None] * predicted_noise)
                    )
                else:
                    denoised_emb = noisy_emb - predicted_noise

                adv_attention_mask = torch.ones(
                    denoised_emb.shape[0], 1, 1, denoised_emb.shape[1],
                    device=self.device
                )

                preds = self.classifier(
                    input_embeddings=denoised_emb,
                    attention_mask=adv_attention_mask
                )

                adv_loss = F.binary_cross_entropy_with_logits(
                    preds,
                    torch.ones_like(preds)
                )

                phys_loss = self.calculate_biophysical_loss(denoised_emb, batch_emb)

                loss = recon_loss + 1.5 * adv_loss + 0.2 * phys_loss
                loss = loss / accumulation_steps
                loss.backward()

                if (i + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

                total_loss += loss.item() * accumulation_steps

            avg_loss = total_loss / len(loader)
            print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f}")

        final_model_path = os.path.join(self.config.output_dir, "adv_diffuser_final.pth")
        os.makedirs(self.config.output_dir, exist_ok=True)
        torch.save(self.diffuser.model.state_dict(), final_model_path)
        print(f"Saved final adversarial fine-tuned model to {final_model_path}")

    def calculate_biophysical_loss(self, denoised_emb, original_emb):
        len_loss = F.mse_loss(
            denoised_emb.norm(dim=2).mean(dim=1),
            original_emb.norm(dim=2).mean(dim=1)
        )

        hydrophobic_loss = F.mse_loss(
            denoised_emb[:, :, :100].mean(),
            original_emb[:, :, :100].mean()
        )

        charge_loss = F.mse_loss(
            denoised_emb[:, :, 500:600].mean(),
            original_emb[:, :, 500:600].mean()
        )

        return len_loss + hydrophobic_loss + charge_loss