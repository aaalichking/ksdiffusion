import numpy as np
import torch
import torch.nn.functional as F
import random
from sklearn.cluster import KMeans


class SampleSelector:
    def __init__(self, classifier, tokenizer, config, diffuser=None):
        self.classifier = classifier
        self.tokenizer = tokenizer
        self.config = config
        self.device = config.device
        self.diffuser = diffuser
        self.embedding_cache = {}

    def calculate_average_quality(self, sequences):
        if not sequences:
            return 0.0
        quality_scores = self.calculate_quality(sequences)
        return float(np.mean(quality_scores)) if len(quality_scores) > 0 else 0.0

    def is_biologically_valid(self, sequence):
        if len(sequence) < 6 or len(sequence) > 25:
            return False

        valid_aa = "ACDEFGHIKLMNPQRSTVWY"
        if any(aa not in valid_aa for aa in sequence):
            return False

        return True

    def select_samples(self, sequences, original_sequences, target_count,
                       quality_threshold=0.6, diversity_threshold=0.5,
                       max_attempts=5):
        try:
            all_embs = torch.tensor([]).to(self.device)
            valid_sequences = []
            sequence_scores = {}

            total_generated = sequences.copy() if sequences is not None else []

            if sequences:
                valid_sequences = [seq for seq in total_generated if self.is_biologically_valid(seq)]

            attempts = 0
            selected_sequences = []

            while len(selected_sequences) < target_count and attempts < max_attempts:
                if attempts > 0:
                    current_gap = target_count - len(selected_sequences)
                    new_sequences = self.diffuser.generate_samples(
                        condition_sequences=original_sequences,
                        num_samples=current_gap * self.config.generation_factor
                    )
                    if new_sequences:
                        total_generated.extend(new_sequences)
                    valid_sequences = [seq for seq in total_generated if self.is_biologically_valid(seq)]

                if total_generated:
                    current_embs = []
                    for seq in total_generated:
                        emb = self.get_sequence_embedding(seq)
                        if emb is not None and emb.numel() > 0:
                            current_embs.append(emb)

                    if not current_embs:
                        print(f"Attempt {attempts + 1}/{max_attempts}: No valid embeddings generated. Skipping.")
                        attempts += 1
                        continue

                    all_embs = torch.stack(current_embs).to(self.device)
                else:
                    print(f"Attempt {attempts + 1}/{max_attempts}: No sequences generated.")
                    attempts += 1
                    continue

                original_embs = []
                for seq in original_sequences:
                    emb = self.get_sequence_embedding(seq)
                    if emb is not None and emb.numel() > 0:
                        original_embs.append(emb)

                if not original_embs:
                    dummy_emb = torch.zeros_like(all_embs[0])
                    original_embs = [dummy_emb] * len(original_sequences)
                    print("Warning: Using dummy embeddings for original sequences")

                original_embs = torch.stack(original_embs).to(self.device)

                if all_embs.dim() == 2:
                    all_embs = all_embs.unsqueeze(1)
                if original_embs.dim() == 2:
                    original_embs = original_embs.unsqueeze(1)

                all_embs_norm = F.normalize(all_embs, p=2, dim=2)
                original_embs_norm = F.normalize(original_embs, p=2, dim=2)

                sim_matrix = torch.einsum('bse,ose->bso', all_embs_norm, original_embs_norm)
                avg_sim = sim_matrix.mean(dim=(1, 2))
                diversity_scores = 1 - avg_sim.cpu().numpy()

                quality_scores = self.calculate_quality(total_generated)

                if len(quality_scores) == 0 or len(quality_scores) != len(total_generated):
                    print(f"Attempt {attempts + 1}/{max_attempts}: Quality scores mismatch. Skipping.")
                    attempts += 1
                    continue

                combined_scores = quality_scores * diversity_scores
                sequence_scores = dict(zip(total_generated, combined_scores))

                strict_mask = (quality_scores >= quality_threshold) & (diversity_scores >= diversity_threshold)
                strict_selected = [seq for i, seq in enumerate(total_generated) if strict_mask[i]]

                print(
                    f"Attempt {attempts + 1}/{max_attempts}: "
                    f"Total samples: {len(total_generated)}, "
                    f"Strict filtered: {len(strict_selected)}"
                )

                for seq in strict_selected:
                    if seq not in selected_sequences and len(selected_sequences) < target_count:
                        selected_sequences.append(seq)

                if len(selected_sequences) >= target_count:
                    selected_sequences = selected_sequences[:target_count]
                    print(f"Reached target count after {attempts + 1} attempts.")
                    break

                attempts += 1

            if len(selected_sequences) < target_count:
                print(
                    f"Using fallback selection. Current: {len(selected_sequences)}, "
                    f"Target: {target_count}, Needed: {target_count - len(selected_sequences)}"
                )
                num_needed = target_count - len(selected_sequences)

                if all_embs.numel() > 0:
                    if all_embs.dim() == 3:
                        flat_embs = all_embs.mean(dim=1).cpu().numpy()
                    else:
                        flat_embs = all_embs.cpu().numpy()

                    cluster_selected = self.select_by_clustering(
                        total_generated,
                        flat_embs,
                        num_needed,
                        combined_scores,
                        diversity_scores
                    )

                    for seq in cluster_selected:
                        if seq not in selected_sequences and len(selected_sequences) < target_count:
                            selected_sequences.append(seq)
                    print(f"Selected {len(cluster_selected)} samples by clustering.")
                else:
                    print("No valid embeddings for clustering. Falling back to combined scores.")
                    if 'quality_scores' in locals() and 'diversity_scores' in locals():
                        combined_scores_fallback = quality_scores * diversity_scores
                    else:
                        quality_scores_fallback = self.calculate_quality(total_generated)
                        diversity_scores_fallback = np.ones(len(total_generated))
                        combined_scores_fallback = quality_scores_fallback * diversity_scores_fallback

                    sorted_indices = np.argsort(combined_scores_fallback)[::-1]
                    additional_selected = [
                        total_generated[i] for i in sorted_indices[:num_needed]
                    ]

                    for seq in additional_selected:
                        if seq not in selected_sequences and len(selected_sequences) < target_count:
                            selected_sequences.append(seq)
                    print(
                        f"Selected {len(additional_selected)} samples by combined scores (fallback)."
                    )

            if len(selected_sequences) < target_count:
                num_still_needed = target_count - len(selected_sequences)
                print(
                    f"Final fallback: Selecting top {num_still_needed} samples "
                    f"by combined scores (including invalid)."
                )

                if not sequence_scores:
                    quality_scores = self.calculate_quality(total_generated)
                    diversity_scores = []
                    for seq in total_generated:
                        emb = self.get_sequence_embedding(seq)
                        if emb is not None and emb.numel() > 0:
                            emb = emb.unsqueeze(0).unsqueeze(0)
                            emb_norm = F.normalize(emb, p=2, dim=2)
                            if original_embs is not None and original_embs.numel() > 0:
                                orig_embs_norm = F.normalize(original_embs, p=2, dim=2)
                                sim_matrix = torch.einsum('bse,ose->bso', emb_norm, orig_embs_norm)
                                avg_sim = sim_matrix.mean().item()
                                diversity = 1 - avg_sim
                            else:
                                diversity = 0.5
                        else:
                            diversity = 0.5
                        diversity_scores.append(diversity)

                    combined_scores = np.array(quality_scores) * np.array(diversity_scores)
                    sequence_scores = dict(zip(total_generated, combined_scores))

                sorted_sequences = sorted(
                    total_generated,
                    key=lambda x: sequence_scores.get(x, 0),
                    reverse=True
                )

                added_count = 0
                for seq in sorted_sequences:
                    if seq not in selected_sequences and len(selected_sequences) < target_count:
                        selected_sequences.append(seq)
                        added_count += 1
                        if len(selected_sequences) >= target_count:
                            break

                print(
                    f"Selected {added_count} additional samples by combined scores."
                )

            final_selected = []
            for seq in selected_sequences:
                if seq not in final_selected and len(final_selected) < target_count:
                    final_selected.append(seq)

            if len(final_selected) < target_count:
                gap = target_count - len(final_selected)
                print(f"WARNING: Still missing {gap} samples after all fallbacks.")

                max_extra_attempts = 5
                extra_attempts = 0
                while len(final_selected) < target_count and extra_attempts < max_extra_attempts:
                    gap = target_count - len(final_selected)
                    extra_needed = max(1, min(10, gap))

                    print(f"Generating {extra_needed} extra samples to fill gap...")
                    extra_sequences = self.diffuser.generate_samples(
                        condition_sequences=original_sequences,
                        num_samples=extra_needed
                    )

                    if extra_sequences:
                        for seq in extra_sequences:
                            if seq not in final_selected and len(final_selected) < target_count:
                                final_selected.append(seq)
                        print(
                            f"Added {len(extra_sequences)} extra generated samples in extra attempt {extra_attempts + 1}."
                        )
                    else:
                        print(
                            f"Extra generation attempt {extra_attempts + 1} failed: no sequences generated."
                        )

                    extra_attempts += 1

                if len(final_selected) < target_count:
                    print(
                        f"CRITICAL WARNING: Failed to generate enough samples after {max_extra_attempts} extra attempts. "
                        f"Only {len(final_selected)} samples selected."
                    )

            return final_selected, len(final_selected)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Critical error in select_samples: {str(e)}")
            return [], 0

    def calculate_quality(self, sequences):
        if not sequences:
            return np.array([])

        self.classifier.eval()
        scores = []

        batch_size = max(2, 32)

        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i + batch_size]
            if len(batch_seqs) == 0:
                continue

            inputs = self.tokenizer(
                batch_seqs,
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=self.config.max_length
            ).to(self.device)

            with torch.no_grad():
                logits = self.classifier(inputs['input_ids'], inputs['attention_mask'])
                preds = torch.sigmoid(logits)
                preds_np = preds.detach().cpu().numpy()
                if preds_np.ndim == 0:
                    preds_np = np.array([preds_np])
                scores.append(preds_np)

        return np.concatenate(scores) if scores else np.array([])

    def get_sequence_embedding(self, sequence):
        if sequence in self.embedding_cache:
            return self.embedding_cache[sequence]

        try:
            inputs = self.tokenizer(
                sequence,
                return_tensors='pt',
                max_length=self.config.max_length,
                padding='max_length',
                truncation=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.diffuser.esm_model(**inputs)
                embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu()
                self.embedding_cache[sequence] = embedding
                return embedding
        except Exception as e:
            print(f"Error generating embedding for sequence: {sequence[:20]}... Error: {str(e)}")
            return None

    def select_by_clustering(self, sequences, embeddings, num_needed, combined_scores=None, diversity_scores=None):
        if num_needed <= 0:
            return []

        if len(sequences) <= num_needed:
            return sequences[:num_needed]

        n_clusters = min(10, max(5, num_needed // 3))
        if len(sequences) < n_clusters:
            n_clusters = len(sequences)

        print(f"Clustering {len(sequences)} samples into {n_clusters} clusters...")

        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.config.seed, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)

            cluster_counts = np.bincount(cluster_labels)
            cluster_quota = (cluster_counts / cluster_counts.sum() * num_needed).astype(int)

            while cluster_quota.sum() < num_needed:
                largest_cluster = np.argmax(cluster_counts)
                cluster_quota[largest_cluster] += 1
                cluster_counts[largest_cluster] = 0

            while cluster_quota.sum() > num_needed:
                largest_quota = np.argmax(cluster_quota)
                cluster_quota[largest_quota] -= 1

            selected_from_clusters = []

            for cluster_id in range(n_clusters):
                indices_in_cluster = np.where(cluster_labels == cluster_id)[0]
                if len(indices_in_cluster) == 0:
                    continue

                cluster_sequences = [sequences[i] for i in indices_in_cluster]

                if combined_scores is None or diversity_scores is None:
                    selected_indices = np.random.choice(
                        len(cluster_sequences),
                        min(cluster_quota[cluster_id], len(cluster_sequences)),
                        replace=False
                    )
                    selected_from_cluster = [cluster_sequences[i] for i in selected_indices]
                else:
                    cluster_scores = combined_scores[indices_in_cluster]
                    sorted_indices = np.argsort(cluster_scores)[::-1]
                    num_to_select = min(cluster_quota[cluster_id], len(cluster_sequences))
                    selected_from_cluster = []
                    for idx in sorted_indices[:num_to_select]:
                        selected_from_cluster.append(cluster_sequences[idx])

                selected_from_clusters.extend(selected_from_cluster)

            if len(selected_from_clusters) < num_needed:
                remaining_indices = []
                for i, seq in enumerate(sequences):
                    if seq not in selected_from_clusters:
                        remaining_indices.append(i)

                if remaining_indices:
                    num_still_needed = num_needed - len(selected_from_clusters)
                    selected_indices = np.random.choice(
                        remaining_indices,
                        min(num_still_needed, len(remaining_indices)),
                        replace=False
                    )
                    for i in selected_indices:
                        selected_from_clusters.append(sequences[i])

            return selected_from_clusters[:num_needed]

        except Exception as e:
            print(f"Clustering failed: {str(e)}")
            if len(sequences) > num_needed:
                return random.sample(sequences, num_needed)
            else:
                return sequences