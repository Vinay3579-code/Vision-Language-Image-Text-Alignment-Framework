# Generic trainer for image-text alignment experiments.

from __future__ import annotations

import time

import torch
from tqdm import tqdm

from src.checkpoint import save_projector
from src.metrics import evaluate_alignment

from src.metrics import (
    evaluate_alignment,
)


class Trainer:

    def __init__(
        self,
        model,
        loss_fn,
        optimizer,
        device,
        checkpoint_path,
        scheduler=None,
        gradient_clip=None,
    ):

        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.device = device

        self.gradient_clip = gradient_clip

        self.checkpoint_path = checkpoint_path

    def train_epoch(
        self,
        dataloader,
    ):

        self.model.train()

        total_loss = 0.0
        total_infonce = 0.0
        total_nmse = 0.0

        progress = tqdm(
            dataloader,
            desc="Training",
            leave=False,
        )

        for batch in progress:

            images = batch["image"].to(self.device)

            captions = batch["caption"]

            outputs = self.model(
                images,
                captions,
            )

            loss_dict = self.loss_fn(
                outputs["projected_embeddings"],
                outputs["text_embeddings"],
            )

            loss = loss_dict["loss"]
            info_loss = loss_dict["infonce"]
            nmse_loss = loss_dict["nmse"]

            self.optimizer.zero_grad()

            loss.backward()

            if self.gradient_clip is not None:

                torch.nn.utils.clip_grad_norm_(
                    self.model.projector.parameters(),
                    self.gradient_clip,
                )

            self.optimizer.step()

            total_loss += loss.item()
            total_infonce += info_loss.item()
            total_nmse += nmse_loss.item()

            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                infonce-f"{info_loss.item():.4f}",
                nmse=f"{nmse_loss.item():.4f}",
            )

        if self.scheduler is not None:
            self.scheduler.step()

        return {
            "loss": total_loss / len(dataloader),
            "infonce": total_infonce / len(dataloader),
            "nmse": total_nmse / len(dataloader),
        }

    def fit(
        self,
        dataloader,
        epochs,
    ):

        history = {
            "loss": [],
            "infonce": [],
            "nmse": [],
            "positive_similarity": [],
            "negative_similarity": [],
            "alignment_gap": [],
            "positive_similarity_std": [],
        }

        overall_start = time.perf_counter()

        for epoch in range(1, epochs + 1):

            epoch_start = time.perf_counter()

            train_stats = self.train_epoch(
                dataloader,
            )

            epoch_time = (
                time.perf_counter()
                - epoch_start
            )

            self.model.eval()

            all_projected_embeddings = []
            all_text_embeddings = []
            
            with torch.inference_mode():
            
                for batch in dataloader:
            
                    images = batch["image"].to(self.device)
                    captions = batch["caption"]
            
                    outputs = self.model(
                        images,
                        captions,
                    )
            
                    all_projected_embeddings.append(
                        outputs["projected_embeddings"]
                    )
            
                    all_text_embeddings.append(
                        outputs["text_embeddings"]
                    )
            
            projected_embeddings = torch.cat(
                all_projected_embeddings,
                dim=0,
            )
            
            text_embeddings = torch.cat(
                all_text_embeddings,
                dim=0,
            )
            
            metrics = evaluate_alignment(
                projected_embeddings,
                text_embeddings,
            )

            history["loss"].append(train_stats["loss"])
            history["infonce"].append(train_stats["infonce"])
            history["nmse"].append(train_stats["nmse"])
            
            history["positive_similarity"].append(
                metrics["Mean Positive Similarity"]
            )
            
            history["negative_similarity"].append(
                metrics["Mean Negative Similarity"]
            )
            
            history["alignment_gap"].append(
                metrics["Alignment Gap"]
            )
            
            history["positive_similarity_std"].append(
                metrics["Positive Similarity Standard Deviation"]
            )

            print(
                f"\nEpoch {epoch:02d}/{epochs}"
            )
            
            print(
                f"Loss                     : {train_stats['loss']:.4f}"
            )
            
            print(
                f"InfoNCE                  : {train_stats['infonce']:.4f}"
            )
            
            print(
                f"NMSE                     : {train_stats['nmse']:.4f}"
            )
            
            print(
                f"Mean Positive Similarity : "
                f"{metrics['Mean Positive Similarity']:.4f}"
            )
            
            print(
                f"Mean Negative Similarity : "
                f"{metrics['Mean Negative Similarity']:.4f}"
            )
            
            print(
                f"Alignment Gap            : "
                f"{metrics['Alignment Gap']:.4f}"
            )
            
            print(
                f"Positive Similarity Standard Deviation      : "
                f"{metrics['Positive Similarity Standard Deviation']:.4f}"
            )
            
            print(
                f"Epoch Time               : {epoch_time:.2f}s"
            )

        total_time = (
            time.perf_counter()
            - overall_start
        )

        save_projector(
            self.model.projector,
            self.checkpoint_path,
        )

        print()

        print(f"Training completed in {total_time:.2f} seconds.")

        return history
