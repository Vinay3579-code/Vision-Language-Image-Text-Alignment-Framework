# Generic trainer for image-text alignment experiments.

from __future__ import annotations

import time

import torch
from tqdm import tqdm

from src.checkpoint import save_projector


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

            self.optimizer.zero_grad()

            loss.backward()

            if self.gradient_clip is not None:

                torch.nn.utils.clip_grad_norm_(
                    self.model.projector.parameters(),
                    self.gradient_clip,
                )

            self.optimizer.step()

            total_loss += loss.item()

            progress.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        if self.scheduler is not None:
            self.scheduler.step()

        return total_loss / len(dataloader)

    def fit(
        self,
        dataloader,
        epochs,
    ):

        history = []

        overall_start = time.perf_counter()

        for epoch in range(1, epochs + 1):

            epoch_start = time.perf_counter()

            loss = self.train_epoch(
                dataloader,
            )

            epoch_time = (
                time.perf_counter()
                - epoch_start
            )

            history.append(loss)

            print(
                f"Epoch {epoch:02d}/{epochs} "
                f"| Loss: {loss:.4f} "
                f"| Time: {epoch_time:.2f}s"
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
