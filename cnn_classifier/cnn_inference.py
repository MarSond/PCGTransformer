from typing import Any

import torch

from MLHelper import constants as const
from MLHelper.dataset import AudioDataset
from MLHelper.ml_loop import HookManager, ML_Loop
from run import Run

from .cnn_dataset import CNN_Dataset


class CNN_Inference(ML_Loop):

    def __init__(self, run: Run, dataset: AudioDataset) -> None:
        super().__init__(run, dataset, pytorch_dataset_class=CNN_Dataset)

    def prepare_kfold_run(self) -> None:
        # Spezifische Vorbereitungen für die Inferenz, falls erforderlich
        pass

    def plot_batch(self, data, target):
        import matplotlib.pyplot as plt
        _, axs = plt.subplots(2, 5, figsize=(20, 10))
        for i in range(10):
            axs[i // 5, i % 5].imshow(data[i].squeeze().cpu().numpy(), cmap="gray")
            axs[i // 5, i % 5].set_title(f"Label: {target[i]}")
        plt.show()

    @HookManager.hook_wrapper("validation_epoch")
    def validation_epoch_loop(self, epoch: int, fold: int, **kwargs: Any) -> None:
        self.pbars.update_total(bar_name=self.pbars.NAME_VALID, total=len(self.valid_loader))

        y_true = []
        y_pred = []
        for _, (loader_data, loader_target) in enumerate(self.valid_loader):
            data, target = loader_data.to(self.device), loader_target.to(self.device)
            with torch.no_grad():
                loss, probabilities = self.predict_step(
                    model=self.model, inputs=data, labels=target)
                prediction = probabilities.argmax(dim=1)
                y_true += target.cpu().numpy().tolist()
                y_pred += prediction.cpu().numpy().tolist()
                self.metrics.update_step(
                    probabilities=probabilities, labels=target, loss=loss, validation=True)
            self.pbars.increment(bar_name=self.pbars.NAME_VALID)

            if self.run.config[const.SINGLE_BATCH_MODE]:
                break

    def start_inference_task(self, model: torch.nn.Module) -> None:
        self.model = model
        self.model.eval()

        return self.kfold_loop(start_epoch=1)
