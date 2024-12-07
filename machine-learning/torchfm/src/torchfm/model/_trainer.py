import time
from typing import Literal
from loguru import logger
from tqdm import tqdm
from torch.optim import Adam
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from torchfm.model._torchfm import TorchFM
from torchfm.data._fmdataset import FMDataset


class ModelTrainer:
    def __init__(
        self,
        model: TorchFM,
        dataset: FMDataset,
        batch_size: int = 128,
        shuffle: bool = True,
        learning_rate: float=1e-3,
        weight_decay: float=0
    ) -> None:
        self._model = model
        self._optimizer = Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self._criterion = MSELoss()
        self._data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )

    def train(self, num_epochs: int = 5) -> None:
        start = time.time()
        
        logger.info(f"Completed create DataLoader.")
        epoch_bar = tqdm(range(num_epochs), desc=f"Training")
        with epoch_bar:
            for _ in epoch_bar:
                for batch_idx, batch in enumerate(self._data_loader):
                    self._optimizer.zero_grad()
                    u, i, y = batch
                    z = self._model(u, i)
                    loss = self._criterion(z, y)
                    loss.backward()
                    self._optimizer.step()
                    epoch_bar.set_postfix(loss=f"{loss:.4f}", batch=f"{batch_idx + 1}")

        taken = time.time() - start
        if taken >= 3600:
            logger.info(f"Total training time: {taken/3600:.2f} hours")
        elif taken >= 60:
            logger.info(f"Total training time: {taken/60:.2f} minutes")
        else:  # Less than a minute
            logger.info(f"Total training time: {taken:.2f} seconds")

    def predict_by_id(self, user_id: int, item_id: int):
        dataset = self._data_loader.dataset
        user_features = dataset.get_user_features_by_id(user_id)
        item_features = dataset.get_item_features_by_id(item_id)
        u_vector, i_vector = dataset.transform(user_features, item_features, user_id, item_id)
        score = self._model(u_vector, i_vector).item()
        return score

    def evaluate(
        self,
        dataset: FMDataset,
        metrics: list[Literal["mse", "auc", "precision@k", "recall@k", "map@k"]]
    ):
        scores = []
        actuals = []
        for user_id, item_id, actual in dataset._interactions:
            scores.append(self.predict_by_id(user_id, item_id))
            actuals.append(actual)
        return scores