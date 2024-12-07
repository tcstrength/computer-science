import torch
from loguru import logger
from torch import Tensor
from torch import FloatTensor
from torch import nn

class TorchFM(nn.Module):
    def __init__(
        self,
        user_input_dim: int,
        item_input_dim: int,
        embedding_dim: int = 8
    ):
        super().__init__()
        self._user_embeddings = nn.Linear(user_input_dim, embedding_dim)
        self._item_embeddings = nn.Linear(item_input_dim, embedding_dim)
        self._user_biases = nn.Linear(user_input_dim, 1)
        self._item_biases = nn.Linear(item_input_dim, 1)

    def forward(self, u: Tensor, i: Tensor, verbose: bool=False) -> Tensor:
        """Process in batch of size B, the tensor `u` represents the many-hot encoded features of 
        users and the tensor `i` represents the many-hot encoded features of items. The function
        takes these inputs and compute scores for B pairs.

        Args:
            u (Tensor): User inputs, should be in shape (B, U)
            i (Tensor): Item inputs, should be in shape (B, I)
        Returns:
            Tensor: Output in shape (B, 1)
        """

        if u.dtype != torch.float32:
            u = u.to(torch.float32)
        
        if i.dtype != torch.float32:
            i = i.to(torch.float32)

        if len(u.shape) == 1:
            u = u.reshape(1, -1)

        if len(i.shape) == 1:
            i = i.reshape(1, -1)

        u_emb = self._user_embeddings(u) # Output (B, K)
        i_emb = self._item_embeddings(i) # Output (B, K)
        u_bias = self._user_biases(u).squeeze(1) # Output (B, )
        i_bias = self._item_biases(i).squeeze(1) # Output (B, )

        if verbose:
            logger.info((
                "Verbose:\n"
                f"User embeddings: {u_emb.shape}\n"
                f"Item embeddings: {i_emb.shape}\n"
                f"User biases: {u_bias.shape}\n"
                f"Item biases: {i_bias.shape}."
            ))
        scores = (u_emb * i_emb).sum(axis=1)
        scores = scores + u_bias + i_bias
        return scores

if __name__ == "__main__":
    model = TorchFM(
        user_input_dim=2,
        item_input_dim=3,
        embedding_dim=8
    )
    u = FloatTensor([
        [1, 0],
        [0, 1],
        [1, 1]
    ])

    i = FloatTensor([
        [1, 0, 0],
        [1, 1, 0],
        [0, 0, 1]
    ])

    scores = model.forward(u, i, True)
    print(f"Model compute: {scores[0]:.4f}")
    u_emb = model._user_embeddings(u[0])
    i_emb = model._item_embeddings(i[0])
    u_bias = model._user_biases(u[0])
    i_bias = model._item_biases(i[0])
    score = u_emb.dot(i_emb) + u_bias + i_bias
    print(f"Manual compute: {score.item():.4f}")