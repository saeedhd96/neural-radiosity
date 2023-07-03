from typing import Any

import tinycudann as tcnn
import torch
from nerad.model.embedding import Embedding



class TcnnEmbedding(Embedding):
    def __init__(self, config: dict[str, Any]) -> None:
        super(TcnnEmbedding, self).__init__()

        self.embedding = tcnn.Encoding(3, config, dtype=torch.float32)
        self.embedding_type = config['otype']
        self.n_output_dims = self.embedding.n_output_dims

    def forward(self, x):
        return self.embedding.forward(x)
