from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

# code referenced from https://github.com/kipgparker/soft-prompt-tuning/blob/main/soft_embedding.py
class InterptEmbedding(nn.Module):
    def __init__(self,
                wte: nn.Embedding,
                n_tokens: int = 10,
                random_range: float = 0.5,
                init_text: List[int] = None):
        """appends learned embedding to

        Args:
            wte (nn.Embedding): original transformer word embedding
            n_tokens (int, optional): number of tokens for task. Defaults to 10.
            random_range (float, optional): range to init embedding (if not initialize from vocab). Defaults to 0.5.
            init_text (List[int], optional): initalizes from list of token ids. Defaults to None.
        """
        super(InterptEmbedding, self).__init__()
        self.wte = wte
        self.n_tokens = n_tokens
        self.learned_embedding = nn.parameter.Parameter(self.initialize_embedding(wte,
                                                                               n_tokens,
                                                                               random_range,
                                                                               init_text))

    def initialize_embedding(self,
                             wte: nn.Embedding,
                             n_tokens: int = 10,
                             random_range: float = 0.5,
                             init_text: List[int] = None):
        """initializes learned embedding

        Args:
            same as __init__

        Returns:
            torch.float: initialized using original schemes
        """
        if init_text != None:
            indices = torch.tensor(init_text)
            return 15 * F.one_hot(indices, wte.weight.size(0)).float() # maybe I should make 15 a hyperparameter
        return torch.FloatTensor(n_tokens, wte.weight.size(0)).uniform_(-random_range, random_range)

    def forward(self, tokens):
        """run forward pass

        Args:
            tokens (torch.long): input tokens before encoding

        Returns:
            torch.float: encoding of text concatenated with learned task specifc embedding
        """
        input_embedding = self.wte(tokens[:, self.n_tokens:])
        learned_embedding = F.softmax(self.learned_embedding, dim=1) @ self.wte.weight # (n_tokens, n_dim)
        learned_embedding = learned_embedding.repeat(input_embedding.size(0), 1, 1)
        # hugging face wants the concat then remove
        return torch.cat([learned_embedding, input_embedding], 1)