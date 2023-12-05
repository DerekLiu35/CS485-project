# CS485-project

Lester et al. 2021 propose prompt tuning models by freezing a pretrained model and learning “soft prompts” to prepend with the embedded word inputs. The performance seems pretty good and and with certain model sizes prompt tuning is competitive with full model fine tuning. The benefits prompts are: it is computationally less expensive then full fine tuning and less space is needed to store the prompts (num parameters = num tokens * embedding dim). In addition, prompt tuning seems to reduce overfitting on the training set.

However, the soft prompts learned by prompt tuning are not that interpretable. The cosine similarity of the learned prompt embeddings and the embeddings of words in the vocab can lead to some sense of interpretablity, but I wanted to find a better way to create interpretable prompts. 

The method I came up with (at least to my knowledge, it’s totally plausible that someone else has come up with this ideas as it is a pretty simple one), is to learn some unnormalized log weights as the soft prompt of size (num tokens, vocab size).

So, when you take the softmax of the soft prompt you get a probability distributions over the vocabulary for each “token”. Then, you take the probability matrix and matrix-multiply it by the embedding matrix and concatenate it to your embedded word inputs.  After multiplying the probability matrix by the embedding matrix the concatenated “tokens” can be interpreted as a weighted sum of the embeddings of words in the vocabulary. Intuitively, to me, I expected this to lead to more human interpretable prompts.

Vocabulary Tuning

<img src="./Vocabulary tuning.png" width="300">

Prompt Tuning

<img src="./Prompt tuning.png" width="300">
