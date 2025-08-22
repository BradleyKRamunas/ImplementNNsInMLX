import csv
from collections import Counter, defaultdict

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np


def normalize_and_tokenize_text(text: str) -> list[str]:
    cleaned: str = "".join(char for char in text if char.isalpha() or char.isspace())
    lowered: str = cleaned.lower()
    return lowered.split(" ")


def get_sentiment_one_hot_target(sentiment: str) -> mx.array:
    return mx.array([1, 0] if sentiment == "positive" else [0, 1])


def load_training_and_test_data(
    path: str, max_sequence_length: int, max_vocabulary_size: int, test_data_size: int
) -> tuple[mx.array, mx.array, mx.array, mx.array, int]:
    row_data: list[tuple[list[str], str]] = []
    with open(path, newline="") as csv_file:
        csv_reader = csv.reader(csv_file)
        row_data = [(normalize_and_tokenize_text(row[0]), row[1]) for row in csv_reader]

    training_data = row_data[:-test_data_size]
    test_data = row_data[test_data_size:]

    token_counter: Counter = Counter()
    for row in training_data:
        token_counter.update(row[0])
    token_mapping: dict[str, int] = dict(
        {
            (tup[1][0], tup[0])
            for tup in list(enumerate(token_counter.most_common(max_vocabulary_size)))
        }
    )
    unk: int = len(token_mapping)
    padding: int = len(token_mapping) + 1
    token_mapper: defaultdict[str, int] = defaultdict(lambda: unk, token_mapping)

    training_inputs: list[mx.array] = []
    training_labels: list[mx.array] = []
    for row in training_data:
        full_sequence = row[0]
        sentiment_one_hot_target = get_sentiment_one_hot_target(row[1])
        sequence_length = len(full_sequence)
        for start in range(0, sequence_length, max_sequence_length):
            end = min(start + max_sequence_length, sequence_length)
            sub_sequence = full_sequence[start:end]
            mapped_sub_sequence = [token_mapper[s] for s in sub_sequence]
            if len(mapped_sub_sequence) < max_sequence_length:
                mapped_sub_sequence.extend(
                    [padding] * (max_sequence_length - len(mapped_sub_sequence))
                )
            final_sub_sequence = mx.array(mapped_sub_sequence)
            training_inputs.append(final_sub_sequence)
            training_labels.append(sentiment_one_hot_target)

    test_inputs: list[mx.array] = []
    test_labels: list[mx.array] = []
    for row in test_data:
        full_sequence = row[0]
        sentiment_one_hot_target = get_sentiment_one_hot_target(row[1])
        sequence_length = len(full_sequence)
        sub_sequence = full_sequence[:max_sequence_length]
        mapped_sub_sequence = [token_mapper[s] for s in sub_sequence]
        # TODO: for testing sequences longer than max sequence length,
        # maybe we should keep more than max sequence length and average results or
        # take majority vote?
        if len(mapped_sub_sequence) < max_sequence_length:
            mapped_sub_sequence.extend(
                [padding] * (max_sequence_length - len(mapped_sub_sequence))
            )
        final_sub_sequence = mx.array(mapped_sub_sequence)
        test_inputs.append(final_sub_sequence)
        test_labels.append(sentiment_one_hot_target)

    return (
        mx.stack(training_inputs),
        mx.stack(training_labels),
        mx.stack(test_inputs),
        mx.stack(test_labels),
        len(token_mapper) + 2,
    )


# h_candidate = sigmoid(x_t @ W_ih.T + h_previous @ W_hh.T + b)
# h_t = h_candidate * mask + h_{t-1} * (1 - mask)
class RNNModule(nn.Module):
    def __init__(self, input_dims: int, output_dims: int):
        super().__init__()

        self.input_to_hidden_layer = nn.Linear(
            input_dims=input_dims, output_dims=output_dims
        )
        self.hidden_to_hidden_layer = nn.Linear(
            input_dims=output_dims, output_dims=output_dims
        )

    def __call__(self, x: mx.array, mask: mx.array, previous: mx.array):
        return mx.sigmoid(
            self.input_to_hidden_layer(x) + self.hidden_to_hidden_layer(previous)
        ) * mask + previous * (1 - mask)


class RNNSentimentClassifier(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        embedding_dims: int,
        hidden_dims: int,
    ):
        super().__init__()

        self.embedding_dims = embedding_dims
        self.hidden_dims = hidden_dims

        # why do i have a +1 lol
        self.embedding_layer = nn.Embedding(
            num_embeddings=vocabulary_size + 1, dims=embedding_dims
        )
        # lazily initialize to (batch_size, embedding_dim)
        self.rnn_layer = RNNModule(input_dims=embedding_dims, output_dims=hidden_dims)
        self.output_layer = nn.Linear(input_dims=hidden_dims, output_dims=2)

    def __call__(self, x: mx.array, mask: mx.array):
        # expecting x to have shape (batch_size, max_sequence_length) where each value is a token index
        # expect mask to have shape (batch_size, max_sequence_length) where each index is a 0 or 1
        batch_size, max_sequence_length = x.shape
        # expecting output from rnn to always have shape (batch_size, embedding_dim)
        previous_output = mx.ones((batch_size, self.hidden_dims)) * 0.5

        # should have shape (batch_size, max_sequence_length, embedding_dim)
        embedded = self.embedding_layer(x)
        for i in range(max_sequence_length):
            # take a single index/token slice of each sequence
            x_slice = embedded[:, i, :]
            # make mask go from e.g., (batch_size,) to (batch_size, 1)
            mask_slice = mx.expand_dims(mask[:, i], axis=1)
            previous_output = self.rnn_layer(x_slice, mask_slice, previous_output)
        # should have shape (batch_size, 2)
        return self.output_layer(previous_output)


def loss_fn(model: nn.Module, input: mx.array, mask: mx.array, target: mx.array):
    logits = model(input, mask)
    return nn.losses.binary_cross_entropy(logits, target, reduction="mean")


def main():
    # TODO: implement eval method so I can see training vs loss eval at intervals
    debug = False
    max_sequence_length = 64
    word_embedding_dims = 32
    hidden_layer_dims = 64
    batch_size = 32
    learning_rate = 0.1
    max_vocabulary_size = 4_000
    test_data_size = 20_000
    total_epochs = 20
    mx.set_default_device(mx.Device(mx.gpu))

    training_inputs, training_labels, test_inputs, test_labels, vocabulary_size = (
        load_training_and_test_data(
            path="imdb_binary_sent_ds.csv",
            max_sequence_length=max_sequence_length,
            max_vocabulary_size=max_vocabulary_size,
            test_data_size=test_data_size,
        )
    )

    print("training inputs size: ", training_inputs.shape[0])
    print("training inputs shape: ", training_inputs.shape)
    print("training labels shape: ", training_labels.shape)
    print("test inputs shape: ", test_inputs.shape)
    print("test labels shape: ", test_labels.shape)

    model = RNNSentimentClassifier(
        vocabulary_size=vocabulary_size,
        embedding_dims=word_embedding_dims,
        hidden_dims=hidden_layer_dims,
    )
    mx.eval(model.parameters())

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    optimizer = optim.SGD(learning_rate=learning_rate)

    # import pdb; pdb.set_trace()

    current_loss = 0
    losses = []

    for epoch in range(total_epochs):
        perm = mx.array(np.random.permutation(training_inputs.shape[0]))
        for start in range(0, training_inputs.shape[0], batch_size):
            ids = perm[start : start + batch_size]
            inputs = training_inputs[ids]
            labels = training_labels[ids]
            mask = mx.ones((inputs.shape))
            loss, grads = loss_and_grad_fn(model, inputs, mask, labels)
            optimizer.update(model, grads)
            mx.eval(model.state, optimizer.state, loss, grads)
            if debug and (start % 4096 == 0 or start % 3968 == 0):
                print(f"======= stats for start {start} =======")
                print(
                    "embedding layer weight: ",
                    model.parameters()["embedding_layer"]["weight"][0],
                )
                print("embedding layer grad: ", grads["embedding_layer"]["weight"][0])
                print(
                    "linear layer weight: ",
                    model.parameters()["layers"][0]["weight"][0],
                )
                print("linear layer grad: ", grads["layers"][0]["weight"][0])
                print("loss: ", loss)
                print(f"======= end for start {start} =======")

            current_loss = loss.item()
        print(
            f"*+*+*+*+*+*+*+*+*+*+ EPOCH STATS FOR EPOCH {epoch} *+*+*+*+*+*+*+*+*+*+"
        )
        print(" loss: ", current_loss)
        losses.append(current_loss)


if __name__ == "__main__":
    main()

# TODOS:
# 1. remove max sequence length, normalize all inputs to be of max length of input
# 2. return the relevant padding tensors _OR_ derive the tensors from padding token in inputs?
# 3. clean things up, make progress bar or something for epoch
# 4. add eval function