from turtle import pd
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import csv
from collections import Counter, defaultdict
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


class SentimentClassifier(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        sequence_length: int,
        word_embedding_dims: int,
        hidden_layer_dims: int,
    ):
        super().__init__()

        self.embedding_layer = nn.Embedding(
            num_embeddings=vocabulary_size + 1, dims=word_embedding_dims
        )
        self.layers = [
            nn.Linear(
                input_dims=word_embedding_dims * sequence_length,
                output_dims=hidden_layer_dims,
            ),
            nn.Linear(input_dims=hidden_layer_dims, output_dims=hidden_layer_dims),
            nn.Linear(input_dims=hidden_layer_dims, output_dims=2),
        ]

    def __call__(self, x: mx.array):
        embedded = self.embedding_layer(x)
        input = mx.flatten(embedded, start_axis=1)
        for layer in self.layers[:-1]:
            input = layer(input)
            input = nn.relu(input)
        return self.layers[-1](input)


def loss_fn(model: nn.Module, input: mx.array, target: mx.array):
    logits = model(input)
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

    model = SentimentClassifier(
        vocabulary_size=vocabulary_size,
        sequence_length=max_sequence_length,
        word_embedding_dims=word_embedding_dims,
        hidden_layer_dims=hidden_layer_dims,
    )
    mx.eval(model.parameters())

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    optimizer = optim.SGD(learning_rate=learning_rate)

    current_loss = 0
    losses = []

    for epoch in range(total_epochs):
        perm = mx.array(np.random.permutation(training_inputs.shape[0]))
        for start in range(0, training_inputs.shape[0], batch_size):
            ids = perm[start : start + batch_size]
            inputs = training_inputs[ids]
            labels = training_labels[ids]
            loss, grads = loss_and_grad_fn(model, inputs, labels)
            optimizer.update(model, grads)
            mx.eval(model.state, optimizer.state, loss, grads)
            if debug and (start % 4096 == 0 or start % 3968 == 0):
                print(f"======= stats for start {start} =======")
                print("embedding layer weight: ", model.parameters()['embedding_layer']['weight'][0])
                print("embedding layer grad: ", grads['embedding_layer']['weight'][0])
                print("linear layer weight: ", model.parameters()['layers'][0]['weight'][0])
                print("linear layer grad: ", grads['layers'][0]['weight'][0])
                print("loss: ", loss)
                print(f"======= end for start {start} =======")
                
            current_loss = loss.item()
        print(f"*+*+*+*+*+*+*+*+*+*+ EPOCH STATS FOR EPOCH {epoch} *+*+*+*+*+*+*+*+*+*+")
        print(
            " loss: ", current_loss
        )
        losses.append(current_loss)
    print("!!!!!!!!!! Training Complete !!!!!!!!!!")
    print("losses: ", losses)


if __name__ == "__main__":
    main()
