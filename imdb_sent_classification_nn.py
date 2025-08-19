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
    
    def eval(self, x: mx.array):
        embedded = self.embedding_layer(x)
        input = mx.flatten(embedded, start_axis=1)
        for layer in self.layers[:-1]:
            input = layer(input)
            input = nn.relu(input)
        logits = self.layers[-1](input)
        return mx.argmax(logits, axis=1)


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
    print("!!!!!!!!!! Training Complete !!!!!!!!!!")
    print("losses: ", losses)
    train_predictions = model.eval(training_inputs)
    train_labels_argmax = mx.argmax(training_labels, axis=1)
    train_count_correct = mx.sum(train_predictions == train_labels_argmax)
    train_count_total = train_labels_argmax.shape[0]
    
    print(f'\nTrain Evaluation result: Got {train_count_correct} / {train_count_total} ({100 * train_count_correct / train_count_total}%) correct')
    
    test_predictions = model.eval(test_inputs)
    test_labels_argmax = mx.argmax(test_labels, axis=1)
    test_count_correct = mx.sum(test_predictions == test_labels_argmax)
    test_count_total = test_labels_argmax.shape[0]
    
    print(f'\nTest Evaluation result: Got {test_count_correct} / {test_count_total} ({100 * test_count_correct / test_count_total}%) correct')


if __name__ == "__main__":
    main()

"""
Example output:

training inputs size:  122847
training inputs shape:  (122847, 64)
training labels shape:  (122847, 2)
test inputs shape:  (30001, 64)
test labels shape:  (30001, 2)
*+*+*+*+*+*+*+*+*+*+ EPOCH STATS FOR EPOCH 0 *+*+*+*+*+*+*+*+*+*+
 loss:  0.5929447412490845
*+*+*+*+*+*+*+*+*+*+ EPOCH STATS FOR EPOCH 1 *+*+*+*+*+*+*+*+*+*+
 loss:  0.5483018755912781
*+*+*+*+*+*+*+*+*+*+ EPOCH STATS FOR EPOCH 2 *+*+*+*+*+*+*+*+*+*+
 loss:  0.5887638926506042
*+*+*+*+*+*+*+*+*+*+ EPOCH STATS FOR EPOCH 3 *+*+*+*+*+*+*+*+*+*+
 loss:  0.49543753266334534
*+*+*+*+*+*+*+*+*+*+ EPOCH STATS FOR EPOCH 4 *+*+*+*+*+*+*+*+*+*+
 loss:  0.5944880247116089
*+*+*+*+*+*+*+*+*+*+ EPOCH STATS FOR EPOCH 5 *+*+*+*+*+*+*+*+*+*+
 loss:  0.2962203025817871
*+*+*+*+*+*+*+*+*+*+ EPOCH STATS FOR EPOCH 6 *+*+*+*+*+*+*+*+*+*+
 loss:  0.4095418453216553
*+*+*+*+*+*+*+*+*+*+ EPOCH STATS FOR EPOCH 7 *+*+*+*+*+*+*+*+*+*+
 loss:  0.284853994846344
*+*+*+*+*+*+*+*+*+*+ EPOCH STATS FOR EPOCH 8 *+*+*+*+*+*+*+*+*+*+
 loss:  0.30930444598197937
*+*+*+*+*+*+*+*+*+*+ EPOCH STATS FOR EPOCH 9 *+*+*+*+*+*+*+*+*+*+
 loss:  0.11571268737316132
*+*+*+*+*+*+*+*+*+*+ EPOCH STATS FOR EPOCH 10 *+*+*+*+*+*+*+*+*+*+
 loss:  0.10726296156644821
*+*+*+*+*+*+*+*+*+*+ EPOCH STATS FOR EPOCH 11 *+*+*+*+*+*+*+*+*+*+
 loss:  0.17406384646892548
*+*+*+*+*+*+*+*+*+*+ EPOCH STATS FOR EPOCH 12 *+*+*+*+*+*+*+*+*+*+
 loss:  0.040653545409440994
*+*+*+*+*+*+*+*+*+*+ EPOCH STATS FOR EPOCH 13 *+*+*+*+*+*+*+*+*+*+
 loss:  0.08426908403635025
*+*+*+*+*+*+*+*+*+*+ EPOCH STATS FOR EPOCH 14 *+*+*+*+*+*+*+*+*+*+
 loss:  0.05074501037597656
*+*+*+*+*+*+*+*+*+*+ EPOCH STATS FOR EPOCH 15 *+*+*+*+*+*+*+*+*+*+
 loss:  0.13719087839126587
*+*+*+*+*+*+*+*+*+*+ EPOCH STATS FOR EPOCH 16 *+*+*+*+*+*+*+*+*+*+
 loss:  0.10115986317396164
*+*+*+*+*+*+*+*+*+*+ EPOCH STATS FOR EPOCH 17 *+*+*+*+*+*+*+*+*+*+
 loss:  0.05432279035449028
*+*+*+*+*+*+*+*+*+*+ EPOCH STATS FOR EPOCH 18 *+*+*+*+*+*+*+*+*+*+
 loss:  0.20951376855373383
*+*+*+*+*+*+*+*+*+*+ EPOCH STATS FOR EPOCH 19 *+*+*+*+*+*+*+*+*+*+
 loss:  0.12259147316217422
!!!!!!!!!! Training Complete !!!!!!!!!!
losses:  [0.5929447412490845, 0.5483018755912781, 0.5887638926506042, 0.49543753266334534, 0.5944880247116089, 0.2962203025817871, 0.4095418453216553, 0.284853994846344, 0.30930444598197937, 0.11571268737316132, 0.10726296156644821, 0.17406384646892548, 0.040653545409440994, 0.08426908403635025, 0.05074501037597656, 0.13719087839126587, 0.10115986317396164, 0.05432279035449028, 0.20951376855373383, 0.12259147316217422]

Train Evaluation result: Got 119432 / 122847 (97.2201156616211%) correct

Test Evaluation result: Got 24300 / 30001 (80.99729919433594%) correct

So... quite a bit of overfitting! But not super bad
"""