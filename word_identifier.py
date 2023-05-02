import json
from typing import Tuple
import AI
from random import randint


def initialize_prepare(vectors_size) -> Tuple[AI.Network, dict, int]:
    print("Initializing")
    with open("./distribution.json", "r") as f:
        distribution = json.load(f)

    num_words = len(distribution)
    network = AI.Network()
    network.initialize(num_words, 1, vectors_size, num_words)
    print([len(layer) for layer in network])

    print("Preparing")
    targets = []
    for index in range(num_words):
        t = [0] * num_words
        for word, probability in distribution[str(index)].items():
            t[int(word)] = probability
        targets.append(t)
    return network, targets, num_words


def train(network, targets, num_words):
    for index in range(num_words):
        input_ = [0] * num_words
        input_[index] = 1
        network.forward(input_)
        network.backpropagation(targets[index])


def test(network, targets, num_words):
    n = randint(0, num_words - 1)
    input_ = [0] * num_words
    input_[n] = 1
    network.forward(input_)
    return sum(map(lambda k: (network.get_output()[k] - targets[n][k]) ** 2, range(num_words)))


def get_vectors(network, num_words):
    vectors = []
    for i in range(num_words):
        inp = [0] * num_words
        inp[i] = 1
        network.forward(inp)
        vectors.append(network.get_output(1))
    return vectors


def to_vectors():
    net, targets, num_words = initialize_prepare(50)
    print("Training")
    for num in range(200):
        print(num)
        train(net, targets, num_words)

    print("Testing")
    error = 0
    for _ in range(20):
        error += test(net, targets, num_words)
    print("Error: ", error ** 0.5)

    print("Saving")
    vs = get_vectors(net, num_words)
    with open("./vectors.json", "w") as f:
        json.dump(vs, f)


def to_vectors_timed():
    import cProfile
    import pstats

    with cProfile.Profile() as pr:
        to_vectors()

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats()
    stats.dump_stats(filename='needs_profiling.prof')


def similarity():
    with open("./words.json", "r") as f:
        words = json.load(f)
    with open("./vectors.json", "r") as f:
        vectors = json.load(f)

    def distance(v1, v2): return sum((c2 - c1) * (c2 - c1) for c1, c2 in zip(v1, v2)) ** 0.5

    while True:
        word1 = input("Word: ")
        try:
            v = vectors[words[word1]]
        except KeyError:
            print(word1, "not in words.")
            continue
        print(sorted([(word, distance(v, vectors[index])) for word, index in words.items()], key=lambda x: x[1]))


def main():
    to_vectors_timed()
    similarity()


if __name__ == '__main__':
    main()
