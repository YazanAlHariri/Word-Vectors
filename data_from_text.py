import json


# def words_filtering(words: set):
#     command = input(";")
#     while command != "end":
#         if command == "remove":
#             pass


def make_distribution(file):
    def get_range(pos, max_):
        r = list(range(pos - 5, pos + 5))
        r.remove(pos)
        try:
            r = r[r.index(-1) + 1:]
        except ValueError:
            pass
        try:
            r = r[:r.index(max_)]
        except ValueError:
            pass
        return r

    with open(file, "r") as f:
        data = f.read().replace(".", "\n").lower()
    with open(file, "w") as f:
        f.write(data)

    data = "".join(map(lambda x: x if x not in ',[]()?!:;' else "", data)).split("\n")

    words = set()
    for p in data:
        for word in p.split():
            words.add(word)
    print(len(words), words)
    words = {word: n for n, word in enumerate(words)}

    distribution = {}
    for p in data:
        part = p.split()
        m = len(part)
        for n, word in enumerate(part):
            w = words[word]
            if w not in distribution:
                distribution[w] = {}
            for k in get_range(n, m):
                w1 = words[part[k]]
                if w1 in distribution[w]:
                    distribution[w][w1] += 1
                else:
                    distribution[w][w1] = 1

    with open("./distribution.json", "w") as f:
        json.dump(distribution, f)
    with open("./words.json", "w") as f:
        json.dump(words, f)


def to_percentage(using=sum):
    with open("./distribution.json", "r") as f:
        distribution = json.load(f)

    for word_distribution in distribution.values():
        m = using(word_distribution.values())
        for word in word_distribution.keys():
            word_distribution[word] /= m

    with open("./distribution.json", "w") as f:
        json.dump(distribution, f)


def main():
    make_distribution("./data.txt")
    to_percentage(max)


if __name__ == '__main__':
    main()
