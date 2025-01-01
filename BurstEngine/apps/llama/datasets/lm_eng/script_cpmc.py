import random


def rand(n: int, r: random.Random):
    return int(r.random() * n)


def transform(data, num_sample: int, r: random.Random):
    if len(data["below"]) > 0:
        ipt = data["above"] + data["<ans>"]
        ans = data["below"]
    else:
        ipt = data["above"]
        ans = data["<ans>"]
    return {"input": "", "output": ipt + " " + ans}
