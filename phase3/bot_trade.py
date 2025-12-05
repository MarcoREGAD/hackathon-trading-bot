

import random

history = []

def get_delta(history: list[dict[str, int]]) -> float:
    return history[-1]["priceA"] - history[-2]["priceA"]

def make_decision(epoch: int, priceA: float, priceB: float):
    history.append({"epoch": epoch, "priceA": priceA, "priceB": priceB})
    if (len(history) < 2):
        return {'Asset A':1/3, 'Asset B':1/3, 'Cash': 1/3}
    if get_delta(history) > 0:
        return {'Asset A':1/3, 'Asset B':1/3, 'Cash': 1/3}
    else:
        return {'Asset A':1/3, 'Asset B':1/3, 'Cash': 1/3}