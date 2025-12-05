

import random

history = []

def get_delta(history: list[dict[str, int]]) -> float:
    return history[-1]["price"] - history[-2]["price"]


def make_decision(epoch: int, price: float):
    # On ajoute le prix au fur et à mesure
    history.append(price)

    # On ne peut rien faire avant d'avoir au moins 500 points
    if len(history) < 500:
        return {"Asset A": 0.5, "Cash": 0.5}

    # Calcul des moyennes mobiles
    mm8 = sum(history[-8:]) / 8
    mm5 = sum(history[-5:]) / 5

    # Logique de décision
    if mm8 > mm5:
        # Tendance haussière : on investit (ex : 80%)
        return {"Asset A": 0.8, "Cash": 0.2}
    else:
        # Tendance baissière : on se protège (ex : 20%)
        return {"Asset A": 0.2, "Cash": 0.8}
