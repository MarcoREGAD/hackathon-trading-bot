

import random

history = []

MM_A_LENGTH = 17
MM_B_LENGTH = 7

def get_delta(history: list[dict[str, int]]) -> float:
    return history[-1]["price"] - history[-2]["price"]


def make_decision(epoch: int, price: float):
    # On ajoute le prix au fur et à mesure
    history.append(price)

    # On ne peut rien faire avant d'avoir au moins 500 points
    if len(history) < 500:
        return {"Asset A": 0.5, "Cash": 0.5}

    # Calcul des moyennes mobiles
    mm_a = sum(history[-MM_A_LENGTH:]) / MM_A_LENGTH
    mm_b = sum(history[-MM_B_LENGTH:]) / MM_B_LENGTH

    # Logique de décision
    if mm_a > mm_b:
        # Tendance haussière : on investit (ex : 80%)
        return {"Asset A": 0.8, "Cash": 0.2}
    else:
        # Tendance baissière : on se protège (ex : 20%)
        return {"Asset A": 0.2, "Cash": 0.8}
