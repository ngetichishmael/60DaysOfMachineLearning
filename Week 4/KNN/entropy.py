import math


# Function to calculate entropy
def entropy(pos, neg):
    total = pos + neg
    p_pos = pos / total
    p_neg = neg / total

    if p_pos == 0:
        entropy_pos = 0
    else:
        entropy_pos = -p_pos * math.log2(p_pos)

    if p_neg == 0:
        entropy_neg = 0
    else:
        entropy_neg = -p_neg * math.log2(p_neg)

    return entropy_pos + entropy_neg
