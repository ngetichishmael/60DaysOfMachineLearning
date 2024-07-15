from entropy import entropy
from total_entropy import total_entropy

# Wind attribute
# Weak
weak_yes = 6
weak_no = 2
weak_entropy = entropy(weak_yes, weak_no)

# Strong
strong_yes = 3
strong_no = 3
strong_entropy = entropy(strong_yes, strong_no)

# Weighted entropy for Wind
weighted_entropy_wind = (8 / 14) * weak_entropy + (6 / 14) * strong_entropy

# Information Gain for Wind
information_gain_wind = total_entropy - weighted_entropy_wind

# Store Wind results in the dictionary
wind_results = {
    "Weak Entropy": weak_entropy,
    "Strong Entropy": strong_entropy,
    "Weighted Entropy (Wind)": weighted_entropy_wind,
    "Information Gain (Wind)": information_gain_wind
}
