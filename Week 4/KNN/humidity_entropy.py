from entropy import entropy
from total_entropy import total_entropy

# Humidity attribute
# High
high_yes = 3
high_no = 4
high_entropy = entropy(high_yes, high_no)

# Normal
normal_yes = 6
normal_no = 1
normal_entropy = entropy(normal_yes, normal_no)

# Weighted entropy for Humidity
weighted_entropy_humidity = (7 / 14) * high_entropy + (7 / 14) * normal_entropy

# Information Gain for Humidity
information_gain_humidity = total_entropy - weighted_entropy_humidity

# Store Humidity results in the dictionary
humidity_results = {
    "High Entropy": high_entropy,
    "Normal Entropy": normal_entropy,
    "Weighted Entropy (Humidity)": weighted_entropy_humidity,
    "Information Gain (Humidity)": information_gain_humidity
}
