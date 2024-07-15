from entropy import entropy
from total_entropy import total_entropy

# Temperature attribute
# Hot
hot_yes = 2
hot_no = 2
hot_entropy = entropy(hot_yes, hot_no)

# Mild
mild_yes = 4
mild_no = 2
mild_entropy = entropy(mild_yes, mild_no)

# Cool
cool_yes = 3
cool_no = 1
cool_entropy = entropy(cool_yes, cool_no)

# Weighted entropy for Temp
weighted_entropy_temp = (4 / 14) * hot_entropy + (6 / 14) * mild_entropy + (4 / 14) * cool_entropy

# Information Gain for Temp
information_gain_temp = total_entropy - weighted_entropy_temp

# Store Temp results in the dictionary
temp_results = {
    "Hot Entropy": hot_entropy,
    "Mild Entropy": mild_entropy,
    "Cool Entropy": cool_entropy,
    "Weighted Entropy (Temp)": weighted_entropy_temp,
    "Information Gain (Temp)": information_gain_temp
}
