from entropy import entropy

# Subset for Outlook = Sunny
sunny_yes = 2
sunny_no = 3
sunny_entropy = entropy(sunny_yes, sunny_no)

# Sunny subset counts for Temp, Humidity, and Wind
# Temp
sunny_hot_yes = 0
sunny_hot_no = 2
sunny_hot_entropy = entropy(sunny_hot_yes, sunny_hot_no)

sunny_mild_yes = 1
sunny_mild_no = 1
sunny_mild_entropy = entropy(sunny_mild_yes, sunny_mild_no)

sunny_cool_yes = 1
sunny_cool_no = 0
sunny_cool_entropy = entropy(sunny_cool_yes, sunny_cool_no)

# Weighted entropy for Temp within Sunny subset
weighted_entropy_sunny_temp = (2 / 5) * sunny_hot_entropy + (2 / 5) * sunny_mild_entropy + (1 / 5) * sunny_cool_entropy

# Information Gain for Temp within Sunny subset
information_gain_sunny_temp = sunny_entropy - weighted_entropy_sunny_temp

# Humidity
sunny_high_yes = 0
sunny_high_no = 3
sunny_high_entropy = entropy(sunny_high_yes, sunny_high_no)

sunny_normal_yes = 2
sunny_normal_no = 0
sunny_normal_entropy = entropy(sunny_normal_yes, sunny_normal_no)

# Weighted entropy for Humidity within Sunny subset
weighted_entropy_sunny_humidity = (3 / 5) * sunny_high_entropy + (2 / 5) * sunny_normal_entropy

# Information Gain for Humidity within Sunny subset
information_gain_sunny_humidity = sunny_entropy - weighted_entropy_sunny_humidity

# Wind
sunny_weak_yes = 1
sunny_weak_no = 2
sunny_weak_entropy = entropy(sunny_weak_yes, sunny_weak_no)

sunny_strong_yes = 1
sunny_strong_no = 1
sunny_strong_entropy = entropy(sunny_strong_yes, sunny_strong_no)

# Weighted entropy for Wind within Sunny subset
weighted_entropy_sunny_wind = (3 / 5) * sunny_weak_entropy + (2 / 5) * sunny_strong_entropy

# Information Gain for Wind within Sunny subset
information_gain_sunny_wind = sunny_entropy - weighted_entropy_sunny_wind

# Store Sunny subset results in the dictionary
sunny_subset_results = {
    "Sunny Subset Entropy": sunny_entropy,
    "Sunny Hot Entropy": sunny_hot_entropy,
    "Sunny Mild Entropy": sunny_mild_entropy,
    "Sunny Cool Entropy": sunny_cool_entropy,
    "Weighted Entropy (Sunny Temp)": weighted_entropy_sunny_temp,
    "Information Gain (Sunny Temp)": information_gain_sunny_temp,
    "Sunny High Entropy": sunny_high_entropy,
    "Sunny Normal Entropy": sunny_normal_entropy,
    "Weighted Entropy (Sunny Humidity)": weighted_entropy_sunny_humidity,
    "Information Gain (Sunny Humidity)": information_gain_sunny_humidity,
    "Sunny Weak Entropy": sunny_weak_entropy,
    "Sunny Strong Entropy": sunny_strong_entropy,
    "Weighted Entropy (Sunny Wind)": weighted_entropy_sunny_wind,
    "Information Gain (Sunny Wind)": information_gain_sunny_wind
}

print(sunny_subset_results)
