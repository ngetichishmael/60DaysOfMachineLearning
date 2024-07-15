from entropy import entropy

# Subset for Outlook = Rain
rain_yes = 3
rain_no = 2
rain_entropy = entropy(rain_yes, rain_no)

# Rain subset counts for Temp, Humidity, and Wind
# Temp
rain_mild_yes = 2
rain_mild_no = 1
rain_mild_entropy = entropy(rain_mild_yes, rain_mild_no)

rain_cool_yes = 1
rain_cool_no = 1
rain_cool_entropy = entropy(rain_cool_yes, rain_cool_no)

# Weighted entropy for Temp within Rain subset
weighted_entropy_rain_temp = (3 / 5) * rain_mild_entropy + (2 / 5) * rain_cool_entropy

# Information Gain for Temp within Rain subset
information_gain_rain_temp = rain_entropy - weighted_entropy_rain_temp

# Humidity
rain_high_yes = 1
rain_high_no = 1
rain_high_entropy = entropy(rain_high_yes, rain_high_no)

rain_normal_yes = 2
rain_normal_no = 1
rain_normal_entropy = entropy(rain_normal_yes, rain_normal_no)

# Weighted entropy for Humidity within Rain subset
weighted_entropy_rain_humidity = (2 / 5) * rain_high_entropy + (3 / 5) * rain_normal_entropy

# Information Gain for Humidity within Rain subset
information_gain_rain_humidity = rain_entropy - weighted_entropy_rain_humidity

# Wind
rain_weak_yes = 2
rain_weak_no = 0
rain_weak_entropy = entropy(rain_weak_yes, rain_weak_no)

rain_strong_yes = 1
rain_strong_no = 2
rain_strong_entropy = entropy(rain_strong_yes, rain_strong_no)

# Weighted entropy for Wind within Rain subset
weighted_entropy_rain_wind = (3 / 5) * rain_weak_entropy + (2 / 5) * rain_strong_entropy

# Information Gain for Wind within Rain subset
information_gain_rain_wind = rain_entropy - weighted_entropy_rain_wind

# Store Rain subset results in the dictionary
rain_subset_results = {
    "Rain Subset Entropy": rain_entropy,
    "Rain Mild Entropy": rain_mild_entropy,
    "Rain Cool Entropy": rain_cool_entropy,
    "Weighted Entropy (Rain Temp)": weighted_entropy_rain_temp,
    "Information Gain (Rain Temp)": information_gain_rain_temp,
    "Rain High Entropy": rain_high_entropy,
    "Rain Normal Entropy": rain_normal_entropy,
    "Weighted Entropy (Rain Humidity)": weighted_entropy_rain_humidity,
    "Information Gain (Rain Humidity)": information_gain_rain_humidity,
    "Rain Weak Entropy": rain_weak_entropy,
    "Rain Strong Entropy": rain_strong_entropy,
    "Weighted Entropy (Rain Wind)": weighted_entropy_rain_wind,
    "Information Gain (Rain Wind)": information_gain_rain_wind
}
print(rain_subset_results)