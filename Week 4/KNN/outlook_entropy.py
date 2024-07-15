from entropy import entropy
from total_entropy import total_entropy

# Outlook attribute
# Sunny
sunny_yes = 2
sunny_no = 3
sunny_entropy = entropy(sunny_yes, sunny_no)

# Overcast
overcast_yes = 4
overcast_no = 0
overcast_entropy = entropy(overcast_yes, overcast_no)

# Rain
rain_yes = 3
rain_no = 2
rain_entropy = entropy(rain_yes, rain_no)

# Weighted entropy for Outlook
weighted_entropy_outlook = (5 / 14) * sunny_entropy + (4 / 14) * overcast_entropy + (5 / 14) * rain_entropy

# Information Gain for Outlook
information_gain_outlook = total_entropy - weighted_entropy_outlook

# Store results in a dictionary for easier display
outlook_results = {
    "Total Entropy": total_entropy,
    "Sunny Entropy": sunny_entropy,
    "Overcast Entropy": overcast_entropy,
    "Rain Entropy": rain_entropy,
    "Weighted Entropy (Outlook)": weighted_entropy_outlook,
    "Information Gain (Outlook)": information_gain_outlook
}
