import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

# Data points and their labels
data_points = np.array([
    [2, 10],
    [2, 5],
    [8, 4],
    [5, 8],
    [7, 5],
    [6, 4],
    [1, 2],
    [4, 9]
])

labels = np.array(['A', 'C', 'B', 'B', 'B', 'B', 'C', 'A'])


# Function to calculate Euclidean distance
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))


# KNN function
def knn_predict(new_point, data_points, labels, k=3):
    distances = []
    for index in range(len(data_points)):
        distance = euclidean_distance(new_point, data_points[index])
        distances.append((distance, labels[index]))
    distances.sort(key=lambda x: x[0])
    nearest_neighbors = distances[:k]
    votes = [label for _, label in nearest_neighbors]
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result


# New point to classify
new_point = np.array([3, 7])

# Predict the class of the new point
predicted_class = knn_predict(new_point, data_points, labels, k=3)
print(f'The predicted class for point {new_point} is {predicted_class}')

# Plotting the data points and the new point
for i, point in enumerate(data_points):
    if labels[i] == 'A':
        plt.scatter(point[0], point[1], c='red', label='A' if i == 0 else "")
    elif labels[i] == 'B':
        plt.scatter(point[0], point[1], c='blue', label='B' if i == 2 else "")
    else:
        plt.scatter(point[0], point[1], c='green', label='C' if i == 1 else "")

# Plotting the new point
plt.scatter(new_point[0], new_point[1], c='black', marker='x', s=100, label='New Point')
plt.legend()
plt.title('K-Nearest Neighbors (KNN)')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()
