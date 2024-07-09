import matplotlib.pyplot as plt

# Data points
data_points = [(2, 10), (2, 5), (8, 4), (5, 8), (7, 5), (6, 4), (1, 2), (4, 9)]
clusters = ['A1', 'C1', 'B1', 'B1', 'B1', 'B1', 'C1', 'A1']

# Plotting the data points
for i, point in enumerate(data_points):
    if clusters[i] == 'A1':
        plt.scatter(point[0], point[1], c='red', label='A1' if i == 0 else "")
    elif clusters[i] == 'B1':
        plt.scatter(point[0], point[1], c='blue', label='B1' if i == 2 else "")
    else:
        plt.scatter(point[0], point[1], c='green', label='C1' if i == 1 else "")

# Plotting centroids
plt.scatter(2, 10, c='red', marker='x', s=100)
plt.scatter(6, 6, c='blue', marker='x', s=100)
plt.scatter(1.5, 3.5, c='green', marker='x', s=100)

plt.legend()
plt.title('K-Means Clustering')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()
