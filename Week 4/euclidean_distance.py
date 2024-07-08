import math


def euclidian(point1, point2):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(point1, point2)))


def find_distance(inner_points, inner_centroids):
    inner_distances = []
    for point in inner_points:
        inner_distances.append([euclidian(point, centroid) for centroid in inner_centroids])

    return inner_distances


points = [(2, 10), (2, 5), (8, 4), (5, 8), (7, 5), (6, 4), (1, 2), (4, 9)]
centroids = [(3.67, 9), (7, 4.33), (1.5, 3.5)]

distances = find_distance(points, centroids)

for index, distance in enumerate(distances):
    print(index + 1, distance)
