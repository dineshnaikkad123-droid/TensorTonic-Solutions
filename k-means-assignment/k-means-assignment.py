def k_means_assignment(points, centroids):
    """
    Assign each point to the nearest centroid.
    """
    assignments = []

    for point in points:
        best_cluster = 0
        best_distance = float('inf')

        for i, centroid in enumerate(centroids):
            dist = sum((p - c) ** 2 for p, c in zip(point, centroid))

            if dist < best_distance:
                best_distance = dist
                best_cluster = i

        assignments.append(best_cluster)

    return assignments