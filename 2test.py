import random

# Define the number of dimensions and the range for the coordinates
k = 2
min_coord, max_coord = -15, 15

# Generate 10 random points with k dimensions
num_points = 5
points = []
for i in range(num_points):
    point = []
    for j in range(k):
        coord = random.uniform(min_coord, max_coord)
        point.append(coord)
    points.append(point)

# Print the generated points
print(points)