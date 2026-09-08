import math

# Dataset
data = [
    (4, 3, 'B'),
    (3, 3, 'A'),
    (5, 5, 'A'),
    (2, 4, 'A'),
    (8, 8, 'B'),
    (7, 2, 'B')
]

# New point
Q = (4, 4)

# Calculate distances
distances = []

for x1, x2, cls in data:
    d = math.sqrt((x1 - Q[0])**2 + (x2 - Q[1])**2)
    distances.append((d, cls))

# Sort distances
distances.sort()

print("Distances:")
for d, cls in distances:
    print(round(d, 3), cls)

# KNN prediction
for k in [1, 3, 5]:

    neighbors = distances[:k]

    A = 0
    B = 0

    for d, cls in neighbors:
        if cls == 'A':
            A += 1
        else:
            B += 1

    if A > B:
        prediction = 'A'
    else:
        prediction = 'B'

    print("K =", k, "=> Class =", prediction)