import os

dataset_path = "PlantVillage"

classes = os.listdir(dataset_path)

print("Total Classes:", len(classes))
print(classes[:10])