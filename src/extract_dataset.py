import zipfile

with zipfile.ZipFile("plantdisease.zip", 'r') as zip_ref:
    zip_ref.extractall("PlantVillage")

print("Dataset Extracted Successfully!")