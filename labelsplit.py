import pandas as pd

# Path to your dataset labels CSV file
csv_path = r"D:\BoneFracture\Dataset\BoneFractureYolo8\splitted_datasetv2\dataset_labels.csv"

# Load the combined CSV file
df = pd.read_csv(csv_path)

# Create separate dataframes for each subset
train_df = df[df['subset'] == 'train']
val_df = df[df['subset'] == 'val']
test_df = df[df['subset'] == 'test']

# Save them to separate CSV files
train_csv_path = r"D:\BoneFracture\Dataset\BoneFractureYolo8\splitted_datasetv2\train_labels.csv"
val_csv_path = r"D:\BoneFracture\Dataset\BoneFractureYolo8\splitted_datasetv2\val_labels.csv"
test_csv_path = r"D:\BoneFracture\Dataset\BoneFractureYolo8\splitted_datasetv2\test_labels.csv"

train_df.to_csv(train_csv_path, index=False)
val_df.to_csv(val_csv_path, index=False)
test_df.to_csv(test_csv_path, index=False)

print("Separate CSV files created:")
print(f"Train: {train_csv_path}")
print(f"Validation: {val_csv_path}")
print(f"Test: {test_csv_path}")
