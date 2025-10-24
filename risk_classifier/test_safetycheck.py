import pandas as pd
from safetycheck import SafetyCheck

# Initialize model
sc = SafetyCheck()

# Load CSV file
df = pd.read_csv("test_safetycheck_examples.csv")

# Create a new column for predictions
predictions = []

for text in df["text"]:
    try:
        pred = sc(text)
        predictions.append(pred)
    except Exception as e:
        print(f"Error processing: {text} → {e}")
        predictions.append(None)

df["prediction"] = predictions

# Save results
output_path = "safetycheck_results.csv"
df.to_csv(output_path, index=False)

print(f"\n Predictions complete! Saved to '{output_path}'")
print(df.head(10))
