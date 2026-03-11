import pandas as pd
from sklearn.datasets import make_moons
import os

os.makedirs('partB/data', exist_ok=True)
X, y = make_moons(n_samples=2000, noise=0.15, random_state=42)

df = pd.DataFrame(X, columns=['feature_1', 'feature_2'])
df['label'] = y

df.to_csv('partB/data/dataset.csv', index=False)
print("Dataset generated at partB/data/dataset.csv")
