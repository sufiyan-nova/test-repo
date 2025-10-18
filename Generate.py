---

## `generate_data.py`
```python
"""
generate_data.py
Creates a simple synthetic dataset of housing-like data and saves to data/sample_data.csv
"""

import numpy as np
import pandas as pd
import os

def generate_sample_data(n=500, random_seed=42):
    np.random.seed(random_seed)
    # Features
    sqft = np.random.normal(loc=1500, scale=400, size=n).clip(300)
    bedrooms = np.random.choice([1,2,3,4], size=n, p=[0.1,0.3,0.4,0.2])
    age = np.random.exponential(scale=20, size=n).clip(0, 100)
    distance_to_city = np.random.normal(loc=10, scale=6, size=n).clip(0)
    has_garden = np.random.choice([0,1], size=n, p=[0.6, 0.4])

    # Target variable (price) with some noise
    base_price = 50_000
    price = (base_price +
             sqft * 120 +
             bedrooms * 10_000 -
             age * 200 -
             distance_to_city * 1_500 +
             has_garden * 7_000 +
             np.random.normal(0, 25_000, size=n))

    df = pd.DataFrame({
        "sqft": sqft.round(0).astype(int),
        "bedrooms": bedrooms,
        "age": age.round(1),
        "distance_to_city": distance_to_city.round(2),
        "has_garden": has_garden,
        "price": price.round(0).astype(int)
    })
    return df

def main():
    os.makedirs("data", exist_ok=True)
    df = generate_sample_data()
    df.to_csv("data/sample_data.csv", index=False)
    print("Saved sample data to data/sample_data.csv")
    print(df.head())

if __name__ == "__main__":
    main()
