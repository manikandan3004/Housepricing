import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

# Load the cleaned dataset
cleaned_file_path = 'Housing.csv'
df = pd.read_csv(cleaned_file_path)

# Ensure 'area' and 'price' columns are one-dimensional
assert df['area'].ndim == 1, "Column 'area' is not one-dimensional"
assert df['price'].ndim == 1, "Column 'price' is not one-dimensional"

# Interaction terms: Create interaction terms between features
df['area_bedrooms'] = df['area'] * df['bedrooms']
df['area_bathrooms'] = df['area'] * df['bathrooms']
df['bedrooms_bathrooms'] = df['bedrooms'] * df['bathrooms']

# Polynomial features: Create polynomial features (degree 2)
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(df[['area', 'bedrooms', 'bathrooms', 'stories']])
poly_features_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(['area', 'bedrooms', 'bathrooms', 'stories']))
df = pd.concat([df, poly_features_df], axis=1)

# Log transformation: Apply log transformation to skewed features
df['log_price'] = np.log(df['price'])

# Binning: Create bins for 'area' and 'price' before normalization
df['area_bin'] = pd.cut(df['area'], bins=5, labels=False)
df['price_bin'] = pd.cut(df['price'], bins=5, labels=False)

# Save binning columns before normalization for clarity
area_bins = df['area_bin']
price_bins = df['price_bin']

# Normalization: Normalize the continuous features
df[['area', 'price', 'bedrooms', 'bathrooms', 'stories', 'parking']] = df[['area', 'price', 'bedrooms', 'bathrooms', 'stories', 'parking']].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

# Re-add the bins to the dataframe after normalization
df['area_bin'] = area_bins
df['price_bin'] = price_bins

# Save the feature-engineered dataset to a new CSV file
feature_engineered_file_path = 'path_to_your_file/Feature_Engineered_Housing.csv'
df.to_csv(feature_engineered_file_path, index=False)

print(f"Feature-engineered dataset saved to {feature_engineered_file_path}")
