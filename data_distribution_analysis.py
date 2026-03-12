import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('Traffic_Collisions_Open_Data.csv')

print("=== TRAFFIC COLLISIONS DATA DISTRIBUTION ANALYSIS ===")
print(f"Dataset Shape: {df.shape}")
print(f"Total Records: {df.shape[0]:,}")
print(f"Total Columns: {df.shape[1]}")
print()

# Basic info about the dataset
print("=== COLUMN INFORMATION ===")
print(df.info())
print()

# Check for missing values
print("=== MISSING VALUES ANALYSIS ===")
missing_data = df.isnull().sum()
missing_percentage = (missing_data / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing Count': missing_data,
    'Missing Percentage': missing_percentage
})
print(missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Percentage', ascending=False))
print()

# Year distribution
print("=== YEAR DISTRIBUTION ===")
year_dist = df['OCC_YEAR'].value_counts().sort_index()
print(year_dist)
print(f"Year Range: {df['OCC_YEAR'].min()} - {df['OCC_YEAR'].max()}")
print()

# Month distribution
print("=== MONTH DISTRIBUTION ===")
month_dist = df['OCC_MONTH'].value_counts()
print(month_dist)
print()

# Day of week distribution
print("=== DAY OF WEEK DISTRIBUTION ===")
dow_dist = df['OCC_DOW'].value_counts()
print(dow_dist)
print()

# Hour distribution
print("=== HOUR DISTRIBUTION ===")
hour_dist = df['OCC_HOUR'].value_counts().sort_index()
print(hour_dist)
print()

# Division distribution
print("=== DIVISION DISTRIBUTION (Top 10) ===")
div_dist = df['DIVISION'].value_counts().head(10)
print(div_dist)
print(f"Total Divisions: {df['DIVISION'].nunique()}")
print()

# Collision types distribution
print("=== COLLISION TYPES DISTRIBUTION ===")
injury_dist = df['INJURY_COLLISIONS'].value_counts()
ftr_dist = df['FTR_COLLISIONS'].value_counts()
pd_dist = df['PD_COLLISIONS'].value_counts()

print("Injury Collisions:")
print(injury_dist)
print("\nFTR Collisions:")
print(ftr_dist)
print("\nPD Collisions:")
print(pd_dist)
print()

# Vehicle types distribution
print("=== VEHICLE TYPES INVOLVED ===")
vehicle_cols = ['AUTOMOBILE', 'MOTORCYCLE', 'PASSENGER', 'BICYCLE', 'PEDESTRIAN']
for col in vehicle_cols:
    if col in df.columns:
        print(f"{col}:")
        print(df[col].value_counts())
        print()

# Fatalities distribution
print("=== FATALITIES DISTRIBUTION ===")
fatal_dist = df['FATALITIES'].value_counts().sort_index()
print(fatal_dist)
print(f"Total Fatalities: {df['FATALITIES'].sum()}")
print(f"Collisions with Fatalities: {(df['FATALITIES'] > 0).sum()}")
print()

# Neighborhood distribution (top 15)
print("=== NEIGHBORHOOD DISTRIBUTION (Top 15) ===")
neighborhood_dist = df['NEIGHBOURHOOD_158'].value_counts().head(15)
print(neighborhood_dist)
print()

# Create visualizations
plt.style.use('seaborn-v0_8')
fig, axes = plt.subplots(3, 3, figsize=(20, 15))
fig.suptitle('Traffic Collisions Data Distribution Analysis', fontsize=16, y=0.98)

# Year distribution
year_dist.plot(kind='bar', ax=axes[0,0], color='skyblue')
axes[0,0].set_title('Collisions by Year')
axes[0,0].set_xlabel('Year')
axes[0,0].set_ylabel('Count')
axes[0,0].tick_params(axis='x', rotation=45)

# Month distribution
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
               'July', 'August', 'September', 'October', 'November', 'December']
month_data = df['OCC_MONTH'].value_counts().reindex(month_order)
month_data.plot(kind='bar', ax=axes[0,1], color='lightgreen')
axes[0,1].set_title('Collisions by Month')
axes[0,1].set_xlabel('Month')
axes[0,1].set_ylabel('Count')
axes[0,1].tick_params(axis='x', rotation=45)

# Day of week distribution
dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
dow_data = df['OCC_DOW'].value_counts().reindex(dow_order)
dow_data.plot(kind='bar', ax=axes[0,2], color='lightcoral')
axes[0,2].set_title('Collisions by Day of Week')
axes[0,2].set_xlabel('Day of Week')
axes[0,2].set_ylabel('Count')
axes[0,2].tick_params(axis='x', rotation=45)

# Hour distribution
hour_dist.plot(kind='line', ax=axes[1,0], color='purple', marker='o')
axes[1,0].set_title('Collisions by Hour of Day')
axes[1,0].set_xlabel('Hour')
axes[1,0].set_ylabel('Count')
axes[1,0].grid(True, alpha=0.3)

# Collision types
collision_types = pd.DataFrame({
    'Injury': df['INJURY_COLLISIONS'].value_counts().get('YES', 0),
    'FTR': df['FTR_COLLISIONS'].value_counts().get('YES', 0),
    'PD': df['PD_COLLISIONS'].value_counts().get('YES', 0)
}, index=['Count'])
collision_types.T.plot(kind='bar', ax=axes[1,1], color=['red', 'orange', 'blue'])
axes[1,1].set_title('Collision Types')
axes[1,1].set_xlabel('Collision Type')
axes[1,1].set_ylabel('Count')
axes[1,1].tick_params(axis='x', rotation=45)

# Fatalities
fatal_dist.plot(kind='bar', ax=axes[1,2], color='darkred')
axes[1,2].set_title('Fatalities per Collision')
axes[1,2].set_xlabel('Number of Fatalities')
axes[1,2].set_ylabel('Count')

# Vehicle types
vehicle_counts = []
for col in vehicle_cols:
    if col in df.columns:
        yes_count = (df[col] == 'YES').sum()
        vehicle_counts.append(yes_count)
    else:
        vehicle_counts.append(0)

vehicle_df = pd.DataFrame({'Count': vehicle_counts}, index=vehicle_cols)
vehicle_df.plot(kind='bar', ax=axes[2,0], color='teal')
axes[2,0].set_title('Vehicle Types Involved')
axes[2,0].set_xlabel('Vehicle Type')
axes[2,0].set_ylabel('Count')
axes[2,0].tick_params(axis='x', rotation=45)

# Top divisions
div_dist.head(10).plot(kind='bar', ax=axes[2,1], color='gold')
axes[2,1].set_title('Top 10 Police Divisions')
axes[2,1].set_xlabel('Division')
axes[2,1].set_ylabel('Count')
axes[2,1].tick_params(axis='x', rotation=45)

# Top neighborhoods
neighborhood_dist.plot(kind='bar', ax=axes[2,2], color='brown')
axes[2,2].set_title('Top 15 Neighborhoods')
axes[2,2].set_xlabel('Neighborhood')
axes[2,2].set_ylabel('Count')
axes[2,2].tick_params(axis='x', rotation=90)

plt.tight_layout()
plt.savefig('traffic_collisions_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# Summary statistics
print("=== SUMMARY STATISTICS ===")
print(f"Average collisions per year: {len(df) / df['OCC_YEAR'].nunique():.0f}")
print(f"Peak collision hour: {hour_dist.idxmax()} ({hour_dist.max()} collisions)")
print(f"Peak collision month: {month_dist.idxmax()} ({month_dist.max()} collisions)")
print(f"Peak collision day: {dow_dist.idxmax()} ({dow_dist.max()} collisions)")
print(f"Most active division: {df['DIVISION'].value_counts().index[0]} ({df['DIVISION'].value_counts().iloc[0]} collisions)")
print(f"Most affected neighborhood: {df['NEIGHBOURHOOD_158'].value_counts().index[0]} ({df['NEIGHBOURHOOD_158'].value_counts().iloc[0]} collisions)")
print()

# Geographic distribution check
print("=== GEOGRAPHIC DISTRIBUTION CHECK ===")
print(f"Valid coordinates: {((df['LONG_WGS84'] != 0) & (df['LAT_WGS84'] != 0)).sum()}")
print(f"Invalid/Zero coordinates: {((df['LONG_WGS84'] == 0) | (df['LAT_WGS84'] == 0)).sum()}")
print(f"Longitude range: {df[df['LONG_WGS84'] != 0]['LONG_WGS84'].min():.4f} to {df[df['LONG_WGS84'] != 0]['LONG_WGS84'].max():.4f}")
print(f"Latitude range: {df[df['LAT_WGS84'] != 0]['LAT_WGS84'].min():.4f} to {df[df['LAT_WGS84'] != 0]['LAT_WGS84'].max():.4f}")

print("\nAnalysis complete! Check 'traffic_collisions_distribution.png' for visualizations.")
