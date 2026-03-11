# OIBSIP_DataAnalytics-_task2-L1
Customer segmentation analysis on e-commerce data to group customers based on purchasing behavior. The project includes data cleaning, descriptive statistics, K-Means clustering, and visualizations to identify distinct customer segments that help businesses create targeted marketing strategies.
# Customer Segmentation Analysis

## Objective
To analyze customer purchasing behavior and segment customers into distinct groups using clustering techniques. This helps businesses understand customer patterns and design targeted marketing strategies.

## Dataset
Dataset Link: uploaded as ifood_df

The dataset contains customer information such as purchase history, spending behavior, and other relevant attributes used for segmentation.

## Tools & Technologies
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- VS Code

## Steps Performed
1. Collected the dataset containing customer and purchase information.
2. Explored the dataset to understand its structure and variables.
3. Cleaned the data by handling missing values and inconsistencies.
4. Calculated descriptive statistics such as average purchase value and purchase frequency.
5. Applied the K-Means clustering algorithm to segment customers.
6. Visualized the clusters using scatter plots and charts.
7. Analyzed the characteristics of each customer segment.

## Code (Example)

```python
import pandas as pd
from sklearn.cluster import KMeans

data = pd.read_csv("customer_data.csv")

kmeans = KMeans(n_clusters=4)
data['Cluster'] = kmeans.fit_predict(data[['Annual_Spend','Purchase_Frequency']])


  #Key Insights

  Customers can be grouped based on spending and purchasing behavior.

  Some segments show high purchase frequency and high spending.

  Other segments represent occasional or low-value customers.

#Outcome

 Customer segmentation helps businesses understand their customer base and create  targeted marketing strategies to improve engagement and sales.

# Author

Ayesha Asna
