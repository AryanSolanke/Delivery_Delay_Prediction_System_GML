# Preprocessing and Design Decisions Documentation

## New Features and the reason

## 1. Calculated Target Variable (`actual_transit_days`): 
This is our target variable/Feature which GML model will try to predict. It is the actual and precise time needed for the order to reach from the store to a customer.

- **How it was calculated:** It was calculated using the already known features, subtracting "order_delivered_customer_date
" and "order_purchase_timestamp".

- **Design Decision:** The columns delivered and the purchase_timestamp have both date and time(eg., 2018-07-24 20.41.37PM).
So, instead of just taking dates into consideration we calculated the total seconds and converted them into days by dividing "86400" which are the number of seconds in a day. This gives us a very precise "actual_transit_time" and thus hopefully increases the model accuracy and precision.

## 2. Calculated Haversine Distance(`haversine_distance_km`): 

- **What is Haversine Distance?** 
The Haversine distance is the shortest distance between two points on the surface of a sphere, measured along the surface. Unlike standard Euclidean distance, which draws a straight line through the Earth, the Haversine formula calculates the "great-circle" distance, accounting for the Earth's curvature using latitudes and longitudes.

- **How is Haversine Distance Calculated**
The Haversine distance is the shortest distance between two points on the surface of a sphere, measured along the surface. Unlike standard Euclidean distance, which draws a straight line through the Earth, the Haversine formula calculates the "great-circle" distance, accounting for the Earth's curvature using latitudes and longitudes.

### The Mathematical Formula

To calculate the Haversine distance $d$ between two coordinates, you need their latitudes ($\phi_1, \phi_2$) and longitudes ($\lambda_1, \lambda_2$) converted from degrees to radians.

Using the Earth's mean radius $R$ (approximately 6,371 kilometers), the formula is calculated in three steps:

1. **Calculate the square of half the chord length ($a$):**

$$a = \sin^2\left(\frac{\phi_2 - \phi_1}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\lambda_2 - \lambda_1}{2}\right)$$


2. **Calculate the angular distance in radians ($c$):**

$$c = 2 \cdot \text{atan2}\left(\sqrt{a}, \sqrt{1-a}\right)$$


3. **Calculate the final physical distance ($d$):**

$$d = R \cdot c$$


- **Design Decision:** Instead of taking normal Euclidean, Manhattan, etc. distances we took haversine distance which takes the curvature of the earth into consideration and thus will hopefully increase the knowledge GML models learn from this feature.

# Statistical Analysis Results

The descriptive statistics below are computed via the `describe()` function on the `dataset.csv` (97,331 rows) for every numerical feature.

| Feature | Count | Mean | Std | Min | 25% | 50% | 75% | Max |
|---|---|---|---|---|---|---|---|---|
| seller_zip_code_prefix | 97331 | 24670.23 | 27736.29 | 1001.00 | 6503.50 | 13613.00 | 29156.00 | 99730.00 |
| seller_lat | 97331 | -22.7965 | 2.7489 | -32.0792 | -23.6127 | -23.4249 | -21.7573 | -2.5012 |
| seller_lng | 97331 | -47.2308 | 2.3496 | -63.8936 | -48.8073 | -46.7558 | -46.5187 | -34.8556 |
| customer_zip_code_prefix | 97331 | 35112.59 | 29837.96 | 1003.00 | 11310.00 | 24360.00 | 58680.00 | 99980.00 |
| customer_lat | 97331 | -21.2091 | 5.5937 | -33.6899 | -23.5902 | -22.9263 | -20.1404 | 42.1840 |
| customer_lng | 97331 | -46.1927 | 4.0494 | -72.6689 | -48.1192 | -46.6328 | -43.6324 | -8.7238 |
| actual_transit_days | 97331 | 12.5026 | 9.5101 | 0.5334 | 6.7266 | 10.1797 | 15.5932 | 209.6286 |
| haversine_distance_km | 97331 | 600.2440 | 592.6135 | 0.0000 | 187.6377 | 433.8786 | 798.0305 | 8677.9116 |

