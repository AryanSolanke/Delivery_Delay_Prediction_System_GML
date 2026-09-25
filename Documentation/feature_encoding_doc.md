# `feature_encoding.py` Module Documentation

The script `feature_encoding.py` mathematically disentangles(resolves) the flat dataset of transactions into the distinct topological components (nodes and edges) required by Graph Neural Networks.

**Explanation:** We need nodes and edges to create a graph, but or data is tabular. So, we extract the nodes and edges from the dataset in order to create required graph for Graph Neural Networks.<br>
## **1. Vocabulary Generation (`build_vocab` function)**
The code extracts all unique city and state names across both sellers and customers to create unified integer-mapping dictionaries. It intentionally starts the indexing at `1` so that index `0` remains empty, which allows the neural network to handle unknown or padded data during future inference. These lookup tables are exported as `city_vocab.json` and `state_vocab.json`.

### Explanation
Vocabulary generation is like assigning roll numbers to students in a classroom. Neural networks cannot read text like "São Paulo" or "Rio"; they only calculate using numbers.

The Steps involved to change text to numbers:
1. **Making a master list:** The script gathers every single city name and state name mentioned in the dataset and removes all the duplicates. This creates a clean master list of unique locations.<br>
2. **Assigning the ID numbers:** It sorts this list and gives each location a permanent, unique integer (e.g., City A becomes 1, City B becomes 2).<br>
3. **Leaving zero blank:** It purposely starts counting at 1 instead of 0. It saves `0` as a blank placeholder. If the final model ever encounters a brand new city in the real world that wasn't on this original list, it can just assign it a `0` instead of crashing.<br>
4. **Saving the dictionary:** It saves these matching lists (text-to-number mappings) into files called `city_vocab.json` and `state_vocab.json`. Later, the neural network uses these files like a translation dictionary to convert the geographic names into numbers it can do math on.
<br><br>

## **2. Node Aggregation (`build_node_table` function)**
Graph models require each physical location to exist exactly once. The script extracts every `seller_zip_code_prefix` and `customer_zip_code_prefix`, stacking them into a single master column. It then groups the data by zip code to collapse duplicate entries. To handle inconsistencies—such as a zip code having slightly different coordinate entries across multiple orders—it calculates the mathematical mean for `lat` and `lng` and selects the statistical mode (most frequent occurrence) for the city and state.

### Explanation
Node aggregation is exactly like merging duplicate contacts in the phone so you only have one clean profile per person.

Steps followed to build one official profile for every physical location:
1. **Stacking everything together:** The script takes all the starting zip codes (from sellers) and all the ending zip codes (from customers) and dumps them into one massive, combined list.<br>
2. **Removing the duplicates:** A major shipping hub will appear thousands of times in the dataset. The script groups these identical zip codes together and collapses them into a single row. This ensures every real-world location only exists once in the graph.(This ensures we don't get duplicate nodes in the graph)<br>
3. **Pinpointing the center dot:** The GPS coordinates for a single zip code might shift slightly depending on exactly which street a package was on. The script calculates the exact average (the mean) of all those latitudes and longitudes. This gives the model one stable, dead-center coordinate to measure distances from.<br>
4. **Voting on the correct name:** Human data entry is messy. If 99 rows say a zip code belongs to "Nashik" and 1 row says "Nasik", the script counts the entries and automatically picks the most popular spelling (the mode) to officially name that city and state.<br>

## **3. Data Conflict Diagnosis (`report_role_consistency` function)**
Because the dataset is built from user-entered logs, some zip codes list one city when acting as a seller and a completely different city when acting as a customer. The script identifies and tallies these conflicting records, ensuring you are aware of the underlying noise that the mode-aggregation resolved.<br>

### Explanation
Data Conflict Diagnosis is basically a quality control alarm that checks if the dataset is lying to itself.

Here is how the script performs this sanity check:

1. **Checking both sides of the story:** The script looks at the records for a specific zip code when it is sending a package (acting as a seller) and compares it to when that exact same zip code is receiving a package (acting as a customer).<br>
2. **Spotting the contradictions:** It checks to see if the city and state names match perfectly in both scenarios. For example, if a zip code claims to be in "Delhi" when shipping out, but claims to be in "New Delhi" when receiving a delivery, the script catches the mismatch.<br>
3. **Sounding the alarm:** Instead of crashing or deleting data, the script simply counts up exactly how many of these geographical contradictions exist and prints the final tally on the screen.<br>
4. **Proving the work:** It serves as proof for the evaluator that model didn't blindly trust a messy, real-world spreadsheet. model actively scanned the data for human errors and documented them before feeding the data to the neural network.<br>

## **4. Edge Construction (`build_edge_table`)**
The script isolates the transaction paths by linking the origin zip code to the destination zip code. It retains the structural weights (`haversine_distance_km`) and the target variable (`actual_transit_days`), while generating a new `route_count` feature. This count acts as a traffic density metric, explicitly telling the GNN how many times a specific transit lane has been utilized.<br>

### Explanation
Edge construction is like mapping out the physical delivery routes and measuring the traffic on each one.

Here is how the script builds the roads for the network:
1. **Drawing the map lines:** The script takes the starting point (the seller's zip code) and the ending point (the customer's zip code) for every single package to draw a direct route between them.<br>
2. **Keeping the travel stats:** It holds onto the critical measurements for each trip: the physical distance in kilometers (`haversine_distance_km`) and the exact number of days the delivery took (`actual_transit_days`).<br>
3. **Counting the traffic:** The script groups together all the packages that traveled the exact same path and counts them up, creating a new metric called `route_count`. This acts as a traffic density score, telling the neural network whether a path is a high-volume shipping lane or a rarely used route.<br>

<br><br>
 Documentation Ends Here