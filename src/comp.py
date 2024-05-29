"""
This script calculates the difference between no of detected FPs in Manual and AMAP-APP results.
It is not used in the application.
"""
# Python Imports
import glob

# Libary Imports
import os
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np

# Local Imports

manual_path = "/home/arash/Desktop/UoC-AMAP/publication/nat_met_data/manual/results/"
amap_path = "/home/arash/Desktop/UoC-AMAP/publication/nat_met_data/AMAP/results_dl/"
app_path = "/home/arash/Desktop/UoC-AMAP/publication/nat_met_data/AMAP-APP/"

manual_list = glob.glob(manual_path + '/**/*', recursive=True)
amap_list = glob.glob(amap_path + '/**/*', recursive=True)
app_list = glob.glob(app_path + '/**/*', recursive=True)

manual_list = [x for x in manual_list if x.endswith(".xls")]
amap_list = [x for x in amap_list if x.endswith(".xls")]
app_list = [x for x in app_list if (x.endswith(".csv") and ("all_params" not in x) and ("SD_length_grid_index" not in x))]


manual_dict = {os.path.splitext(os.path.basename(x))[0].rstrip("_overview_Results_FP_analysis").lower(): x for x in manual_list}
amap_dict = {os.path.splitext(os.path.basename(x))[0].rstrip("_Results_FP_analysis").lower(): x for x in amap_list}
app_dict = {os.path.splitext(os.path.basename(x))[0].rstrip("_fp_params").lower(): x for x in app_list}

print("Missing in APP")
for item in manual_dict:
    if item not in app_dict.keys():
        print(item)

print("Missing in Manual")
for item in app_dict:
    if item not in manual_dict.keys():
        print(item)

print("----------------")
app_no_list = list()
ama_no_list = list()
for key in app_dict.keys():
    print(key)
    manual_fp = pd.read_csv(manual_dict[key])
    auto_fp = pd.read_csv(app_dict[key])
    amap_fp = pd.read_csv(amap_dict[key])
    print("Manual FP: ", manual_fp.shape[0])
    print("APP FP: ", auto_fp.shape[0])
    print("AMAP FP: ", amap_fp.shape[0])
    app_no_list.append(auto_fp.shape[0])
    ama_no_list.append(amap_fp.shape[0])

no_correlation, no_p_value = stats.pearsonr(app_no_list,
                                            ama_no_list)
print("Correlation coefficient:", no_correlation)
print("p-value:", no_p_value)

app_data = np.array(app_no_list)
ama_data = np.array(ama_no_list)

diff = app_data - ama_data
ratio = diff / ama_data
mean_diff = np.mean(ratio)
std_diff = np.std(ratio)

print("Mean difference:", mean_diff)
print("STD difference:", std_diff)

x = np.arange(len(app_data))

# Width of the bars
width = 0.35

# Create bar chart
fig, ax = plt.subplots()
bars1 = ax.bar(x - width/2, app_data, width, label='AMAP-APP', color='blue')
bars2 = ax.bar(x + width/2, ama_data, width, label='AMAP', color='red')

# Add labels and legend
ax.set_xlabel('Data Point Index')
ax.set_ylabel('No of FPs')
ax.set_title('AMAP-APP vs AMAP')
ax.set_xticks([])
ax.legend()

# Show plot
plt.savefig('amap-vs-app-FPs-BarChart.png', bbox_inches='tight', dpi=300)

# Calculate absolute difference between corresponding elements
abs_diff = np.abs(np.array(ama_data) - np.array(app_data))

ratio_diff = abs_diff / ama_data

# Calculate percentiles
percentiles = np.percentile(ratio_diff, [0, 25, 50, 75, 100])


plt.plot(range(len(ratio_diff)),
         ratio_diff, 'o', color='blue', alpha=0.3, label='_nolegend_')

# Mark percentiles with vertical lines
for percentile in percentiles:
    plt.axhline(y=percentile, color='red', linestyle='--')

# Add labels and title
plt.xlabel('Data Point Index')
plt.ylabel('Ratio of Absolute Difference to AMAP')
plt.title('Percentiles of Ratio of Absolute Difference')

plt.savefig('amap-vs-app-FPs-Percentile.png', bbox_inches='tight', dpi=300)

# Create a box plot of the ratio
plt.figure(figsize=(4, 6))
plt.boxplot(ratio_diff)

# Remove x-axis values
plt.xticks([])

# Add labels and title
plt.xlabel('')
plt.ylabel('Ratio of Absolute Difference')
plt.title('Ratio of Absolute Difference to AMAP')

# Show plot
plt.savefig('amap-vs-app-FPs-BoxPlot.png', bbox_inches='tight', dpi=300)
