"""
This script calculates the correlation between the AMAP and AMAP-APP results.
It is not used in the application.
"""

import glob

import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

excel_path = "/home/arash/Desktop/UoC-AMAP/publication/nat_met_data/parameters_per_file_combined.xlsx"
output_path = "/home/arash/Desktop/UoC-AMAP/publication/nat_met_data/parameters_per_file_combined.xlsx"
app_results = "/home/arash/Desktop/UoC-AMAP/publication/nat_met_data/AMAP-APP/"
app_list = glob.glob(app_results + '/**/*', recursive=True)
app_list = [x for x in app_list if (x.endswith(".xls") and ("all_params" in x))]

df = pd.read_excel(excel_path)

for result_path in app_list:
    tmp_df = pd.read_excel(result_path, engine='xlrd')
    for index, row in tmp_df.iterrows():
        for index2, row2 in df.iterrows():
            if row['file'].lower() == row2['File'].lower():
                print("Found the file")
                df.at[index2, 'Area APP'] = row['FP Area']
                df.at[index2, 'Perim. APP'] = row['FP Perim.']
                df.at[index2, 'Circ. APP'] = row['FP Circ.']
df.to_excel(output_path, index=False)

area_manual = df['Area manual'].values
area_dl = df['Area DL'].values
area_app = df['Area APP'].values

perim_manual = df['Perim. manual'].values
perim_dl = df['Perim. DL'].values
perim_app = df['Perim. APP'].values

circ_manual = df['Circ. manual'].values
circ_dl = df['Circ. DL'].values
circ_app = df['Circ. APP'].values

# Correlation between APP and Manual
print("-----------\nCorrelation between AMAP-APP and Manual")
area_correlation, area_p_value = stats.pearsonr(area_manual, area_app)
perim_correlation, perim_p_value = stats.pearsonr(perim_manual, perim_app)
circ_correlation, circ_p_value = stats.pearsonr(circ_manual, circ_app)

# Print results
print("Area Correlation coefficient:", area_correlation)
print("Area p-value:", area_p_value)
print("Perim Correlation coefficient:", perim_correlation)
print("Perim p-value:", perim_p_value)
print("Circ Correlation coefficient:", circ_correlation)
print("Circ p-value:", circ_p_value)

# Correlation between APP and AMAP
print("-----------\nCorrelation between AMAP-APP and AMAP")
area_correlation, area_p_value = stats.pearsonr(area_dl, area_app)
perim_correlation, perim_p_value = stats.pearsonr(perim_dl, perim_app)
circ_correlation, circ_p_value = stats.pearsonr(circ_dl, circ_app)

# Print results
print("Area Correlation coefficient:", area_correlation)
print("Area p-value:", area_p_value)
print("Perim Correlation coefficient:", perim_correlation)
print("Perim p-value:", perim_p_value)
print("Circ Correlation coefficient:", circ_correlation)
print("Circ p-value:", circ_p_value)

# Plotting
matplotlib.use('Agg')
sns.set_theme(style="ticks")

sns.scatterplot(x=area_dl, y=area_app)

plt.title('Podocydes Area')
plt.xlabel('AMAP')
plt.ylabel('AMAP-APP')

ax = plt.gca()  # Get a matplotlib's axes instance
plt.text(.05, .8, "Pearson's r ={:.2f}".format(area_correlation),
         transform=ax.transAxes)
plt.savefig('amap-vs-app-area.png', bbox_inches='tight', dpi=300)
plt.close()

sns.scatterplot(x=perim_dl, y=perim_app)

plt.title('Podocydes Perimeter')
plt.xlabel('AMAP')
plt.ylabel('AMAP-APP')

ax = plt.gca()  # Get a matplotlib's axes instance
plt.text(.05, .8, "Pearson's r ={:.2f}".format(perim_correlation),
         transform=ax.transAxes)
plt.savefig('amap-vs-app-perim.png', bbox_inches='tight', dpi=300)
plt.close()

sns.scatterplot(x=circ_dl, y=circ_app)

plt.title('Podocydes Circularity')
plt.xlabel('AMAP')
plt.ylabel('AMAP-APP')

ax = plt.gca()  # Get a matplotlib's axes instance
plt.text(.05, .8, "Pearson's r ={:.2f}".format(circ_correlation),
         transform=ax.transAxes)
plt.savefig('amap-vs-app-circ.png', bbox_inches='tight', dpi=300)
plt.close()
