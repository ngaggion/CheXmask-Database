import os
import random
import pandas as pd
import numpy as np
import shutil
import matplotlib.pyplot as plt

# Function to create histogram and sample files
def process_dataframe(df, name):
    # Create histogram with 10 bins
    hist, bin_edges = np.histogram(df['Dice_RCA_Max'], bins=10)

    # Sample 1 file per bin
    samples_per_bin = []
    for i in range(len(bin_edges) - 1):
        filtered_df = df[(df['Dice_RCA_Max'] >= bin_edges[i]) & (df['Dice_RCA_Max'] < bin_edges[i+1])]
        sample = filtered_df.sample()
        samples_per_bin.append(sample)
        
    # Sample 20 random files
    random_samples = df.sample(20)

    # Create a new folder
    os.makedirs(name, exist_ok=True)

    # Save the 30 files and create a dictionary with the original and new names
    original_to_new = {}
        
    combined_samples = pd.concat(samples_per_bin + [random_samples])
    for i, (_, sample) in enumerate(combined_samples.iterrows()):
        image_path = sample['Image']
        new_name = f'image_{i + 1}.png'
        
        new_path = os.path.join(name, new_name)

        # Load the segmentation map (replace with the path to your segmentation map)
        if "Padchest" in image_path:
            landmark = image_path.replace('Images', 'Output').replace('.png', '.txt')               
        elif "MIMIC" in image_path or "CANDID" in image_path:
            landmark = image_path.replace('Images', 'Output').replace('.jpg', '.txt')
        elif "VinBigData" in image_path:
            landmark = image_path.replace('pngs', 'Output').replace('.png', '.txt')
        elif "CheXpert" in image_path:
            landmark = image_path.replace('Preprocessed', 'Output').replace('.png', '.txt')
        elif "Chest8" in image_path:
            landmark = image_path.replace("ChestX-ray8", "Output").replace('images/','')[0:-3] + 'txt'
        
        # Copy the file to the new location (assumes it's an image file)
        shutil.copy(image_path, new_path)
        shutil.copy(landmark, new_path.replace('.png', '.txt'))
            
        original_to_new[image_path] = new_name

    # Save the histogram of sampled values
    plt.figure()
    plt.hist(combined_samples['Dice_RCA_Max'], bins=bin_edges)
    plt.xlabel('Dice_RCA_Max')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of sampled values for {name}')
    plt.savefig(os.path.join(name, 'histogram.png'))

    return original_to_new

# Read CSV files
df1 = pd.read_csv('Datasets/CANDID/CANDID_RCA.csv')
name1 = 'CANDID'
df1 = df1[df1['Dice_RCA_Max'] > 0.7]

df2 = pd.read_csv('Datasets/Chest8/CHEST8_RCA.csv')
name2 = 'ChestX-Ray8'
df2 = df2[df2['Dice_RCA_Max'] > 0.7]

df3 = pd.read_csv('Datasets/CheXpert/CHEXPERT_RCA.csv')
name3 = 'CheXpert'
df3 = df3[df3['Dice_RCA_Max'] > 0.7]

df4 = pd.read_csv('Datasets/MIMIC/MIMIC_RCA.csv')
name4 = "MIMIC"
df4 = df4[df4['Dice_RCA_Max'] > 0.7]

df5 = pd.read_csv('Datasets/Padchest/Padchest_RCA.csv')
name5 = 'Padchest'
df5 = df5[df5['Dice_RCA_Max'] > 0.7]

df6 = pd.read_csv('Datasets/VinBigData/VinBigData_RCA.csv')
name6 = 'VinBigData'
df6 = df6[df6['Dice_RCA_Max'] > 0.7]

# Process dataframes and get dictionaries with original and new names
orig_to_new1 = process_dataframe(df1, name1)
orig_to_new2 = process_dataframe(df2, name2)
orig_to_new3 = process_dataframe(df3, name3)
orig_to_new4 = process_dataframe(df4, name4)
orig_to_new5 = process_dataframe(df5, name5)
orig_to_new6 = process_dataframe(df6, name6)

# Save the dictionaries as CSV files
pd.DataFrame(orig_to_new1.items(), columns=['Original', 'New']).to_csv(f'{name1}_name_mapping.csv', index=False)
pd.DataFrame(orig_to_new2.items(), columns=['Original', 'New']).to_csv(f'{name2}_name_mapping.csv', index=False)
pd.DataFrame(orig_to_new3.items(), columns=['Original', 'New']).to_csv(f'{name3}_name_mapping.csv', index=False)
pd.DataFrame(orig_to_new4.items(), columns=['Original', 'New']).to_csv(f'{name4}_name_mapping.csv', index=False)
pd.DataFrame(orig_to_new5.items(), columns=['Original', 'New']).to_csv(f'{name5}_name_mapping.csv', index=False)
pd.DataFrame(orig_to_new6.items(), columns=['Original', 'New']).to_csv(f'{name6}_name_mapping.csv', index=False)