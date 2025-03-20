# %%
# Log my print statements to a log file
import sys
import datetime

class Logger:
    """Logs both to stdout (terminal) and a log file."""
    def __init__(self, filename="log.txt"):
        self.terminal = sys.stdout  # Keep reference to original stdout
        self.log = open(filename, "a", encoding="utf-8")  # Append mode

    def write(self, message):
        self.terminal.write(message)  # Print to terminal
        self.log.write(message)  # Write to log file
        self.log.flush()  # Force flush after each write

    def flush(self):
        """Ensure real-time logging."""
        self.terminal.flush()
        self.log.flush()

# Redirect stdout to Logger (All print statements will now go to log.txt)
sys.stdout = Logger("log.txt")

# Log start time
print(f"\n===== Workflow Started: {datetime.datetime.now()} =====\n")

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

import seaborn as sns

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# %%
# NOTE: directly using model_summary csv files of /data/modelzoo/data_storage by Jason: 
# data_storage_path = "../../../../data/modelzoo/data_storage"

# copied from this path, and use it locally in project directory

# %% [markdown]
# # Generate Model Summaries CSV Files

# %%
# After getting JSON files for model data ready, we can run model_summaries.py to get attribute matrices for each origin.
# The generated model summary csv files will be defaultly saved in the data_storage/results folder.
# takes a while to run the following commands

# currently, didn't re-run these commands and use the existing csv files here
# if you want to re-run these commands, please uncomment the following lines

# !python model_summaries.py give_summaries_timm
# !python model_summaries.py give_summaries_pytorch
# !python model_summaries.py give_summaries_tfhub
# !python model_summaries.py give_summaries_keras

# %% [markdown]
# # Analysis on Previous Model Collection's Model Metadata

# %%
folder_save_plot = "plots"

# Read in the model_summaries csv files
# Use new csv files if the previous JSON files are updated
# right now it uses existing csv files

# combine data_storage_path with file relative paths (/modelData/models_summary_*.csv)
models_summaries_file_paths = [
    "data/models_summaries/models_summary_keras.csv",
    "data/models_summaries/models_summary_pytorch.csv",
    "data/models_summaries/models_summary_tfhub.csv",
    "data/models_summaries/models_summary_timm_new.csv"
]
#     "../../../../data/modelzoo/data_storage/modelData/models_summary_keras.csv",
#     "../../../../data/modelzoo/data_storage/modelData/models_summary_pytorch.csv",
#     "../../../../data/modelzoo/data_storage/modelData/models_summary_tfhub.csv",
#     "../../../../data/modelzoo/data_storage/modelData/models_summary_timm_new.csv"

print(models_summaries_file_paths[0])
print(os.path.exists(models_summaries_file_paths[0]))

# %% [markdown]
# ## Concatenate all the model summaries CSV files into one

# %% [markdown]
# ### Sort columns and Remove models if redundent

# %%
models_summaries_all = []
# Concatenate all the dataframes, ensuring columns match

# Add "Origin" column based on file name
for file in models_summaries_file_paths:
    df = pd.read_csv(file)
    if "Origin" not in df.columns:
        if "keras" in file:
            df["Origin"] = "keras"
        elif "pytorch" in file:
            df["Origin"] = "pytorch"
        elif "tfhub" in file:
            df["Origin"] = "tfhub"
        elif "timm" in file:
            df["Origin"] = "timm"
    models_summaries_all.append(df)


# Find the union of all columns
all_columns = set()
for df in models_summaries_all:
    all_columns.update(df.columns)

# Reindex dataframes to have the same columns
models_summaries_all = [df.reindex(columns=all_columns) for df in models_summaries_all]

# Concatenate the dataframes
models_summaries_all = pd.concat(models_summaries_all, ignore_index=True)

# %%
# # Load all sims
# sims_files = os.listdir("../../../../../../data/modelzoo/data_storage/sims")
# sims_files = sorted(sims_files)

# # Load the first dataset to get info
# sims_matrix_data = pd.read_csv(f"../../../../../../data/modelzoo/data_storage/sims/{sims_files[0]}", index_col=0)

# model_columns_sims = list(sims_matrix_data.columns)
# print(len(model_columns_sims))

# # Turn sims_matrix_data into an array
# sims_matrix_data = sims_matrix_data.values

# # Loop through the rest
# for file in sims_files[1:]:
#     # Load the sims_matrix_data
#     tmp = pd.read_csv(f"../../../../../../data/modelzoo/data_storage/sims/{file}", index_col=0)

#     # Make sure the columns are in the same order
#     tmp = tmp[model_columns_sims]

#     # Make sure the rows are in the same order
#     tmp = tmp.loc[model_columns_sims]

#     # Add to sims_matrix_data
#     sims_matrix_data += tmp.values

# sims_matrix_data = sims_matrix_data / len(sims_files)
# distance_matrix_data = 1 - sims_matrix_data
# distance_matrix_data[distance_matrix_data < 0] = 0

# # clean models
# # Remove models if not in sims
# models_summaries_all_cleanned = models_summaries_all.loc[models_summaries_all["Model"].isin(model_columns_sims)]
# print(len(models_summaries_all_cleanned))
# # Remove duplicates (drop the row with more empty data)
# models_summaries_all_cleanned = models_summaries_all_cleanned.sort_values(by=var, ascending=False).drop_duplicates(subset="Model", keep="first")
# print(len(models_summaries_all_cleanned))
# models_names_cleanned = models_summaries_all_cleanned["Model"].values

# # Find the models not in clusterModelInfo
# missingModels = [model for model in model_columns_sims if model not in models_names_cleanned]

# # Remove these models from the distance matrix
# missingIdx = [model_columns_sims.index(model) for model in missingModels]
# distance_matrix_data = np.delete(distance_matrix_data, missingIdx, axis=0)
# distance_matrix_data = np.delete(distance_matrix_data, missingIdx, axis=1)
# model_columns_sims = [model for model in model_columns_sims if model in models_names_cleanned]

# # Only keep rows in model_columns_sims
# models_summaries_all_cleanned = models_summaries_all_cleanned.loc[models_summaries_all_cleanned["Model"].isin(model_columns_sims)]

# # Reorder the rows to match model_columns_sims
# models_summaries_all_cleanned = models_summaries_all_cleanned.set_index("Model").loc[model_columns_sims]


# %% [markdown]
# ### Check if the amount of models align with existing sims file

# %%
# check if the total amount of models == 774 showed in the sims files

# read in a sims data
sims_file_path = "data/sims/yufos.csv"
sims_data = pd.read_csv(sims_file_path)

# check if the models in sims are in the models_summaries_all
sims_data_models = sims_data.columns[1:]
models_summaries_all_models = models_summaries_all["Model"].values
print("Number of models in sims data:", len(sims_data_models))
print("Number of models in models_summaries_all:", len(models_summaries_all_models))
print()
print("Number of models in sims but not in models_summaries_all:", len(set(sims_data_models) - set(models_summaries_all_models)))
print("Number of models in models_summaries_all but not in sims:", len(set(models_summaries_all_models) - set(sims_data_models)))
print()


# print the models in models_summaries_all but not in sims
print("Models in models_summaries_all but not in sims:")
for model in set(models_summaries_all_models) - set(sims_data_models):
    # print these models in a list
    print(model)
print()

    
# print the models in sims but not in models_summaries_all
# print("Models in sims but not in models_summaries_all:")
# for model in set(sims_data_models) - set(models_summaries_all_models):
#     print(model)
# print()

    

# %%
models_summaries_all = []
# Concatenate all the dataframes, ensuring columns match

# Add "Origin" column based on file name
for file in models_summaries_file_paths:
    df = pd.read_csv(file)
    if "Origin" not in df.columns:
        if "keras" in file:
            df["Origin"] = "keras"
        elif "pytorch" in file:
            df["Origin"] = "pytorch"
        elif "tfhub" in file:
            df["Origin"] = "tfhub"
        elif "timm" in file:
            df["Origin"] = "timm"
    models_summaries_all.append(df)



# Find the union of all columns
all_columns = set()
for df in models_summaries_all:
    all_columns.update(df.columns)

# Reindex dataframes to have the same columns
models_summaries_all = [df.reindex(columns=all_columns) for df in models_summaries_all]

# Concatenate the dataframes
models_summaries_all = pd.concat(models_summaries_all, ignore_index=True)

# Check for duplicates who has the same "Model" string
duplicated_models = models_summaries_all["Model"].duplicated()
count = 0
if duplicated_models.any():
    print("Duplicated models:", models_summaries_all[duplicated_models]["Model"].values)
    for model in models_summaries_all[duplicated_models]["Model"].values:
        # see if the duplicated model has different origin
        if len(set(models_summaries_all[models_summaries_all["Model"] == model]["Origin"])) > 1:
            # print("Duplicated model from different origin:", model)
            # drop the row for the duplicates if the origin is keras (keras has less information than timm)
            models_summaries_all = models_summaries_all.drop(models_summaries_all[(models_summaries_all["Model"] == model) & (models_summaries_all["Origin"] == "keras")].index)
            # print("Dropped the row for the duplicated model from keras and keep timm:", model)
            count += 1
    print("Dropped", count, "duplicated models from keras")


# Only include models if they are both in the model summaries and the sims data
sims_data_models = sims_data.columns
print(len(sims_data_models), "models in sims data")

models_summaries_all = models_summaries_all.loc[models_summaries_all["Model"].isin(sims_data_models)]
print("Dropped models that are not in sims data", len(models_summaries_all), "models left")


print("Number of models:", len(models_summaries_all))
print("Number of unique models:", len(models_summaries_all["Model"].unique()))
print("Yes, Models are all unique" if len(models_summaries_all) == len(models_summaries_all["Model"].unique()) else "Still have duplicates: Models are not all unique")

# %% [markdown]
# ### Add family data from previous framework

# %%
# Load family and left join
family_data_path = "data/models_summaries/family_data.csv"
family_data = pd.read_csv(family_data_path)

# Add binary attributes
attributes_binary = [
    "Convolutional Layers",
    "Residual Blocks",
    "Dense Layers",
    "Bottlenecks",
    "Recurrent Layers",
    "Skip Connections",
    "Attention Layers",
    "ReLU",
    "GeLU",
]

family_data = family_data.rename(columns={col: "Has " + col for col in attributes_binary})


# Merge family data into the models summaries
models_summaries_all = pd.merge(models_summaries_all, family_data, on="Family", how="left")

# %% [markdown]
# ### Scale down parameters and layers (Jason)

# %%
# scale down parameters to be in the millions
models_summaries_all["Parameters"] = models_summaries_all["Parameters"] / 1_000_000

# Scale down layers by 10
models_summaries_all["Layers"] = models_summaries_all["Layers"] / 10

# rename pooling to pooling type
models_summaries_all = models_summaries_all.rename(columns={"Pooling": "Pooling Type"})

# %% [markdown]
# ## Save the concatenated file

# %%
# Save the concatenated dataframe
models_summaries_all.to_csv(f"data/models_summary_all.csv", index=False)

# %% [markdown]
# ## Pre-Analysis on the concatenated file of all models' attributes

# %%
# Read the combined CSV file
models_summaries_all = pd.read_csv("data/models_summary_all.csv")

# %% [markdown]
# ### Get attributes' types

# %%
# print all column names
print(f"All {len(models_summaries_all.columns)} attributes: {models_summaries_all.columns}")
print() # consistent with Jason's manuscript code

# Find the categorical and numerical columns
attribute_categorical_columns = []
attribute_continuous_columns = []
attribute_binary_columns = []

# Find the columns with missing data
# TODO: Find why they are missing and how we deal with them (right now filled them with zeros)
attribute_missing_data_columns = []

# Find the columns with missing data, categorical columns, binary columns, and continuous columns
for column in models_summaries_all.columns:
    # print(f"Checking column: {column}")

    # Check if have missing data
    if models_summaries_all[column].isnull().any():
        attribute_missing_data_columns.append(column)
        # print(f"    {column} has missing data")
        # print(f"    Number of missing data: {models_summaries_all[column].isnull().sum()}")
        # print(f"    Percentage of missing data: {models_summaries_all[column].isnull().sum() / models_summaries_all.shape[0] * 100}%")
        # print()

    # Check if the column is categorical
    if models_summaries_all[column].dtype == "object":
        attribute_categorical_columns.append(column)
        # print(f"    {column} is categorical")
        # print(f"    Number of unique values: {models_summaries_all[column].nunique()}")
        # print()

    # Check if the column is binary (only have 0 and 1)
    # TODO: Jason's manuscript code has categorical columns list categoricalCols
    elif models_summaries_all[column].dtype in ["int64", "float64"] and models_summaries_all[column].nunique() <= 2:
        attribute_binary_columns.append(column)
        # print(f"    {column} is binary")
        # print()

    # Check if the column is numerical and not binary
    elif models_summaries_all[column].dtype in ["int64", "float64"] and models_summaries_all[column].nunique() > 3:
        if column in [
        "Parameters",
        "Layers",
        "Residual Blocks",
        "Conv Layers",
        "Dense Layers",
        "Bottlenecks",
        "Pooling Layers",
        "Normalization Layers",
        "Recurrent Layers",
        "Attention Layers",
        "Output Features",
        "First Layer Parameters",
        "Highest Internal Layer Parameters",
        "Lowest Internal Layer Parameters",
        "Last Layer Parameters",
        ]: # refer to Jason's manuscript code
            attribute_continuous_columns.append(column)
            # print(f"    {column} is numerical")
            # print()


    

print("Finished checking columns")

# %% [markdown]
# ### Overview of the attributes' types and empty data

# %%
# show a summary on the attributes' types
attribute_table = pd.DataFrame({
    "Attribute": models_summaries_all.columns,
    "Type": ["Categorical" if col in attribute_categorical_columns else "Binary" if col in attribute_binary_columns else "Continuous" for col in models_summaries_all.columns],
    "Empty data": [models_summaries_all[col].isnull().sum() for col in models_summaries_all.columns],
    "Unique values": [models_summaries_all[col].nunique() for col in models_summaries_all.columns]
})

# order by the attribute
attribute_table = attribute_table.sort_values(by="Attribute")
# order by the type
attribute_table = attribute_table.sort_values(by="Type")
# order by missing data percentage
attribute_table = attribute_table.sort_values(by="Empty data", ascending=True)

# show the table (without column index)
print(attribute_table.to_string(index=False))

# save the table
attribute_table.to_csv("data/attribute_table.csv", index=False)

# %% [markdown]
# ### fill in empty data

# %%
# fill in empty data with 0 or NA respectively
# NOTE: in Jason's previous analysis, models with specific attribute data missing are dropped
# alternatives: fill in with mean or median?

for column in attribute_continuous_columns:
    if column in attribute_categorical_columns:
        models_summaries_all[column] = models_summaries_all[column].fillna("NA")
    elif column in attribute_continuous_columns:
        # fill in median for continuous columns
        models_summaries_all[column] = models_summaries_all[column].fillna(models_summaries_all[column].median())
    elif column in attribute_binary_columns:
        models_summaries_all[column] = models_summaries_all[column].fillna("NA")


# %% [markdown]
# ### Clean the continuous data before plotting

# %%
attribute_continuous_cleanned =[]

for column in attribute_continuous_columns:
    if column == "id":
        print(f"Skipping {column}")
        continue

    # if the every value is unique, skip this column
    elif models_summaries_all[column].nunique() == len(models_summaries_all):
        print(f"Skipping {column} because it has unique values for each row.")
        continue

    # if the column is all the same value (such as 0), skip this column
    elif models_summaries_all[column].nunique() == 1:
        print(f"Skipping {column} because it has only one unique value.")
        continue

    else:
        # include column only if they have at least 500 non-empty data (refer to Jason's manuscript code)
        if models_summaries_all[column].isnull().sum() < len(models_summaries_all) - 500:
            attribute_continuous_cleanned.append(column)

# sort the columns alphabetically and by Layers or contains layers, parameters and contains parameters
attribute_continuous_cleanned = sorted(attribute_continuous_cleanned, key=lambda x: ("Layers" in x, "Parameters" in x, x))

# adjust Layers to be the first of the columns with layers, do so on Parameters
attribute_continuous_cleanned = ["Layers"] + [col for col in attribute_continuous_cleanned if "Layers" in col and col != "Layers"] + [col for col in attribute_continuous_cleanned if "Layers" not in col]
attribute_continuous_cleanned = ["Parameters"] + [col for col in attribute_continuous_cleanned if "Parameters" in col and col != "Parameters"] + [col for col in attribute_continuous_cleanned if "Parameters" not in col]

        
print()
print(f"Analyzing {len(attribute_continuous_cleanned)} continuous columns: {attribute_continuous_cleanned}")
# consistent with Jason's manuscript code

# 

# %% [markdown]
# ### Histograms of the attributes: see their distributions
# Save plots in a folder rather than displaying them

# %%
# for each column of the attribute matrix, plot the distribution of models by that column

folder_save = "plots"

for column in models_summaries_all.columns:
    # if half of the rows are missing, skip this column
    if models_summaries_all[column].isnull().sum() > len(models_summaries_all) / 2:
        print(f"Skipping {column} because more than half of the rows are missing.")
        continue
    
    if column == "id":
        print(f"Skipping {column} because it's an index.")
        continue

    print(f"...Plotting {column}...")

    # categorical columns
    if column in attribute_categorical_columns:
        num_unique = int(models_summaries_all[column].nunique())
        plt.figure(figsize=(max(10, int(num_unique * 0.5)), 12))
        
        column_counts = models_summaries_all[column].value_counts()
        print(f"Number of unique models: {num_unique}")
        print(f"Total rows: {len(models_summaries_all)}")
        
        if num_unique == len(models_summaries_all):
            print(f"Alert: {column} has unique values for each row. Skipping for plotting.")
            continue
        elif num_unique > 100:
            num_unique = int(30)
            # when num_unique > 30, combine the least frequent values into "Others"
            column_counts = column_counts[:30]
            column_counts["Others"] = models_summaries_all[column].value_counts()[30:].sum()
            print(f"Number of unique models after combining: {num_unique}")
            print(f"Total rows after combining: {column_counts.sum()}")
        
        column_counts.plot(kind='bar', edgecolor='black')
        for i in range(int(num_unique)):
            plt.text(i, column_counts.iloc[i], column_counts.iloc[i], ha='center', va='bottom')
        plt.xlabel(column)
        plt.ylabel('Number of models')
        plt.title(f'Distribution of models by {column}')
        if num_unique > 30:
            plt.xticks(rotation=45, ha='right')
        else:
            plt.xticks(rotation=10, ha='right')
        plt.grid(axis='y', alpha=0.4)
        plt.gca().set_xticks(range(num_unique))
        plt.savefig(f'{folder_save}/hist_categorical_{column}_model_distribution.png')
        plt.close()

    # Continuous columns
    elif column in attribute_continuous_columns:
        plt.figure(figsize=(10, 6))
        models_summaries_all[column].plot(kind='hist', bins=30, edgecolor='black')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.title(f'Distribution of models by {column}')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.4)
        plt.axvline(models_summaries_all[column].mean(), linestyle='dashed', color='red', alpha=0.5, linewidth=1)
        models_summaries_all[column].plot(kind='kde', secondary_y=True, color="red", linewidth=1, alpha=0.5, label='KDE Density')
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        plt.savefig(f'{folder_save}/hist_continuous_{column}_model_distribution.png')
        plt.close()

    # Binary columns
    elif column in attribute_binary_columns:
        plt.figure(figsize=(6, 6))
        models_summaries_all[column].value_counts().plot(kind='bar', edgecolor='black')
        plt.xlabel(column)
        plt.ylabel('Number of models')
        plt.title(f'Distribution of models by {column}')
        plt.xticks(rotation=0, ha='center')
        plt.grid(axis='y', alpha=0.4)
        # add number above each bar
        for i in range(2):
            plt.text(i, models_summaries_all[column].value_counts().iloc[i], models_summaries_all[column].value_counts().iloc[i], ha='center', va='bottom')
        plt.savefig(f'{folder_save}/hist_binary_{column}_model_distribution.png')
        plt.close()
        
    else:
        print(f"--> Skipping {column} because it's not categorical or numerical. The type is {models_summaries_all[column].dtype}")


# %%
# for the "Training Dataset == ImageNet-1K", we want to have a bar plot based on family

# filter the data
models_summaries_all_imagenet = models_summaries_all[models_summaries_all["Training Dataset"] == "ImageNet-1K"]

# group by family
models_summaries_all_imagenet_family = models_summaries_all_imagenet.groupby("Family").size().reset_index(name="Count")

# sort by count
models_summaries_all_imagenet_family = models_summaries_all_imagenet_family.sort_values(by="Count", ascending=True)

# plot the bar plot
plt.figure(figsize=(10, 15))
plt.barh(models_summaries_all_imagenet_family["Family"], models_summaries_all_imagenet_family["Count"])
# mark the number of models
for i, v in enumerate(models_summaries_all_imagenet_family["Count"]):
    plt.text(v, i, str(v), color='black', va='center')
plt.title("Number of models in ImageNet-1K by Family")
plt.xlabel("Number of models")
plt.ylabel("Family")
plt.tight_layout()
plt.savefig("plots/family_distribution_imagenet1k.png")
plt.close()


# %% [markdown]
# ## Analysis on Continuous Attributes

# %% [markdown]
# ### Scatterplot matrix of the continuous attributes

# %%
# plot the scatterplot matrix of continuous attributes
sns.pairplot(models_summaries_all[attribute_continuous_cleanned])
# plt.suptitle("Scatterplot Matrix of continuous Attributes")
plt.savefig(f'{folder_save_plot}/continuous_attributes_scatterplot_matrix.png')
plt.show()

# %% [markdown]
# ### Correlation matrix (Heatmap) of continuous attributes

# %%
# plot the correlation matrix of continuous attributes
plt.figure(figsize=(12, 10))
sns.heatmap(
    models_summaries_all[attribute_continuous_cleanned].corr(), 
    annot=True, cmap="coolwarm", fmt=".2f", center=0
)
# tilt the x-axis labels for better readability
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.title("Correlation Matrix of continuous Attributes")
plt.savefig(f'{folder_save_plot}/continuous_attributes_correlation_matrix.png')
plt.show()

# %% [markdown]
# ### Calculate VIF for each continuous attribute

# %%
# Standardize the continuous attributes
Xs_continuous = StandardScaler().fit_transform(models_summaries_all[attribute_continuous_cleanned])


# Calculate the VIF (Variance Inflation Factor) for each continuous attribute
vif_data = pd.DataFrame({
    "Attributes": attribute_continuous_cleanned,
    "VIF": [variance_inflation_factor(Xs_continuous, i) for i in range(Xs_continuous.shape[1])]
})

# sort the VIF from largest to smallest
vif_data = vif_data.sort_values(by="VIF", ascending=False)

print("\nVariance Inflation Factor (VIF):")
print(vif_data)

# plot the VIF
plt.figure(figsize=(12, 6))
plt.bar(vif_data["Attributes"], vif_data["VIF"])
plt.xlabel("Attributes")
plt.ylabel("VIF")
plt.title("Variance Inflation Factor (VIF)")
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.4)
# draw a line of VIF = 5 and VIF = 10
plt.axhline(y=5, color='r', linestyle='--', linewidth=1)
plt.axhline(y=10, color='r', linestyle='--', linewidth=1)
plt.savefig(f'{folder_save_plot}/continuous_attributes_vif.png')
plt.show()

# %% [markdown]
# ### Use PCA to find the most important continuous attributes

# %%
# PCA

pca = PCA()
pca_result = pca.fit_transform(Xs_continuous)

# explained variance ratio: how much variance each principal component explains 
explained_variance_ratio = pca.explained_variance_ratio_ # PC1 to n explains the most to the least variance
cumulative_variance = np.cumsum(explained_variance_ratio)

# goal is to use fewer PCs to capture most of the variances


# Explained variance plot (Scree Plot)

# plot the explained variance ratio (individual and cumulative) of each principal component
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, 
         marker='o', linestyle='-', label='Individual Variance')
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 
         marker='s', linestyle='-', label='Cumulative Variance')
plt.axhline(y=0.1, linestyle='--', label='10% Threshold', color='grey', alpha=0.5)
plt.axhline(y=0.9, linestyle='--', label='90% Threshold', color='grey', alpha=0.5)
plt.xlabel('Principal Components \n (Ordered by Explained Variance)')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance: Scree Plot')
plt.xticks(range(1, len(explained_variance_ratio) + 1))
plt.legend(loc='best')
plt.grid(True)
plt.savefig(f'{folder_save_plot}/continuous_attributes_pca_scree_plot.png')
plt.show()




# PCA Loadings

# show how much each attribute contributes to each principal component
pca_loadings = pd.DataFrame(
    pca.components_,
    columns=attribute_continuous_cleanned,
    index=[f"PC{i+1}" for i in range(len(pca.components_))]
)
# visualize the loadings
plt.figure(figsize=(12, 8))
sns.heatmap(pca_loadings.T, 
            annot=True, cmap='coolwarm', linewidths=0.5, fmt=".2f", center=0
            )
plt.title('PCA Loadings Heatmap')
plt.xlabel('Principal Components')
plt.ylabel('Attributes')
plt.savefig(f'{folder_save_plot}/continuous_attributes_pca_loadings_heatmap.png')
plt.show()



# PCA Biplot
# visualize samples and attributes together
plt.figure(figsize=(15, 10))
x_scores = pca_result[:, 0]
y_scores = pca_result[:, 1]
plt.scatter(x_scores, y_scores, alpha=0.5, label='')

for i, var in enumerate(attribute_continuous_cleanned):
    plt.arrow(0, 0, pca_loadings.iloc[i, 0]*3, pca_loadings.iloc[i, 1]*3,
              color='red', alpha=0.7, head_width=0.05)
    plt.text(pca_loadings.iloc[i, 0]*3.3, pca_loadings.iloc[i, 1]*3.3, var, fontsize=10)

plt.xlabel(f'PC1 ({explained_variance_ratio[0]*100:.1f}% variance)')
plt.ylabel(f'PC2 ({explained_variance_ratio[1]*100:.1f}% variance)')
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.title('PCA Biplot: Samples and Attribute Influences')
plt.grid(True)
plt.savefig(f'{folder_save_plot}/continuous_attributes_pca_biplot.png')
plt.show()



# Calculate the number of PCs to capture 90% of the variance
threshold_cum_variance = 0.9
num_components = np.argmax(cumulative_variance >= threshold_cum_variance) + 1

print(f"Number of Principal Components to capture {threshold_cum_variance*100:.0f}% of the variance: {num_components}")


