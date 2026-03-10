# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 15:40:08 2025

@author: Glass
"""
######################

# Nutzung der Open Membrane Database (https://www.openmembranedatabase.org/ ; aufgerufen am 08.04.2025) zur Suche
# von Membranen für die Trennung eines Katalysators von Methanol/Methylvalerat Gemisch

######################



#######################
# %% Import Libraries

import warnings
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import textwrap
import re

from sklearn.preprocessing import StandardScaler, RobustScaler, TargetEncoder

# Own Code Files
from data_preprocessing import preprocessing, solvent_visco, feature_list

warnings.filterwarnings("ignore")

np.random.seed(42)

# Importing the data
ONF_Database = "OMD_SRNF_2025-04-08.csv"

###################################################
# %% Preprocessing of data

# ---- use Data_Preprocessing file to import several DataFrames
srnf, perm_given, mwco_given, both_given, experiments, membranes, unique_filtrations = preprocessing("OMD_SRNF_2025-04-08.csv")

# ---- Histrograms of the two results MWCO and permeance
sns.histplot(x="filtrationResults.solventPermeance", data=perm_given, kde=False,
             stat="density", log_scale=10) # stat = "density" normalizes the values
plt.show()

sns.histplot(x="filtrationResults.mwco", data=mwco_given, kde=False,
             stat="density") # stat="density" normalizes the values
plt.show()

# ---- plot  permeance vs. mwco and highlight the datapoints of interest
plt.scatter(both_given["filtrationResults.solventPermeance"], both_given["filtrationResults.mwco"])
plt.axhline(y=1000, xmin=0, xmax=0.775, color="black") # MW (Cat) = 590 g/mol
plt.axhline(y=200, xmin=0, xmax=0.775, color="black") # MW (Methylvalerat) = 116 g/mol
plt.axvline(x=500, ymin=200/2000, ymax=1000/2000, color="black")
plt.ylim(0, 2000)
plt.xscale("log")
plt.xlabel("Permeance [LMH/bar]")
plt.ylabel("MWCO [g/mol]")
plt.show()

########################################
# %% Plot data statistics

# Description and plotting of experiments and membranes that are relevant for the project
membranes_descriptor = membranes.describe().T
experiment_descriptor = experiments.describe().T

sns.boxplot(data=membranes, x="filtrationResults.mwco", y="family", hue="soluteChoice.solute1.category", legend="brief")
plt.title("MWCO of interesting membranes sorted by family")
plt.legend(loc="lower right")
plt.show()

sns.boxplot(data=membranes, x="filtrationResults.solventPermeance", y="family", log_scale=True)
plt.title("Permeance of interesting membranes sorted by family")
plt.show()

sns.boxplot(data=membranes, x="filtrationResults.mwco", y="chemistry")
plt.title("MWCO of interesting membranes sorted by chemistry")
plt.show()

sns.boxplot(data=membranes, x="filtrationResults.mwco", y="structure")
plt.title("MWCO of interesting membranes sorted by structure")
plt.show()

sns.boxplot(data=membranes, x="filtrationResults.solventPermeance", y="structure", log_scale=True)
plt.title("Permeance of interesting membranes sorted by structure")
plt.show()

sns.countplot(data=membranes, x="family")
plt.title("Amount of membranes by family")
plt.show()

sns.boxplot(data=both_given, x="filtrationResults.mwco", y="family", log_scale=True)
plt.title("MWCO of all filtration experiments sorted by structure")
plt.show()

sns.boxplot(data=both_given, x="filtrationResults.solventPermeance", y="family", log_scale=True)
plt.title("Permeance of all filtration experiments sorted by structure")
plt.show()

sns.boxplot(data=both_given, x="filtrationResults.solventPermeance", y="testConditions.solvent1", log_scale=True)
plt.title("Permeance of all filtration experiments sorted by solvent1")
plt.show()

sns.countplot(data=both_given, x="testConditions.solventMixture")
plt.title("Amount of all experiments analyses with solvent mixture")
plt.show()

sns.countplot(data=both_given, x="testConditions.solvent1")
plt.title("Amount of all experiments analyses with solvent1")
plt.xticks(rotation=90)
plt.show()

sns.boxplot(data=membranes, x="filtrationResults.solventPermeance", y="testConditions.solvent1", log_scale=True)
plt.title("Permeance of membranes sorted by solvent")
plt.show()

##########################################
# %% Target/Label selection and Feature engineering



# define label
y = mwco_given["filtrationResults.mwco"].astype(float)

# ---- define Features
features = mwco_given[feature_list]

features["solvent1.viscosity"] = features["testConditions.solvent1"].map(solvent_visco)
features = features.drop(["testConditions.solvent1"], axis=1)

# Using numbers such as Layer thickness as float by preplacing missing values (nan) with the median of all existing values or with 0
features = features.astype({"isaSupportLayer.castingThickness":"float", # "testConditions.hydraulicP":"float"
                            })
median = features["isaSupportLayer.castingThickness"].median()
median_roughness = features["characterizationResults.roughness"].median()
median_CA = features["characterizationResults.contactAngle"].median()
median_total_thickness = features["characterizationResults.totalThickness"].median()
features.fillna({"isaSupportLayer.castingThickness":0, "soluteChoice.solute1.concentration":0, "characterizationResults.topLayerThickness":0},
                inplace=True) # if median should be used replace 0 with median

features.fillna({"characterizationResults.roughness":median_roughness,
                 "characterizationResults.contactAngle":median_CA,
                 "characterizationResults.totalThickness": median_total_thickness},
                 inplace=True)

# divide into numerical and categorical values; done separately for training and test set
categorical_cols = ["testConditions.filtrationMode", "soluteChoice.solute1.category",
                    "structure", "chemistry", "chemistryOther", "family",
                    "supportLayerChemistry", "supportLayerType", "supportLayer.postTreatment",
                    "postTreatment", "testConditions.temperature",
                    "isaSupportLayer.solvent", 
                    "topLayerDepositionMethod",
                    "tfnData.nanomaterialName"]

numerical_cols = ["testConditions.hydraulicP",
                  "soluteChoice.solute1.concentration",
                  "isaSupportLayer.castingThickness", 
                  "characterizationResults.topLayerThickness", "characterizationResults.roughness",
                  "characterizationResults.contactAngle", "characterizationResults.totalThickness",
                  "solvent1.viscosity"]

X_categorical = features[categorical_cols]

X_numerical = features[numerical_cols]

# Target encode categorical values 
encoder = TargetEncoder("auto")

X_encoded_cat = encoder.fit_transform(X_categorical, y)

# join encoded catergorical and numerical values
X_new = pd.concat([pd.DataFrame(X_encoded_cat, index=features.index, 
                                columns = X_categorical.columns),
                   X_numerical], axis=1)

# ---- Scaling 
scaler = StandardScaler()
robust = RobustScaler()

# Scaling features
X_scaled = robust.fit_transform(X_new)
X_scaled = pd.DataFrame(X_scaled, index=X_new.index, columns = X_new.columns)

correlation_matrix = X_new.corr()

plt.colorbar(plt.matshow(correlation_matrix, cmap='RdBu', vmin=-1, vmax=1))
plt.xlabel("")
plt.xticks(ticks=range(len(X_new.columns)), labels=X_new.columns, rotation=90)
plt.yticks(ticks=range(len(X_new.columns)), labels=X_new.columns)
plt.show()


# %% Missing data



missing_in_mwco = mwco_given[feature_list].isnull().mean().sort_values(ascending=False) * 100
missing_in_mwco = missing_in_mwco[missing_in_mwco > 0]

missing_in_perm = perm_given[feature_list].isnull().mean().sort_values(ascending=False) * 100
missing_in_perm = missing_in_perm[missing_in_perm > 0]

df = pd.concat([missing_in_mwco, missing_in_perm], axis=1)

plt.figure(figsize=(10,6))
ax = df.plot(kind="bar")

labels = ['\n'.join(textwrap.fill(part.strip(), 25) for part in re.split(r'\.\s*', label, maxsplit=1) if part)
    for label in df.index]
ax.set_xticklabels(labels, rotation=90)

plt.legend(["MWCO", "Permeance"])

plt.ylabel('% Missing')
plt.title('Missing Data by Feature')
plt.tight_layout()
plt.show()
