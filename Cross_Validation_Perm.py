# -*- coding: utf-8 -*-
"""
Created on Fri Jul 11 09:48:55 2025

@author: Glass
"""

#%% 
# import libraries
import random
import os
import warnings
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

from sklearn.preprocessing import RobustScaler, TargetEncoder
from sklearn.model_selection import train_test_split,  KFold
from sklearn.metrics import mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity

# Own Code Files
from data_preprocessing import preprocessing

from Perm_Model import gradient_boost, median_CA, median_total_thickness, categorical_cols, numerical_cols, X, y

warnings.filterwarnings("ignore")
random.seed(42)
os.environ["PYTHONHASHSEED"] = "42"
np.random.seed(42)

# Importing the data
ONF_Database = "OMD_SRNF_2025-04-08.csv"

# %% Preprocessing of data

# ---- use Data_Preprocessing file to import several DataFrames
srnf, perm_given, mwco_given, both_given = preprocessing("OMD_SRNF_2025-04-08.csv")

# %% Target/Label selection and Feature engineering
y_binned = pd.qcut(y, q=4, labels=False)

# ---- split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y_binned)

# ---- Target-Encoding
# divide into numerical and categorical values; done separately for training and test set
X_train_categorical = X_train[categorical_cols]
X_test_categorical = X_test[categorical_cols]

X_train_numerical = X_train[numerical_cols]
X_test_numerical = X_test[numerical_cols]

# Target encode categorical values
encoder = TargetEncoder("auto")

X_train_encoded_cat = encoder.fit_transform(X_train_categorical, y_train)
X_test_encoded_cat = encoder.transform(X_test_categorical)

# join encoded catergorical and numerical values
X_train_new = pd.concat([pd.DataFrame(X_train_encoded_cat,
                                      index=X_train.index, columns = X_train_categorical.columns),
                         X_train_numerical], axis=1)

X_test_new = pd.concat([pd.DataFrame(X_test_encoded_cat,
                                    index=X_test.index, columns = X_test_categorical.columns),
                        X_test_numerical], axis=1)

# ---- Scaling
robust = RobustScaler()

# Scaling features
X_train_scaled = robust.fit_transform(X_train_new)
X_test_scaled = robust.transform(X_test_new)
X_train_scaled = pd.DataFrame(X_train_scaled, index=X_train_new.index, columns = X_train_new.columns)

#%% predict unknown, new membranes
m1 = ["Dead-end", 4, 2000,  20, "[\"TFC\"]",
    "[\"Polydimethylsiloxane\"]", None, "[\"Polyacrylonitrile\",\"Polyester\"]", "ISA", "[]",
    "[\"Crosslinking\",\"Drying\"]","[\"Dimethylformamide (DMF)\"]", "Casting", 102,
    34.2,  0.543]
    # 1xradiation crosslinked PDMS on PAN, measured Perm = 0.07 LMH/bar

m2 = ["Dead-end", 4, 2000, 20, "[\"TFC\"]",
    "[\"Polydimethylsiloxane\"]", None, "[\"Polyacrylonitrile\"]", "ISA", "[\"None\"]",
    "[\"Crosslinking\"]","[]", "Casting", 101,
    35.7,  0.543] # Puramem Flux, measured Perm = 0.47 LMH/bar

m3 = ["Dead-end", 4, 2000, 20, "[\"TFC\"]",
    "[\"Polymers of intrinsic microporosity\"]", None, "[\"Polyacrylonitrile\",\"Polyester\"]", "ISA", "[\"Drying\"]",
    "[\"Crosslinking\"]","[Dimethylformamide (DMF)]", "Casting", 79,
    50.5, 0.543] # PIM B 1% crosslinked, measured Perm = 0.52 LMH/bar

m4 = ["Dead-end", 4, 2000, 20, "[\"TFC\"]",
    "[\"Polyamide\"]", None, "[\"Polyetherimide\"]", "ISA", "[\"Drying\"]",
    "[\"Crosslinking\",\"Drying\"]","[Dimethylformamide (DMF)]", "Dipcoating", 52,
    20, 0.543] # PEBAX, measured Perm = 0.75 LMH/bar

ml_1 = ["Crossflow", 10, 500, 25, "[\"TFC\"]",
    "[\"Polyimide\"]", None, "[\"Polyetherimide\"]", "ISA", "[\"None\"]",
    "[]","[Dimethylformamide (DMF)]", "Dipcoating", median_CA,
    median_total_thickness, 0.34] # Puramem 280 von https://doi.org/10.1016/j.memsci.2014.10.030; ACN

ml_2 = ["Crossflow", 10, 500, 25, "[\"TFC\"]",
    "[\"Polyimide\"]", None, "[\"Polyetherimide\"]", "ISA", "[\"None\"]",
    "[]","[Dimethylformamide (DMF)]", "Dipcoating", median_CA,
    median_total_thickness, 0.3] # Puramem 280 von https://doi.org/10.1016/j.memsci.2014.10.030; Aceton

ml_3 = ["Crossflow", 10, 500, 25, "[\"TFC\"]",
    "[\"Polyimide\"]", None, "[\"Polyetherimide\"]", "ISA", "[\"None\"]",
    "[]","[Dimethylformamide (DMF)]", "Dipcoating", median_CA,
    median_total_thickness, 0.59] # Puramem 280 von https://doi.org/10.1016/j.memsci.2014.10.030; Toluene

# %% CrossValidation
# k-fold split and training
y_true_all = []
y_pred_all = []

kf = KFold(n_splits=10, shuffle=True, random_state=42)

for train_idx, test_idx in kf.split(X):
    # Split
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    #Giving weights to the samples to lower outlier impact (added after review)
    train_bins = pd.qcut(y_train, q=10, labels=False, duplicates="drop")

    # Calculate inverse-frequency weights
    bin_counts = train_bins.value_counts()
    weights = train_bins.map(lambda b: len(train_bins) / bin_counts[b]).values

    # Normalize weights
    weights = weights / np.mean(weights)

    # Divide columns
    X_train_cat = X_train[categorical_cols]
    X_train_num = X_train[numerical_cols]
    X_test_cat = X_test[categorical_cols]
    X_test_num = X_test[numerical_cols]

    # Target-Encoding 
    X_train_cat_enc = encoder.fit_transform(X_train_cat, y_train)
    X_test_cat_enc = encoder.transform(X_test_cat)

    # combine categorical and numerical columns
    X_train_new = pd.concat([pd.DataFrame(X_train_cat_enc,
                                          index=X_train.index, columns=X_train_categorical.columns), X_train_num],
                            axis=1)
    X_test_new = pd.concat([pd.DataFrame(X_test_cat_enc,
                                          index=X_test.index, columns=X_test_categorical.columns), X_test_num],
                           axis=1)

    # scale
    X_train_scaled = robust.fit_transform(X_train_new)
    X_test_scaled = robust.transform(X_test_new)
    
    y_train = np.log1p(y_train.array.astype(float))
    y_test = np.log1p(y_test.array.astype(float))

    # Modelltraining
    gradient_boost.fit(X_train_scaled, y_train, sample_weight=weights)

    # Vorhersage
    y_pred = gradient_boost.predict(X_test_scaled)

    # Ergebnisse sammeln
    y_true_all.extend(y_test)
    y_pred_all.extend(y_pred)
    
y_true_all = np.expm1(y_true_all)
y_pred_all = np.expm1(y_pred_all)

#%% Membranes for evaluation
#mem_names = ["Hereon PDMS", "Puramem", "Hereon PIM", "Hereon PEBAX", "Lit Toluene", "Lit Aceton", "Lit ACN"]
m_true = [0.07, 0.47, 0.52, 0.75, 3, 8, 12]

mem_names = ["Hereon PDMS", "Puramem", "Hereon PIM", "Hereon PEBAX", "Lit Toluene", "Lit Aceton", "Lit ACN"]
hereon_1 = pd.DataFrame([m1, m2, m3, m4, ml_3, ml_2, ml_1], columns=X_test.columns, index=mem_names)
hereon_cat = hereon_1[categorical_cols]
hereon_encoded_cat = encoder.transform(hereon_cat)

hereon_encoded = pd.concat([pd.DataFrame(hereon_encoded_cat, index=mem_names,
                                    columns = X_test_categorical.columns), 
                       hereon_1[numerical_cols]], axis=1)

hereon_scaled = robust.transform(hereon_encoded)

m_pred = gradient_boost.predict(hereon_scaled)
m_pred_t = np.absolute(np.expm1(m_pred.reshape(1, -1)))
m_pred_t =m_pred_t[0].tolist()

print(mem_names, m_pred_t)

# %%
# ---- Plot
# plt.figure(figsize=(6, 6))
plt.scatter(y_true_all, y_pred_all, alpha=0.3)
plt.plot([min(y_true_all), max(y_true_all)],
         [min(y_true_all), max(y_true_all)],
         color='red', linestyle='--', label='Ideal')
for i in range(len(mem_names)):
    if i<=3: 
        plt.text(m_true[i], m_pred_t[i], mem_names[i], color="black",
                fontsize=9, horizontalalignment='right',
                bbox=dict(alpha=0.4, facecolor="white", edgecolor="white", boxstyle='round,pad=-1'))
    elif i==4:
        plt.text(m_true[i], m_pred_t[i], mem_names[i], color="black",
                        fontsize=9, horizontalalignment='right', verticalalignment="top" ,
                        bbox=dict(alpha=0.4, facecolor="white", edgecolor="white", boxstyle='round,pad=-1'))
    elif i == 6:
        plt.text(m_true[i], m_pred_t[i], mem_names[i], color="black",
                        fontsize=9, horizontalalignment='left', verticalalignment="bottom" , 
                        bbox=dict(alpha=0.4, facecolor="white", edgecolor="white", boxstyle='round,pad=-1'))        
    else:
        plt.text(m_true[i], m_pred_t[i], mem_names[i], color="black",
                        fontsize=9, horizontalalignment='left', verticalalignment="top" , 
                        bbox=dict(alpha=0.4, facecolor="white", edgecolor="white", boxstyle='round,pad=-1'))
plt.scatter(m_true[0:4], m_pred_t[0:4], color="red", alpha=0.6)
plt.scatter(m_true[4:7], m_pred_t[4:7], color="yellow", alpha=0.6)
plt.xlabel("Measured Permeance [LHM/bar]")
plt.ylabel("Predicted Permeance [LHM/bar]")
plt.xscale("log")
plt.yscale("log")
plt.title("10-fold Cross-Validation: Prediction vs. Truth")
plt.legend()
plt.tight_layout()
plt.show()

# %% Check similarity
# Vector mean of training data and test data
mean_vec = np.mean(X_train_scaled, axis=0)
mean_vec_test = np.mean(X_test_scaled, axis=0)

# calculate euclidian distance and cosinus similarity of new data
differences = pd.DataFrame(None, index=mem_names, columns=["euclidian distance","cosinus similarity"])
for i,name in enumerate(mem_names):
    avg_dist = np.mean(np.linalg.norm(np.array(X_train_scaled) - hereon_scaled[[i]], axis=1))
    differences["euclidian distance"][name] = avg_dist
    cos_sim = cosine_similarity(hereon_scaled[[i]].reshape(1, -1), np.array(mean_vec).reshape(1, -1))[0, 0]
    differences["cosinus similarity"][name] = cos_sim

avg_dist = np.mean(np.linalg.norm(np.mean(X_train_scaled, axis=0) - np.mean(X_test_scaled, axis=0)))
cos_sim = cosine_similarity(np.array(mean_vec_test).reshape(1, -1), np.array(mean_vec).reshape(1, -1))[0, 0]
differences.loc["Test data set"] = [avg_dist, cos_sim]
print(differences.astype(float).round(2))

# %% Residual-Analysis incl. plot
residuals = list(map(lambda true, pred: true - pred, y_true_all, y_pred_all))
mae = mean_absolute_error(y_true_all, y_pred_all)
std = np.std(residuals)

plt.figure(figsize=(7, 5))
plt.scatter(y_pred_all, residuals, alpha=0.3)
plt.axhline(0, color='red', linestyle='--')
plt.axhline(mae, color='grey', linestyle='--', label = "Mean absolute error (true-pred)")
plt.axhline(-mae, color='grey', linestyle='--')
plt.axhline(std, color='lightgrey', linestyle='--', label = "Standard deviation residuals")
plt.axhline(-std, color='lightgrey', linestyle='--')
plt.xlabel("Predicted values [LMH/bar]")
plt.ylabel("Residues [LMH/bar]")
plt.legend()
plt.ylim(-max(y), max(y))
plt.tight_layout()
plt.show()

y_true, y_pred = np.array(y_true_all), np.array(y_pred_all)
mape = np.mean(np.abs((y_true[y_true != 0] - y_pred[y_true != 0]) / y_true[y_true != 0])) * 100
print(f"MAPE: {mape:.2f}%", f"STD: {std:.1f} LMH/bar", f"MAE: {mae:.1f} LMH/bar")

#%%