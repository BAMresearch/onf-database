# -*- coding: utf-8 -*-
"""
Created on Wed May 28 11:09:47 2025

@author: Glass
"""
#%%
import warnings
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import shap
import re
import textwrap

# import tensorflow as tf

# from keras.models import Sequential
# from keras.layers import Dense
# from keras.regularizers import l2

from sklearn.preprocessing import StandardScaler, RobustScaler, TargetEncoder
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor
import sklearn.ensemble as se
from sklearn.metrics import mean_absolute_error, get_scorer, get_scorer_names
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.metrics import r2_score

# from xgboost import XGBRegressor

# Own Code Files
from data_preprocessing import preprocessing, solvent_visco

warnings.filterwarnings("ignore")

np.random.seed(42)

# Importing the data
ONF_Database = "OMD_SRNF_2025-04-08.csv"

###################################################
# %% Preprocessing of data

# ---- use Data_Preprocessing file to import several DataFrames
srnf, perm_given, mwco_given, both_given, experiments, membranes, unique_filtrations = preprocessing("OMD_SRNF_2025-04-08.csv")

# %% Target/Label selection and Feature engineering
# ---- define Features
features = mwco_given[["testConditions.filtrationMode", "testConditions.solvent1", 
                       # "testConditions.hydraulicP",
                       # "soluteChoice.solute1.category", 
                       "soluteChoice.solute1.concentration",
                       "structure", 
                       "chemistry", "chemistryOther",
                       # "family",
                       "supportLayerChemistry", "supportLayerType", "supportLayer.postTreatment",
                       "postTreatment",
                       "isaSupportLayer.castingThickness", # "isaSupportLayer.solvent",
                       "topLayerDepositionMethod",
                       "tfnData.nanomaterialName",
                       "characterizationResults.topLayerThickness", # "characterizationResults.roughness",
                       "characterizationResults.contactAngle", 
                       #"characterizationResults.totalThickness",
                       "testConditions.temperature"
                       ]]

features["solvent1.viscosity"] = features["testConditions.solvent1"].map(solvent_visco)
features = features.drop(["testConditions.solvent1"], axis=1)

# Using numbers such as Layer thickness as float by preplacing missing values (nan) with the median of all existing values or with 0
features = features.astype({"isaSupportLayer.castingThickness":"float", # "testConditions.hydraulicP":"float"
                            })
# median = features["isaSupportLayer.castingThickness"].median()
# median_roughness = features["characterizationResults.roughness"].median()
median_CA = features["characterizationResults.contactAngle"].median()
# median_total_thickness = features["characterizationResults.totalThickness"].median()
median_T = features["testConditions.temperature"].median()
features.fillna({"isaSupportLayer.castingThickness":0, "soluteChoice.solute1.concentration":0, "characterizationResults.topLayerThickness":0},
                inplace=True) # if median should be used replace 0 with median

features.fillna({# "characterizationResults.roughness":median_roughness,
                 "characterizationResults.contactAngle":median_CA,
                 #"characterizationResults.totalThickness": median_total_thickness,
                 "testConditions.temperature": median_T
                 }, inplace=True)
                 
X = features

# ---- Define label
y = mwco_given["filtrationResults.mwco"].astype(float)
y_binned = pd.qcut(y, q=4, labels=False)

# ---- split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y_binned)

# ---- Target-Encoding
# divide into numerical and categorical values; done separately for training and test set
categorical_cols = ["testConditions.filtrationMode", # "soluteChoice.solute1.category",
                    "structure", "chemistry", "chemistryOther",
                    # "family",
                    "supportLayerChemistry", "supportLayerType", "supportLayer.postTreatment",
                    "postTreatment", 
                    # "isaSupportLayer.solvent", 
                    "topLayerDepositionMethod",
                    "tfnData.nanomaterialName",
                    "testConditions.temperature"]

numerical_cols = [# "testConditions.hydraulicP",
                  "soluteChoice.solute1.concentration",
                  "isaSupportLayer.castingThickness", 
                  "characterizationResults.topLayerThickness", # "characterizationResults.roughness",
                  "characterizationResults.contactAngle", 
                  # "characterizationResults.totalThickness",
                  "solvent1.viscosity"]

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
scaler = StandardScaler()
robust = RobustScaler()

# Scaling features
X_train_scaled = robust.fit_transform(X_train_new)
X_test_scaled = robust.transform(X_test_new)

X_train_scaled = pd.DataFrame(X_train_scaled, index=X_train_new.index, columns = X_train_new.columns)

# Scaling of target y
y_train = scaler.fit_transform(y_train.values.reshape((-1, 1)))
y_test = scaler.transform(y_test.values.reshape((-1, 1)))

plt.colorbar(plt.matshow(X_train_new.corr(), cmap='RdBu', vmin=-1, vmax=1))
plt.xlabel("")
plt.xticks(ticks=range(len(X_train_new.columns)), labels=X_train_new.columns, rotation=90)
plt.yticks(ticks=range(len(X_train_new.columns)), labels=X_train_new.columns)
plt.show()

# %% Training models
# %%% Linear Regression
print("\n Linear Regression:")
linear = LinearRegression()

linear.fit(X_train_scaled, y_train)
y_pred = linear.predict(X_test_scaled)

# ---- evaluate model
train_R2 = linear.score(X_train_scaled, y_train)
test_R2 = linear.score(X_test_scaled, y_test)
mae = mean_absolute_error(scaler.inverse_transform(y_test).reshape(1, -1).T, scaler.inverse_transform(y_pred).reshape(1, -1).T)

print("R2 train: ", round(train_R2, 2), " & R2 test: ", round(test_R2, 2), "& MAE:",  round(mae, 1))

# %%% Elastic net
print("\n Elastic Net:")
elan = ElasticNet(alpha=0.2, l1_ratio=0.05) #optimized

elan.fit(X_train_scaled, y_train)
y_pred = elan.predict(X_test_scaled)

# ---- evaluate model
train_R2 = elan.score(X_train_scaled, y_train)
test_R2 = elan.score(X_test_scaled, y_test)

mae = mean_absolute_error(scaler.inverse_transform(np.array(y_test).reshape(1, -1)).T, scaler.inverse_transform(np.array(y_pred).reshape(1, -1)).T)

print("R2 train: ", round(train_R2, 2), " & R2 test: ", round(test_R2, 2) , "& MAE:",  round(mae, 1))

coefficients = pd.DataFrame(elan.coef_, X.columns.values)

# %%% kNN with PCA beforehead
print("\n PCA + kNN:") # optimized

pca = PCA(12) 

X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

knn_pca = KNeighborsRegressor(n_neighbors=3, metric="nan_euclidean")

knn_pca.fit(X_train_pca, y_train)
y_pred = knn_pca.predict(X_test_pca)

# ---- evaluate model
train_R2 = knn_pca.score(X_train_pca, y_train)
test_R2 = knn_pca.score(X_test_pca, y_test)
mae = mean_absolute_error(scaler.inverse_transform(np.array(y_test).reshape(1, -1)).T, scaler.inverse_transform(np.array(y_pred).reshape(1, -1)).T)

print("R2 train: ", round(train_R2, 2), " & R2 test: ", round(test_R2, 2), "& MAE:",  round(mae, 1))

# %%% Random Forest
print("\n Random Forest: ")

random_forest = RandomForestRegressor(n_estimators=25, max_depth=13, random_state=42) # optimized
random_forest.fit(X_train_scaled, y_train)
y_pred = random_forest.predict(X_test_scaled)

# ---- evaluate model
train_R2 = random_forest.score(X_train_scaled, y_train)
test_R2 = random_forest.score(X_test_scaled, y_test)
mae = mean_absolute_error(scaler.inverse_transform(np.array(y_test).reshape(1, -1)).T, scaler.inverse_transform(np.array(y_pred).reshape(1, -1)).T)

print("R2 train: ", round(train_R2, 2), " & R2 test: ", round(test_R2, 2), "& MAE:",  round(mae, 1))

# %%% Gradient Boosting
print("\n Gradient Boosting: ") # optimized
gradient_boost = se.GradientBoostingRegressor(n_estimators=100, max_depth=6,min_samples_leaf=5, learning_rate=0.1, random_state=42) 
gradient_boost.fit(X_train_scaled, y_train)
y_pred = gradient_boost.predict(X_test_scaled)

# ---- evaluate model
train_R2 = gradient_boost.score(X_train_scaled, y_train)
test_R2 = gradient_boost.score(X_test_scaled, y_test)
mae = mean_absolute_error(scaler.inverse_transform(np.array(y_test).reshape(1, -1)).T, scaler.inverse_transform(np.array(y_pred).reshape(1, -1)).T)

print("R2 train: ", round(train_R2, 2), " & R2 test: ", round(test_R2, 2), "& MAE:",  round(mae, 1))

# %%% Extra Tree
print("\n Extra Tree: ") #optimized
extra_tree = se.ExtraTreesRegressor(n_estimators=225, max_depth=13, max_features = 0.7, min_samples_leaf= 2, random_state=42, n_jobs=-1)
extra_tree.fit(X_train_scaled, y_train)
y_pred = extra_tree.predict(X_test_scaled)

# ---- evaluate model
train_R2 = extra_tree.score(X_train_scaled, y_train)
test_R2 = extra_tree.score(X_test_scaled, y_test)
mae = mean_absolute_error(scaler.inverse_transform(np.array(y_test).reshape(1, -1)).T, scaler.inverse_transform(np.array(y_pred).reshape(1, -1)).T)

print("R2 train: ", round(train_R2, 2), " & R2 test: ", round(test_R2, 2), "& MAE:",  round(mae, 1))

#%% predict unknown, new membranes
m1 = ["Dead-end", 2000, "[\"TFC\"]", "[\"Polydimethylsiloxane\"]", 
      None, "[\"Polyacrylonitrile\]", "ISA", "[\"Drying\"]", "[\"Crosslinking\"]", 
      0.0, "Dipcoating", None, 
      2500, 101, 20, 0.543] # Hereon-PDMS, measured MWCO = 506 Da

m2 = ["Dead-end", 2000, "[\"TFC\"]", "[\"Polydimethylsiloxane\"]", 
      None, "[\"Polyacrylonitrile\"]", "ISA", "[\"None\"]","[\"Crosslinking\"]", 
      0.0, "Dipcoating", None, 
      86.5, 101, 20, 0.543] # Puramem Flux, measured MWCO = 766 Da

# m3 = ["Dead-end", 2000, "[\"TFC\"]", "[\"Polymers of intrinsic microporosity\"]", 
#       None, "[\"Polyacrylonitrile\"]", "ISA", "[\"None\"]","[\"Crosslinking\",\"Drying\"]", 
#       0.0, "Dipcoating", None, 
#       86.5, 79, 20, 0.543] # PIM A high Flux, measured MWCO = ???

m3 = ["Dead-end", 2000, "[\"TFC\"]", "[\"Polymers of intrinsic microporosity\"]", 
      None, "[\"Polyacrylonitrile\"]", "ISA", "[\"None\"]","[\"Crosslinking\",\"Drying\"]", 
      0.0, "Dipcoating", None, 
      90, 75, 20, 0.543] # PIM B low flux, measured MWCO = 680 LMH/bar

m4 = ["Dead-end", 2000, "[\"TFC\"]", "[\"Polyamine\"]",  
    None, "[\"Polyetherimide\"]", "ISA", "[\"None\"]","[\"Crosslinking\",\"Drying\"]",
    0.0,  "Dipcoating", None, 
    1900, 52, 20, 0.543] # PEBAX, measured Perm = 0.75 LMH/bar

mem_names = ["Hereon \nPDMS", "Puramem", "Hereon \nPIM", "Hereon \nPEBAX"]
m_true = [563, 766, 680, 990] # measured MWCO using 90% Retention

# ['testConditions.filtrationMode', 'soluteChoice.solute1.concentration', 'structure', 'chemistry',
#      'chemistryOther', 'supportLayerChemistry', 'supportLayerType', 'supportLayer.postTreatment', 'postTreatment',
#      'isaSupportLayer.castingThickness', 'topLayerDepositionMethod', 'tfnData.nanomaterialName', 
#      'characterizationResults.topLayerThickness','characterizationResults.contactAngle', 'testConditions.temperature','solvent1.viscosity']

hereon_1 = pd.DataFrame([m1, m2, m3 ,m4], columns=X_test.columns, index=mem_names)
hereon_cat = hereon_1[categorical_cols]
hereon_encoded_cat = encoder.transform(hereon_cat)

#%%
hereon_encoded = pd.concat([pd.DataFrame(hereon_encoded_cat, index=mem_names,
                                    columns = X_test_categorical.columns), 
                       hereon_1[numerical_cols]], axis=1)

hereon_scaled = robust.transform(hereon_encoded)
# %%
m_pred = extra_tree.predict(hereon_scaled)
m_pred_t = scaler.inverse_transform(m_pred.reshape(1, -1))
m_pred_t =m_pred_t[0].tolist()

print(mem_names, m_pred_t)
# %% Check similarity
# Vector mean of training data and test data
mean_vec = np.mean(X_train_scaled, axis=0)
mean_vec_test = np.mean(X_test_scaled, axis=0)

# %%
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

#%%
# %% XAI & feature analysis

# ---- shap plot # auskommentiert für Schnelligkeit
import random
import os
random.seed(42)
os.environ["PYTHONHASHSEED"] = "42"
np.random.seed(42)

explainer = shap.TreeExplainer(extra_tree, X_train_scaled)
runs = []
  
shap_values = explainer.shap_values(X_train_scaled)
shap_values_2d = shap_values[:, :]

X_train_scaled = pd.DataFrame(X_train_scaled, index=X_train_new.index, columns = X_train_new.columns)
# %%
labels = ['\n'.join(textwrap.fill(part.strip(), 25) for part in re.split(r'\.\s*', label, maxsplit=1) if part)
    for label in X_train_scaled.columns]
shap.summary_plot(shap_values_2d,X_train_scaled, feature_names=labels)
plt.show()

# %%
# SHAP-values for original names of categorical values

def category_plot(X, shap_values, category_name:str):
    shap_values_df = pd.DataFrame(shap_values, columns=X.columns)
    X_with_cats = X.copy()
    X_with_cats["shap_category"] = X_with_cats[category_name]
        
    # all_categories = X_with_cats["shap_category"].unique()
    
    # mean of SHAP values per category
    shap_cat_importance = shap_values_df.groupby(X_with_cats["shap_category"]).mean()
    
    # shap_cat_importance = (shap_values_df.groupby(X_with_cats["shap_category"]).mean()
    #                     .reindex(all_categories, fill_value=0))   # Kategorien mit SHAP=0 erzwingen

    print(shap_cat_importance[category_name].sort_values())
    colors = ['#1f77b4' if e >= 0 else '#ff0051' for e in shap_cat_importance[category_name].sort_values()]
    
    shap_cat_importance[category_name].sort_values().plot(kind="barh", color=colors)
    plt.ylabel("")
    plt.xlabel("SHAP value (impact on model output)")
    plt.xlim(-0.2, 0.1)
    plt.tight_layout()
    plt.show()
    
# ----- structure
category_plot(X_train, shap_values, "topLayerDepositionMethod")

# ---- Mode
category_plot(X_train, shap_values, "chemistryOther")

# ---- chemistry 
category_plot(X_train, shap_values, "chemistry")

# ---- post treatment 
category_plot(X_train, shap_values, "postTreatment")

# ---- supportLayer.post treatment 
category_plot(X_train, shap_values, "supportLayer.postTreatment")
# %%% Tree Explainer

importances_rf = extra_tree.feature_importances_
indices = np.argsort(importances_rf)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Extra Trees")
plt.bar(range(len(importances_rf)), importances_rf[indices], align="center")
plt.xticks(range(len(importances_rf)), [X_train_scaled.columns[i] for i in indices], rotation=45, ha='right')
plt.tight_layout()
plt.show()

# %%% Feature Importances
# ---- Permutation Importance
permutation_results = permutation_importance(extra_tree,
                                             X=X_test_scaled,
                                             y=y_test,
                                             n_repeats=5,
                                             random_state=42,
                                             scoring=get_scorer("r2"))

permutation_results["feature"] = X_train.columns

sorted_importances_idx = permutation_results.importances_mean.argsort()
importances = pd.DataFrame(np.flip(permutation_results.importances[sorted_importances_idx]).T,
                           columns=np.flip(X.columns[sorted_importances_idx]))

ax = importances.plot.box(vert=True, whis=10)
# ax.set_title("Permutation Importances (test set)")
ax.axhline(y=0, color="k", linestyle="--")
labels = ['\n'.join(textwrap.fill(part.strip(), 25) for part in re.split(r'\.\s*', label, maxsplit=1) if part)
    for label in importances.columns]
ax.set_xticklabels(labels, rotation=90)
ax.set_ylabel("Decrease in R2 score")
ax.figure.tight_layout()

# %% Checking Correlations
# X_train_encoded_cat = pd.DataFrame(X_train_encoded_cat, columns = X_train_categorical.columns)

# plt.scatter(X_train_categorical["testConditions.filtrationMode"], X_train_encoded_cat["testConditions.filtrationMode"])
# plt.show()

# # plt.scatter(X_train_categorical["soluteChoice.solute1.category"].astype(str), X_train_encoded_cat["soluteChoice.solute1.category"])
# # plt.show()

# plt.scatter(X_train_categorical["structure"].astype(str), X_train_encoded_cat["structure"])
# plt.show()

# %% CrossValidation

# ---- Initiales Setup

# k-fold split and training
y_true_all = []
y_pred_all = []

kf = KFold(n_splits=10, shuffle=True, random_state=42)

for train_idx, test_idx in kf.split(X):
    # Split
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Spalten trennen
    X_train_cat = X_train[categorical_cols]
    X_train_num = X_train[numerical_cols]
    X_test_cat = X_test[categorical_cols]
    X_test_num = X_test[numerical_cols]

    # Target-Encoding (wird fold-weise neu codiert!)
    encoder = TargetEncoder()
    X_train_cat_enc = encoder.fit_transform(X_train_cat, y_train)
    X_test_cat_enc = encoder.transform(X_test_cat)

    # Zusammenfügen
    X_train_new = pd.concat([pd.DataFrame(X_train_cat_enc,
                                          index=X_train.index, columns = X_train_categorical.columns), X_train_num], axis=1)
    X_test_new = pd.concat([pd.DataFrame(X_test_cat_enc,
                                          index=X_test.index, columns = X_test_categorical.columns), X_test_num], axis=1)

    # Skalieren
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_new)
    X_test_scaled = scaler.transform(X_test_new)
    
    y_train = scaler.fit_transform(y_train.values.reshape((-1, 1)))
    y_test = scaler.transform(y_test.values.reshape((-1, 1)))

    # Modelltraining
    extra_tree.fit(X_train_scaled, y_train)

    # Vorhersage
    y_pred = extra_tree.predict(X_test_scaled)

    # Ergebnisse sammeln
    y_true_all.extend(y_test)
    y_pred_all.extend(y_pred)

y_true_all = scaler.inverse_transform(np.array(y_true_all).reshape(1, -1)).T
y_pred_all = scaler.inverse_transform(np.array(y_pred_all).reshape(1, -1)).T
# %%
# ---- Plot
# plt.figure(figsize=(6, 6))
plt.scatter(y_true_all, y_pred_all, alpha=0.3)
plt.plot([min(y_true_all), max(y_true_all)],
         [min(y_true_all), max(y_true_all)],
         color='red', linestyle='--', label='Ideal')
plt.scatter(m_true, m_pred_t, color="red", alpha=0.6)
for i in range(len(mem_names)):
    plt.text(m_true[i], m_pred_t[i], mem_names[i],
            fontsize=9, horizontalalignment='left', verticalalignment = "bottom",
            bbox=dict(alpha=0.4, facecolor="white", edgecolor="white", boxstyle='round,pad=-1'))
plt.xlabel("Measured MWCO")
plt.ylabel("Predicted MWCO")
plt.title("10-fold Cross-Validation: Prediction vs. Truth")
plt.legend()
# plt.grid(True)
plt.tight_layout()
plt.show()

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
plt.xlabel("Predicted values")
plt.ylabel("Residues (y_true - y_pred)")
plt.title("Residual Analysis")
plt.legend()
# plt.grid(True)
plt.ylim(-max(y_true_all)[0], max(y_true_all)[0])
plt.tight_layout()
plt.show()

y_true, y_pred = np.array(y_true_all), np.array(y_pred_all)
mape = np.mean(np.abs((y_true[y_true != 0] - y_pred[y_true != 0]) / y_true[y_true != 0])) * 100
print(f"MAPE: {mape:.2f}%", f"STD: {std:.1f} Da", f"MAE: {mae:.1f} Da")

# %%
