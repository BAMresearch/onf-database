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
import textwrap
import re
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import shap
import sklearn.ensemble as se

from sklearn.preprocessing import StandardScaler, RobustScaler, TargetEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, get_scorer, root_mean_squared_error
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance

# Own Code Files
from data_preprocessing import preprocessing, solvent_visco, feature_list

random.seed(42)
os.environ["PYTHONHASHSEED"] = "42"
np.random.seed(42)
warnings.filterwarnings("ignore")

# Importing the data
ONF_Database = "OMD_SRNF_2025-04-08.csv"

# %% Preprocessing of data

# ---- use Data_Preprocessing file to import several DataFrames
srnf, perm_given, mwco_given, both_given = preprocessing("OMD_SRNF_2025-04-08.csv")

# %% Target/Label selection and Feature engineering
# ---- define Features
features = perm_given[feature_list].drop(["soluteChoice.solute1.category", "family",
                                        "isaSupportLayer.castingThickness","tfnData.nanomaterialName",
                                        "characterizationResults.topLayerThickness","characterizationResults.roughness"],
                                        axis=1)

features["solvent1.viscosity"] = features["testConditions.solvent1"].map(solvent_visco)
features = features.drop(["testConditions.solvent1"], axis=1)

# Using numbers such as Layer thickness as float by preplacing missing values (nan) 
# with the median of all existing values or with 0
median_CA = features["characterizationResults.contactAngle"].median()
median_total_thickness = features["characterizationResults.totalThickness"].median()
median_T = features["testConditions.temperature"].median()
features.fillna({"soluteChoice.solute1.concentration":0,
                 }, inplace=True) # if median should be used replace 0 with median

features.fillna({"characterizationResults.contactAngle":median_CA,
                 "characterizationResults.totalThickness": median_total_thickness,
                 "testConditions.temperature": median_T
                 }, inplace=True)

X = features

# ---- Define label
y = perm_given["filtrationResults.solventPermeance"].astype(float)
y_binned = pd.qcut(y, q=4, labels=False)

# ---- split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y_binned)

#%% Giving weights to the samples to lower outlier impact (added after review)
train_bins = pd.qcut(y_train, q=10, labels=False, duplicates="drop")

# Calculate inverse-frequency weights
bin_counts = train_bins.value_counts()
weights = train_bins.map(lambda b: len(train_bins) / bin_counts[b]).values

# Normalize weights
weights = weights / np.mean(weights)

# ---- Target-Encoding
# divide into numerical and categorical values; done separately for training and test set
categorical_cols = ["testConditions.filtrationMode",
                    "structure", "chemistry", "chemistryOther",
                    "supportLayerChemistry", "supportLayerType", "supportLayer.postTreatment",
                    "postTreatment",
                    "isaSupportLayer.solvent",
                    "topLayerDepositionMethod"]

numerical_cols = ["testConditions.hydraulicP",
                "soluteChoice.solute1.concentration",
                "characterizationResults.contactAngle",
                "characterizationResults.totalThickness",
                "solvent1.viscosity",
                "testConditions.temperature"]

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
y_train = np.log1p(y_train.array.astype(float))
y_test = np.log1p(y_test.array.astype(float))

plt.colorbar(plt.matshow(X_train_new.corr(), cmap='RdBu', vmin=-1, vmax=1))
plt.xlabel("")
plt.xticks(ticks=range(len(X_train_new.columns)), labels=X_train_new.columns, rotation=90)
plt.yticks(ticks=range(len(X_train_new.columns)), labels=X_train_new.columns)
plt.show()

# %% Training models (Hyperparameter already optimized)
# %%% Linear Regression
print("\n Linear Regression:")
linear = LinearRegression()

linear.fit(X_train_scaled, y_train, sample_weight=weights)
y_pred = linear.predict(X_test_scaled)

# ---- evaluate model
train_R2 = linear.score(X_train_scaled, y_train)
test_R2 = linear.score(X_test_scaled, y_test)
mae =  mean_absolute_error(np.expm1(y_test), np.expm1(y_pred))
rmse = root_mean_squared_error(np.expm1(y_test), np.expm1(y_pred))

print("R2 train: ", round(train_R2, 2), " & R2 test: ", round(test_R2, 2), "& MAE:",  round(mae, 1), "& RMSE:",  round(rmse, 1))

# %%% Elastic net
print("\n Elastic Net:")
elan = ElasticNet(alpha=0.04, l1_ratio=0.05)

elan.fit(X_train_scaled, y_train, sample_weight=weights)
y_pred = elan.predict(X_test_scaled)

# ---- evaluate model
train_R2 = elan.score(X_train_scaled, y_train)
test_R2 = elan.score(X_test_scaled, y_test)
mae =  mean_absolute_error(np.expm1(y_test), np.expm1(y_pred))
rmse = root_mean_squared_error(np.expm1(y_test), np.expm1(y_pred))

print("R2 train: ", round(train_R2, 2), " & R2 test: ", round(test_R2, 2), "& MAE:",  round(mae, 1), "& RMSE:",  round(rmse, 1))

# %%% kNN with PCA beforehead
print("\n PCA + kNN:") 

pca = PCA(11)

X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

knn_pca = KNeighborsRegressor(n_neighbors=3, metric="nan_euclidean") 
knn_pca.fit(X_train_pca, y_train)
y_pred = knn_pca.predict(X_test_pca)

# ---- evaluate model
train_R2 = knn_pca.score(X_train_pca, y_train)
test_R2 = knn_pca.score(X_test_pca, y_test)
mae =  mean_absolute_error(np.expm1(y_test), np.expm1(y_pred))
rmse = root_mean_squared_error(np.expm1(y_test), np.expm1(y_pred))

print("R2 train: ", round(train_R2, 2), " & R2 test: ", round(test_R2, 2), "& MAE:",  round(mae, 1), "& RMSE:",  round(rmse, 1))

# %%% Random Forest
print("\n Random Forest: ") 

random_forest = RandomForestRegressor(n_estimators=225, max_depth=12, random_state=42)
random_forest.fit(X_train_scaled, y_train, sample_weight=weights)
y_pred = random_forest.predict(X_test_scaled)

# ---- evaluate model
train_R2 = random_forest.score(X_train_scaled, y_train)
test_R2 = random_forest.score(X_test_scaled, y_test)
mae =  mean_absolute_error(np.expm1(y_test), np.expm1(y_pred))
rmse = root_mean_squared_error(np.expm1(y_test), np.expm1(y_pred))

print("R2 train: ", round(train_R2, 2), " & R2 test: ", round(test_R2, 2), "& MAE:",  round(mae, 1), "& RMSE:",  round(rmse, 1))

# %%%% Tree Explainer
importances_rf = random_forest.feature_importances_
indices = np.argsort(importances_rf)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Random Forest")
plt.bar(range(len(importances_rf)), importances_rf[indices], align="center")
plt.xticks(range(len(importances_rf)), [X_train_scaled.columns[i] for i in indices], rotation=45, ha='right')
plt.ylabel("Feature importance")
plt.tight_layout()
plt.show()

# %%% Gradient Boosting
print("\n Gradient Boosting: ")
gradient_boost = se.GradientBoostingRegressor(n_estimators=325, max_depth=5, min_samples_leaf=5, learning_rate=0.088, random_state=42)
gradient_boost.fit(X_train_scaled, y_train, sample_weight=weights)
y_pred = gradient_boost.predict(X_test_scaled)

# ---- evaluate model
train_R2 = gradient_boost.score(X_train_scaled, y_train)
test_R2 = gradient_boost.score(X_test_scaled, y_test)
mae =  mean_absolute_error(np.expm1(y_test), np.expm1(y_pred))
rmse = root_mean_squared_error(np.expm1(y_test), np.expm1(y_pred))

print("R2 train: ", round(train_R2, 2), " & R2 test: ", round(test_R2, 2), "& MAE:",  round(mae, 1), "& RMSE:",  round(rmse, 1))

# %%%
print("\n Extra Tree: ")
extra_tree = se.ExtraTreesRegressor(n_estimators=175, max_depth=19, min_samples_leaf=3, random_state=42, n_jobs=-1)
extra_tree.fit(X_train_scaled, y_train, sample_weight=weights)
y_pred = extra_tree.predict(X_test_scaled)

# ---- evaluate model
train_R2 = extra_tree.score(X_train_scaled, y_train)
test_R2 = extra_tree.score(X_test_scaled, y_test)
mae =  mean_absolute_error(np.expm1(y_test), np.expm1(y_pred))
rmse = root_mean_squared_error(np.expm1(y_test), np.expm1(y_pred))

print("R2 train: ", round(train_R2, 2), " & R2 test: ", round(test_R2, 2), "& MAE:",  round(mae, 1), "& RMSE:",  round(rmse, 1))

# %%%% Tree Explainer
# Feature Importance
importances_rf = gradient_boost.feature_importances_
indices = np.argsort(importances_rf)[::-1]

X_train_scaled = pd.DataFrame(X_train_scaled, index=X_train_new.index, columns = X_train_new.columns)

labels = ['\n'.join(textwrap.fill(part.strip(), 25) for part in re.split(r'\.\s*', label, maxsplit=1) if part)
    for label in X_train_scaled.columns]
plt.figure(figsize=(10, 6))
plt.title("Gradient Boosting")
plt.bar(range(len(importances_rf)), importances_rf[indices], align="center")
plt.xticks(range(len(importances_rf)), [labels[i] for i in indices],
           rotation=90, ha='right')
plt.ylabel("Feature importance")
plt.tight_layout()
plt.show()

# %% XAI & feature analysis
# ---- shap plot 
explainer = shap.TreeExplainer(gradient_boost, X_train_scaled)
runs = []
  
shap_values = explainer.shap_values(X_train_scaled)
shap_values_2d = shap_values[:, :]

X_train_scaled = pd.DataFrame(X_train_scaled, index=X_train_new.index, columns = X_train_new.columns)

# %%
shap.summary_plot(shap_values_2d,X_train_scaled, feature_names=labels)
plt.show()

# %%
# SHAP-values for original names of categorical values
# def category_plot(X, shap_values, category_name:str):
#     shap_values_df = pd.DataFrame(shap_values, columns=X.columns)
#     X_with_cats = X.copy()
#     X_with_cats["shap_category"] = X_with_cats[category_name]
#     X_with_cats = X_with_cats.replace("[\"Polysulfone; Sulfonated poly(ether ether) ketone\"]", "[\"Polysulfone; Sulfonated PEEK\"]")
#     X_with_cats = X_with_cats.replace("[\"Pure solvent treatment\",\"Thermal treatment\", \"Crosslinking\"]",
#                                       "[\"Pure solvent t.\",\"Thermal t.\",\n\"Crosslinking\"]")
#     X_with_cats = X_with_cats.replace("[\"Gutter layer synthesis\",\"Thermal treatment\", \"Surface treatment\"]",
#                                       "[\"Gutter layer synthesis\",\n\"Thermal t.\",\"Surface t.\"]")
#     X_with_cats = X_with_cats.replace("[\"Crosslinking\",\"Gutter layer synthesis\"]","[\"Crosslinking\",\n\"Gutter layer synthesis\"]")
#     X_with_cats = X_with_cats.replace("[\"Pure solvent treatment\",\"Thermal treatment\",\"Crosslinking\"]",
#                                       "[\"Pure solvent treatment\",\n\"Thermal treatment\",\"Crosslinking\"]")
        
#     # mean of SHAP values per category
#     shap_cat_importance = shap_values_df.groupby(X_with_cats["shap_category"]).mean()

#     print(shap_cat_importance[category_name].sort_values())
#     colors = ['#1f77b4' if e >= 0 else '#ff0051' for e in shap_cat_importance[category_name].sort_values()]
    
#     shap_cat_importance[category_name].sort_values().plot(kind="barh", color=colors)
#     plt.ylabel("")
#     plt.xlabel("SHAP value (impact on model output)")
#     plt.xlim(-0.5, 0.65)
#     plt.tight_layout()
#     plt.show()
    
# # ----- structure
# category_plot(X_train, shap_values, "structure")

# # ---- Mode
# category_plot(X_train, shap_values, "testConditions.filtrationMode")

# # ---- chemistry 
# category_plot(X_train, shap_values, "chemistry")

# # ---- support chemistry 
# category_plot(X_train, shap_values, "supportLayerChemistry")

# # ---- post treatment 
# category_plot(X_train, shap_values, "postTreatment")

# # ---- supportLayer.post treatment 
# category_plot(X_train, shap_values, "supportLayer.postTreatment")

# %%
# ---- Permutation Importance
permutation_results = permutation_importance(gradient_boost,
                                             X=X_test_scaled,
                                             y=y_test,
                                             n_repeats=5,
                                             random_state=42,
                                             scoring=get_scorer("r2"))

permutation_results["feature"] = X_train.columns
# %% 
sorted_importances_idx = permutation_results.importances_mean.argsort()
importances = pd.DataFrame(np.flip(permutation_results.importances[sorted_importances_idx]).T,
                           columns=np.flip(X_train_scaled.columns[sorted_importances_idx]))

ax = importances.plot.box(vert=True, whis=10)
# ax.set_title("Permutation Importances (test set)")
ax.axhline(y=0, color="k", linestyle="--")
labels = ['\n'.join(textwrap.fill(part.strip(), 25) for part in re.split(r'\.\s*', label, maxsplit=1) if part)
    for label in importances.columns]
ax.set_xticklabels(labels, rotation=90)
ax.set_ylabel("Decrease in R2 score")
ax.figure.tight_layout()

# %%