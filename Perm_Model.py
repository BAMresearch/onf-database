# -*- coding: utf-8 -*-
"""
Created on Fri Jul 11 09:48:55 2025

@author: Glass
"""

#%%
import warnings
import textwrap
import re
# import time
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import shap

from sklearn.preprocessing import StandardScaler, RobustScaler, TargetEncoder
from sklearn.model_selection import train_test_split,  KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor
import sklearn.ensemble as se
from sklearn.metrics import mean_absolute_error, get_scorer
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.metrics.pairwise import cosine_similarity

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

# X = pd.get_dummies(features) # One-Hot-Encoding
X = features

# ---- Define label
y = perm_given["filtrationResults.solventPermeance"].astype(float)
y_binned = pd.qcut(y, q=4, labels=False)

# ---- split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y_binned)

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

# %% Training models
# %%% Linear Regression
print("\n Linear Regression:")
linear = LinearRegression()

linear.fit(X_train_scaled, y_train)
y_pred = linear.predict(X_test_scaled)

# ---- evaluate model
train_R2 = linear.score(X_train_scaled, y_train)
test_R2 = linear.score(X_test_scaled, y_test)
mae =  mean_absolute_error(np.expm1(y_test), np.expm1(y_pred))

print("R2 train: ", round(train_R2, 2), " & R2 test: ", round(test_R2, 2), "& MAE:",  round(mae, 1))

# %%% Elastic net
print("\n Elastic Net:")
elan = ElasticNet(alpha=0.04, l1_ratio=0.05)

elan.fit(X_train_scaled, y_train)
y_pred = elan.predict(X_test_scaled)

# ---- evaluate model
train_R2 = elan.score(X_train_scaled, y_train)
test_R2 = elan.score(X_test_scaled, y_test)
mae =  mean_absolute_error(np.expm1(y_test), np.expm1(y_pred))

print("R2 train: ", round(train_R2, 2), " & R2 test: ", round(test_R2, 2), "& MAE:",  round(mae, 1))

# coefficients = pd.DataFrame(elan.coef_, X.columns.values)

# # %%%% Hyperparameter tuning

# neighbors = list(range(1, 16, 1))
# n_components = list(range(1,len(X.columns),1))
# scores, n_list, c_list = [], [], []

# # ---- evaluate model

# counter = 0
# start = time.time()

# for c in n_components:
#     counter += 1
#     pca = PCA(n_components=c)
#     X_train_pca = pca.fit_transform(X_train_scaled)
#     X_test_pca = pca.transform(X_test_scaled)
#     for n in neighbors:
#         knn = KNeighborsRegressor(n_neighbors=n, metric = "nan_euclidean")
#         score = cross_val_score(knn, X_train_pca, y_train, cv=10)
#         scores.append(np.mean(score))
#         n_list.append(n)
#         c_list.append(c)
#     print(counter)
#     now = time.time()
#     run_time = np.round((now -start)/60, 2)
#     print(run_time)

# end = time.time()
# total_run_time = np.round((end-start)/60, 2)

# best_k = n_list[np.argmax(scores)]
# best_c = c_list[np.argmax(scores)]

# # %% Plot k vs scores
# ax = plt.figure().add_subplot(projection='3d')

# ax.scatter(n_list, c_list, scores, c=scores)
# ax.set_xlabel('Neighbors')
# ax.set_ylabel('Components')
# ax.set_zlabel('CV R2')
# plt.show()

# print("Selected k:", best_k)
# print("Selected c:", best_c)


# %%% kNN with PCA beforehead
print("\n PCA + kNN:") # optimized

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

print("R2 train: ", round(train_R2, 2), " & R2 test: ", round(test_R2, 2), "& MAE:",  round(mae, 1))

# %%% Random Forest
print("\n Random Forest: ") # optimized

random_forest = RandomForestRegressor(n_estimators=225, max_depth=12, random_state=42)
random_forest.fit(X_train_scaled, y_train)
y_pred = random_forest.predict(X_test_scaled)

# ---- evaluate model
train_R2 = random_forest.score(X_train_scaled, y_train)
test_R2 = random_forest.score(X_test_scaled, y_test)
mae =  mean_absolute_error(np.expm1(y_test), np.expm1(y_pred))

print("R2 train: ", round(train_R2, 2), " & R2 test: ", round(test_R2, 2), "& MAE:",  round(mae, 1))

# %%%% Tree Explainer
importances_rf = random_forest.feature_importances_
indices = np.argsort(importances_rf)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Random Forest")
plt.bar(range(len(importances_rf)), importances_rf[indices], align="center")
plt.xticks(range(len(importances_rf)), [X_train_scaled.columns[i] for i in indices], rotation=45, ha='right')
plt.tight_layout()
plt.show()

# %%% Gradient Boosting
print("\n Gradient Boosting: ")
gradient_boost = se.GradientBoostingRegressor(n_estimators=325, max_depth=5, min_samples_leaf=5, learning_rate=0.088, random_state=42)
gradient_boost.fit(X_train_scaled, y_train)
y_pred = gradient_boost.predict(X_test_scaled)

# ---- evaluate model
train_R2 = gradient_boost.score(X_train_scaled, y_train)
test_R2 = gradient_boost.score(X_test_scaled, y_test)
mae =  mean_absolute_error(np.expm1(y_test), np.expm1(y_pred))

print("R2 train: ", round(train_R2, 2), " & R2 test: ", round(test_R2, 2), "& MAE:",  round(mae, 1))

#%% predict unknown, new membranes
m1 = ["Dead-end", 4, 2000,  20, "[\"TFC\"]",
    "[\"Polydimethylsiloxane\"]", None, "[\"Polyacrylonitrile\"]", "ISA", "[\"None\"]",
    "[\"Crosslinking\",\"Drying\"]","[\"Dimethylformamide (DMF)\"]", "Dipcoating", 101,
    34.2,  0.543]
    # 1xradiation crosslinked PDMS on PAN, measured Perm = 0.07 LMH/bar

m2 = ["Dead-end", 4, 2000, 20, "[\"TFC\"]",
    "[\"Polydimethylsiloxane\"]", None, "[\"Polyacrylonitrile\"]", "Commercial", "[\"None\"]",
    "[\"Crosslinking\"]","[]", None, 101,
    35.7,  0.543] # Puramem Flux, measured Perm = 0.685 LMH/bar

m3 = ["Dead-end", 4, 2000, 20, "[\"TFC\"]",
    "[\"Polymers of intrinsic microporosity\"]", None, "[\"Polyacrylonitrile\"]", "ISA", "[\"None\"]",
    "[\"Crosslinking\",\"Drying\"]","[Dimethylformamide (DMF)]", "Dipcoating", 96,
    55.5, 0.543] # PIM A 5% crosslinked, measured Perm = 0.625 LMH/bar

m4 = ["Dead-end", 4, 2000, 20, "[\"TFC\"]",
    "[\"Polymers of intrinsic microporosity\"]", None, "[\"Polyacrylonitrile\"]", "ISA", "[\"None\"]",
    "[\"Crosslinking\",\"Drying\"]","[Dimethylformamide (DMF)]", "Dipcoating", 79,
    50.5, 0.543] # PIM B 1% crosslinked, measured Perm = 0.475 LMH/bar

m5 = ["Dead-end", 4, 2000, 20, "[\"TFC\"]",
    "[\"Polyamine\"]", None, "[\"Polyetherimide\"]", "ISA", "[\"None\"]",
    "[\"Crosslinking\",\"Drying\"]","[Dimethylformamide (DMF)]", "Dipcoating", 79,
    20, 0.543] # PEBAX, measured Perm = 0.75 LMH/bar

# ['testConditions.filtrationMode', 'testConditions.hydraulicP','soluteChoice.solute1.concentration','testConditions.Temperature' ,  'structure', 
# 'chemistry','supportLayerChemistry', 'supportLayerChemistry', 'supportLayerType', 'supportLayer.postTreatment', 
# 'postTreatment', 'isaSupportLayer.solvent', 'topLayerDepositionMethod', 'characterizationResults.contactAngle',
# 'characterizationResults.totalThickness', 'solvent1.viscosity']

mem_names = ["Hereon PDMS", "Puramem", "Hereon PIM", "Hereon PEBAX"]
m_true = [0.07, 0.47, 0.625, 0.75] 

hereon_1 = pd.DataFrame([m1, m2, m4, m5], columns=X_test.columns, index=mem_names)
hereon_cat = hereon_1[categorical_cols]
hereon_encoded_cat = encoder.transform(hereon_cat)

#%%
hereon_encoded = pd.concat([pd.DataFrame(hereon_encoded_cat, index=mem_names,
                                    columns = X_test_categorical.columns), 
                       hereon_1[numerical_cols]], axis=1)

hereon_scaled = robust.transform(hereon_encoded)
# %%
m_pred = gradient_boost.predict(hereon_scaled)
m_pred_t = np.expm1(m_pred.reshape(1, -1))
m_pred_t =m_pred_t[0].tolist()

print(mem_names, m_pred_t)

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

# %%%
print("\n Extra Tree: ")
extra_tree = se.ExtraTreesRegressor(n_estimators=175, max_depth=19, min_samples_leaf=3, random_state=42, n_jobs=-1)
extra_tree.fit(X_train_scaled, y_train)
y_pred = extra_tree.predict(X_test_scaled)

# ---- evaluate model
train_R2 = extra_tree.score(X_train_scaled, y_train)
test_R2 = extra_tree.score(X_test_scaled, y_test)
mae =  mean_absolute_error(np.expm1(y_test), np.expm1(y_pred))

print("R2 train: ", round(train_R2, 2), " & R2 test: ", round(test_R2, 2), "& MAE:",  round(mae, 1))

# %%%% Tree Explainer
importances_rf = gradient_boost.feature_importances_
indices = np.argsort(importances_rf)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Gradient Boosting")
plt.bar(range(len(importances_rf)), importances_rf[indices], align="center")
plt.xticks(range(len(importances_rf)), [X_train_scaled.columns[i] for i in indices],
           rotation=45, ha='right')
plt.tight_layout()
plt.show()

# # %% XAI & feature analysis

# # ---- shap plot # auskommentiert für Schnelligkeit
# X_train_scaled = pd.DataFrame(X_train_scaled)
import random
import os
random.seed(42)
os.environ["PYTHONHASHSEED"] = "42"
np.random.seed(42)

explainer = shap.TreeExplainer(gradient_boost, X_train_scaled)
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
    X_with_cats = X_with_cats.replace("[\"Polysulfone; Sulfonated poly(ether ether) ketone\"]", "[\"Polysulfone; Sulfonated PEEK\"]")
    X_with_cats = X_with_cats.replace("[\"Pure solvent treatment\",\"Thermal treatment\", \"Crosslinking\"]",
                                      "[\"Pure solvent t.\",\"Thermal t.\",\n\"Crosslinking\"]")
    X_with_cats = X_with_cats.replace("[\"Gutter layer synthesis\",\"Thermal treatment\", \"Surface treatment\"]",
                                      "[\"Gutter layer synthesis\",\n\"Thermal t.\",\"Surface t.\"]")
    X_with_cats = X_with_cats.replace("[\"Crosslinking\",\"Gutter layer synthesis\"]","[\"Crosslinking\",\n\"Gutter layer synthesis\"]")
    X_with_cats = X_with_cats.replace("[\"Pure solvent treatment\",\"Thermal treatment\",\"Crosslinking\"]",
                                      "[\"Pure solvent treatment\",\n\"Thermal treatment\",\"Crosslinking\"]")
        
    # mean of SHAP values per category
    shap_cat_importance = shap_values_df.groupby(X_with_cats["shap_category"]).mean()

    print(shap_cat_importance[category_name].sort_values())
    colors = ['#1f77b4' if e >= 0 else '#ff0051' for e in shap_cat_importance[category_name].sort_values()]
    
    shap_cat_importance[category_name].sort_values().plot(kind="barh", color=colors)
    plt.ylabel("")
    plt.xlabel("SHAP value (impact on model output)")
    plt.xlim(-0.5, 0.65)
    plt.tight_layout()
    plt.show()
    
# ----- structure
category_plot(X_train, shap_values, "structure")

# ---- Mode
category_plot(X_train, shap_values, "testConditions.filtrationMode")

# ---- chemistry 
category_plot(X_train, shap_values, "chemistry")

# ---- support chemistry 
category_plot(X_train, shap_values, "supportLayerChemistry")

# ---- post treatment 
category_plot(X_train, shap_values, "postTreatment")

# ---- supportLayer.post treatment 
category_plot(X_train, shap_values, "supportLayer.postTreatment")

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

    # Target-Encoding (foldweise codieren)
    encoder = TargetEncoder()
    X_train_cat_enc = encoder.fit_transform(X_train_cat, y_train)
    X_test_cat_enc = encoder.transform(X_test_cat)

    # Zusammenfügen
    X_train_new = pd.concat([pd.DataFrame(X_train_cat_enc,
                                          index=X_train.index, columns=X_train_categorical.columns), X_train_num],
                            axis=1)
    X_test_new = pd.concat([pd.DataFrame(X_test_cat_enc,
                                          index=X_test.index, columns=X_test_categorical.columns), X_test_num],
                           axis=1)

    # Skalieren
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_new)
    X_test_scaled = scaler.transform(X_test_new)
    
    y_train = np.log1p(y_train.array.astype(float))
    y_test = np.log1p(y_test.array.astype(float))

    # Modelltraining
    gradient_boost.fit(X_train_scaled, y_train)

    # Vorhersage
    y_pred = gradient_boost.predict(X_test_scaled)

    # Ergebnisse sammeln
    y_true_all.extend(y_test)
    y_pred_all.extend(y_pred)
    
y_true_all = np.expm1(y_true_all)
y_pred_all = np.expm1(y_pred_all)

# %%
# ---- Plot
# plt.figure(figsize=(6, 6))
plt.scatter(y_true_all, y_pred_all, alpha=0.3)
plt.plot([min(y_true_all), max(y_true_all)],
         [min(y_true_all), max(y_true_all)],
         color='red', linestyle='--', label='Ideal')
# for i in range(len(mem_names)):
#     plt.text(m_true[i], m_pred_t[i], mem_names[i], color="black",
#             fontsize=9, horizontalalignment='right',
#             bbox=dict(alpha=0.4, facecolor="white", edgecolor="white", boxstyle='round,pad=-1'))
# plt.scatter(m_true, m_pred_t, color="red", alpha=0.6)
plt.xlabel("Measured Permeance [LHM/bar]")
plt.ylabel("Predicted Permeance [LHM/bar]")
plt.xscale("log")
plt.yscale("log")
plt.title("10-fold Cross-Validation: Prediction vs. Truth")
plt.legend()
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
plt.ylim(-max(y), max(y))
plt.tight_layout()
plt.show()

y_true, y_pred = np.array(y_true_all), np.array(y_pred_all)
mape = np.mean(np.abs((y_true[y_true != 0] - y_pred[y_true != 0]) / y_true[y_true != 0])) * 100
print(f"MAPE: {mape:.2f}%", f"STD: {std:.1f} LMH/bar", f"MAE: {mae:.1f} LMH/bar")

# %%
