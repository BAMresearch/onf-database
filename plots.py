# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 08:46:20 2025

@author: Glass
"""

import pandas as pd
from matplotlib import pyplot as plt

r2_mwco = pd.DataFrame(index=["R²(Train)", "R²(Test)"], 
                       data={"Linear Regression": [0.49, 0.53],
                             "Elastic Net":	[0.49, 0.52],	
                             "Principal Component \nAnalysis + \n k Nearest Neighbors": [0.79, 0.61],
                             "Random Forest": [0.90, 0.66],
                             "Gradient Boosting": [0.90, 0.68],
                             "Extra Tree": [0.86, 0.70]	
                             }).T
delta_mwco = r2_mwco["R²(Train)"]-r2_mwco["R²(Test)"]

delta_mwco.name = "R² Gap"

r2_mwco = pd.concat([r2_mwco, delta_mwco], axis=1)

r2_mwco.plot(kind="bar")
plt.ylim(0,1)
plt.show()

r2_perm =  pd.DataFrame(index=["R²(Train)", "R²(Test)"], 
                       data={"Linear Regression": [0.43, 0.45],
                             "Elastic Net":	[0.43, 0.45],	
                             "Principal Component \nAnalysis + \n k Nearest Neighbors": [0.80, 0.65],
                             "Random Forest": [0.89, 0.73],
                             "Gradient Boosting": [0.92, 0.74],
                             "Extra Tree": [0.86, 0.72]	
                             }).T

delta_perm = r2_perm["R²(Train)"]-r2_perm["R²(Test)"]

delta_perm.name = "R² Gap"

r2_perm = pd.concat([r2_perm, delta_perm], axis=1)
r2_perm.plot(kind="bar")
plt.ylim(0,1)
plt.show()