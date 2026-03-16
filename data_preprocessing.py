# -*- coding: utf-8 -*-
"""
Created on Mon May 26 16:00:32 2025

@author: Glass
"""
#%%
import pandas as pd

def preprocessing(data_file):
    """Function to extract relevant data from the original database.
    Provides several smaller dataframes that are chosen by parameter meaning. Excludes unneaded columns
    and columns that are fully empty. Is used for all training purposes. """
    # Importing the data
    srnf = pd.read_csv(data_file, sep=",")

    # find all data points that were rejected from the data base
    rejected = srnf["status"].str.contains("Rejected")

    # exclude columns with no or useless information
    new = srnf.drop(["_id", "dates.created", "report._id", "suspicious", "report.dates.created", "report.publicationDate",
                     "report.source", "report.title", "report.creator", "report.author.lastName", "report.link", "report.__v",
                     "characterizationResults.characterizationData", 
                     "status", "type", "__v", "published", "modification"],
                     axis=1)[~rejected]
    
    # convert all solvent1 concentrations to the same unit (all to µmol/L)
    set(srnf["soluteChoice.solute1.concentrationUnit"]) 
    # Out: {'g/L', 'mg/L', 'mmol/L', 'mol%', 'mol/L', nan, 'wt%', 'µg/L', 'µmol/L'}
    
    for index, unit in new["soluteChoice.solute1.concentrationUnit"].items():
        if unit == "mol/L":
            new["soluteChoice.solute1.concentration"][index]= new["soluteChoice.solute1.concentration"][index]*1e6
            new["soluteChoice.solute1.concentrationUnit"][index] = 'µmol/L'
        elif unit == "mmol/L":
            new["soluteChoice.solute1.concentration"][index]= new["soluteChoice.solute1.concentration"][index]*1e3
            new["soluteChoice.solute1.concentrationUnit"][index] = 'µmol/L'
        elif unit == "µg/L":
            new["soluteChoice.solute1.concentration"][index]= new["soluteChoice.solute1.concentration"][index]/new["soluteChoice.solute1.molecularWeight"][index]
            new["soluteChoice.solute1.concentrationUnit"][index] = 'µmol/L'
        elif unit == "mg/L":
            new["soluteChoice.solute1.concentration"][index]= new["soluteChoice.solute1.concentration"][index]/new["soluteChoice.solute1.molecularWeight"][index]*1e3
            new["soluteChoice.solute1.concentrationUnit"][index] = 'µmol/L'
        elif unit == "g/L":
            new["soluteChoice.solute1.concentration"][index]= new["soluteChoice.solute1.concentration"][index]/new["soluteChoice.solute1.molecularWeight"][index]*1e6
            new["soluteChoice.solute1.concentrationUnit"][index] = 'µmol/L'  

    # ---- find all data points with a permeance given
    perm_given = new.dropna(subset=["filtrationResults.solventPermeance"])

    # find all data points with amolecular weight cut-off (mwco) given
    mwco_given = new.dropna(subset=["filtrationResults.mwco"])
    
    # drop values not in the NF range
    mwco_given = mwco_given.drop(mwco_given[mwco_given["filtrationResults.mwco"]>1000].index)
    mwco_given = mwco_given.drop(mwco_given[mwco_given["filtrationResults.mwco"]<200].index)

    mwco_given = mwco_given.drop_duplicates(subset=["name", "testConditions.solvent1",
                                                   "testConditions.solvent2", "testConditions.filtrationMode"])

    perm_given = perm_given.drop(perm_given[perm_given["filtrationResults.solventPermeance"]>500].index)
    perm_given = perm_given.drop_duplicates(subset=["name", "testConditions.solvent1",
                                                    "testConditions.solvent2", "testConditions.filtrationMode"])

    # ---- find data point which have a permeance and a mwco reported
    both_given = new.dropna(subset=["filtrationResults.mwco", "filtrationResults.solventPermeance"])

    return srnf, perm_given, mwco_given, both_given

# Convert Solvent name to Viscosity
solvent_visco = {'1,4-dioxane': 1.177, # https://14d-1.itrcweb.org/appendix-b/
                 '1-Hexanol': 4.594, # https://pubs.acs.org/doi/10.1021/je060353z
                 '1-Methyl-2-pyrrolidinone (NMP)': 1.65, #https://www.sigmaaldrich.com/deepweb/assets/sigmaaldrich/product/documents/890/214/m6762pis.pdf?srsltid=AfmBOoqExprqBeHM-7kQduxiRLSPMTRySQdc655iinYhqTB4WYH7GRNe
                 '1-Pentanol': 3.77, # https://www.researchgate.net/publication/335118628_Viscosity_of_pentan-1-ol#:~:text=The%20kinematic%20viscosity%20data%20were%20fitted%20using,0.1%20MPa%20over%20the%20entire%20composition%20range.
                 '2-Butanol': 3.8, # https://pubchem.ncbi.nlm.nih.gov/compound/2-Butanol#section=Autoignition-Temperature
                 'Acetone': 0.306, # https://en.wikipedia.org/wiki/Acetone
                 'Acetonitrile': 0.334, # https://wiki.anton-paar.com/uk-en/acetonitrile/
                 'Butanol': 2.573, # https://en.wikipedia.org/wiki/1-Butanol
                 'Carbon Tetrachloride': 0.901, # https://en.wikipedia.org/wiki/Carbon_tetrachloride_(data_page)
                 'Chloroform': 0.542,# https://en.wikipedia.org/wiki/Chloroform_(data_page)
                 'Cyclohexane': 0.89, # https://en.wikipedia.org/wiki/Cyclohexane_(data_page)
                 'Cyclohexanone': 0.942, # https://wiki.anton-paar.com/de-de/cyclohexanon/
                 'Dichloromethane (DCM)': 0.413,# https://en.wikipedia.org/wiki/Dichloromethane
                 'Dimethylacetamide (DMAC)': 0.945, # https://pubs.acs.org/doi/10.1021/je050538q
                 'Dimethylformamide (DMF)': 0.802, # http://www.kianresin.com/userfiles/files/eastman_Dimethylformamide%20(DMF).pdf
                 'Dimethylsulfoxide (DMSO)': 1.99, # https://wiki.anton-paar.com/en/dimethyl-sulfoxide/
                 'Ethanol': 1.1, # https://wiki.anton-paar.com/de-de/ethanol/
                 'Ethyl acetate': 0.426, # https://en.wikipedia.org/wiki/Ethyl_acetate
                 'Heptane': 0.389, # https://en.wikipedia.org/wiki/Heptane
                 'Hexane': 0.3, # https://wiki.anton-paar.com/de-de/hexan/
                 'Isobutanol': 3.8, # https://en.wikipedia.org/wiki/Isobutanol
                 'Isopropanol (IPA)': 2.2, # https://www.eamaterials.com/products/economic/isopropyl-alcohol-labplus              
                 'Isopropyl acetate': 0.5, # https://www.hedinger.de/fileadmin/content/03_Geschaeftsbereiche/Industrie/Isoproyl_Acetate.pdf
                 'Methanol': 0.543, #https://wiki.anton-paar.com/en/methanol/
                 'Methyl acetate': 0.364, # https://pubchem.ncbi.nlm.nih.gov/compound/Methyl-Acetate#section=Decomposition
                 'Methyl ethyl ketone (MEK)': 0.4, # https://www.inchem.org/documents/icsc/icsc/eics0179.htm
                 'Methylene chloride': 0.413, # https://en.wikipedia.org/wiki/Dichloromethane
                 'N-butyl-2-pyrrolidinone (tamisolve)': 4.0, # https://www.carlroth.com/medias/SDB-1E8A-GB-EN.pdf?context=bWFzdGVyfHNlY3VyaXR5RGF0YXNoZWV0c3wyNTMyMjB8YXBwbGljYXRpb24vcGRmfGFEWTVMMmhqWkM4NU1UWTVPVFkxTWpFNU9EY3dMMU5FUWw4eFJUaEJYMGRDWDBWT0xuQmtaZ3w2YjM2NmFlZGJjNTljYjg4ZTgwN2NhNTQ0NDZjMzk4MjE0ZTEwMDkwMDAxYWJiNjYyOGFkMzg5MzZiM2JiYTg2
                 'Pentane': 0.214, # https://wiki.anton-paar.com/de-de/pentan/
                 'Propanol': 1.959, # https://en.wikipedia.org/wiki/1-Propanol
                 'Propylene glycol methyl ether acetate': 1.07, # https://www.carlroth.com/medias/SDB-7966-IE-EN.pdf?context=bWFzdGVyfHNlY3VyaXR5RGF0YXNoZWV0c3wyODM4ODh8YXBwbGljYXRpb24vcGRmfGFHSTJMMmc1WVM4NU1UYzVOREk0TlRVeU56TTBMMU5FUWw4M09UWTJYMGxGWDBWT0xuQmtaZ3w2Yjk0N2UwZDRlOGFjYzliZWRjYzE5MDAxZDIwMzRjNWRmMTI2ZGVmMDg3ZmEwMWIzNzcwOTI2YWE1NDI2YjFm
                 'Tetrahydrofuran (THF)': 0.46, # https://wiki.anton-paar.com/de-de/tetrahydrofuran/
                 'Toluene': 0.56, # https://wiki.anton-paar.com/en/toluene/
                 'water': 0.89 # https://wiki.anton-paar.com/en/water/
                  } # dynamic viscosity at 25°C in mPa*s

feature_list = ["testConditions.filtrationMode", "testConditions.solvent1", "testConditions.hydraulicP",
                "soluteChoice.solute1.category", "soluteChoice.solute1.concentration", "testConditions.temperature",
                "structure", "family", "chemistry", "chemistryOther",
                "supportLayerChemistry", "supportLayerType", "supportLayer.postTreatment",
                "postTreatment",
                "isaSupportLayer.castingThickness", "isaSupportLayer.solvent",
                "topLayerDepositionMethod",
                "tfnData.nanomaterialName",
                "characterizationResults.topLayerThickness", "characterizationResults.roughness",
                "characterizationResults.contactAngle", "characterizationResults.totalThickness"]

if __name__ == "__main__":
    srnf, perm_given, mwco_given, both_given = preprocessing("OMD_SRNF_2025-04-08.csv")

# %%
