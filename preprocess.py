import numpy as np
import pandas as pd

#the timestep needs to be scaled separately to encode information on the time of day
def fraction_time(tod):
    """
    All timestamps are put into fractions out of 48 (for 48 half-hours in a day)
    """
    tod = str(tod).zfill(4)
    hh = int(tod[:2])
    mm = int(tod[2:])
    return (hh + mm / 60) / 24

def featureengineering(sitedata):

    """
    Takes flux data and prepares it for PITE partitioning
    outputs:
        1. scaledfeatures: all input features for PITE scaled for a mean of 0 and std. deviation of 1
        2. modeldata: unscaled input features will the same indicies as scaledfeatures, used to create 
            some boundary conditions
    """
    
    variables = list(sitedata.columns.values)

    #checking for the following variables in the dataset
    targetfeatures = ['TA_F','VPD_F', 'USTAR', 'NETRAD', 'NEE_VUT_REF', 'P_F', 'H_F_MDS', 'G_F_MDS', 'SW_IN_POT', 'TS_F_MDS_1', 'SWC_F_MDS_1', 'LE_F_MDS']
    finalfeatures = []

    for var in targetfeatures:
        if var in variables:
            #skip features with more than 30% of missing data
            if sitedata[var].isna().mean() > 0.3:
                continue
            else:
                finalfeatures.append(var)

    #if all variables are found, we want to get rid of some due to multicollinearity
    if 'NETRAD' in finalfeatures and 'SW_IN_POT' in finalfeatures:
        finalfeatures.remove('NETRAD')
    if 'G_F_MDS' in finalfeatures:
        for var in ['TS_F_MDS_1', 'SWC_F_MDS_1']:
            if var in finalfeatures:
                finalfeatures.remove(var)

    #only want the desired variables to be used as input features
    modeldata = sitedata[finalfeatures].copy()
    modeldata['timestamp_asstring'] = sitedata['TIMESTAMP_START'].astype(str)
    modeldata['TOD'] = modeldata['timestamp_asstring'].str[8:]
    modeldata['TOD'] = modeldata['TOD'].astype(int)

    #rough calculation of ecosystem WUE
    #ET mm/s or Kg/m2/s
    modeldata['ET'] = modeldata['LE_F_MDS'] / ((2.501 - (0.00237 * modeldata['TA_F'] )) * 1e+06)
    #WUE g C hpa^0.5 / Kg H2O 
    modeldata['WUE'] = (-(modeldata['NEE_VUT_REF'] * 12.01 / 1000000) * np.sqrt(modeldata['VPD_F'])) / modeldata['ET']
    modeldata['WUE'] = modeldata['WUE'].replace([np.inf, -np.inf], 0)

    #dNEE/dt, 48 HHs, 96HHs, etc., wont ever be able to partition for the first four days of the dataset
    modeldata['dnee_d48t'] = modeldata['NEE_VUT_REF'].shift(48)
    modeldata['dnee_d96t'] = modeldata['NEE_VUT_REF'].shift(96)
    modeldata['dnee_d144t'] = modeldata['NEE_VUT_REF'].shift(144)
    modeldata['dnee_d192t'] = modeldata['NEE_VUT_REF'].shift(192)
    #then drop ET and LE as input feature
    modeldata = modeldata.drop(columns=['ET', 'LE_F_MDS', 'timestamp_asstring'])
    modeldata = modeldata.dropna()
    finalfeatures = modeldata.columns
    finalfeatures = finalfeatures.drop(['TOD'])

    #scale all features (needed for neural networks) to a mean of 0 and a standard deviation of 1
    scaledfeatures = pd.DataFrame()
    for var in finalfeatures:
        scaled = (modeldata[var] - modeldata[var].mean()) / (modeldata[var].std())
        scaledfeatures[var] = scaled

    #insert TOD as first column
    scaledfeatures.insert(0, 'TOD', modeldata['TOD'])

    scaledfeatures['TOD'] = scaledfeatures['TOD'].apply(fraction_time)

    return scaledfeatures, modeldata