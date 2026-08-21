import numpy as np
import torch
def defineboundaries(modeldata, scaledfeatures, device):
    # lower bounds
    x_fallowunscaled = modeldata[(modeldata['NEE_VUT_REF'] > np.percentile(modeldata['NEE_VUT_REF'], 85))
                                & (modeldata['NEE_VUT_REF'] > 0)]
    fallow_indices = x_fallowunscaled.index
    x_fallow = scaledfeatures.loc[fallow_indices]

    # upper bound
    x_upper_tavpd = scaledfeatures[(scaledfeatures['TA_F']  >= np.percentile(scaledfeatures['TA_F'], 99))
                                & (scaledfeatures['VPD_F'] >= np.percentile(scaledfeatures['VPD_F'], 99))]

    #tensors
    x_hotanddry = torch.tensor(x_upper_tavpd.values, dtype=torch.float32).to(device)
    y_hotanddry_target = torch.ones((x_hotanddry.shape[0], 1), dtype=torch.float32).to(device)

    x_fallow_t = torch.tensor(x_fallow.values, dtype=torch.float32).to(device)
    y_fallow_target = torch.zeros((x_fallow_t.shape[0], 1), dtype=torch.float32).to(device)

    x_pde = torch.tensor(scaledfeatures.values, dtype=torch.float32, requires_grad=True).to(device)

    #precip?
    P_mean = modeldata['P_F'].mean()
    P_std = modeldata['P_F'].std()
    threshold_p_scaled = (0 - P_mean) / P_std
    P_time_pde = torch.tensor(scaledfeatures['P_F'].values, dtype=torch.float32).to(device)

    return x_hotanddry, y_hotanddry_target, x_fallow_t, y_fallow_target, x_pde, P_time_pde, threshold_p_scaled
