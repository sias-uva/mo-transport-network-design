#%%
import os
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import json
import seaborn as sns
plt.rcParams.update({'font.size': 18})
from pymoo.indicators.hv import HV
from morl_baselines.common.performance_indicators import hypervolume

# %% LCN vs PCN
def gini(x, normalized=True):
    sorted_x = np.sort(x, axis=1)
    n = x.shape[1]
    cum_x = np.cumsum(sorted_x, axis=1, dtype=float)
    gi = (n + 1 - 2 * np.sum(cum_x, axis=1) / cum_x[:, -1]) / n
    if normalized:
        gi = gi * (n / (n - 1))
    return gi

all_objectives = [
    ## XIAN -- PCN vs LCN (ND, OPTMAX, NDMEAN)
    {'nr_groups': 2,
     'models': [
        {'name': 'PCN', 'dirs': ['pcn_xian_20231025_00_29_11.009308', 'pcn_xian_20231221_17_05_37.776445', 'pcn_xian_20231221_17_06_46.045896']},
        # {'name': 'PCN_HV', 'dirs': ['pcn_xian_20231025_14_57_58.229993']},
        {'name': 'LCN_ND', 'dirs': ['lcn_xian_20231024_15_11_42.594057', 'lcn_xian_20231220_13_53_13.263588', 'lcn_xian_20231220_13_53_29.501767']},
        {'name': 'LCN_OPTMAX', 'dirs': ['lcn_xian_20231031_19_50_00.078706', 'lcn_xian_20231220_14_44_33.522291', 'lcn_xian_20231220_14_44_51.798498']},
        {'name': 'LCN_NDMEAN', 'dirs': ['lcn_xian_20231031_18_20_59.173870', 'lcn_xian_20231220_14_47_26.220395', 'lcn_xian_20231220_14_47_57.984685']},
     ]},

     {'nr_groups': 3, 
     'models': [
        {'name': 'PCN', 'dirs': ['pcn_xian_20231026_19_57_12.259730', 'pcn_xian_20231221_17_13_28.741533', 'pcn_xian_20231221_17_16_07.823677']},
        {'name': 'LCN_ND', 'dirs': ['lcn_xian_20231027_10_35_51.656132', 'lcn_xian_20231220_14_59_53.519271', 'lcn_xian_20231220_15_00_10.564421']},
        {'name': 'LCN_OPTMAX', 'dirs': ['lcn_xian_20231030_17_47_36.629611', 'lcn_xian_20231220_15_02_57.746395', 'lcn_xian_20240105_14_47_03.507169']},
        {'name': 'LCN_NDMEAN', 'dirs': ['lcn_xian_20231028_01_07_53.729884', 'lcn_xian_20231220_17_03_12.414722', 'lcn_xian_20231220_17_03_18.736912']},
        # {'name': 'LCN_Lambda_0.0', 'dirs': ['lcn_xian_20231220_17_30_54.791998', 'lcn_xian_20231230_13_29_42.430687', 'lcn_xian_20231230_13_31_16.726711']},
        # {'name': 'LCN_Lambda_0.1', 'dirs': ['lcn_xian_20231220_19_23_57.505711', 'lcn_xian_20231230_13_32_50.819328', 'lcn_xian_20231230_16_11_31.788882']},
        # {'name': 'LCN_Lambda_0.2', 'dirs': ['lcn_xian_20231220_13_40_05.161123', 'lcn_xian_20231230_16_11_46.903158', 'lcn_xian_20231230_16_15_58.578610']},
        # {'name': 'LCN_Lambda_0.3', 'dirs': ['lcn_xian_20231220_17_31_54.882410', 'lcn_xian_20231230_16_26_58.092950', 'lcn_xian_20231230_16_51_50.555284']},
        # {'name': 'LCN_Lambda_0.4', 'dirs': ['lcn_xian_20231220_14_57_49.642641', 'lcn_xian_20231230_16_25_40.149452', 'lcn_xian_20231230_16_25_45.420788']},
        # {'name': 'LCN_Lambda_0.5', 'dirs': ['lcn_xian_20231220_18_18_21.304292', 'lcn_xian_20231231_18_38_25.815554', 'lcn_xian_20231231_18_38_32.040668']},
        # {'name': 'LCN_Lambda_0.6', 'dirs': ['lcn_xian_20231220_19_56_38.709599', 'lcn_xian_20231231_18_39_11.034438', 'lcn_xian_20231231_18_53_06.497753']},
        # {'name': 'LCN_Lambda_0.7', 'dirs': ['lcn_xian_20231220_14_05_26.124152', 'lcn_xian_20231231_18_53_39.088964', 'lcn_xian_20231231_18_53_43.368080']},
        # {'name': 'LCN_Lambda_0.8', 'dirs': ['lcn_xian_20231221_15_02_14.233385', 'lcn_xian_20231231_19_05_50.681596', 'lcn_xian_20231231_19_05_58.900443']},
        # {'name': 'LCN_Lambda_0.9', 'dirs': ['lcn_xian_20231221_17_39_23.693903', 'lcn_xian_20231231_19_06_35.791660', 'lcn_xian_20231231_19_22_46.233640']},
        # {'name': 'LCN_Lambda_1.0', 'dirs': ['lcn_xian_20231221_10_41_50.036536', 'lcn_xian_20231231_19_23_02.379536', 'lcn_xian_20231231_19_23_46.110507']},

        ## Interpolate3
        {'name': 'LCN_Lambda_0.0', 'dirs': ['lcn_xian_20240105_10_42_52.606690', 'lcn_xian_20240105_10_17_58.219304', 'lcn_xian_20240106_16_16_45.800515']},
        {'name': 'LCN_Lambda_0.1', 'dirs': ['lcn_xian_20240105_10_42_56.580745', 'lcn_xian_20240105_10_18_06.591934', 'lcn_xian_20240106_16_27_27.764975']},
        {'name': 'LCN_Lambda_0.2', 'dirs': ['lcn_xian_20240105_10_43_03.423714', 'lcn_xian_20240105_10_18_19.858721', 'lcn_xian_20240106_16_30_02.294221']},
        {'name': 'LCN_Lambda_0.3', 'dirs': ['lcn_xian_20240105_10_56_23.761950', 'lcn_xian_20240106_15_50_19.671774', 'lcn_xian_20240106_16_31_50.850046']},
        {'name': 'LCN_Lambda_0.4', 'dirs': ['lcn_xian_20240105_10_56_39.236767', 'lcn_xian_20240106_15_50_41.058354', 'lcn_xian_20240106_16_37_21.072217']},
        {'name': 'LCN_Lambda_0.5', 'dirs': ['lcn_xian_20240105_10_57_50.800221', 'lcn_xian_20240106_15_51_07.942022', 'lcn_xian_20240106_16_44_32.902128']},
        {'name': 'LCN_Lambda_0.6', 'dirs': ['lcn_xian_20240110_14_21_50.354585', 'lcn_xian_20240110_12_09_02.587302', 'lcn_xian_20240110_12_09_03.257627']}, 
        {'name': 'LCN_Lambda_0.7', 'dirs': ['lcn_xian_20240105_11_32_45.469236', 'lcn_xian_20240106_16_04_08.057850', 'lcn_xian_20240106_17_00_27.340156']},
        {'name': 'LCN_Lambda_0.8', 'dirs': ['lcn_xian_20231221_13_04_18.503687', 'lcn_xian_20240110_11_58_16.516068', 'lcn_xian_20240110_11_58_19.538591']},
        {'name': 'LCN_Lambda_0.9', 'dirs': ['lcn_xian_20240105_12_06_39.994789', 'lcn_xian_20240106_16_07_06.454882', 'lcn_xian_20240106_17_01_04.146726']},
        {'name': 'LCN_Lambda_1.0', 'dirs': ['lcn_xian_20231221_15_06_26.599246', 'lcn_xian_20240110_11_39_14.608825', 'lcn_xian_20240110_11_39_18.362828']},

     ]},

     {'nr_groups': 4,
     'models': [
        {'name': 'PCN', 'dirs': ['pcn_xian_20231221_23_50_49.667392', 'pcn_xian_20231222_10_07_26.733823', 'pcn_xian_20231222_10_07_34.709387']},
        {'name': 'LCN_ND', 'dirs': ['lcn_xian_20231031_20_34_46.319655', 'lcn_xian_20231221_16_46_35.929493', 'lcn_xian_20231221_16_46_49.155431']},
        {'name': 'LCN_OPTMAX', 'dirs': ['lcn_xian_20231031_20_57_36.686604', 'lcn_xian_20231221_16_51_28.384988', 'lcn_xian_20231221_17_02_48.227904']},
        {'name': 'LCN_NDMEAN', 'dirs': ['lcn_xian_20231031_14_06_09.833760', 'lcn_xian_20231221_18_00_15.460010', 'lcn_xian_20231221_18_00_24.285662']},
     ]},

     {'nr_groups': 5,
     'models': [
        {'name': 'PCN', 'dirs': ['pcn_xian_20231222_06_43_24.520264', 'pcn_xian_20231222_10_12_34.260974', 'pcn_xian_20231222_10_22_47.341518']},
        {'name': 'LCN_ND', 'dirs': ['lcn_xian_20231031_15_37_43.607061', 'lcn_xian_20231221_18_07_11.697683', 'lcn_xian_20231221_18_13_20.184699']},
        {'name': 'LCN_OPTMAX', 'dirs': ['lcn_xian_20231031_17_22_15.390900', 'lcn_xian_20231221_18_16_19.629580', 'lcn_xian_20231221_18_17_27.264538']},
        {'name': 'LCN_NDMEAN', 'dirs': ['lcn_xian_20231101_03_27_35.010116', 'lcn_xian_20231222_10_35_27.721735', 'lcn_xian_20231222_10_36_54.179537']},
     ]},

     {'nr_groups': 6,
     'models': [
        {'name': 'PCN', 'dirs': ['pcn_xian_20231222_05_18_23.713060', 'pcn_xian_20231222_10_42_08.861957', 'pcn_xian_20231222_13_51_29.156008']},
        {'name': 'LCN_ND', 'dirs': ['lcn_xian_20231101_20_50_50.262098', 'lcn_xian_20231222_13_55_28.792287', 'lcn_xian_20231222_13_55_14.499330']},
        {'name': 'LCN_OPTMAX', 'dirs': ['lcn_xian_20231101_23_59_04.892749', 'lcn_xian_20231222_14_09_29.475197', 'lcn_xian_20231222_14_20_13.646705']},
        {'name': 'LCN_NDMEAN', 'dirs': ['lcn_xian_20231102_00_39_34.815751', 'lcn_xian_20231222_14_21_49.604156', 'lcn_xian_20231222_14_30_21.727029']},
        # {'name': 'LCN_Lambda_0.0', 'dirs': ['lcn_xian_20231223_03_51_02.035482', 'lcn_xian_20240103_12_39_41.243528', 'lcn_xian_20240103_12_39_50.651435']},
        # {'name': 'LCN_Lambda_0.1', 'dirs': ['lcn_xian_20231222_15_35_54.370796', 'lcn_xian_20240103_12_43_21.486401', 'lcn_xian_20240103_12_58_03.785256']},
        # {'name': 'LCN_Lambda_0.2', 'dirs': ['lcn_xian_20231223_02_22_19.068841', 'lcn_xian_20240103_12_59_05.165989', 'lcn_xian_20240103_12_59_13.866006']},
        # {'name': 'LCN_Lambda_0.3', 'dirs': ['lcn_xian_20231222_23_50_40.792027', 'lcn_xian_20240103_13_22_20.152186', 'lcn_xian_20240103_13_22_27.755196']},
        # {'name': 'LCN_Lambda_0.4', 'dirs': ['lcn_xian_20231223_03_50_28.413054', 'lcn_xian_20240103_13_24_46.625112', 'lcn_xian_20240103_15_22_06.047546']},
        # {'name': 'LCN_Lambda_0.5', 'dirs': ['lcn_xian_20231223_03_38_58.678816', 'lcn_xian_20240103_15_23_58.995186', 'lcn_xian_20240103_15_24_04.417342']},
        # {'name': 'LCN_Lambda_0.6', 'dirs': ['lcn_xian_20240103_13_15_07.614476', 'lcn_xian_20240105_09_54_58.256737', 'lcn_xian_20240105_09_55_02.372001']},
        # {'name': 'LCN_Lambda_0.7', 'dirs': ['lcn_xian_20231229_23_13_00.075207', 'lcn_xian_20240103_15_38_14.852001', 'lcn_xian_20240103_15_41_31.690977']},
        # {'name': 'LCN_Lambda_0.8', 'dirs': ['lcn_xian_20231227_10_30_37.066763', 'lcn_xian_20240103_15_41_55.425827', 'lcn_xian_20240103_15_56_10.839843']},
        # {'name': 'LCN_Lambda_0.9', 'dirs': ['lcn_xian_20231227_14_31_33.262752', 'lcn_xian_20240103_16_06_08.909584', 'lcn_xian_20240103_16_08_55.025827']},
        # {'name': 'LCN_Lambda_1.0', 'dirs': ['lcn_xian_20231230_17_58_12.046057', 'lcn_xian_20240103_16_09_32.898444', 'lcn_xian_20240103_17_06_29.331688']},

        ## Interpolate3
        {'name': 'LCN_Lambda_0.0', 'dirs': ['lcn_xian_20240105_12_43_13.935811', 'lcn_xian_20240106_17_10_32.866164', 'lcn_xian_20240108_09_35_47.242270']},
        {'name': 'LCN_Lambda_0.1', 'dirs': ['lcn_xian_20240105_12_43_30.919111', 'lcn_xian_20240106_17_11_02.437616', 'lcn_xian_20240108_10_00_55.067102']},
        {'name': 'LCN_Lambda_0.2', 'dirs': ['lcn_xian_20240105_12_43_48.156219', 'lcn_xian_20240106_17_11_37.252309', 'lcn_xian_20240108_10_01_09.645822']},
        {'name': 'LCN_Lambda_0.3', 'dirs': ['lcn_xian_20240105_13_21_45.148803', 'lcn_xian_20240106_17_30_47.711754', 'lcn_xian_20240108_10_02_14.761305']},
        {'name': 'LCN_Lambda_0.4', 'dirs': ['lcn_xian_20240105_13_21_56.386628', 'lcn_xian_20240106_17_31_00.054289', 'lcn_xian_20240108_10_42_52.703536']},
        {'name': 'LCN_Lambda_0.5', 'dirs': ['lcn_xian_20240105_13_22_08.118364', 'lcn_xian_20240106_17_31_33.566415', 'lcn_xian_20240108_10_43_12.203160']},
        {'name': 'LCN_Lambda_0.6', 'dirs': ['lcn_xian_20240105_13_50_00.004743', 'lcn_xian_20240106_17_51_42.304328', 'lcn_xian_20240108_10_43_49.043968']},
        {'name': 'LCN_Lambda_0.7', 'dirs': ['lcn_xian_20240105_14_16_16.501580', 'lcn_xian_20240106_17_51_52.976563', 'lcn_xian_20240108_11_07_46.944352']},
        {'name': 'LCN_Lambda_0.8', 'dirs': ['lcn_xian_20240105_13_50_19.726513', 'lcn_xian_20240106_17_52_16.733108', 'lcn_xian_20240108_11_08_03.770924']},
        {'name': 'LCN_Lambda_0.9', 'dirs': ['lcn_xian_20240105_14_16_31.714213', 'lcn_xian_20240108_09_34_55.039430', 'lcn_xian_20240108_11_09_02.291641']},
        {'name': 'LCN_Lambda_1.0', 'dirs': ['lcn_xian_20240105_13_50_34.828994', 'lcn_xian_20240108_09_35_03.251690', 'lcn_xian_20240108_11_27_56.174596']},

     ]},

     {'nr_groups': 7,
     'models': [
        {'name': 'PCN', 'dirs': ['pcn_xian_20231222_08_55_25.941723', 'pcn_xian_20231222_14_57_23.739874', 'pcn_xian_20231222_14_57_31.976834']},
        {'name': 'LCN_ND', 'dirs': ['lcn_xian_20231101_15_20_51.805160', 'lcn_xian_20231222_14_57_56.348703', 'lcn_xian_20231222_15_30_31.084748']},
        {'name': 'LCN_OPTMAX', 'dirs': ['lcn_xian_20231101_21_16_46.111166', 'lcn_xian_20231222_15_31_35.361593', 'lcn_xian_20231222_15_31_44.709426']},
        {'name': 'LCN_NDMEAN', 'dirs': ['lcn_xian_20231101_11_07_18.996482', 'lcn_xian_20231222_16_01_12.895824', 'lcn_xian_20231222_16_01_22.677730']},
     ]},

     {'nr_groups': 8,
     'models': [
        {'name': 'PCN', 'dirs': ['pcn_xian_20231222_12_09_13.742678', 'pcn_xian_20231227_15_28_42.251814', 'pcn_xian_20231227_15_28_57.920106']},
        {'name': 'LCN_ND', 'dirs': ['lcn_xian_20231104_02_19_49.123322', 'lcn_xian_20231222_16_02_26.783621', 'lcn_xian_20231227_15_33_05.170445']},
        {'name': 'LCN_OPTMAX', 'dirs': ['lcn_xian_20231104_09_50_45.815954', 'lcn_xian_20231227_15_54_02.878568', 'lcn_xian_20231227_15_54_13.305009']},
        {'name': 'LCN_NDMEAN', 'dirs': ['lcn_xian_20231104_00_20_11.895885', 'lcn_xian_20231227_15_55_14.416467', 'lcn_xian_20231227_16_23_37.917198']},
     ]},

     {'nr_groups': 9,
     'models': [
        {'name': 'PCN', 'dirs': ['pcn_xian_20231223_06_22_56.997405', 'pcn_xian_20231227_17_08_40.492328', 'pcn_xian_20231227_17_09_24.836241']},
        {'name': 'LCN_ND', 'dirs': ['lcn_xian_20231104_09_15_53.872422', 'lcn_xian_20231227_16_25_38.142788', 'lcn_xian_20231227_16_25_44.687983']},
        {'name': 'LCN_OPTMAX', 'dirs': ['lcn_xian_20231103_11_44_11.962378', 'lcn_xian_20231227_17_11_46.744104', 'lcn_xian_20231227_17_37_05.654135']},
        {'name': 'LCN_NDMEAN', 'dirs': ['lcn_xian_20231103_21_09_16.023286', 'lcn_xian_20231227_17_37_21.273219', 'lcn_xian_20231227_17_39_32.436277']},
     ]},

     {'nr_groups': 10,
     'models': [
        {'name': 'PCN', 'dirs': ['pcn_xian_20231230_09_50_11.556705', 'pcn_xian_20231230_16_48_04.893434', 'pcn_xian_20231230_16_48_16.788576',]},
        {'name': 'LCN_ND', 'dirs': ['lcn_xian_20231115_20_13_10.771362', 'lcn_xian_20231227_18_12_34.258525', 'lcn_xian_20231227_18_13_21.258877']},
        {'name': 'LCN_OPTMAX', 'dirs': ['lcn_xian_20231107_23_24_04.859340', 'lcn_xian_20231227_18_15_40.794295', 'lcn_xian_20231228_19_49_15.985623']},
        {'name': 'LCN_NDMEAN', 'dirs': ['lcn_xian_20231106_10_49_36.941661', 'lcn_xian_20231228_19_50_36.283374', 'lcn_xian_20231228_19_54_38.446733']},
     ]},
]

# AMSTERDAM - 10 Stations
all_objectives = [
    {'nr_groups': 2,
     'models': [
        {'name': 'PCN', 'dirs': ['pcn_amsterdam_20240108_13_01_00.778807', 'pcn_amsterdam_20240108_17_20_07.542513', 'pcn_amsterdam_20240108_17_26_09.093184']},
        {'name': 'LCN_ND', 'dirs': ['lcn_amsterdam_20240107_10_11_32.394826', 'lcn_amsterdam_20240108_17_38_08.076197', 'lcn_amsterdam_20240108_17_46_19.928274']},
        {'name': 'LCN_OPTMAX', 'dirs': ['lcn_amsterdam_20240108_12_26_27.169590', 'lcn_amsterdam_20240109_12_49_43.217241', 'lcn_amsterdam_20240109_12_49_50.515894']},
        {'name': 'LCN_NDMEAN', 'dirs': ['lcn_amsterdam_20240112_00_30_56.974278', 'lcn_amsterdam_20240112_15_54_08.068799', 'lcn_amsterdam_20240112_16_13_34.153684']},
     ]},

     {'nr_groups': 3, 
     'models': [
        {'name': 'PCN', 'dirs': ['pcn_amsterdam_20240109_21_39_25.509057', 'pcn_amsterdam_20240110_16_05_16.003156', 'pcn_amsterdam_20240110_16_05_32.771992']},
        {'name': 'LCN_ND', 'dirs': ['lcn_amsterdam_20240107_18_10_19.466760', 'lcn_amsterdam_20240108_20_58_24.047456', 'lcn_amsterdam_20240108_21_40_38.580133']},
        {'name': 'LCN_OPTMAX', 'dirs': ['lcn_amsterdam_20240108_21_41_36.541590', 'lcn_amsterdam_20240110_10_47_17.814055', 'lcn_amsterdam_20240110_10_47_22.195133']},
        {'name': 'LCN_NDMEAN', 'dirs': ['lcn_amsterdam_20240112_06_42_28.137646', 'lcn_amsterdam_20240113_22_47_13.821275', 'lcn_amsterdam_20240113_22_47_17.305739']},
     ]},

     {'nr_groups': 4,
     'models': [
        {'name': 'PCN', 'dirs': ['pcn_amsterdam_20240104_18_09_06.462886', 'pcn_amsterdam_20240108_12_58_10.631250', 'pcn_amsterdam_20240108_12_58_16.652467']},
        {'name': 'LCN_ND', 'dirs': ['lcn_amsterdam_20240108_20_58_49.749453', 'lcn_amsterdam_20240108_21_39_47.553554', 'lcn_amsterdam_20240108_22_08_07.191329']},
        {'name': 'LCN_OPTMAX', 'dirs': ['lcn_amsterdam_20240109_18_03_40.856486', 'lcn_amsterdam_20240111_11_15_13.976013', 'lcn_amsterdam_20240111_11_15_18.613571']},
        {'name': 'LCN_NDMEAN', 'dirs': ['lcn_amsterdam_20240112_12_57_51.113202', 'lcn_amsterdam_20240113_22_50_41.907029']},
     ]},

     {'nr_groups': 5,
     'models': [
        {'name': 'PCN', 'dirs': ['pcn_amsterdam_20240103_20_27_53.063974', 'pcn_amsterdam_20240108_13_58_08.926091', 'pcn_amsterdam_20240108_15_38_47.417417']},
        {'name': 'LCN_ND', 'dirs': ['lcn_amsterdam_20240107_16_21_51.303601', 'lcn_amsterdam_20240108_22_09_31.479469', 'lcn_amsterdam_20240108_22_13_03.190803']},
        {'name': 'LCN_OPTMAX', 'dirs': ['lcn_amsterdam_20240109_15_09_24.960638', 'lcn_amsterdam_20240111_11_23_19.171292']},
        {'name': 'LCN_NDMEAN', 'dirs': ['lcn_amsterdam_20240113_07_01_47.678304']},
     ]},

     {'nr_groups': 6,
     'models': [
        {'name': 'PCN', 'dirs': ['pcn_amsterdam_20240105_18_12_44.252718', 'pcn_amsterdam_20240108_16_30_15.035406', 'pcn_amsterdam_20240108_16_30_18.519311']},
        {'name': 'LCN_ND', 'dirs': ['lcn_amsterdam_20240107_22_04_25.859926', 'lcn_amsterdam_20240109_09_28_20.393661', 'lcn_amsterdam_20240109_09_28_24.902641']},
        {'name': 'LCN_OPTMAX', 'dirs': ['lcn_amsterdam_20240110_23_55_56.733360', 'lcn_amsterdam_20240112_14_38_37.581301', 'lcn_amsterdam_20240112_14_38_43.749497']},
        {'name': 'LCN_NDMEAN', 'dirs': []},
     ]},

     {'nr_groups': 7,
     'models': [
        {'name': 'PCN', 'dirs':        ['pcn_amsterdam_20240104_08_25_38.900589', 'pcn_amsterdam_20240108_16_33_52.512832', 'pcn_amsterdam_20240108_17_34_48.315408']},
        {'name': 'LCN_ND', 'dirs':     ['lcn_amsterdam_20240107_22_24_36.910516', 'lcn_amsterdam_20240109_09_33_21.491671', 'lcn_amsterdam_20240109_10_18_44.316108']},
        {'name': 'LCN_OPTMAX', 'dirs': []},
        {'name': 'LCN_NDMEAN', 'dirs': []},
     ]},

     {'nr_groups': 8,
     'models': [
        {'name': 'PCN', 'dirs':        ['pcn_amsterdam_20240104_21_20_01.222785', 'pcn_amsterdam_20240108_17_59_10.349946', 'pcn_amsterdam_20240108_18_03_27.554833']},
        {'name': 'LCN_ND', 'dirs':     ['lcn_amsterdam_20240108_19_18_01.962049', 'lcn_amsterdam_20240112_13_28_11.272913', 'lcn_amsterdam_20240112_13_28_13.723613']},
        {'name': 'LCN_OPTMAX', 'dirs': ['lcn_amsterdam_20240109_19_12_06.840031', 'lcn_amsterdam_20240112_15_52_13.230587', 'lcn_amsterdam_20240112_15_52_24.219015']},
        {'name': 'LCN_NDMEAN', 'dirs': []},
     ]},

     {'nr_groups': 9,
     'models': [
        {'name': 'PCN', 'dirs':        ['pcn_amsterdam_20240105_15_08_43.307181', 'pcn_amsterdam_20240108_18_29_39.047196', 'pcn_amsterdam_20240108_19_02_17.164682']},
        {'name': 'LCN_ND', 'dirs':     ['lcn_amsterdam_20240111_10_07_28.175932', 'lcn_amsterdam_20240111_10_07_28.283603', 'lcn_amsterdam_20240111_10_07_32.119925']},
        {'name': 'LCN_OPTMAX', 'dirs': []},
        {'name': 'LCN_NDMEAN', 'dirs': []},
     ]},

     {'nr_groups': 10,
     'models': [
        {'name': 'PCN', 'dirs':        ['pcn_amsterdam_20240106_06_59_51.676341', 'pcn_amsterdam_20240108_19_05_38.185490', 'pcn_amsterdam_20240108_20_57_34.062783']}, 
        {'name': 'LCN_ND', 'dirs':     ['lcn_amsterdam_20240110_15_57_25.825464', 'lcn_amsterdam_20240112_13_31_09.736475', 'lcn_amsterdam_20240112_14_37_46.363796']},
        {'name': 'LCN_OPTMAX', 'dirs': []},
        {'name': 'LCN_NDMEAN', 'dirs': []},
     ]},
]

all_results = pd.DataFrame()
REQ_SEEDS = 3 # to control if a model was not run for sufficient seeds
for oidx, objective in enumerate(all_objectives):
    nr_groups = objective['nr_groups']
    ref_point = np.array([0] * nr_groups)
    models_to_plot = pd.DataFrame(objective['models'])

    results_by_objective = {}
    groups = None
    for i, model_name in enumerate(models_to_plot['name'].unique()):
        models = models_to_plot[models_to_plot['name'] == model_name].to_dict('records')
        if len(models[0]['dirs']) < REQ_SEEDS:
            print(f"!WARNING! {objective['nr_groups']} nrgroups, {model_name} does not have enough seeds (has {len(models[0]['dirs'])}, while {REQ_SEEDS} are required)")

        for j, model in enumerate(models):
            # metrics = pd.read_csv(f"./results/{model['dir']}/metrics.csv")
            # Read the content of the output file
            results_by_objective[model_name] = {'gini': [], 'total_efficiency': [], 'avg_efficiency': [], 'hv': [], 'sen_welfare': [], 'nash_welfare': [], 'avg_per_group': []}
            for i in range(len(model['dirs'])):
                # Check if the output file exists
                if not os.path.exists(f"./results/{model['dirs'][i]}/output.txt"):
                    print(f"!WARNING! {model_name} - {model['dirs'][i]} does not have output.txt file")
                    continue

                with open(f"./results/{model['dirs'][i]}/output.txt", "r") as file:
                    output = file.read()
                    fronts = json.loads(output)['best_front_r']
                    groups = len(fronts[0])
                    if groups != nr_groups:
                        print(f"!ERROR! {objective['nr_groups']} nrgroups, {model_name} has {groups} groups, while {nr_groups} are required")

                    gini_index = gini(np.array(fronts))
                    total_efficiency = np.sum(fronts, axis=1)
                    avg_efficiency = np.mean(fronts, axis=1)
                    # if model['dirs'][i] == 'lcn_xian_20240110_12_09_02.587302':
                        # print('asdas')
                    hv = hypervolume(ref_point, fronts)
                    nash_welfare = np.prod(fronts, axis=1)
                    results_by_objective[model_name]['fronts'] = fronts
                    results_by_objective[model_name]['total_efficiency'] = results_by_objective[model_name]['total_efficiency'] + total_efficiency.tolist()
                    results_by_objective[model_name]['avg_efficiency'] = results_by_objective[model_name]['avg_efficiency'] + avg_efficiency.tolist()
                    results_by_objective[model_name]['hv'] = results_by_objective[model_name]['hv'] + [hv]
                    results_by_objective[model_name]['gini'] = results_by_objective[model_name]['gini'] + gini_index.tolist()
                    results_by_objective[model_name]['sen_welfare'] = results_by_objective[model_name]['sen_welfare'] + (total_efficiency * (1-gini_index)).tolist()
                    results_by_objective[model_name]['nash_welfare'] = results_by_objective[model_name]['nash_welfare'] + nash_welfare.tolist()
                    results_by_objective[model_name]['avg_per_group'] = results_by_objective[model_name]['avg_per_group'] + np.mean(fronts, axis=0).tolist()
                    
        # results_by_objective[model_name]['lambda'] = model['lambda'] if 'lambda' in model else ''
    # Quite a hacky way to get the results in a dataframe, but didn't have time to do it properly (thanks copilot)
    # Convert all_results to a dataframe, with columns 'model', 'metric', 'value', and each row is a different value and not a list
    # results_by_objective = pd.DataFrame([(name, metric, value) for name in results_by_objective.keys() for metric in results_by_objective[name].keys() for value in results_by_objective[name][metric]], columns=['model', 'metric', 'value'])
    # Convert all_results to a dataframe, with columns 'model', 'lambda; 'metric', 'value', and each row is a different value and not a list
    results_by_objective = pd.DataFrame([(name, model['lambda'] if 'lambda' in model else None, metric, value) for name in results_by_objective.keys() 
                                         for metric in results_by_objective[name].keys() 
                                         for value in results_by_objective[name][metric]], columns=['model', 'lambda', 'metric', 'value'])
    results_by_objective['nr_groups'] = nr_groups
    results_by_objective['lambda'] = results_by_objective[results_by_objective['model'].str.contains('Lambda')]['model'].str.split('_').str[-1].astype(float)
    results_by_objective.loc[results_by_objective['model'].str.contains('Lambda'), 'model'] = 'LCN_Lambda'
    all_results = pd.concat([all_results, results_by_objective])
    
#%%
# Plot Total Efficiency, Gini Index, Sen Welfare for PCN vs LCN (ND, OPTMAX, NDMEAN)
fig, axs = plt.subplots(4, 1, figsize=(10, 12))
pcnvlcn = all_results[all_results['model'].isin(['PCN', 'LCN_ND', 'LCN_OPTMAX', 'LCN_NDMEAN'])]
pcnvlcn.loc[pcnvlcn['model'] == 'LCN_ND', 'model'] = 'LCN'
pcnvlcn.loc[pcnvlcn['model'] == 'LCN_OPTMAX', 'model'] = 'LCN-Redist'
pcnvlcn.loc[pcnvlcn['model'] == 'LCN_NDMEAN', 'model'] = 'LCN-Mean'

colors = ["#BFBFBF", "#1A85FF", "#E66100", "#D41159"]
sns.set_palette(sns.color_palette(colors))
LINEWIDTH = 1.5

hv = pcnvlcn[pcnvlcn['metric'] == 'hv']
# hv['value'] = hv.groupby('nr_groups')['value'].transform(lambda x: x / x.max())
# Normalize hv['value'] 
hv['value'] = hv.groupby('nr_groups')['value'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))
hv['value'] = hv['value'].fillna(0)

sns.boxplot(data=hv, x="nr_groups", y="value", hue="model", ax=axs[0], legend=False, linewidth=LINEWIDTH)
axs[0].set_title('Normalized Hypervolume')
axs[0].set_xlabel(None)
axs[0].set_ylabel(None)

eff = pcnvlcn[pcnvlcn['metric'] == 'total_efficiency']
eff['value'] = eff.groupby('nr_groups')['value'].transform(lambda x: x / x.max())
bxplot = sns.boxplot(data=eff, x="nr_groups", y="value", hue="model", ax=axs[1], legend='brief', linewidth=LINEWIDTH)
bxplot.legend_.set_title(None)
bxplot.legend(fontsize=12)
axs[1].set_title('Total Efficiency')
axs[1].set_xlabel(None)
axs[1].set_ylabel(None)

gini_ = pcnvlcn[pcnvlcn['metric'] == 'gini']
sns.boxplot(data=gini_, x="nr_groups", y="value", hue="model", ax=axs[2], legend=None, linewidth=LINEWIDTH)
axs[2].set_title('Gini')
axs[2].set_xlabel(None)
axs[2].set_ylabel(None)

sen_welfare = pcnvlcn[pcnvlcn['metric'] == 'sen_welfare']
sen_welfare['value'] = sen_welfare.groupby('nr_groups')['value'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))
sns.boxplot(data=sen_welfare, x="nr_groups", y="value", hue="model", ax=axs[3], legend=False, linewidth=LINEWIDTH)
axs[3].set_title('Normalized Sen Welfare (Efficiency * (1 - Gini Index))')
axs[3].set_xlabel('Number of Groups')
axs[3].set_ylabel(None)
fig.tight_layout()

# %%

#%% Plot Total Efficiency, Gini Index, Sen Welfare for lambda-LCN (0.0-1.0)
NR_GROUPS_TO_PLOT = 3
colors = ["#1A85FF", "#E66100", "#D41159", "#BFBFBF"]
sns.set_palette(sns.color_palette(colors))
LINEWIDTH = 2

# Get the results for PCN, to compare them to the lambda-LCN results
pcn_eff = all_results[(all_results['model']=='PCN') & (all_results['nr_groups']==NR_GROUPS_TO_PLOT) & (all_results['metric']=='total_efficiency')]
pcn_hv = all_results[(all_results['model']=='PCN') & (all_results['nr_groups']==NR_GROUPS_TO_PLOT) & (all_results['metric']=='hv')]
pcn_gini = all_results[(all_results['model']=='PCN') & (all_results['nr_groups']==NR_GROUPS_TO_PLOT) & (all_results['metric']=='gini')]
pcn_sen_welfare = all_results[(all_results['model']=='PCN') & (all_results['nr_groups']==NR_GROUPS_TO_PLOT) & (all_results['metric']=='sen_welfare')]

fig, axs = plt.subplots(4, 1, figsize=(12, 14))
lambda_lcn = all_results[(all_results['model'].isin(['LCN_Lambda'])) & (all_results['lambda'].isin([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]))]
lambda_lcn = lambda_lcn[lambda_lcn['nr_groups'] == NR_GROUPS_TO_PLOT]

hyperv = lambda_lcn[lambda_lcn['metric'] == 'hv']
sns.boxplot(data=hyperv, x="lambda", y="value", ax=axs[0], linewidth=LINEWIDTH)
axs[0].set_title('Hypervolume')
axs[0].set_ylabel('Hypervolume')
axs[0].set_xlabel(None)

# axs[0].axhline(pcn_hv['value'].quantile(0.25), ls='--', color='black', alpha=0.5)
# axs[0].axhline(pvn_hv['value'].quantile(0.75), ls='--', color='black', alpha=0.5)
axs[0].axhline(pcn_hv['value'].max(), ls='--', color='black', alpha=0.5)

eff = lambda_lcn[lambda_lcn['metric'] == 'total_efficiency']
sns.boxplot(data=eff, x="lambda", y="value", ax=axs[1], linewidth=LINEWIDTH)
axs[1].set_title('Total Efficiency')
axs[1].set_ylabel('Efficiency')

# axs[1].axhline(pcn_eff['value'].quantile(0.25), ls='--', color='black', alpha=0.5)
# axs[1].axhline(pcn_eff['value'].quantile(0.75), ls='--', color='black', alpha=0.5)
axs[1].axhline(pcn_eff['value'].max(), ls='--', color='black', alpha=0.5)

gini_ = lambda_lcn[lambda_lcn['metric'] == 'gini']
sns.boxplot(data=gini_, x="lambda", y="value", ax=axs[2], linewidth=LINEWIDTH)
axs[2].set_title('Gini')
axs[2].set_ylabel('Gini Index')
# axs[2].axhline(pcn_gini['value'].quantile(0.25), ls='--', color='black', alpha=0.5)
# axs[2].axhline(pcn_gini['value'].quantile(0.75), ls='--', color='black', alpha=0.5)
axs[2].axhline(pcn_gini['value'].min(), ls='--', color='black', alpha=0.5)


sen_welfare = lambda_lcn[lambda_lcn['metric'] == 'sen_welfare']
sns.boxplot(data=sen_welfare, x="lambda", y="value", ax=axs[3], linewidth=LINEWIDTH)
axs[3].set_title('Sen Welfare (Efficiency * (1 - Gini Index))')
axs[3].set_ylabel('Sen Welfare')
# axs[3].axhline(pcn_sen_welfare['value'].quantile(0.25), ls='--', color='black', alpha=0.5)
# axs[3].axhline(pcn_sen_welfare['value'].quantile(0.75), ls='--', color='black', alpha=0.5)
axs[3].axhline(pcn_sen_welfare['value'].max(), ls='--', color='black', alpha=0.5)
fig.suptitle(f'λ-LCN with {NR_GROUPS_TO_PLOT} groups', fontsize=20)
fig.text(0.5, 0.95, 'Gray Horizontal Line indicates the best solution found by PCN', ha='center', va='center', fontsize=14, color='gray')
fig.tight_layout()


#%% SAME BUT ONLY for Hypervolume and Sen Welfare
colors = ["#1A85FF", "#E66100", "#D41159", "#BFBFBF"]
sns.set_palette(sns.color_palette(colors))
LINEWIDTH = 2

fig, axs = plt.subplots(2, 1, figsize=(7, 6))
lambda_lcn = all_results[(all_results['model'].isin(['LCN_Lambda'])) & (all_results['lambda'].isin([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]))]
lambda_lcn = lambda_lcn[lambda_lcn['nr_groups'] == NR_GROUPS_TO_PLOT]

hyperv = lambda_lcn[lambda_lcn['metric'] == 'hv']
sns.boxplot(data=hyperv, x="lambda", y="value", ax=axs[0], linewidth=2)
axs[0].set_title('Hypervolume')
axs[0].set_ylabel(None)
axs[0].set_xlabel(None)
axs[0].axhline(pcn_hv['value'].max(), ls='--', color='black', alpha=0.5)

sen_welfare = lambda_lcn[lambda_lcn['metric'] == 'sen_welfare']
sns.boxplot(data=sen_welfare, x="lambda", y="value", ax=axs[1], linewidth=2)
axs[1].set_title('Sen Welfare (Efficiency * (1 - Gini Index))')
axs[1].set_ylabel(None)
axs[1].set_xlabel('λ')
# axs[1].axhline(pcn_sen_welfare['value'].quantile(0.25), ls='--', color='black', alpha=0.5)
# axs[1].axhline(pcn_sen_welfare['value'].quantile(0.75), ls='--', color='black', alpha=0.5)
axs[1].axhline(pcn_sen_welfare['value'].max(), ls='--', color='black', alpha=0.5)
# fig.suptitle(f'λ-LCN with {NR_GROUPS_TO_PLOT} groups', fontsize=20)
# fig.text(0.5, 0.95, 'Gray Horizontal Line indicates the best solution found by PCN', ha='center', va='center', fontsize=14, color='gray')
fig.tight_layout()





# %% Plot the fronts for each Lambda -- NR GROUPS = 3
# plt.rcParams.update({'font.size': 12})
# fronts_3 = lambda_lcn[lambda_lcn['nr_groups'] == 3]
# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(111, projection='3d')
# cmap = plt.get_cmap("Blues")
# # for l in fronts_3['lambda'].unique():
# # for l in [0, 0.3, 0.6, 0.9]:
# for l in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
#     fronts = fronts_3[(fronts_3['lambda'] == l) & (fronts_3['metric'] == 'fronts')]['value']
#     fronts = np.array([np.array(front) for front in fronts])
#     # Plot the fronts in a 3D scatter plot
#     ax.scatter(fronts[:, 0], fronts[:, 1], fronts[:, 2], label=f'Lambda = {l}', alpha=0.5, c=cmap(l), s=50)
#     ax.set_xlabel('Group 1')
#     ax.set_ylabel('Group 2')
#     ax.set_zlabel('Group 3')
#     # ax.set_ylim()
#     ax.legend()

# # lim = fronts_3[(fronts_3['lambda'] == 1.0) & (fronts_3['metric'] == 'fronts')]['value']
# ax.set_xlim([0, 0.007])
# ax.set_ylim([0, 0.007])
# ax.set_zlim([0, 0.007])
# fig.tight_layout()


# # Add to the scatter plot all the points in the range (0, 0.007) with step 0.001, where each point is the element in the range repeated 3 times
# for i in np.arange(0, 0.007, 0.001):
#     ax.scatter([i]*3, [i]*3, [i]*3, alpha=0.5, c='Red', s=50)
# ax.scatter([0.003]*3, [0.003]*3, [0.003]*3, alpha=0.5, c='Red', s=50, label='Equality')
# ax.legend()

# %%
