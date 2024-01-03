#%%
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import json
import seaborn as sns
plt.rcParams.update({'font.size': 18})

#%%
#% Plot all models
# models_to_plot = [
#                 {'dir': 'pcn_amsterdam_20230705_11_14_58.880354', 'name': 'baseline'},
#                 {'dir': 'pcn_amsterdam_20230705_13_06_15.465554', 'name': 'baseline'},
#                 {'dir': 'pcn_amsterdam_20230705_14_24_30.249039', 'name': 'baseline'},
#                 {'dir': 'pcn_amsterdam_20230706_12_39_41.825288', 'name': 'baseline'},
#                 # {'dir': 'pcn_amsterdam_20230706_14_33_35.160305', 'name': 'baseline@500k'},
#               {'dir': 'pcn_amsterdam_20230705_19_31_40.423497', 'name': 'dtf'},
#               {'dir': 'pcn_amsterdam_20230705_19_33_45.190523', 'name': 'dtf'},

#                 {'dir': 'pcn_amsterdam_20230706_10_25_46.730664', 'name': 'dtf2'},
#                 {'dir': 'pcn_amsterdam_20230706_10_40_06.784805', 'name': 'dtf2'},
#                 {'dir': 'pcn_amsterdam_20230706_11_20_11.151743', 'name': 'dtf2'},
#                 {'dir': 'pcn_amsterdam_20230706_11_48_12.178011', 'name': 'dtf2'},
                
#                 # {'dir': 'pcn_amsterdam_20230706_13_25_22.795103', 'name': 'baselineDTK50k'},
#                 # {'dir': 'pcn_amsterdam_20230706_14_49_12.030486', 'name': 'baselineDTK50k'},
#                 #   {'dir': 'pcn_amsterdam_20230705_11_22_38.891947', 'name': 'dtf_er'},
#                 #   {'dir': 'pcn_amsterdam_20230705_13_04_01.109243', 'name': 'dtf_er'},
#                 #   {'dir': 'pcn_amsterdam_20230705_15_13_12.034815', 'name': 'dtf_er'},
#                 #   {'dir': 'pcn_amsterdam_20230705_11_26_05.996728', 'name': 'dtf_top10'},
#                 #   {'dir': 'pcn_amsterdam_20230705_13_58_50.833274', 'name': 'dtf_top10'},
#                 #   {'dir': 'pcn_amsterdam_20230705_15_55_34.410131', 'name': 'dtf_top10'},
# ]

# 10 Stations
all_objectives = [
    {'dir': 'pcn_amsterdam_20230706_19_18_15.208962', 'name': 'baseline'},
    {'dir': 'pcn_amsterdam_20230706_19_32_03.366079', 'name': 'baseline'},
    {'dir': 'pcn_amsterdam_20230706_21_26_17.265377', 'name': 'baseline'},
    {'dir': 'pcn_amsterdam_20230706_22_47_16.110920', 'name': 'baseline'},
    {'dir': 'pcn_amsterdam_20230707_09_31_10.537691', 'name': 'baseline'},
    
    {'dir': 'pcn_amsterdam_20230707_14_02_29.515148', 'name': 'baseline_MODEL_UPDATES_ev500'},
    {'dir': 'pcn_amsterdam_20230707_14_02_20.287639', 'name': 'baseline_MODEL_UPDATES_ev500'},

    {'dir': 'pcn_amsterdam_20230707_15_06_52.175708', 'name': 'dtf_MODEL_UPDATES_ev500'},
    {'dir': 'pcn_amsterdam_20230707_15_07_14.306401', 'name': 'dtf_MODEL_UPDATES_ev500'},
    {'dir': 'pcn_amsterdam_20230707_16_22_13.623109', 'name': 'dtf_MODEL_UPDATES_ev500'},
    {'dir': 'pcn_amsterdam_20230707_16_47_27.392052', 'name': 'dtf_MODEL_UPDATES_ev500'},
    {'dir': 'pcn_amsterdam_20230707_16_48_14.390614', 'name': 'dtf_MODEL_UPDATES_ev500'},

    {'dir': 'pcn_amsterdam_20230710_10_58_34.852196', 'name': 'dtf2_MODEL_UPDATES_ev500'},
    {'dir': 'pcn_amsterdam_20230710_10_58_53.485685', 'name': 'dtf2_MODEL_UPDATES_ev500'},
    {'dir': 'pcn_amsterdam_20230710_11_32_24.437619', 'name': 'dtf2_MODEL_UPDATES_ev500'},
    {'dir': 'pcn_amsterdam_20230710_11_32_36.391045', 'name': 'dtf2_MODEL_UPDATES_ev500'},

    # {'dir': 'pcn_amsterdam_20230710_12_04_28.040372', 'name': 'dtf2_MODEL_UPDATES_ev500_best'},
    # {'dir': 'pcn_amsterdam_20230710_12_04_39.002849', 'name': 'dtf2_MODEL_UPDATES_ev500_best'},


    # {'dir': '', 'name': 'baseline_MODEL_UPDATES_ev1000'},
    
    # {'dir': 'pcn_amsterdam_20230706_16_42_36.528509', 'name': 'baselineDTF50k'},
    # {'dir': 'pcn_amsterdam_20230706_18_32_58.959206', 'name': 'baselineDTF50k'},
    # {'dir': 'pcn_amsterdam_20230706_21_31_09.215754', 'name': 'baselineDTF50k'},
    # {'dir': 'pcn_amsterdam_20230706_22_44_08.284618', 'name': 'baselineDTF50k'},
    # {'dir': 'pcn_amsterdam_20230707_10_06_40.925056', 'name': 'baselineDTF50k'},
    # {'dir': 'pcn_amsterdam_20230706_16_41_44.443906', 'name': 'dtf'},
    # {'dir': 'pcn_amsterdam_20230706_18_32_17.561493', 'name': 'dtf'},
    # {'dir': 'pcn_amsterdam_20230706_21_28_37.933000', 'name': 'dtf'},
    # {'dir': 'pcn_amsterdam_20230706_22_44_43.427270', 'name': 'dtf'},
    # {'dir': 'pcn_amsterdam_20230707_09_32_43.600904', 'name': 'dtf'},
    # {'dir': 'pcn_amsterdam_20230707_10_40_21.328566', 'name': 'dtf2'},
    # {'dir': 'pcn_amsterdam_20230707_10_40_36.378424', 'name': 'dtf2'},
]

# # 10 Stations @ 200k
all_objectives = [
    {'dir': 'pcn_amsterdam_20230710_14_05_48.908216', 'name': 'baseline'},
    {'dir': 'pcn_amsterdam_20230710_15_19_26.791523', 'name': 'baseline'},
    {'dir': 'pcn_amsterdam_20230710_16_28_03.135061', 'name': 'baseline'},

    {'dir': 'pcn_amsterdam_20230711_14_30_00.272314', 'name': 'baseline_20step_episodes'},
    {'dir': 'pcn_amsterdam_20230711_14_30_00.285458', 'name': 'baseline_20step_episodes'},
    {'dir': 'pcn_amsterdam_20230711_14_29_57.375035', 'name': 'baseline_20step_episodes'},

    # {'dir': 'pcn_amsterdam_20230710_14_07_41.087041', 'name': 'baseline_MODEL_UPDATES_ev500'},
    # {'dir': 'pcn_amsterdam_20230710_15_17_11.407820', 'name': 'baseline_MODEL_UPDATES_ev500'},
    # {'dir': 'pcn_amsterdam_20230710_16_30_12.391358', 'name': 'baseline_MODEL_UPDATES_ev500'},

    # {'dir': 'pcn_amsterdam_20230710_14_10_37.927414', 'name': 'dtf_MODEL_UPDATES_ev500'},
    # {'dir': 'pcn_amsterdam_20230710_15_21_00.612220', 'name': 'dtf_MODEL_UPDATES_ev500'},
    # {'dir': 'pcn_amsterdam_20230710_16_32_16.712395', 'name': 'dtf_MODEL_UPDATES_ev500'},

    # {'dir': 'pcn_amsterdam_20230710_14_15_24.150347', 'name': 'dtf2_MODEL_UPDATES_ev500'},
    # {'dir': 'pcn_amsterdam_20230710_15_22_35.421954', 'name': 'dtf2_MODEL_UPDATES_ev500'},
    # {'dir': 'pcn_amsterdam_20230710_16_54_39.203461', 'name': 'dtf2_MODEL_UPDATES_ev500'},

]

# # # 10 Stations @ 200k, crowding distance threshold
# models_to_plot = [
#     {'dir': 'pcn_amsterdam_20230706_19_18_15.208962', 'name': 'baseline'},
#     {'dir': 'pcn_amsterdam_20230706_19_32_03.366079', 'name': 'baseline'},
#     {'dir': 'pcn_amsterdam_20230706_21_26_17.265377', 'name': 'baseline'},
#     {'dir': 'pcn_amsterdam_20230706_22_47_16.110920', 'name': 'baseline'},
#     {'dir': 'pcn_amsterdam_20230707_09_31_10.537691', 'name': 'baseline'},
    
    
#     {'dir': 'pcn_amsterdam_20230713_13_09_58.097758', 'name': 'baseline_sf'},
#     {'dir': 'pcn_amsterdam_20230713_13_26_15.266139', 'name': 'baseline_sf'},




#      {'dir': 'pcn_amsterdam_20230711_10_47_38.126387', 'name': 'baseline_0.3'},
#      {'dir': 'pcn_amsterdam_20230711_14_47_21.373754', 'name': 'baseline_0.3'},
#      {'dir': 'pcn_amsterdam_20230711_14_47_27.374584', 'name': 'baseline_0.3'},
#      {'dir': 'pcn_amsterdam_20230711_14_47_21.412165', 'name': 'baseline_0.3'},

#      {'dir': 'pcn_amsterdam_20230711_14_51_21.115324', 'name': 'baseline_0.5'},
#      {'dir': 'pcn_amsterdam_20230711_14_51_20.937086', 'name': 'baseline_0.5'},

#      {'dir': 'pcn_amsterdam_20230711_12_09_55.281764', 'name': 'baseline_0.15'},
#      {'dir': 'pcn_amsterdam_20230711_10_59_06.797089', 'name': 'baseline_0.1'},
#      {'dir': 'pcn_amsterdam_20230711_12_12_36.552336', 'name': 'baseline_0.1'},
#      {'dir': 'pcn_amsterdam_20230711_13_19_20.510421', 'name': 'baseline_0.05'},
# ]

# # 10 Stations @ 200k, new scaling factor
all_objectives = [
    {'dir': 'pcn_amsterdam_20230713_13_52_59.686475', 'name': 'baseline_sf'},
    {'dir': 'pcn_amsterdam_20230713_13_52_59.686486', 'name': 'baseline_sf'},
    {'dir': 'pcn_amsterdam_20230713_13_52_59.686430', 'name': 'baseline_sf'},
    {'dir': 'pcn_amsterdam_20230713_13_55_02.074199', 'name': 'dtf_sf'},
    {'dir': 'pcn_amsterdam_20230713_13_55_07.698058', 'name': 'dtf_sf'},
    {'dir': 'pcn_amsterdam_20230713_13_55_07.702153', 'name': 'dtf_sf'},
]

# 20 Stations @ 30k
all_objectives = [
    {'dir': 'pcn_amsterdam_20230713_17_02_44.311874', 'name': 'baseline'},
    {'dir': 'pcn_amsterdam_20230713_17_02_44.176873', 'name': 'baseline'},
    {'dir': 'pcn_amsterdam_20230713_17_02_44.273859', 'name': 'baseline'},
   
    {'dir': 'pcn_amsterdam_20230713_18_26_38.070319', 'name': 'dtf'},
    {'dir': 'pcn_amsterdam_20230713_18_26_38.062950', 'name': 'dtf'},
    {'dir': 'pcn_amsterdam_20230713_18_26_37.903533', 'name': 'dtf'},

]

# 20 stations @ 50k, new scaling factor
# models_to_plot = [
#     {'dir': 'pcn_amsterdam_20230713_19_23_25.170750', 'name': 'baseline'},
#     {'dir': 'pcn_amsterdam_20230713_19_23_25.111063', 'name': 'baseline'},
   
#     {'dir': 'pcn_amsterdam_20230713_18_26_38.070319', 'name': 'dtf'},
#     {'dir': 'pcn_amsterdam_20230713_18_26_38.062950', 'name': 'dtf'},
#     # {'dir': 'pcn_amsterdam_20231018_10_26_00.342164', 'name': 'dtf'},

# ]

# # LCN
# models_to_plot = [
#     {'dir': 'lcn_amsterdam_20231023_11_30_35.331255', 'name': 'baseline'},
#     {'dir': 'lcn_amsterdam_20231023_11_30_35.299109', 'name': 'baseline'},
   
#     {'dir': 'lcn_amsterdam_20231023_11_50_32.217883', 'name': 'dtf'},
#     {'dir': 'lcn_amsterdam_20231023_11_44_54.212951', 'name': 'dtf'},
#     {'dir': 'lcn_amsterdam_20231023_11_44_54.211632', 'name': 'dtf'},
# ]


all_objectives = pd.DataFrame(all_objectives)
colors = {
        'baseline':'red', 
        'dtf':'blue', 
        # 'dtf2':'green', 
        # 'baseline_MODEL_UPDATES_ev1000': 'orange', 
        'baseline_20step_episodes': 'orange',
        'baseline_MODEL_UPDATES_ev500': 'green',
        'dtf_MODEL_UPDATES_ev500': 'yellow',
        'dtf2_MODEL_UPDATES_ev500': 'blue',
        'dtf2_MODEL_UPDATES_ev500_best': 'orange',
        'baseline_0.5': 'green',
        'baseline_0.3': 'orange',
        'baseline_0.1': 'blue',
        'baseline_0.15': 'black',
        'baseline_0.05': 'purple',
        'baseline_sf': 'gray',
        'dtf_sf': 'blue',
        }

fig, axs = plt.subplots(2, 2, figsize=(15, 8))
# [ax.set_ylim(0, 2.5e-5) for ax in axs.flatten()]
for _, model in all_objectives.iterrows():
    metrics = pd.read_csv(f"./results/{model['dir']}/metrics.csv")
    axs[0][0].plot(metrics['step'], metrics['train_hv'], label=model['name'])
    axs[0][0].title.set_text('Train Hypervolume')
    axs[0][1].plot(metrics['step'], metrics['greedy_hv'], label=model['name'])
    axs[0][1].title.set_text('Greedy Hypervolume')
    axs[1][0].plot(metrics['step'], metrics['eval_hv'], label=model['name'])
    axs[1][0].title.set_text('Eval. Hypervolume')
    axs[1][1].plot(metrics['step'], metrics['greedy_hv']/metrics['train_hv'], label=model['name'])
    axs[1][1].title.set_text('Greedy/Train Hypervolume')
    axs[1][1].set_ylim(0, 1.01)
    axs[0][0].legend()
    

# # %% Confidence intervals per method

# def calc_ci(models, metric, z=1.96):
#     data = []
#     for model in models:
#         data.append(pd.read_csv(f"./results/{model['dir']}/metrics.csv")[metric].tolist())
#     data = np.array(data)

#     m = data.mean(axis=0)
#     std = data.std(axis=0)
#     se = std/np.sqrt(data.shape[0])

#     lb = m - z * se
#     ub = m + z * se

#     return np.vstack((lb, m, ub))


# fig, axs = plt.subplots(2, 2, figsize=(15, 9))

# for name in models_to_plot['name'].unique():
#     models = models_to_plot[models_to_plot['name'] == name].to_dict('records')

#     ci_train = calc_ci(models, 'train_hv')
#     x = range(ci_train.shape[1])
#     axs[0][0].plot(x, ci_train[1], label=name, color=colors[name])
#     axs[0][0].fill_between(x, ci_train[0], ci_train[2], color=colors[name], alpha=0.3)
#     axs[0][0].title.set_text('Train Hypervolume')
#     axs[0][0].legend()

#     ci_greedy = calc_ci(models, 'greedy_hv')
#     axs[0][1].plot(x, ci_greedy[1], label=name, color=colors[name])
#     axs[0][1].fill_between(x, ci_greedy[0], ci_greedy[2], color=colors[name], alpha=0.3)
#     axs[0][1].title.set_text('Greedy Hypervolume')

#     ci_eval = calc_ci(models, 'eval_hv')
#     axs[1][0].plot(x, ci_eval[1], label=name, color=colors[name])
#     axs[1][0].fill_between(x, ci_eval[0], ci_eval[2], color=colors[name], alpha=0.3)
#     axs[1][0].title.set_text('Eval. Hypervolume')

#     ci_greedy_train = ci_greedy/ci_train
#     axs[1][1].plot(x, ci_greedy_train[1], label=name, color=colors[name])
#     axs[1][1].fill_between(x, ci_greedy_train[0], ci_greedy_train[2], color=colors[name], alpha=0.3)
#     axs[1][1].title.set_text('Greedy/Train Hypervolume')

# %% Plot the final fronts
# import matplotlib.image as mpimg

# fig, axs = plt.subplots(models_to_plot['name'].nunique(), 
#                         models_to_plot.groupby('name')['dir'].count().max(), 
#                         figsize=(30, 30))
# for i, name in enumerate(models_to_plot['name'].unique()):
#     models = models_to_plot[models_to_plot['name'] == name].to_dict('records')
#     for j, model in enumerate(models):
#         try:
#             axs[i][j].imshow(mpimg.imread(f"./results/{model['dir']}/Front_99.png"))
#             axs[i][j].title.set_text(model['name'])

#             # axs[i][j].axis('off')
#             axs[i][j].set_ylabel(model['dir'])
#             # axs[i][j].xaxis.set_visible(False)
#             axs[i][j].get_xaxis().set_ticks([])
#             axs[i][j].get_yaxis().set_ticks([])
#             axs[i][j].spines['top'].set_visible(False)
#             axs[i][j].spines['right'].set_visible(False)
#             axs[i][j].spines['bottom'].set_visible(False)
#             axs[i][j].spines['left'].set_visible(False)


#         except FileNotFoundError:
#             pass
# fig.tight_layout()

    
# #%% Plot the final ERs
# import matplotlib.image as mpimg

# fig, axs = plt.subplots(models_to_plot['name'].nunique(), 
#                         models_to_plot.groupby('name')['dir'].count().max(), 
#                         figsize=(30, 30))
# for i, name in enumerate(models_to_plot['name'].unique()):
#     models = models_to_plot[models_to_plot['name'] == name].to_dict('records')
#     for j, model in enumerate(models):
#         metrics = pd.read_csv(f"./results/{model['dir']}/metrics.csv")
#         try:
#             axs[i][j].imshow(mpimg.imread(f"./results/{model['dir']}/ER_99.png"))
#             axs[i][j].title.set_text(f"{model['name']}\nTrain HV: {metrics['train_hv'].iloc[-1]:.4}")
#             axs[i][j].title.set_fontsize(30)

#             # axs[i][j].axis('off')
#             axs[i][j].set_ylabel(model['dir'], fontsize=17)
#             # axs[i][j].xaxis.set_visible(False)
#             axs[i][j].get_xaxis().set_ticks([])
#             axs[i][j].get_yaxis().set_ticks([])
#             axs[i][j].spines['top'].set_visible(False)
#             axs[i][j].spines['right'].set_visible(False)
#             axs[i][j].spines['bottom'].set_visible(False)
#             axs[i][j].spines['left'].set_visible(False)


#         except FileNotFoundError:
#             pass
# fig.tight_layout()

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
        {'name': 'LCN_ND', 'dirs': ['lcn_xian_20231024_15_11_42.594057', 'lcn_xian_20231220_13_53_13.263588', 'lcn_xian_20231220_13_53_29.501767']},
        {'name': 'LCN_OPTMAX', 'dirs': ['lcn_xian_20231031_19_50_00.078706', 'lcn_xian_20231220_14_44_33.522291', 'lcn_xian_20231220_14_44_51.798498']},
        {'name': 'LCN_NDMEAN', 'dirs': ['lcn_xian_20231031_18_20_59.173870', 'lcn_xian_20231220_14_47_26.220395', 'lcn_xian_20231220_14_47_57.984685']},
     ]},

     {'nr_groups': 3, 
     'models': [
        {'name': 'PCN', 'dirs': ['pcn_xian_20231026_19_57_12.259730', 'pcn_xian_20231221_17_13_28.741533', 'pcn_xian_20231221_17_16_07.823677']},
        {'name': 'LCN_ND', 'dirs': ['lcn_xian_20231027_10_35_51.656132', 'lcn_xian_20231220_14_59_53.519271', 'lcn_xian_20231220_15_00_10.564421']},
        {'name': 'LCN_OPTMAX', 'dirs': ['lcn_xian_20231030_17_47_36.629611', 'lcn_xian_20231220_15_02_57.746395', 'lcn_xian_20231220_14_44_51.798498']},
        {'name': 'LCN_NDMEAN', 'dirs': ['lcn_xian_20231028_01_07_53.729884', 'lcn_xian_20231220_17_03_12.414722', 'lcn_xian_20231220_17_03_18.736912']},
        {'name': 'LCN_Lambda_0.0', 'dirs': ['lcn_xian_20231220_17_30_54.791998', 'lcn_xian_20231230_13_29_42.430687', 'lcn_xian_20231230_13_31_16.726711']},
        {'name': 'LCN_Lambda_0.1', 'dirs': ['lcn_xian_20231220_19_23_57.505711', 'lcn_xian_20231230_13_32_50.819328', 'lcn_xian_20231230_16_11_31.788882']},
        {'name': 'LCN_Lambda_0.2', 'dirs': ['lcn_xian_20231220_13_40_05.161123', 'lcn_xian_20231230_16_11_46.903158', 'lcn_xian_20231230_16_15_58.578610']},
        {'name': 'LCN_Lambda_0.3', 'dirs': ['lcn_xian_20231220_17_31_54.882410', 'lcn_xian_20231230_16_26_58.092950', 'lcn_xian_20231230_16_51_50.555284']},
        {'name': 'LCN_Lambda_0.4', 'dirs': ['lcn_xian_20231220_14_57_49.642641', 'lcn_xian_20231230_16_25_40.149452', 'lcn_xian_20231230_16_25_45.420788']},
        {'name': 'LCN_Lambda_0.5', 'dirs': ['lcn_xian_20231220_18_18_21.304292', 'lcn_xian_20231231_18_38_25.815554', 'lcn_xian_20231231_18_38_32.040668']},
        {'name': 'LCN_Lambda_0.6', 'dirs': ['lcn_xian_20231220_19_56_38.709599', 'lcn_xian_20231231_18_39_11.034438', 'lcn_xian_20231231_18_53_06.497753']},
        {'name': 'LCN_Lambda_0.7', 'dirs': ['lcn_xian_20231220_14_05_26.124152', 'lcn_xian_20231231_18_53_39.088964', 'lcn_xian_20231231_18_53_43.368080']},
        {'name': 'LCN_Lambda_0.8', 'dirs': ['lcn_xian_20231221_15_02_14.233385', 'lcn_xian_20231231_19_05_50.681596', 'lcn_xian_20231231_19_05_58.900443']},
        {'name': 'LCN_Lambda_0.9', 'dirs': ['lcn_xian_20231221_17_39_23.693903', 'lcn_xian_20231231_19_06_35.791660', 'lcn_xian_20231231_19_22_46.233640']},
        {'name': 'LCN_Lambda_1.0', 'dirs': ['lcn_xian_20231221_10_41_50.036536', 'lcn_xian_20231231_19_23_02.379536', 'lcn_xian_20231231_19_23_46.110507']},
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
        {'name': 'LCN_Lambda_0.0', 'dirs': ['lcn_xian_20231223_03_51_02.035482', 'lcn_xian_20240103_12_39_41.243528', 'lcn_xian_20240103_12_39_50.651435']},
        {'name': 'LCN_Lambda_0.1', 'dirs': ['lcn_xian_20231222_15_35_54.370796', 'lcn_xian_20240103_12_43_21.486401', 'lcn_xian_20240103_12_58_03.785256']},
        {'name': 'LCN_Lambda_0.2', 'dirs': ['lcn_xian_20231223_02_22_19.068841', 'lcn_xian_20240103_12_59_05.165989', 'lcn_xian_20240103_12_59_13.866006']},
        {'name': 'LCN_Lambda_0.3', 'dirs': ['lcn_xian_20231222_23_50_40.792027', 'lcn_xian_20240103_13_22_20.152186', 'lcn_xian_20240103_13_22_27.755196']},
        {'name': 'LCN_Lambda_0.4', 'dirs': ['lcn_xian_20231223_03_50_28.413054', 'lcn_xian_20240103_13_24_46.625112', 'lcn_xian_20240103_15_22_06.047546']},
        {'name': 'LCN_Lambda_0.5', 'dirs': ['lcn_xian_20231223_03_38_58.678816', 'lcn_xian_20240103_15_23_58.995186', 'lcn_xian_20240103_15_24_04.417342']},
        # {'name': 'LCN_Lambda_0.6', 'dirs': ['']},
        {'name': 'LCN_Lambda_0.7', 'dirs': ['lcn_xian_20231229_23_13_00.075207', 'lcn_xian_20240103_15_38_14.852001', 'lcn_xian_20240103_15_41_31.690977']},
        {'name': 'LCN_Lambda_0.8', 'dirs': ['lcn_xian_20231227_10_30_37.066763', 'lcn_xian_20240103_15_41_55.425827', 'lcn_xian_20240103_15_56_10.839843']},
        {'name': 'LCN_Lambda_0.9', 'dirs': ['lcn_xian_20231227_14_31_33.262752', 'lcn_xian_20240103_16_06_08.909584', 'lcn_xian_20240103_16_08_55.025827']},
        {'name': 'LCN_Lambda_1.0', 'dirs': ['lcn_xian_20231230_17_58_12.046057', 'lcn_xian_20240103_16_09_32.898444', 'lcn_xian_20240103_17_06_29.331688']},

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

all_results = pd.DataFrame()
REQ_SEEDS = 3 # to control if a model was not run for sufficient seeds
for oidx, objective in enumerate(all_objectives):
    nr_groups = objective['nr_groups']
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
            results_by_objective[model_name] = {'gini': [], 'total_efficiency': [], 'sen_welfare': [], 'nash_welfare': [], 'avg_per_group': []}
            for i in range(len(model['dirs'])):

                with open(f"./results/{model['dirs'][i]}/output.txt", "r") as file:
                    output = file.read()
                    fronts = json.loads(output)['best_front_r']
                    groups = len(fronts[0])
                    gini_index = gini(np.array(fronts))
                    total_efficiency = np.sum(fronts, axis=1)
                    nash_welfare = np.prod(fronts, axis=1)
                    results_by_objective[model_name]['fronts'] = fronts
                    results_by_objective[model_name]['gini'] = results_by_objective[model_name]['gini'] + gini_index.tolist()
                    results_by_objective[model_name]['total_efficiency'] = results_by_objective[model_name]['total_efficiency'] + total_efficiency.tolist()
                    results_by_objective[model_name]['sen_welfare'] = results_by_objective[model_name]['sen_welfare'] + (total_efficiency * (1-gini_index)).tolist()
                    results_by_objective[model_name]['nash_welfare'] = results_by_objective[model_name]['nash_welfare'] + nash_welfare.tolist()
                    results_by_objective[model_name]['avg_per_group'] = results_by_objective[model_name]['avg_per_group'] + np.mean(fronts, axis=0).tolist()
                    
        # results_by_objective[model_name]['lambda'] = model['lambda'] if 'lambda' in model else ''
    # Quite a hacky way to get the results in a dataframe, but didn't have time to do it properly (thanks copilot)
    # Convert all_results to a dataframe, with columns 'model', 'metric', 'value', and each row is a different value and not a list
    # results_by_objective = pd.DataFrame([(name, metric, value) for name in results_by_objective.keys() for metric in results_by_objective[name].keys() for value in results_by_objective[name][metric]], columns=['model', 'metric', 'value'])
    # Convert all_results to a dataframe, with columns 'model', 'lambda; 'metric', 'value', and each row is a different value and not a list
    results_by_objective = pd.DataFrame([(name, model['lambda'] if 'lambda' in model else None, metric, value) for name in results_by_objective.keys() for metric in results_by_objective[name].keys() for value in results_by_objective[name][metric]], columns=['model', 'lambda', 'metric', 'value'])
    results_by_objective['nr_groups'] = nr_groups
    results_by_objective['lambda'] = results_by_objective[results_by_objective['model'].str.contains('Lambda')]['model'].str.split('_').str[-1].astype(float)
    results_by_objective.loc[results_by_objective['model'].str.contains('Lambda'), 'model'] = 'LCN_Lambda'
    all_results = pd.concat([all_results, results_by_objective])
    
# Plot Total Efficiency, Gini Index, Sen Welfare for PCN vs LCN (ND, OPTMAX, NDMEAN)
fig, axs = plt.subplots(3, 1, figsize=(20, 16))
pcnvlcn = all_results[all_results['model'].isin(['PCN', 'LCN_ND', 'LCN_OPTMAX', 'LCN_NDMEAN'])]
eff = pcnvlcn[pcnvlcn['metric'] == 'total_efficiency']
eff['value'] = eff.groupby('nr_groups')['value'].transform(lambda x: x / x.max())
sns.boxplot(data=eff, x="nr_groups", y="value", hue="model", ax=axs[0])
axs[0].set_title('Total Efficiency')

gini_ = pcnvlcn[pcnvlcn['metric'] == 'gini']
sns.boxplot(data=gini_, x="nr_groups", y="value", hue="model", ax=axs[1])
axs[1].set_title('Gini')

sen_welfare = pcnvlcn[pcnvlcn['metric'] == 'sen_welfare']
sen_welfare['value'] = sen_welfare.groupby('nr_groups')['value'].transform(lambda x: x / x.max())
sns.boxplot(data=sen_welfare, x="nr_groups", y="value", hue="model", ax=axs[2])
axs[2].set_title('Sen Welfare (Efficiency * (1 - Gini Index))')
fig.tight_layout()

#%% Plot Total Efficiency, Gini Index, Sen Welfare for lambda-LCN (0.0-1.0)
NR_GROUPS_TO_PLOT = 6

fig, axs = plt.subplots(3, 1, figsize=(20, 16))
lambda_lcn = all_results[all_results['model'].isin(['LCN_Lambda'])]
lambda_lcn = lambda_lcn[lambda_lcn['nr_groups'] == NR_GROUPS_TO_PLOT]
eff = lambda_lcn[lambda_lcn['metric'] == 'total_efficiency']
eff['value'] = eff.groupby(['nr_groups', 'lambda'])['value'].transform(lambda x: x / x.max())
sns.boxplot(data=eff, x="lambda", y="value", ax=axs[0])
axs[0].set_title('Total Efficiency')

gini_ = lambda_lcn[lambda_lcn['metric'] == 'gini']
sns.boxplot(data=gini_, x="lambda", y="value", ax=axs[1])
axs[1].set_title('Gini')

sen_welfare = lambda_lcn[lambda_lcn['metric'] == 'sen_welfare']
sen_welfare['value'] = sen_welfare.groupby(['nr_groups', 'lambda'])['value'].transform(lambda x: x / x.max())
sns.boxplot(data=sen_welfare, x="lambda", y="value", ax=axs[2])
axs[2].set_title('Sen Welfare (Efficiency * (1 - Gini Index))')
fig.suptitle(f'λ-LCN with {NR_GROUPS_TO_PLOT} groups', fontsize=20)
fig.tight_layout()
# # %% Plot the fronts for each Lambda -- NR GROUPS = 3
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
