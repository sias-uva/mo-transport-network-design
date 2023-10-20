#%%
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import json

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
models_to_plot = [
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
models_to_plot = [
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
models_to_plot = [
    {'dir': 'pcn_amsterdam_20230713_13_52_59.686475', 'name': 'baseline_sf'},
    {'dir': 'pcn_amsterdam_20230713_13_52_59.686486', 'name': 'baseline_sf'},
    {'dir': 'pcn_amsterdam_20230713_13_52_59.686430', 'name': 'baseline_sf'},
    {'dir': 'pcn_amsterdam_20230713_13_55_02.074199', 'name': 'dtf_sf'},
    {'dir': 'pcn_amsterdam_20230713_13_55_07.698058', 'name': 'dtf_sf'},
    {'dir': 'pcn_amsterdam_20230713_13_55_07.702153', 'name': 'dtf_sf'},
]

# 20 Stations @ 30k
# models_to_plot = [
#     {'dir': 'pcn_amsterdam_20230713_17_02_44.311874', 'name': 'baseline'},
#     {'dir': 'pcn_amsterdam_20230713_17_02_44.176873', 'name': 'baseline'},
#     {'dir': 'pcn_amsterdam_20230713_17_02_44.273859', 'name': 'baseline'},
   
#     {'dir': 'pcn_amsterdam_20230713_18_26_38.070319', 'name': 'dtf'},
#     {'dir': 'pcn_amsterdam_20230713_18_26_38.062950', 'name': 'dtf'},
#     {'dir': 'pcn_amsterdam_20230713_18_26_37.903533', 'name': 'dtf'},

# ]

# 20 stations @ 50k, new scaling factor
models_to_plot = [
    {'dir': 'pcn_amsterdam_20230713_19_23_25.170750', 'name': 'baseline'},
    {'dir': 'pcn_amsterdam_20230713_19_23_25.111063', 'name': 'baseline'},
   
    {'dir': 'pcn_amsterdam_20230713_18_26_38.070319', 'name': 'dtf'},
    {'dir': 'pcn_amsterdam_20230713_18_26_38.062950', 'name': 'dtf'},
    # {'dir': 'pcn_amsterdam_20231018_10_26_00.342164', 'name': 'dtf'},

]


models_to_plot = pd.DataFrame(models_to_plot)
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
for _, model in models_to_plot.iterrows():
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
    

# %% Confidence intervals per method

def calc_ci(models, metric, z=1.96):
    data = []
    for model in models:
        data.append(pd.read_csv(f"./results/{model['dir']}/metrics.csv")[metric].tolist())
    data = np.array(data)

    m = data.mean(axis=0)
    std = data.std(axis=0)
    se = std/np.sqrt(data.shape[0])

    lb = m - z * se
    ub = m + z * se

    return np.vstack((lb, m, ub))


fig, axs = plt.subplots(2, 2, figsize=(15, 8))

for name in models_to_plot['name'].unique():
    models = models_to_plot[models_to_plot['name'] == name].to_dict('records')

    ci_train = calc_ci(models, 'train_hv')
    x = range(ci_train.shape[1])
    axs[0][0].plot(x, ci_train[1], label=name, color=colors[name])
    axs[0][0].fill_between(x, ci_train[0], ci_train[2], color=colors[name], alpha=0.3)
    axs[0][0].title.set_text('Train Hypervolume')
    axs[0][0].legend()

    ci_greedy = calc_ci(models, 'greedy_hv')
    axs[0][1].plot(x, ci_greedy[1], label=name, color=colors[name])
    axs[0][1].fill_between(x, ci_greedy[0], ci_greedy[2], color=colors[name], alpha=0.3)
    axs[0][1].title.set_text('Greedy Hypervolume')

    ci_eval = calc_ci(models, 'eval_hv')
    axs[1][0].plot(x, ci_eval[1], label=name, color=colors[name])
    axs[1][0].fill_between(x, ci_eval[0], ci_eval[2], color=colors[name], alpha=0.3)
    axs[1][0].title.set_text('Eval. Hypervolume')

    ci_greedy_train = ci_greedy/ci_train
    axs[1][1].plot(x, ci_greedy_train[1], label=name, color=colors[name])
    axs[1][1].fill_between(x, ci_greedy_train[0], ci_greedy_train[2], color=colors[name], alpha=0.3)
    axs[1][1].title.set_text('Greedy/Train Hypervolume')

# %% Plot the final fronts
import matplotlib.image as mpimg

fig, axs = plt.subplots(models_to_plot['name'].nunique(), 
                        models_to_plot.groupby('name')['dir'].count().max(), 
                        figsize=(30, 30))
for i, name in enumerate(models_to_plot['name'].unique()):
    models = models_to_plot[models_to_plot['name'] == name].to_dict('records')
    for j, model in enumerate(models):
        try:
            axs[i][j].imshow(mpimg.imread(f"./results/{model['dir']}/Front_99.png"))
            axs[i][j].title.set_text(model['name'])

            # axs[i][j].axis('off')
            axs[i][j].set_ylabel(model['dir'])
            # axs[i][j].xaxis.set_visible(False)
            axs[i][j].get_xaxis().set_ticks([])
            axs[i][j].get_yaxis().set_ticks([])
            axs[i][j].spines['top'].set_visible(False)
            axs[i][j].spines['right'].set_visible(False)
            axs[i][j].spines['bottom'].set_visible(False)
            axs[i][j].spines['left'].set_visible(False)


        except FileNotFoundError:
            pass
fig.tight_layout()

    
#%% Plot the final ERs
import matplotlib.image as mpimg

fig, axs = plt.subplots(models_to_plot['name'].nunique(), 
                        models_to_plot.groupby('name')['dir'].count().max(), 
                        figsize=(30, 30))
for i, name in enumerate(models_to_plot['name'].unique()):
    models = models_to_plot[models_to_plot['name'] == name].to_dict('records')
    for j, model in enumerate(models):
        metrics = pd.read_csv(f"./results/{model['dir']}/metrics.csv")
        try:
            axs[i][j].imshow(mpimg.imread(f"./results/{model['dir']}/ER_99.png"))
            axs[i][j].title.set_text(f"{model['name']}\nTrain HV: {metrics['train_hv'].iloc[-1]:.4}")
            axs[i][j].title.set_fontsize(30)

            # axs[i][j].axis('off')
            axs[i][j].set_ylabel(model['dir'], fontsize=17)
            # axs[i][j].xaxis.set_visible(False)
            axs[i][j].get_xaxis().set_ticks([])
            axs[i][j].get_yaxis().set_ticks([])
            axs[i][j].spines['top'].set_visible(False)
            axs[i][j].spines['right'].set_visible(False)
            axs[i][j].spines['bottom'].set_visible(False)
            axs[i][j].spines['left'].set_visible(False)


        except FileNotFoundError:
            pass
fig.tight_layout()

# %% LCN vs PCN
def gini(x, normalized=True):
    sorted_x = np.sort(x, axis=1)
    n = x.shape[1]
    cum_x = np.cumsum(sorted_x, axis=1, dtype=float)
    gi = (n + 1 - 2 * np.sum(cum_x, axis=1) / cum_x[:, -1]) / n
    if normalized:
        gi = gi * (n / (n - 1))
    return gi

models_to_plot = [
    # {'dir': 'lcn_amsterdam_20230804_19_40_49.371245', 'name': 'lcn_l2'},
    # {'dir': 'lcn_amsterdam_20230804_21_46_22.960346', 'name': 'lcn_l2'},
    # {'dir': 'lcn_amsterdam_20230804_20_07_37.298332', 'name': 'lcn_linf'},
    # {'dir': 'lcn_amsterdam_20230804_21_23_06.860050', 'name': 'lcn_linf'},
    # {'dir': 'pcn_amsterdam_20230804_19_47_02.560807', 'name': 'pcn'},

    {'dir': 'lcn_amsterdam_20231020_02_58_59.046020', 'name': 'LCN'},
    {'dir': 'pcn_amsterdam_20230713_18_26_38.070319', 'name': 'PCN'}
]

models_to_plot = pd.DataFrame(models_to_plot)
final_metrics = {}

for i, name in enumerate(models_to_plot['name'].unique()):
    models = models_to_plot[models_to_plot['name'] == name].to_dict('records')
    ginis = []

    for j, model in enumerate(models):
        metrics = pd.read_csv(f"./results/{model['dir']}/metrics.csv")
        # Read the content of the output file
        with open(f"./results/{model['dir']}/output.txt", "r") as file:
            output = file.read()
            fronts = json.loads(output)['best_front_r']
            gini_index = gini(np.array(fronts))
            # Get the gini index of the more equal 3 solutions in the front.
            # avg_gini = np.mean(np.sort(gini_index))
            avg_gini = np.mean(np.sort(gini_index))
            ginis.append(avg_gini)

    final_metrics[name] = np.mean(ginis)

fig, ax = plt.subplots(figsize=(5, 5))
ax.bar(final_metrics.keys(), final_metrics.values())

models_to_plot = pd.DataFrame(models_to_plot)
all_results = {}
for i, name in enumerate(models_to_plot['name'].unique()):
    models = models_to_plot[models_to_plot['name'] == name].to_dict('records')

    for j, model in enumerate(models):
        metrics = pd.read_csv(f"./results/{model['dir']}/metrics.csv")
        # Read the content of the output file
        with open(f"./results/{model['dir']}/output.txt", "r") as file:
            output = file.read()
            fronts = json.loads(output)['best_front_r']
            gini_index = gini(np.array(fronts))
            total_efficiency = np.sum(fronts, axis=1)
            all_results[name] = {'gini': gini_index, 'total_efficiency': total_efficiency}

fig, ax = plt.subplots(figsize=(5, 5))
for name in all_results.keys():
    ax.scatter(all_results[name]['total_efficiency'], all_results[name]['gini'], label=name, alpha=0.5)
ax.set_xlabel('Total Efficiency')
ax.set_ylabel('Gini Index (normalized)')
ax.legend()

# %%
