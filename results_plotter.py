#%%
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

#% Plot all models
models_to_plot = [
                {'dir': 'pcn_amsterdam_20230705_11_14_58.880354', 'name': 'baseline'},
                {'dir': 'pcn_amsterdam_20230705_13_06_15.465554', 'name': 'baseline'},
                {'dir': 'pcn_amsterdam_20230705_14_24_30.249039', 'name': 'baseline'},
                {'dir': 'pcn_amsterdam_20230706_12_39_41.825288', 'name': 'baseline'},
                # {'dir': 'pcn_amsterdam_20230706_14_33_35.160305', 'name': 'baseline@500k'},
              {'dir': 'pcn_amsterdam_20230705_19_31_40.423497', 'name': 'dtf'},
              {'dir': 'pcn_amsterdam_20230705_19_33_45.190523', 'name': 'dtf'},

                {'dir': 'pcn_amsterdam_20230706_10_25_46.730664', 'name': 'dtf2'},
                {'dir': 'pcn_amsterdam_20230706_10_40_06.784805', 'name': 'dtf2'},
                {'dir': 'pcn_amsterdam_20230706_11_20_11.151743', 'name': 'dtf2'},
                {'dir': 'pcn_amsterdam_20230706_11_48_12.178011', 'name': 'dtf2'},
                
                # {'dir': 'pcn_amsterdam_20230706_13_25_22.795103', 'name': 'baselineDTK50k'},
                # {'dir': 'pcn_amsterdam_20230706_14_49_12.030486', 'name': 'baselineDTK50k'},
                #   {'dir': 'pcn_amsterdam_20230705_11_22_38.891947', 'name': 'dtf_er'},
                #   {'dir': 'pcn_amsterdam_20230705_13_04_01.109243', 'name': 'dtf_er'},
                #   {'dir': 'pcn_amsterdam_20230705_15_13_12.034815', 'name': 'dtf_er'},
                #   {'dir': 'pcn_amsterdam_20230705_11_26_05.996728', 'name': 'dtf_top10'},
                #   {'dir': 'pcn_amsterdam_20230705_13_58_50.833274', 'name': 'dtf_top10'},
                #   {'dir': 'pcn_amsterdam_20230705_15_55_34.410131', 'name': 'dtf_top10'},
]

# 10 Stations
models_to_plot = [
    {'dir': 'pcn_amsterdam_20230706_19_18_15.208962', 'name': 'baseline'},
    {'dir': 'pcn_amsterdam_20230706_19_32_03.366079', 'name': 'baseline'},
    {'dir': 'pcn_amsterdam_20230706_21_26_17.265377', 'name': 'baseline'},
    {'dir': 'pcn_amsterdam_20230706_22_47_16.110920', 'name': 'baseline'},
    {'dir': 'pcn_amsterdam_20230707_09_31_10.537691', 'name': 'baseline'},
    
    # {'dir': 'pcn_amsterdam_20230707_11_06_34.501789', 'name': 'baseline_MODEL_UPDATES_1000'},
    # {'dir': 'pcn_amsterdam_20230707_12_05_53.426532', 'name': 'baseline_MODEL_UPDATES_1000'},
    # {'dir': 'pcn_amsterdam_20230707_12_05_35.071832', 'name': 'baseline_MODEL_UPDATES_1000'},

    {'dir': 'pcn_amsterdam_20230707_12_53_09.769885', 'name': 'baseline_MODEL_UPDATES_ev1000'},
    {'dir': 'pcn_amsterdam_20230707_12_54_11.658907', 'name': 'baseline_MODEL_UPDATES_ev1000'},
    
    
    # {'dir': 'pcn_amsterdam_20230707_13_27_46.450130', 'name': 'baseline_MODEL_UPDATES_ev2000'},
    # {'dir': 'pcn_amsterdam_20230707_13_27_59.221432', 'name': 'baseline_MODEL_UPDATES_ev2000'},

    {'dir': 'pcn_amsterdam_20230707_14_02_29.515148', 'name': 'baseline_MODEL_UPDATES_ev500'},
    {'dir': 'pcn_amsterdam_20230707_14_02_20.287639', 'name': 'baseline_MODEL_UPDATES_ev500'},
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
# models_to_plot = [{'dir': 'pcn_dilemma_20230705_18_42_35.696519', 'name': 'baseline'},
                #   {'dir': 'pcn_dilemma_20230705_18_44_06.188166', 'name': 'dft'},
                #   ]
models_to_plot = pd.DataFrame(models_to_plot)
colors = {'baseline':'red', 
          'dtf':'blue', 
          'dtf2':'green', 
          'dtf2@200k':'black', 
          'baseline_MODEL_UPDATES_ev1000': 'orange', 
          'baseline_MODEL_UPDATES_ev2000': 'purple',
          'baseline_MODEL_UPDATES_1000': 'black'
        }

fig, axs = plt.subplots(2, 2, figsize=(15, 8))
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


# %%
