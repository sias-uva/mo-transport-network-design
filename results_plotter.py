#%%
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

#% Plot all models
models_to_plot = [{'dir': 'pcn_amsterdam_20230705_11_14_58.880354', 'name': 'baseline'},
                  {'dir': 'pcn_amsterdam_20230705_13_06_15.465554', 'name': 'baseline'},
                  {'dir': 'pcn_amsterdam_20230705_14_24_30.249039', 'name': 'baseline'},
                #   {'dir': 'pcn_amsterdam_20230705_19_31_40.423497', 'name': 'dtf'},
                #   {'dir': 'pcn_amsterdam_20230705_19_33_45.190523', 'name': 'dtf'},

                #   {'dir': 'pcn_amsterdam_20230705_20_45_41.456186', 'name': 'dtf_30k'},
                #   {'dir': 'pcn_amsterdam_20230705_20_57_05.618977', 'name': 'dtf_30k'},
                #   {'dir': 'pcn_amsterdam_20230705_21_42_34.819344', 'name': 'dtf_30k'},

                #   {'dir': 'pcn_amsterdam_20230706_09_52_54.422550', 'name': 'dtf_shenanigans'},
                #   {'dir': 'pcn_amsterdam_20230706_09_53_27.891896', 'name': 'dtf_shenanigans'},

                  {'dir': 'pcn_amsterdam_20230706_10_25_46.730664', 'name': 'dtf2'},
                  {'dir': 'pcn_amsterdam_20230706_10_40_06.784805', 'name': 'dtf2'},
                #   {'dir': 'pcn_amsterdam_20230705_11_22_38.891947', 'name': 'dtf_er'},
                #   {'dir': 'pcn_amsterdam_20230705_13_04_01.109243', 'name': 'dtf_er'},
                #   {'dir': 'pcn_amsterdam_20230705_15_13_12.034815', 'name': 'dtf_er'},
                #   {'dir': 'pcn_amsterdam_20230705_11_26_05.996728', 'name': 'dtf_top10'},
                #   {'dir': 'pcn_amsterdam_20230705_13_58_50.833274', 'name': 'dtf_top10'},
                #   {'dir': 'pcn_amsterdam_20230705_15_55_34.410131', 'name': 'dtf_top10'},
                  ]
# models_to_plot = [{'dir': 'pcn_dilemma_20230705_18_42_35.696519', 'name': 'baseline'},
                #   {'dir': 'pcn_dilemma_20230705_18_44_06.188166', 'name': 'dft'},
                #   ]
models_to_plot = pd.DataFrame(models_to_plot)
colors = {'baseline':'red', 'dtf':'blue'}

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
    x = range(ci.shape[1])

    ci = calc_ci(models, 'train_hv')
    axs[0][0].plot(x, ci[1], label=name, color=colors[name])
    axs[0][0].fill_between(x, ci[0], ci[2], color=colors[name], alpha=0.3)
    axs[0][0].title.set_text('Train Hypervolume')
    axs[0][0].legend()

    ci = calc_ci(models, 'greedy_hv')
    axs[0][1].plot(x, ci[1], label=name, color=colors[name])
    axs[0][1].fill_between(x, ci[0], ci[2], color=colors[name], alpha=0.3)
    axs[0][1].title.set_text('Greedy Hypervolume')

    ci = calc_ci(models, 'eval_hv')
    axs[1][0].plot(x, ci[1], label=name, color=colors[name])
    axs[1][0].fill_between(x, ci[0], ci[2], color=colors[name], alpha=0.3)
    axs[1][0].title.set_text('Eval. Hypervolume')

# %%
