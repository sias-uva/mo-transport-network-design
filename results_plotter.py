#%%
import os
from typing import List
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import json
import seaborn as sns
plt.rcParams.update({'font.size': 18})
from pymoo.indicators.hv import HV
from morl_baselines.common.performance_indicators import hypervolume

#%%
def read_json(file_path):
    with open(file_path, 'r') as file:
        json_content = json.load(file)
    return json_content

def gini(x, normalized=True):
    sorted_x = np.sort(x, axis=1)
    n = x.shape[1]
    cum_x = np.cumsum(sorted_x, axis=1, dtype=float)
    gi = (n + 1 - 2 * np.sum(cum_x, axis=1) / cum_x[:, -1]) / n
    if normalized:
        gi = gi * (n / (n - 1))
    return gi

xian_result_dirs = read_json('./result_dirs_xian.txt')
xian_result_dirs_new = read_json('./result_dirs_xian_new.txt')
amsterdam_result_dirs = read_json('./result_dirs_ams.txt')
amsterdam_result_dirs_new = read_json('./result_dirs_ams_new.txt')

all_objectives = amsterdam_result_dirs_new

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

hvboxplot = sns.boxplot(data=hv, x="nr_groups", y="value", hue="model", ax=axs[0], legend=False, linewidth=LINEWIDTH)
# hvboxplot.legend_.set_title(None)
# hvboxplot.legend(fontsize=12)
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
axs[3].set_title('Normalized Sen Welfare (Efficiency * (1 - Gini))')
axs[3].set_xlabel('Number of Groups')
axs[3].set_ylabel(None)
fig.tight_layout()

# %% Show the mean and SE of the values for the table
sen_welfare.groupby(['model', 'nr_groups']).agg({'value': ['mean', lambda x: np.std(x, ddof=1) / np.sqrt(len(x))]}).round(2)
# hv.groupby(['model', 'nr_groups']).agg({'value': ['mean', lambda x: np.std(x, ddof=1) / np.sqrt(len(x))]}).round(2)
# eff.groupby(['model', 'nr_groups']).agg({'value': ['mean', lambda x: np.std(x, ddof=1) / np.sqrt(len(x))]}).round(2)
# gini_.groupby(['model', 'nr_groups']).agg({'value': ['mean', lambda x: np.std(x, ddof=1) / np.sqrt(len(x))]}).round(2)


#%%
## Same but only hv and sen_welfare
fig, axs = plt.subplots(2, 1, figsize=(7, 6))
pcnvlcn = all_results[all_results['model'].isin(['PCN', 'LCN_ND'])]
pcnvlcn.loc[pcnvlcn['model'] == 'LCN_ND', 'model'] = 'LCN'

hv = pcnvlcn[pcnvlcn['metric'] == 'hv']
hv['value'] = hv.groupby('nr_groups')['value'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))
hv['value'] = hv['value'].fillna(0)

hvboxplot = sns.boxplot(data=hv, x="nr_groups", y="value", hue="model", ax=axs[0], legend=True, linewidth=LINEWIDTH)
hvboxplot.legend_.set_title(None)
hvboxplot.legend(fontsize=14)
axs[0].set_title('Normalized Hypervolume')
axs[0].set_xlabel(None)
axs[0].set_ylabel(None)

sen_welfare = pcnvlcn[pcnvlcn['metric'] == 'sen_welfare']
sen_welfare['value'] = sen_welfare.groupby('nr_groups')['value'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))
sns.boxplot(data=sen_welfare, x="nr_groups", y="value", hue="model", ax=axs[1], legend=False, linewidth=LINEWIDTH)
axs[1].set_title('Normalized Sen Welfare')
axs[1].set_xlabel('Number of Groups')
axs[1].set_ylabel(None)

fig.tight_layout()
#%%
## Boxplot but only gini and sen_welfare
# fig, axs = plt.subplots(2, 1, figsize=(7, 6))
# pcnvlcn = all_results[all_results['model'].isin(['PCN', 'LCN_ND'])]
# pcnvlcn.loc[pcnvlcn['model'] == 'LCN_ND', 'model'] = 'LCN'

# hv = pcnvlcn[pcnvlcn['metric'] == 'hv']
# hv['value'] = hv.groupby('nr_groups')['value'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))
# hv['value'] = hv['value'].fillna(0)

# hvboxplot = sns.boxplot(data=hv, x="nr_groups", y="value", hue="model", ax=axs[0], legend=True, linewidth=LINEWIDTH)
# hvboxplot.legend_.set_title(None)
# hvboxplot.legend(fontsize=14)
# axs[0].set_title('Normalized Hypervolume')
# axs[0].set_xlabel(None)
# axs[0].set_ylabel(None)

# sen_welfare = pcnvlcn[pcnvlcn['metric'] == 'sen_welfare']
# sen_welfare['value'] = sen_welfare.groupby('nr_groups')['value'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))
# sns.boxplot(data=sen_welfare, x="nr_groups", y="value", hue="model", ax=axs[1], legend=False, linewidth=LINEWIDTH)
# axs[1].set_title('Normalized Sen Welfare')
# axs[1].set_xlabel('Number of Groups')
# axs[1].set_ylabel(None)

# fig.tight_layout()

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
sns.boxplot(data=hyperv, x="lambda", y="value", ax=axs[0], linewidth=2, width=0.6)
axs[0].set_title('Hypervolume')
axs[0].set_ylabel(None)
axs[0].set_xlabel(None)
axs[0].axhline(pcn_hv['value'].max(), ls='--', color='black', alpha=0.5)

sen_welfare = lambda_lcn[lambda_lcn['metric'] == 'sen_welfare']
sns.boxplot(data=sen_welfare, x="lambda", y="value", ax=axs[1], linewidth=2, width=0.6)
axs[1].set_title('Sen Welfare')
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

# %% MANUAL PLOT
# RUN IN DOWNLOADS

# lines = pd.read_csv('./wandb_export_2024-04-09T10 45 56.775+02 00.csv')
# fig, ax = plt.subplots(figsize=(10, 5))
# lines.plot(ax=ax, color=["#BFBFBF", "#1A85FF", "#E66100", "#D41159"], linewidth=3)
# ax.set_xlabel('Steps (NOT training steps)')
# ax.set_ylabel('Median Sen Welfare')

# #%% all lines, 2 objectives
# # RUN IN DOWNLOADS
# def generate_results(filenames):
#     lines = pd.DataFrame()
#     for results in filenames:
#         hv = pd.read_csv(results)
#         hv.columns = ['step', f'hv_{results}', 'hv_min', 'hv_max']
#         lines[f'hv_{results}'] = hv[f'hv_{results}']
    
#     lines['mean'] = lines.mean(axis=1)
#     lines['ub'] = lines['mean'] + 1.96 * lines.sem(axis=1)
#     lines['lb'] = lines['mean'] - 1.96 * lines.sem(axis=1)
    
#     return lines
    
# fig, ax = plt.subplots(figsize=(10, 5))
# pcn_results = generate_results(['pcn_2_42.csv', 'pcn_2_1234.csv', 'pcn_2_3405.csv'])
# pcn_results['mean'].plot(ax=ax, color=["#BFBFBF"], linewidth=3, label='PCN')
# ax.fill_between(pcn_results.index, pcn_results['lb'], pcn_results['ub'], facecolor='#BFBFBF', alpha=0.1)

# lcn_results = generate_results(['lcn_2_42.csv', 'lcn_2_1234.csv', 'lcn_2_3405.csv'])
# lcn_results['mean'].plot(ax=ax, color=["#1A85FF"], linewidth=3, label='LCN')
# ax.fill_between(lcn_results.index, lcn_results['lb'], lcn_results['ub'], facecolor='#1A85FF', alpha=0.1)

# lcn_optmax_results = generate_results(['lcn_2_optmax_42.csv', 'lcn_2_optmax_1234.csv', 'lcn_2_optmax_3405.csv'])
# lcn_optmax_results['mean'].plot(ax=ax, color=["#E66100"], linewidth=3, label='LCN_optmax')
# ax.fill_between(lcn_optmax_results.index, lcn_optmax_results['lb'], lcn_optmax_results['ub'], facecolor='#E66100', alpha=0.1)


# lcn_ndmean_results = generate_results(['lcn_2_ndmean_42.csv', 'lcn_2_ndmean_1234.csv', 'lcn_2_ndmean_3405.csv'])
# lcn_ndmean_results['mean'].plot(ax=ax, color=["#D41159"], linewidth=3, label='LCN_ndmean')
# ax.fill_between(lcn_ndmean_results.index, lcn_ndmean_results['lb'], lcn_ndmean_results['ub'], facecolor='#D41159', alpha=0.1)

# plt.legend()

# #% all lines, 5 objectives    
# fig, ax = plt.subplots(figsize=(10, 5))
# pcn_results = generate_results(['pcn_5_42.csv', 'pcn_5_1234.csv'])
# pcn_results['mean'].plot(ax=ax, color=["#BFBFBF"], linewidth=3, label='PCN')
# ax.fill_between(pcn_results.index, pcn_results['lb'], pcn_results['ub'], facecolor='#BFBFBF', alpha=0.1)

# lcn_results = generate_results(['lcn_5_42.csv', 'lcn_5_1234.csv'])
# lcn_results['mean'].plot(ax=ax, color=["#1A85FF"], linewidth=3, label='LCN')
# ax.fill_between(lcn_results.index, lcn_results['lb'], lcn_results['ub'], facecolor='#1A85FF', alpha=0.1)

# lcn_optmax_results = generate_results(['lcn_5_optmax_42.csv', 'lcn_5_optmax_1234.csv'])
# lcn_optmax_results['mean'].plot(ax=ax, color=["#E66100"], linewidth=3, label='LCN_optmax')
# ax.fill_between(lcn_optmax_results.index, lcn_optmax_results['lb'], lcn_optmax_results['ub'], facecolor='#E66100', alpha=0.1)


# lcn_ndmean_results = generate_results(['lcn_5_ndmean_42.csv', 'lcn_5_ndmean_1234.csv'])
# lcn_ndmean_results['mean'].plot(ax=ax, color=["#D41159"], linewidth=3, label='LCN_ndmean')
# ax.fill_between(lcn_ndmean_results.index, lcn_ndmean_results['lb'], lcn_ndmean_results['ub'], facecolor='#D41159', alpha=0.1)

# plt.legend()
# # %%
# x = np.array([[0.0044, 0.0045, 0.0062, 0.0061, 0.0044, 0.0053], 
#               [0.0023, 0.0055, 0.0053, 0.0066, 0.0054, 0.0036],
#               [0.0026, 0.0061, 0.0061, 0.0008, 0.0025, 0.0022],
#               [0.0023, 0.0055, 0.0075, 0.0066, 0.0031, 0.0027],
#               [0.0026, 0.0061, 0.0039, 0.0084, 0.0037, 0.0037],
#               [0.0026, 0.0061, 0.0039, 0.0081, 0.0026, 0.0051],
#               [0.0026, 0.0061, 0.0061, 0.0077, 0.0017, 0.0051]
#              ])

# np.round(x.sum() * (1 - gini(x)), 2)

#%% MANUAL Plot for the DC Paper
# from matplotlib.ticker import FormatStrFormatter
# plt.rcParams.update({'font.size': 16})

# hv = pd.read_csv('./hypervolume.csv')
# sw = pd.read_csv('./sen_welfare.csv')
# front = pd.read_csv('./front.csv')

# fig, axs = plt.subplots(1, 3, figsize=(17, 5))
# hv.plot(ax=axs[0], x='Step', y='hypervolume', linewidth=3, color='#88D18A', legend=False)
# axs[0].set_xlabel('Step')
# axs[0].set_title('Hypervolume')

# sw.plot(ax=axs[1], x='Step', y='sen_welfare_median', linewidth=3, color='#A833B9', legend=False)
# axs[1].set_xlabel('Step')
# axs[1].set_title('Sen Welfare')


# front.plot(kind='scatter', ax=axs[2], x='objective_1', y='objective_2', s=100)
# axs[2].set_title('Pareto Front')
# axs[2].set_xlabel('group 1')
# axs[2].set_ylabel('group 2')
# plt.gca().ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
# %%
