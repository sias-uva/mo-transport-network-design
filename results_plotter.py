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

#%%
xian_result_dirs = read_json('./result_dirs_xian.txt')
xian_result_dirs_new = read_json('./result_dirs_xian_new.txt')
amsterdam_result_dirs = read_json('./result_dirs_ams.txt')
amsterdam_result_dirs_new = read_json('./result_dirs_ams_new.txt')

all_objectives = xian_result_dirs_new

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
pcnvlcn = all_results[all_results['model'].isin(['PCN', 'GPILS', 'LCN_ND', 'LCN_OPTMAX', 'LCN_NDMEAN'])]
pcnvlcn.loc[pcnvlcn['model'] == 'GPILS', 'model'] = 'GPI-LS'
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
## Boxplot but only sen_welfare and cardinality
fig, axs = plt.subplots(2, 1, figsize=(7, 6))
pcnvlcn = all_results[all_results['model'].isin(['PCN', 'LCN_ND'])]
pcnvlcn.loc[pcnvlcn['model'] == 'LCN_ND', 'model'] = 'LCN'


sen_welfare = pcnvlcn[pcnvlcn['metric'] == 'sen_welfare']
sen_welfare['value'] = sen_welfare.groupby('nr_groups')['value'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))
sns.boxplot(data=sen_welfare, x="nr_groups", y="value", hue="model", ax=axs[0], legend=False, linewidth=LINEWIDTH)
axs[0].set_title('Normalized Sen Welfare')
axs[0].set_xlabel('Number of Groups')
axs[0].set_ylabel(None)


cardinality = pcnvlcn[pcnvlcn['metric'] == 'cardinality']
# cardinality['value'] = cardinality.groupby('nr_groups')['value'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))
cardinality['value'] = cardinality['value'].fillna(0)

hvboxplot = sns.boxplot(data=cardinality, x="nr_groups", y="value", hue="model", ax=axs[1], legend=True, linewidth=LINEWIDTH)
hvboxplot.legend_.set_title(None)
hvboxplot.legend(fontsize=14)
axs[1].set_title('Cardinality')
axs[1].set_xlabel(None)
axs[1].set_ylabel(None)


fig.tight_layout()

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

#%%
import wandb

api = wandb.Api()

# Replace with your project and run details
project_name = "MORL-TNDP"
REQ_SEEDS = 3 # to control if a model was not run for sufficient seeds

xian_result_runs = read_json('./result_dirs_xian_wandb.txt')
ams_result_runs = read_json('./result_dirs_ams_wandb.txt')

all_objectives = ams_result_runs

all_results = pd.DataFrame()
hv_over_time = pd.DataFrame()

def average_per_step(hvs_by_seed):
    # Determine the maximum length of the sublists
    max_length = max(len(sublist) for sublist in hvs_by_seed)
    
    # Pad shorter sublists with zeros
    padded_hvs_by_seed = [sublist + [sublist[-1]] * (max_length - len(sublist)) for sublist in hvs_by_seed]
    
    # Calculate the average per step
    averages = []
    for i in range(max_length):
        step_values = [sublist[i] for sublist in padded_hvs_by_seed]
        averages.append(sum(step_values) / len(hvs_by_seed))
    
    return averages

for oidx, objective in enumerate(all_objectives):
    nr_groups = objective['nr_groups']
    ref_point = np.array([0] * nr_groups)
    models_to_plot = pd.DataFrame(objective['models'])

    results_by_objective = {}
    groups = None
    for i, model_name in enumerate(models_to_plot['name'].unique()):
        models = models_to_plot[models_to_plot['name'] == model_name].to_dict('records')
        if len(models[0]['run_ids']) < REQ_SEEDS:
            print(f"!WARNING! {objective['nr_groups']} nrgroups, {model_name} does not have enough seeds (has {len(models[0]['run_ids'])}, while {REQ_SEEDS} are required)")

        hvs_by_seed = []
        for j, model in enumerate(models):
            # Read the content of the output file
            results_by_objective[model_name] = {'gini': [], 'total_efficiency': [], 'avg_efficiency': [], 'hv': [], 'sen_welfare': [], 'nash_welfare': [], 'avg_per_group': [], 'cardinality': []}
            for i in range(len(model['run_ids'])):
                if model['run_ids'][i] == '':
                    print(f"WARNING - Empty run id in {model_name}")
                    continue
                
                run = api.run(f"{project_name}/{model['run_ids'][i]}")
                                
                front_artifact = api.artifact(f'{project_name}/run-{model["run_ids"][i]}-evalfront:latest')
                local_path = f'./artifacts/{front_artifact.name}'
                if not os.path.exists(local_path):
                    front_artifact.download()
                    
                with open(f"{local_path}/eval/front.table.json", "r") as file:
                    fronts = json.load(file)['data']
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
                results_by_objective[model_name]['cardinality'] = results_by_objective[model_name]['cardinality'] + [run.summary['eval/cardinality']]
                
                history = []
                for row in run.scan_history(keys=['global_step', 'eval/hypervolume']):
                    history.append(row)

                # Convert to DataFrame
                history = pd.DataFrame(history)
                hv_values = history[history['eval/hypervolume'] > 0]['eval/hypervolume'].tolist()
                # if len(hv_values) > 0:
                hvs_by_seed.append(hv_values)
                # else:
                    # print(f"WARNING - No hypervolume values in {model_name}, {nr_groups} - {model['run_ids'][i]}")
                ### 
        if len(hvs_by_seed) > 0:
            averages = average_per_step(hvs_by_seed)
            hv_over_time = pd.concat([hv_over_time, pd.DataFrame({f"{model_name}_{nr_groups}_hv": averages})])

                    
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


# %%
plt.rcParams.update({'font.size': 36})
fig, axs = plt.subplots(3, 3, figsize=(50, 20))
LINEWIDTH = 8
for i in range(9):
    row = i // 3
    col = i % 3
    
    group = i + 2
    
    axs[row, col].plot(hv_over_time[f'PCN_{group}_hv'], label='PCN', linewidth=LINEWIDTH)
    # axs[row, col].plot(hv_over_time[f'GPILS_{group}_hv'], label='GPILS')
    axs[row, col].plot(hv_over_time[f'LCN_ND_{group}_hv'], label='LCN_ND', linewidth=LINEWIDTH)
    axs[row, col].plot(hv_over_time[f'LCN_OPTMAX_{group}_hv'], label='LCN_OPTMAX', linewidth=LINEWIDTH)
    axs[row, col].plot(hv_over_time[f'LCN_NDMEAN_{group}_hv'], label='LCN_NDMEAN', linewidth=LINEWIDTH)
    
    
    # axs[row, col].set_title(group)
    axs[row, col].set_xlabel('Step')
    axs[row, col].set_ylabel('Hypervolume')
    axs[row, col].set_title(f'{group} Groups')
    

fig.legend(['PCN', 'LCN_ND', 'LCN_OPTMAX', 'LCN_NDMEAN'], loc='lower center', ncol=4)
fig.tight_layout()