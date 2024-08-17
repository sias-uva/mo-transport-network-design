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
import wandb

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

#%% LOAD DATA FROM W&B
api = wandb.Api()

# Replace with your project and run details
REQ_SEEDS = 5 # to control if a model was not run for sufficient seeds

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

def load_all_results_from_wadb(all_objectives, env_name=None):
    if env_name == 'DST':
        project_name = 'DST'
    else:
        project_name = 'MORL-TNDP'
        
    all_results = pd.DataFrame()
    hv_over_time = pd.DataFrame()
    eum_over_time = pd.DataFrame()
    sw_over_time = pd.DataFrame()

    for oidx, objective in enumerate(all_objectives):
        nr_groups = objective['nr_groups']
        if env_name == 'DST':
            ref_point = np.array([0.0, -200.0])
        else:
            ref_point = np.array([0] * nr_groups)
        models_to_plot = pd.DataFrame(objective['models'])

        results_by_objective = {}
        groups = None
        for i, model_name in enumerate(models_to_plot['name'].unique()):
            models = models_to_plot[models_to_plot['name'] == model_name].to_dict('records')
            if len(models[0]['run_ids']) < REQ_SEEDS:
                print(f"!WARNING! {objective['nr_groups']} nrgroups, {model_name} does not have enough seeds (has {len(models[0]['run_ids'])}, while {REQ_SEEDS} are required)")

            hvs_by_seed = []
            eum_by_seed = []
            sw_by_seed = []
            for j, model in enumerate(models):
                print(f"Processing {env_name} {model_name} ({nr_groups}) - {model['run_ids']}")
                # Read the content of the output file
                results_by_objective[model_name] = {'gini': [], 'total_efficiency': [], 'avg_efficiency': [], 'hv': [], 'sen_welfare': [], 'nash_welfare': [], 'avg_per_group': [], 'cardinality': [], 'eum': []}
                for i in range(len(model['run_ids'])):
                    if model['run_ids'][i] == '':
                        print(f"WARNING - Empty run id in {model_name}")
                        continue
                    
                    run = api.run(f"{project_name}/{model['run_ids'][i]}")
                    
                    if 'lcn_lambda' in run.config and run.config['lcn_lambda'] is not None:
                        run_algo = f"{run.config['algo']}-Lambda-{run.config['lcn_lambda']}"
                    elif 'distance_ref' in run.config and run.config['distance_ref'] is not None:
                        run_algo = f"{run.config['algo']}-{run.config['distance_ref']}"
                    else:
                        run_algo = run.config['algo']
                        
                    if run_algo != model_name:
                        print(f"!ERROR! {model_name} has algo {model_name}, while {run_algo} is required")
                                    
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
                    if env_name != 'DST':
                        results_by_objective[model_name]['sen_welfare'] = results_by_objective[model_name]['sen_welfare'] + (total_efficiency * (1-gini_index)).tolist()
                    results_by_objective[model_name]['nash_welfare'] = results_by_objective[model_name]['nash_welfare'] + nash_welfare.tolist()
                    results_by_objective[model_name]['avg_per_group'] = results_by_objective[model_name]['avg_per_group'] + np.mean(fronts, axis=0).tolist()
                    results_by_objective[model_name]['cardinality'] = results_by_objective[model_name]['cardinality'] + [run.summary['eval/cardinality']]
                    results_by_objective[model_name]['eum'] = results_by_objective[model_name]['eum'] + [run.summary['eval/eum']]

                    if env_name == 'DST':
                        keys_to_load = ['global_step', 'eval/hypervolume', 'eval/eum']
                    else:
                        keys_to_load = ['global_step', 'eval/hypervolume', 'eval/eum', 'eval/sen_welfare_median']
                    history = []
                    for row in run.scan_history(keys=keys_to_load):
                        history.append(row)

                    # Convert to DataFrame
                    history = pd.DataFrame(history)
                    hv_values = history[history['eval/hypervolume'] > 0]['eval/hypervolume'].tolist()
                    if len(hv_values) > 0:
                        hvs_by_seed.append(hv_values)
                    else:
                        print(f"WARNING - No hypervolume values in {model_name}, {nr_groups} - {model['run_ids'][i]}")
                        
                    eum_values = history[history['eval/eum'] > 0]['eval/eum'].tolist()
                    if len(eum_values) > 0:
                        eum_by_seed.append(eum_values)
                    else:
                        print(f"WARNING - No EUM values in {model_name}, {nr_groups} - {model['run_ids'][i]}")
                    
                    if env_name != 'DST':
                        sw_values = history[history['eval/sen_welfare_median'] > 0]['eval/sen_welfare_median'].tolist()
                        if len(sw_values) > 0:
                            sw_by_seed.append(sw_values)
                        else:
                            print(f"WARNING - No SW values in {model_name}, {nr_groups} - {model['run_ids'][i]}")

                    ###
            model_name_adj = model_name.replace(f'-{env_name}', '')
            if len(hvs_by_seed) > 0:
                averages = average_per_step(hvs_by_seed)
                hv_over_time = pd.concat([hv_over_time, pd.DataFrame({f"{model_name_adj}_{nr_groups}": averages})])
            
            if len(eum_by_seed) > 0:
                averages = average_per_step(eum_by_seed)
                eum_over_time = pd.concat([eum_over_time, pd.DataFrame({f"{model_name_adj}_{nr_groups}": averages})])
                
            if len(sw_by_seed) > 0:
                averages = average_per_step(sw_by_seed)
                sw_over_time = pd.concat([sw_over_time, pd.DataFrame({f"{model_name_adj}_{nr_groups}": averages})])


                        
            # results_by_objective[model_name]['lambda'] = model['lambda'] if 'lambda' in model else ''
        # Quite a hacky way to get the results in a dataframe, but didn't have time to do it properly (thanks copilot)
        # Convert all_results to a dataframe, with columns 'model', 'metric', 'value', and each row is a different value and not a list
        # results_by_objective = pd.DataFrame([(name, metric, value) for name in results_by_objective.keys() for metric in results_by_objective[name].keys() for value in results_by_objective[name][metric]], columns=['model', 'metric', 'value'])
        # Convert all_results to a dataframe, with columns 'model', 'lambda; 'metric', 'value', and each row is a different value and not a list
        results_by_objective = pd.DataFrame([(name, model['lambda'] if 'lambda' in model else None, metric, value) for name in results_by_objective.keys() 
                                            for metric in results_by_objective[name].keys() 
                                            for value in results_by_objective[name][metric]], columns=['model', 'lambda', 'metric', 'value'])
        results_by_objective['model'] = results_by_objective['model'].str.replace(f'-{env_name}', '')
        results_by_objective['nr_groups'] = nr_groups
        results_by_objective['lambda'] = results_by_objective[results_by_objective['model'].str.contains('Lambda')]['model'].str.split('-').str[-1].astype(float)
        results_by_objective.loc[results_by_objective['model'].str.contains('Lambda'), 'model'] = 'LCN_Lambda'
        all_results = pd.concat([all_results, results_by_objective])
        
    return all_results, hv_over_time, eum_over_time, sw_over_time

# ams_results, ams_hv_over_time, ams_eum_over_time, ams_sw_over_time = load_all_results_from_wadb(read_json('./result_ids_ams.txt'), 'Amsterdam')
# xian_results, xian_hv_over_time, xian_eum_over_time, xian_sw_over_time = load_all_results_from_wadb(read_json('./result_ids_xian.txt'), 'Xian')

dst_results, dst_hv_over_time, dst_eum_over_time, dst_sw_over_time = load_all_results_from_wadb(read_json('./result_ids_dst.txt'), 'DST')
#%%

# Change this to the results you want to plot
results_to_plot = dst_results

# Plot Total Efficiency, Gini Index, Sen Welfare for PCN vs LCN (ND, OPTMAX, NDMEAN)
fig, axs = plt.subplots(4, 1, figsize=(10, 12))
pcnvlcn = results_to_plot[results_to_plot['model'].isin(['PCN', 'GPI-LS', 'LCN-nondominated', 'LCN-optimal_max', 'LCN-nondominated_mean'])]
pcnvlcn.loc[pcnvlcn['model'] == 'LCN-nondominated', 'model'] = 'LCN'
pcnvlcn.loc[pcnvlcn['model'] == 'LCN-optimal_max', 'model'] = 'LCN-Redist'
pcnvlcn.loc[pcnvlcn['model'] == 'LCN-nondominated_mean', 'model'] = 'LCN-Mean'

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
# sen_welfare.groupby(['model', 'nr_groups']).agg({'value': ['mean', lambda x: np.std(x, ddof=1) / np.sqrt(len(x))]}).round(2)
hv.groupby(['model', 'nr_groups']).agg({'value': ['mean', lambda x: np.std(x, ddof=1) / np.sqrt(len(x))]}).round(2)
# eff.groupby(['model', 'nr_groups']).agg({'value': ['mean', lambda x: np.std(x, ddof=1) / np.sqrt(len(x))]}).round(2)
# gini_.groupby(['model', 'nr_groups']).agg({'value': ['mean', lambda x: np.std(x, ddof=1) / np.sqrt(len(x))]}).round(2)


#%%
## Same but only hv and sen_welfare
fig, axs = plt.subplots(3, 1, figsize=(8, 8))
colors = ["#FFF2E5", "#BFBFBF", "#1A85FF"]
sns.set_palette(sns.color_palette(colors))
plt.rcParams.update({'font.size': 16})
LINEWIDTH = 1.5

pcnvlcn = results_to_plot[results_to_plot['model'].isin(['GPI-LS', 'PCN', 'LCN-nondominated'])]
pcnvlcn.loc[pcnvlcn['model'] == 'LCN-nondominated', 'model'] = 'LCN'

hv = pcnvlcn[pcnvlcn['metric'] == 'hv']
hv['value'] = hv.groupby('nr_groups')['value'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))
hv['value'] = hv['value'].fillna(0)

hvboxplot = sns.boxplot(data=hv, x="nr_groups", y="value", hue="model", ax=axs[0], legend=False, linewidth=LINEWIDTH)
# hvboxplot.legend_.set_title(None)
# hvboxplot.legend(fontsize=14)
axs[0].set_title('Normalized Hypervolume')
axs[0].set_xlabel(None)
axs[0].set_ylabel(None)
axs[0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

eum = pcnvlcn[pcnvlcn['metric'] == 'eum']
eum['value'] = eum.groupby('nr_groups')['value'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))
sns.boxplot(data=eum, x="nr_groups", y="value", hue="model", ax=axs[1], legend=False, linewidth=LINEWIDTH)
axs[1].set_title('Normalized EUM')
axs[1].set_ylabel(None)
axs[1].set_xlabel(None)
axs[1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

sen_welfare = pcnvlcn[pcnvlcn['metric'] == 'sen_welfare']
sen_welfare['value'] = sen_welfare.groupby('nr_groups')['value'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))
eumboxplot = sns.boxplot(data=sen_welfare, x="nr_groups", y="value", hue="model", ax=axs[2], legend=True, linewidth=LINEWIDTH)
axs[2].set_title('Normalized Sen Welfare')
axs[2].set_ylabel(None)
axs[2].set_xlabel('Number of Groups')
axs[2].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

eumboxplot.legend(fontsize=14, loc='upper center', bbox_to_anchor=(0.5, -0.6), ncol=3)
eumboxplot.legend_.set_title(None)

fig.tight_layout()

#%%
## Box plot of HV, Sen Welfare, EUM for all LCN models
fig, axs = plt.subplots(3, 1, figsize=(8, 8))
colors = ["#1A85FF", "#E66100", "#D41159", "#BFBFBF"]
sns.set_palette(sns.color_palette(colors))
plt.rcParams.update({'font.size': 16})

all_lcn = results_to_plot[results_to_plot['model'].isin(['LCN-nondominated', 'LCN-optimal_max', 'LCN-nondominated_mean'])]
all_lcn.loc[all_lcn['model'] == 'LCN-nondominated', 'model'] = 'LCN'
all_lcn.loc[all_lcn['model'] == 'LCN-optimal_max', 'model'] = 'LCN-Redist'
all_lcn.loc[all_lcn['model'] == 'LCN-nondominated_mean', 'model'] = 'LCN-Mean'

hv = all_lcn[all_lcn['metric'] == 'hv']
hv['value'] = hv.groupby('nr_groups')['value'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))
hv['value'] = hv['value'].fillna(0)

sns.boxplot(data=hv, x="nr_groups", y="value", hue="model", ax=axs[0], legend=False, linewidth=LINEWIDTH)
axs[0].set_title('Normalized Hypervolume')
axs[0].set_xlabel(None)
axs[0].set_ylabel(None)

eum = all_lcn[all_lcn['metric'] == 'eum']
eum['value'] = eum.groupby('nr_groups')['value'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))
sns.boxplot(data=eum, x="nr_groups", y="value", hue="model", ax=axs[1], legend=False, linewidth=LINEWIDTH)
axs[1].set_title('Normalized EUM')
axs[1].set_ylabel(None)
axs[1].set_xlabel(None)

sen_welfare = all_lcn[all_lcn['metric'] == 'sen_welfare']
sen_welfare['value'] = sen_welfare.groupby('nr_groups')['value'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))
eumboxplot = sns.boxplot(data=sen_welfare, x="nr_groups", y="value", hue="model", ax=axs[2], legend=True, linewidth=LINEWIDTH)
axs[2].set_title('Normalized Sen Welfare')
axs[2].set_ylabel(None)
axs[2].set_xlabel('Number of Groups')

eumboxplot.legend(fontsize=14, loc='upper center', bbox_to_anchor=(0.5, -0.6), ncol=3)
eumboxplot.legend_.set_title(None)
fig.tight_layout()

#%%
## Boxplot but only sen_welfare and cardinality
fig, axs = plt.subplots(2, 1, figsize=(7, 6))
pcnvlcn = results_to_plot[results_to_plot['model'].isin(['PCN', 'LCN_ND'])]
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
pcn_eff = results_to_plot[(results_to_plot['model']=='PCN') & (results_to_plot['nr_groups']==NR_GROUPS_TO_PLOT) & (results_to_plot['metric']=='total_efficiency')]
pcn_hv = results_to_plot[(results_to_plot['model']=='PCN') & (results_to_plot['nr_groups']==NR_GROUPS_TO_PLOT) & (results_to_plot['metric']=='hv')]
pcn_gini = results_to_plot[(results_to_plot['model']=='PCN') & (results_to_plot['nr_groups']==NR_GROUPS_TO_PLOT) & (results_to_plot['metric']=='gini')]
pcn_sen_welfare = results_to_plot[(results_to_plot['model']=='PCN') & (results_to_plot['nr_groups']==NR_GROUPS_TO_PLOT) & (results_to_plot['metric']=='sen_welfare')]

fig, axs = plt.subplots(4, 1, figsize=(12, 14))
lambda_lcn = results_to_plot[(results_to_plot['model'].isin(['LCN_Lambda'])) & (results_to_plot['lambda'].isin([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]))]
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
lambda_lcn = results_to_plot[(results_to_plot['model'].isin(['LCN_Lambda'])) & (results_to_plot['lambda'].isin([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]))]
lambda_lcn = lambda_lcn[lambda_lcn['nr_groups'] == NR_GROUPS_TO_PLOT]

hyperv = lambda_lcn[lambda_lcn['metric'] == 'hv']
sns.boxplot(data=hyperv, x="lambda", y="value", ax=axs[0], linewidth=2, width=0.6)
axs[0].set_title('Hypervolume')
axs[0].set_ylabel(None)
axs[0].set_xlabel(None)
axs[0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
# axs[0].axhline(pcn_hv['value'].max(), ls='--', color='black', alpha=0.5)

sen_welfare = lambda_lcn[lambda_lcn['metric'] == 'sen_welfare']
sns.boxplot(data=sen_welfare, x="lambda", y="value", ax=axs[1], linewidth=2, width=0.6)
axs[1].set_title('Sen Welfare')
axs[1].set_ylabel(None)
axs[1].set_xlabel('λ')
axs[1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
# axs[1].axhline(pcn_sen_welfare['value'].quantile(0.25), ls='--', color='black', alpha=0.5)
# axs[1].axhline(pcn_sen_welfare['value'].quantile(0.75), ls='--', color='black', alpha=0.5)
# axs[1].axhline(pcn_sen_welfare['value'].max(), ls='--', color='black', alpha=0.5)
# fig.suptitle(f'λ-LCN with {NR_GROUPS_TO_PLOT} groups', fontsize=20)
# fig.text(0.5, 0.95, 'Gray Horizontal Line indicates the best solution found by PCN', ha='center', va='center', fontsize=14, color='gray')
fig.tight_layout()
# %% Plot EUM for 3, 6, 9 objectives

def plot_over_time_results(ams_metric, xian_metric, groups, figsize, ylabel, linewidth=8, font_size=36):
    plt.rcParams.update({'font.size': font_size})
    fig = plt.figure(constrained_layout=True, figsize=figsize)
    colors = ["#BFBFBF", "#1A85FF", "#E66100", "#D41159"]
    sns.set_palette(sns.color_palette(colors))
    LINEWIDTH = linewidth

    subfigs = fig.subfigures(nrows=2, ncols=1, wspace=0.4, hspace=0.1)  # Add spacing between subfigures
    subfigs[0].suptitle(f"Xi'an")
    subfigs[1].suptitle(f"Amsterdam")

    axs_xian = subfigs[0].subplots(nrows=1, ncols=len(groups))
    axs_ams = subfigs[1].subplots(nrows=1, ncols=len(groups))
    axs_xian[0].set_ylabel(ylabel)
    axs_ams[0].set_ylabel(ylabel)
    for i, group in enumerate(groups):
        axs_xian[i].plot(xian_metric[f'PCN_{group}'], label='PCN', linewidth=LINEWIDTH)
        axs_xian[i].plot(xian_metric[f'LCN-nondominated_{group}'], label='LCN_ND', linewidth=LINEWIDTH)
        axs_xian[i].plot(xian_metric[f'LCN-optimal_max_{group}'], label='LCN_OPTMAX', linewidth=LINEWIDTH)
        axs_xian[i].plot(xian_metric[f'LCN-nondominated_mean_{group}'], label='LCN_NDMEAN', linewidth=LINEWIDTH)
    
        axs_xian[i].set_xlabel('Step')
        axs_xian[i].set_title(f'{group} Objectives')
        axs_xian[i].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
        axs_ams[i].plot(ams_metric[f'PCN_{group}'], label='PCN', linewidth=LINEWIDTH)
        axs_ams[i].plot(ams_metric[f'LCN-nondominated_{group}'], label='LCN_ND', linewidth=LINEWIDTH)
        axs_ams[i].plot(ams_metric[f'LCN-optimal_max_{group}'], label='LCN_OPTMAX', linewidth=LINEWIDTH)
        axs_ams[i].plot(ams_metric[f'LCN-nondominated_mean_{group}'], label='LCN_NDMEAN', linewidth=LINEWIDTH)
    
        axs_ams[i].set_xlabel('Step')
        axs_ams[i].set_title(f'{group} Objectives')
        axs_ams[i].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    subfigs[1].legend(['PCN', 'LCN', 'LCN-Redist', 'LCN-Mean'], loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.2))


# Plot EUM for 3, 6, 9 objectives
# plot_over_time_results(ams_eum_over_time, xian_eum_over_time, [3, 6, 9], (40, 15), 'EUM')
# Plot EUM for all objectives
# plot_over_time_results(ams_eum_over_time, xian_eum_over_time, range(2, 11), (50, 15), 'EUM')
plot_over_time_results(ams_eum_over_time, xian_eum_over_time, [3, 10], (15, 8), 'EUM', linewidth=4, font_size=22)

# Plot SW for 3, 6, 9 objectives
# plot_over_time_results(ams_sw_over_time, xian_sw_over_time, [3, 6, 9], (40, 15), 'Sen Welfare')
plot_over_time_results(ams_sw_over_time, xian_sw_over_time, [3, 10], (15, 8), 'Sen Welfare', linewidth=4, font_size=22)
# %%

fig, ax = plt.subplots(figsize=(7, 6))
xian_eum_lambda_lcn = xian_sw_over_time[['LCN-Lambda-0_3', 'LCN-Lambda-0.3_3', 'LCN-Lambda-0.6_3', 'LCN-Lambda-0.9_3']]
xian_eum_lambda_lcn.plot(ax=ax, linewidth=4, colormap='Blues')

# %%
