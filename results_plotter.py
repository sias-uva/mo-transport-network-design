#%%
import os
from typing import List
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import json
import seaborn as sns
plt.rcParams.update({'font.size': 18})
from morl_baselines.common.weights import equally_spaced_weights
from pymoo.indicators.hv import HV
from morl_baselines.common.performance_indicators import hypervolume, expected_utility
import wandb

# Fair weights per nr_obectives, to be used to generate the fair-expected-utility metric.
fair_weights_dict = np.load('fair_weights_dict.npy', allow_pickle=True).item()

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

def euclidean_distance_to_equality_ref_point(ref_point: np.array, fronts: np.array):    
    # Calculate Euclidean distance between each front and equality point
    distances = np.sqrt(np.sum((fronts - ref_point) ** 2, axis=1))
    
    return distances

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
            # THIS IS SPECIFIC TO THE Xi'an and Amsterdam environments, where 0.1 is very large for a single transport line.
            eq_ref_point = np.array([0.1 * nr_groups])
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
            distances_by_seed = []
            for j, model in enumerate(models):
                print(f"Processing {env_name} {model_name} ({nr_groups}) - {model['run_ids']}")
                # Read the content of the output file
                results_by_objective[model_name] = {'gini': [], 'total_efficiency': [], 'avg_efficiency': [], 'hv': [], 'sen_welfare': [], 'nash_welfare': [], 'dist_to_eq_ref_point': [], 'fair_eum': [], 'avg_per_group': [], 'cardinality': [], 'eum': []}
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
                    
                    has_large_point = any(np.all(front >= eq_ref_point) for front in fronts)
                    if has_large_point:
                        print(f"WARNING - Point > equality reference point in {model_name}, {nr_groups} - {model['run_ids'][i]}")
                    else:
                        dist_to_eq_ref = euclidean_distance_to_equality_ref_point(eq_ref_point, np.array(fronts))
                        
                    fair_eum = expected_utility(fronts, fair_weights_dict[objective['nr_groups']])
                                                
                    results_by_objective[model_name]['fronts'] = fronts
                    results_by_objective[model_name]['total_efficiency'] = results_by_objective[model_name]['total_efficiency'] + total_efficiency.tolist()
                    results_by_objective[model_name]['avg_efficiency'] = results_by_objective[model_name]['avg_efficiency'] + avg_efficiency.tolist()
                    results_by_objective[model_name]['hv'] = results_by_objective[model_name]['hv'] + [hv]
                    results_by_objective[model_name]['gini'] = results_by_objective[model_name]['gini'] + gini_index.tolist()
                    if env_name != 'DST':
                        results_by_objective[model_name]['sen_welfare'] = results_by_objective[model_name]['sen_welfare'] + (total_efficiency * (1-gini_index)).tolist()
                    results_by_objective[model_name]['nash_welfare'] = results_by_objective[model_name]['nash_welfare'] + nash_welfare.tolist()
                    results_by_objective[model_name]['dist_to_eq_ref_point'] = results_by_objective[model_name]['dist_to_eq_ref_point'] + dist_to_eq_ref.tolist()
                    results_by_objective[model_name]['fair_eum'] = results_by_objective[model_name]['fair_eum'] + [fair_eum]
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

ams_results, ams_hv_over_time, ams_eum_over_time, ams_sw_over_time = load_all_results_from_wadb(read_json('./result_ids_ams.txt'), 'Amsterdam')
xian_results, xian_hv_over_time, xian_eum_over_time, xian_sw_over_time = load_all_results_from_wadb(read_json('./result_ids_xian.txt'), 'Xian')

# dst_results, dst_hv_over_time, dst_eum_over_time, dst_sw_over_time = load_all_results_from_wadb(read_json('./result_ids_dst.txt'), 'DST')
#%%

# Change this to the results you want to plot
results_to_plot = xian_results

# Plot Total Efficiency, Gini Index, Sen Welfare for PCN vs LCN (ND, OPTMAX, NDMEAN)
fig, axs = plt.subplots(5, 1, figsize=(12, 12))
pcnvlcn = results_to_plot[results_to_plot['model'].isin(['GPI-LS', 'PCN', 'LCN-nondominated', 'LCN-optimal_max', 'LCN-nondominated_mean'])]
pcnvlcn.loc[pcnvlcn['model'] == 'LCN-nondominated', 'model'] = 'LCN'
pcnvlcn.loc[pcnvlcn['model'] == 'LCN-optimal_max', 'model'] = 'LCN-Redist'
pcnvlcn.loc[pcnvlcn['model'] == 'LCN-nondominated_mean', 'model'] = 'LCN-Mean'

colors = ["#FFF2E5", '#BFBFBF', "#1A85FF", "#E66100", "#D41159"]
sns.set_palette(sns.color_palette(colors))
palette = {
    'GPI-LS': colors[0],
    'PCN': colors[1],
    'LCN': colors[2],
    'LCN-Redist': colors[3],
    'LCN-Mean': colors[4]
}

LINEWIDTH = 1.5

hv = pcnvlcn[pcnvlcn['metric'] == 'hv']
# hv['value'] = hv.groupby('nr_groups')['value'].transform(lambda x: x / x.max())
# Normalize hv['value'] 
hv['value'] = hv.groupby('nr_groups')['value'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))
hv['value'] = hv['value'].fillna(0)

hvboxplot = sns.boxplot(data=hv, x="nr_groups", y="value", hue="model", palette=palette, ax=axs[0], legend=None, linewidth=LINEWIDTH)
# hvboxplot.legend_.set_title(None)
# hvboxplot.legend(fontsize=12)
axs[0].set_title('Normalized Hypervolume')
axs[0].set_xlabel(None)
axs[0].set_ylabel(None)
axs[0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

eum = pcnvlcn[pcnvlcn['metric'] == 'eum']
# eum['value'] = eum.groupby('nr_groups')['value'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))
eum['value'] = eum['value'].fillna(0)

sns.boxplot(data=eum, x="nr_groups", y="value", hue="model", ax=axs[1], legend=False, linewidth=LINEWIDTH)
axs[1].set_title('EUM')
axs[1].set_xlabel(None)
axs[1].set_ylabel(None)
axs[1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

eff = pcnvlcn[pcnvlcn['metric'] == 'total_efficiency']
eff['value'] = eff.groupby('nr_groups')['value'].transform(lambda x: x / x.max())
bxplot = sns.boxplot(data=eff, x="nr_groups", y="value", hue="model", ax=axs[2], legend=False, linewidth=LINEWIDTH)
# bxplot.legend_.set_title(None)
# bxplot.legend(fontsize=12)
axs[2].set_title('Total Efficiency')
axs[2].set_xlabel(None)
axs[2].set_ylabel(None)
axs[2].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

gini_ = pcnvlcn[pcnvlcn['metric'] == 'gini']
sns.boxplot(data=gini_, x="nr_groups", y="value", hue="model", ax=axs[3], legend=None, linewidth=LINEWIDTH)
axs[3].set_title('Gini')
axs[3].set_xlabel(None)
axs[3].set_ylabel(None)
axs[3].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

sen_welfare = pcnvlcn[pcnvlcn['metric'] == 'sen_welfare']
sen_welfare['value'] = sen_welfare.groupby('nr_groups')['value'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))
swbxplot = sns.boxplot(data=sen_welfare, x="nr_groups", y="value", hue="model", ax=axs[4], legend='brief', linewidth=LINEWIDTH)
swbxplot.legend_.set_title(None)
axs[4].legend(fontsize=12, loc='upper center', bbox_to_anchor=(0.5, -0.6), ncol=5)
axs[4].set_title('Normalized Sen Welfare (Efficiency * (1 - Gini))')
axs[4].set_xlabel('Number of Groups')
axs[4].set_ylabel(None)
axs[4].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
fig.tight_layout()

# %% Show the mean and SE of the values for the table
# sen_welfare.groupby(['model', 'nr_groups']).agg({'value': ['mean', lambda x: np.std(x, ddof=1) / np.sqrt(len(x))]}).round(2)
hv.groupby(['model', 'nr_groups']).agg({'value': ['mean', lambda x: np.std(x, ddof=1) / np.sqrt(len(x))]}).round(2)
# eff.groupby(['model', 'nr_groups']).agg({'value': ['mean', lambda x: np.std(x, ddof=1) / np.sqrt(len(x))]}).round(2)
# gini_.groupby(['model', 'nr_groups']).agg({'value': ['mean', lambda x: np.std(x, ddof=1) / np.sqrt(len(x))]}).round(2)


#%%
## Same but only hv and sen_welfare
fig, axs = plt.subplots(2, 2, figsize=(16, 8))
colors = ["#FFF2E5", "#BFBFBF", "#1A85FF"]
sns.set_palette(sns.color_palette(colors))
plt.rcParams.update({'font.size': 18})
LINEWIDTH = 2

pcnvlcn = results_to_plot[results_to_plot['model'].isin(['GPI-LS', 'PCN', 'LCN-nondominated'])]
pcnvlcn.loc[pcnvlcn['model'] == 'LCN-nondominated', 'model'] = 'LCN'

hv = pcnvlcn[pcnvlcn['metric'] == 'hv']
hv['value'] = hv.groupby('nr_groups')['value'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))
hv['value'] = hv['value'].fillna(0)

hvboxplot = sns.boxplot(data=hv, x="nr_groups", y="value", hue="model", ax=axs[0,0], legend=False, linewidth=LINEWIDTH)
axs[0,0].set_title('Normalized Hypervolume')
axs[0,0].set_xlabel(None)
axs[0,0].set_ylabel(None)
axs[0,0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

eum = pcnvlcn[pcnvlcn['metric'] == 'eum']
eum['value'] = eum.groupby('nr_groups')['value'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))
sns.boxplot(data=eum, x="nr_groups", y="value", hue="model", ax=axs[0,1], legend=False, linewidth=LINEWIDTH)
axs[0,1].set_title('Normalized EUM')
axs[0,1].set_ylabel(None)
axs[0,1].set_xlabel(None)
axs[0,1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

sen_welfare = pcnvlcn[pcnvlcn['metric'] == 'sen_welfare']
sen_welfare['value'] = sen_welfare.groupby('nr_groups')['value'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))
swboxplot = sns.boxplot(data=sen_welfare, x="nr_groups", y="value", hue="model", ax=axs[1,0], legend=False, linewidth=LINEWIDTH)
axs[1,0].set_title('Normalized Sen Welfare')
axs[1,0].set_ylabel(None)
axs[1,0].set_xlabel('Number of Groups')
axs[1,0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

dist_to_ref = pcnvlcn[pcnvlcn['metric'] == 'dist_to_eq_ref_point']
dist_to_ref['value'] = dist_to_ref.groupby('nr_groups')['value'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))
dtrboxplot = sns.boxplot(data=dist_to_ref, x="nr_groups", y="value", hue="model", ax=axs[1,1], legend='brief', linewidth=LINEWIDTH)
# Necessary to have a legend for it to apepar below, but then we need to remove it.
handles, labels = dtrboxplot.get_legend_handles_labels()
dtrboxplot.get_legend().remove()
axs[1,1].set_title('Normalized Distance to Utopian Point (lower=better)')
axs[1,1].set_ylabel(None)
axs[1,1].set_xlabel('Number of Groups')
axs[1,1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

# Move legend below the entire figure
fig.legend(handles, labels, fontsize=18, loc='center', bbox_to_anchor=(0.5, -0.02), ncol=3)

fig.tight_layout()

#%%
## Box plot of HV, Sen Welfare, EUM for all LCN models
fig, axs = plt.subplots(2, 2, figsize=(16, 8))
colors = ["#1A85FF", "#E66100", "#D41159", "#BFBFBF"]
sns.set_palette(sns.color_palette(colors))
plt.rcParams.update({'font.size': 18})

all_lcn = results_to_plot[results_to_plot['model'].isin(['LCN-nondominated', 'LCN-optimal_max', 'LCN-nondominated_mean'])]
all_lcn.loc[all_lcn['model'] == 'LCN-nondominated', 'model'] = 'LCN'
all_lcn.loc[all_lcn['model'] == 'LCN-optimal_max', 'model'] = 'LCN-Redist'
all_lcn.loc[all_lcn['model'] == 'LCN-nondominated_mean', 'model'] = 'LCN-Mean'

hv = all_lcn[all_lcn['metric'] == 'hv']
hv['value'] = hv.groupby('nr_groups')['value'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))
hv['value'] = hv['value'].fillna(0)

sns.boxplot(data=hv, x="nr_groups", y="value", hue="model", ax=axs[0,0], legend=False, linewidth=LINEWIDTH)
axs[0,0].set_title('Normalized Hypervolume')
axs[0,0].set_xlabel(None)
axs[0,0].set_ylabel(None)

eum = all_lcn[all_lcn['metric'] == 'eum']
eum['value'] = eum.groupby('nr_groups')['value'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))
sns.boxplot(data=eum, x="nr_groups", y="value", hue="model", ax=axs[0,1], legend=False, linewidth=LINEWIDTH)
axs[0,1].set_title('Normalized EUM')
axs[0,1].set_ylabel(None)
axs[0,1].set_xlabel(None)

sen_welfare = all_lcn[all_lcn['metric'] == 'sen_welfare']
sen_welfare['value'] = sen_welfare.groupby('nr_groups')['value'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))
swboxplot = sns.boxplot(data=sen_welfare, x="nr_groups", y="value", hue="model", ax=axs[1,0], legend=False, linewidth=LINEWIDTH)
axs[1,0].set_title('Normalized Sen Welfare')
axs[1,0].set_ylabel(None)
axs[1,0].set_xlabel('Number of Groups')

dist_to_ref = all_lcn[all_lcn['metric'] == 'dist_to_eq_ref_point']
dist_to_ref['value'] = dist_to_ref.groupby('nr_groups')['value'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))
dtrboxplot = sns.boxplot(data=dist_to_ref, x="nr_groups", y="value", hue="model", ax=axs[1,1], legend='brief', linewidth=LINEWIDTH)
# Necessary to have a legend for it to apepar below, but then we need to remove it.
handles, labels = dtrboxplot.get_legend_handles_labels()
dtrboxplot.get_legend().remove()
axs[1,1].set_title('Normalized Distance to Utopian Point (lower=better)')
axs[1,1].set_ylabel(None)
axs[1,1].set_xlabel('Number of Groups')
axs[1,1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

fig.legend(handles, labels, fontsize=18, loc='center', bbox_to_anchor=(0.5, -0.02), ncol=3)
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
plot_over_time_results(ams_eum_over_time, xian_eum_over_time, [3, 6, 10], (40, 15), 'EUM')
# Plot EUM for all objectives
# plot_over_time_results(ams_eum_over_time, xian_eum_over_time, range(2, 11), (50, 15), 'EUM')
# plot_over_time_results(ams_eum_over_time, xian_eum_over_time, [3, 10], (15, 8), 'EUM', linewidth=4, font_size=22)

# Plot SW for 3, 6, 9 objectives
# plot_over_time_results(ams_sw_over_time, xian_sw_over_time, range(2, 11), (50, 15), 'Sen Welfare')
plot_over_time_results(ams_sw_over_time, xian_sw_over_time, [3, 6, 10], (40, 15), 'Sen Welfare')
# plot_over_time_results(ams_sw_over_time, xian_sw_over_time, [3, 10], (15, 8), 'Sen Welfare', linewidth=4, font_size=22)
# %%

fig, ax = plt.subplots(figsize=(7, 6))
xian_eum_lambda_lcn = xian_sw_over_time[['LCN-Lambda-0_3', 'LCN-Lambda-0.3_3', 'LCN-Lambda-0.6_3', 'LCN-Lambda-0.9_3']]
xian_eum_lambda_lcn.plot(ax=ax, linewidth=4, colormap='Blues')

# %% test coordinate plot
    

import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
import json
import os
from matplotlib.colors import ListedColormap
plt.rcParams.update({'font.size': 18})

cm = ListedColormap(["#848181", "#1A85FF"])

def parallel_coordinates_plot(ax, front, labels, model_names, plot_average=False, color_map=None, opacity=0.5, line_width=2, avg_line_width=4, xaxis_rotation=45):
    """
    Plot parallel coordinates for a given front on a specified axis.

    Args:
    - ax (matplotlib.axes.Axes): The axis on which to plot.
    - front (list of lists or list of np.arrays): List containing the front data.
    - labels (list): List of labels for the parallel coordinates.
    - model_names (list): List of model names corresponding to each front.
    - plot_average (bool, optional): Flag to indicate if the average should be plotted. Defaults to True.
    - color_map (dict, optional): Dictionary mapping model names to colors. Defaults to None.
    - opacity (float, optional): Opacity for the individual lines. Defaults to 0.5.
    - linewidth (int, optional): Line width for the average lines. Defaults to 2.
    """
    # Convert front to DataFrame
    front_df = pd.DataFrame(front, columns=labels)
    front_df['model'] = model_names

    # Plot individual fronts with transparency
    for model_name in front_df['model'].unique():
        model_data = front_df[front_df['model'] == model_name]
        if not model_data.empty:
            parallel_coordinates(model_data, 'model', color=color_map[model_name] if color_map else 'blue', linewidth=line_width, alpha=opacity, ax=ax)
    
    # Plot mean lines with dashed style if plot_average is True
    if plot_average:
        for model_name in front_df['model'].unique():
            mean_data = front_df[front_df['model'] == model_name].iloc[:, :-1].mean().to_frame().T  # Exclude 'model' column
            mean_data['model'] = f'Mean_{model_name}'
            parallel_coordinates(mean_data, 'model', color=color_map[model_name] if color_map else 'red', linewidth=avg_line_width, linestyle='--', ax=ax)
    
    ax.tick_params(axis='x', rotation=xaxis_rotation)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)

def plot_parallel_coordinates(objectives, models_to_plot=[], model_labels=[], figtitle=None):
    # Calculate number of rows needed (3 plots per row)
    n_plots = len(objectives)
    n_rows = (n_plots + 2) // 3  # Ceiling division to get enough rows
    n_cols = min(3, n_plots)
    
    # Create figure with subplots
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(7*n_cols, 7*n_rows))
    if n_rows == 1:
        axs = [axs]  # Make axs 2D even if only one row
    if n_cols == 1:
        axs = [[ax] for ax in axs]  # Make axs 2D even if only one column
        
    # Add figure title if provided
    if figtitle:
        fig.suptitle(figtitle)
        
    for plot_idx, objective in enumerate(objectives):
        row = plot_idx // 3
        col = plot_idx % 3
        
        nr_groups = objective['nr_groups']
        models = objective['models']
        
        # Initialize lists
        all_fronts = []
        model_names = []
        cardinality_per_model = {}
        
        # Filter models if models_to_plot is not empty
        if models_to_plot:
            models = [m for m in models if m['name'] in models_to_plot]
        
        # Create color mapping for models
        unique_models = [m['name'] for m in models]
        colors = cm(np.linspace(0, 1, len(unique_models)))
        color_map = dict(zip(unique_models, colors))
        
        # Create label mapping if model_labels provided
        if model_labels:
            label_map = dict(zip(models_to_plot, model_labels))
        
        for model in models:
            model_name = model['name']
            
            for run_id in model['run_ids']:
                if run_id == "":
                    continue
                    
                run = api.run(f"MORL-TNDP/{run_id}")
                front_artifact = api.artifact(f'MORL-TNDP/run-{run_id}-evalfront:latest')
                local_path = f'./artifacts/{front_artifact.name}'
                if not os.path.exists(local_path):
                    front_artifact.download()
                
                with open(f"{local_path}/eval/front.table.json", "r") as file:
                    fronts = json.load(file)['data']
                
                if len(fronts) == 0:
                    continue
                    
                # Add all fronts for this model
                all_fronts.extend(fronts)
                model_names.extend([model_name] * len(fronts))
                
                # Store cardinality
                cardinality_per_model[model_name] = cardinality_per_model.get(model_name, 0) + len(fronts)
        
        if not all_fronts:
            print(f"No data available for {nr_groups} objectives")
            continue
            
        # Plot in the corresponding subplot
        ax = axs[row][col]
        
        parallel_coordinates_plot(ax, all_fronts, [f'Group{i+1}' for i in range(nr_groups)], model_names, plot_average=True, color_map=color_map, opacity=0.1)
        
        ax.set_title(f'{nr_groups} Objectives')
        
        # Create custom legend with cardinality information
        legend_elements = [plt.Line2D([0], [0], color=color_map[model], 
                                    label=f'{label_map[model] if model_labels else model} (n={cardinality_per_model[model]})',
                                    linewidth=2) for model in unique_models]
        ax.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, -0.4), ncol=len(unique_models))
    
    # Remove any empty subplots
    for row in range(n_rows):
        for col in range(n_cols):
            if row * 3 + col >= n_plots:
                fig.delaxes(axs[row][col])
    
    plt.tight_layout()
    plt.show()

# Example usage
run_ids_xian = read_json('./result_ids_xian.txt')
run_ids_ams = read_json('./result_ids_ams.txt')
# Filter run_ids to only contain nr_groups 4 and 6
run_ids_xian = [group for group in run_ids_xian if group['nr_groups'] in [3, 6, 10]]
run_ids_ams = [group for group in run_ids_ams if group['nr_groups'] in [3, 6, 10]]

plot_parallel_coordinates(run_ids_xian,
                          models_to_plot=['PCN-Xian', 'LCN-Xian-nondominated'],
                          model_labels=['PCN', 'LCN'],
                          figtitle="Xi'an")

plot_parallel_coordinates(run_ids_ams,
                          models_to_plot=['PCN-Amsterdam', 'LCN-Amsterdam-nondominated'],
                          model_labels=['PCN', 'LCN'],
                          figtitle='Amsterdam')

# %%
# import numpy as np
# from itertools import permutations

# def generate_symmetric_weights(d, range_min=0.4, range_max=0.6):
#     """
#     Generate a list of symmetric weight vectors for a given dimension `d`.
    
#     Args:
#     - d (int): The number of dimensions.
#     - range_min (float): Minimum bound for the first weight.
#     - range_max (float): Maximum bound for the first weight.
    
#     Returns:
#     - list of numpy arrays: Each array is a weight vector that sums to 1.
#     """
#     if d < 2 or d > 10:
#         raise ValueError("Dimension `d` should be between 2 and 10.")
    
#     # Create equally spaced values between range_min and range_max for the first component
#     first_weights = np.linspace(range_min, range_max, num=d)
    
#     # Generate base weight vectors with the first weight varying
#     weight_vectors = []
#     for w1 in first_weights:
#         # Remaining weight after setting the first component
#         remaining_weight = 1 - w1
#         # Spread the remaining weight equally across the other components
#         other_weights = np.full(d - 1, remaining_weight / (d - 1))
        
#         # Create the initial weight vector
#         base_vector = np.concatenate(([w1], other_weights))
        
#         # Generate all unique permutations of the base vector for symmetry
#         for perm in set(permutations(base_vector)):
#             weight_vectors.append(np.array(perm))
    
#     # Add the perfectly equal distribution vector
#     equal_vector = np.full(d, 1 / d)
#     weight_vectors.append(equal_vector)
    
#     # Remove duplicates and sort for consistency
#     unique_weight_vectors = []
#     for vec in weight_vectors:
#         if not any(np.allclose(vec, uvec) for uvec in unique_weight_vectors):
#             unique_weight_vectors.append(vec)
    
#     return unique_weight_vectors

# weights_dict = {}
# for d in range(2, 11):
#     weights = generate_symmetric_weights(d)
#     weights_dict[d] = weights
    

# np.save('fair_weights_dict.npy', weights_dict)

#%%
lambdas = [0.0, 0.2, 0.4, 1.0]
lambda_lcn_results = results_to_plot[(results_to_plot['model'].isin(['LCN_Lambda'])) & (results_to_plot['lambda'].isin(lambdas))]
lambda_lcn_3 = lambda_lcn_results[lambda_lcn_results['nr_groups'] == 3]
lambda_lcn_6 = lambda_lcn_results[lambda_lcn_results['nr_groups'] == 6]

fig, axs = plt.subplots(2, 2, figsize=(10, 6))

# Loop through each lambda value and plot in a separate subplot
for idx, lambda_value in enumerate(lambdas):
    row = idx // 2
    col = idx % 2
    ax = axs[row, col]
    fronts = np.array(lambda_lcn_results[(lambda_lcn_results['lambda'] == lambda_value) & (lambda_lcn_results['metric'] == 'fronts') & (lambda_lcn_results['nr_groups'] == 3)]['value'].values.tolist())
    model_names = [f'λ={lambda_value}'] * len(fronts)
    
    # Plot parallel coordinates
    parallel_coordinates_plot(ax, fronts, ['Group 1', 'Group 2', 'Group 3'], model_names, plot_average=False, color_map={f'λ={lambda_value}': '#1A85FF'}, line_width=2, avg_line_width=6, opacity=1.0, xaxis_rotation=15)
    ax.set_title(f'λ={lambda_value}')
    ax.set_ylim(0e-3, 9e-3)  # Set the same y-limits for all subplots
    ax.get_legend().remove()  # Hide the legend

    # Remove x-axis labels for the top row
    if row == 0:
        ax.set_xticklabels([])

fig.tight_layout()
