#%%
import os
from typing import List
from matplotlib import font_manager, pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
import json
import seaborn as sns
plt.rcParams.update({'font.size': 18})
from morl_baselines.common.weights import equally_spaced_weights
from pymoo.indicators.hv import HV
from morl_baselines.common.performance_indicators import hypervolume, expected_utility
import wandb


#  Linux libertine font
linlib_font_path = "/Users/dimichai/Library/Fonts/LinLibertine_RB.ttf"
font_manager.fontManager.addfont(linlib_font_path)

plt.rcParams.update({
    'axes.titlesize': 24,
    'axes.labelsize': 20,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 20,
    # 'font.family': 'Georgia', # For thesis
    "font.family":font_manager.FontProperties(fname=linlib_font_path).get_name(), # For JAIR
})

# Fair weights per nr_obectives, to be used to generate the fair-expected-utility metric.
# fair_weights_dict = np.load('fair_weights_dict.npy', allow_pickle=True).item()

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
REQ_SEEDS = 10 # to control if a model was not run for sufficient seeds

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

def average_per_step_with_ci(hvs_by_seed, confidence=0.95):
    """Calculate mean and confidence interval for each step across seeds"""
    import scipy.stats as st
    
    # Determine the maximum length of the sublists
    max_length = max(len(sublist) for sublist in hvs_by_seed)
    
    # Pad shorter sublists with their last value
    padded_hvs_by_seed = [sublist + [sublist[-1]] * (max_length - len(sublist)) for sublist in hvs_by_seed]
    
    # Convert to numpy array for easier calculations
    padded_array = np.array(padded_hvs_by_seed)
    
    # Calculate mean per step
    means = np.mean(padded_array, axis=0)
    
    # Calculate confidence intervals
    if len(hvs_by_seed) > 1:  # Need at least 2 samples for CI
        ci = st.t.interval(confidence, 
                          len(hvs_by_seed) - 1, 
                          loc=means, 
                          scale=st.sem(padded_array, axis=0))
        lower_ci = ci[0]
        upper_ci = ci[1]
    else:
        # If only one seed, no CI possible
        lower_ci = means
        upper_ci = means
    
    return means, lower_ci, upper_ci

def load_all_results_from_wandb(all_objectives, env_name=None):
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
                results_by_objective[model_name] = {'gini': [], 'total_efficiency': [], 'avg_efficiency': [], 'hv': [], 'train_hv': [], 'sen_welfare': [], 'nash_welfare': [], 'dist_to_eq_ref_point': [], 'avg_per_group': [], 'cardinality': [], 'eum': []}
                for i in range(len(model['run_ids'])):
                    if model['run_ids'][i] == '':
                        print(f"WARNING - Empty run id in {model_name}")
                        continue
                    
                    run = api.run(f"{project_name}/{model['run_ids'][i]}")
                    
                    # Sanity check for algo name, so that we don't load wrong results by mistake
                    if 'lcn_lambda' in run.config and run.config['lcn_lambda'] is not None:
                        run_algo = f"{run.config['algo']}-Lambda-{run.config['lcn_lambda']}"
                    elif 'distance_ref' in run.config and run.config['distance_ref'] is not None:
                        run_algo = f"{run.config['algo']}-{run.config['distance_ref']}"
                    else:
                        run_algo = run.config['algo']
                        
                    if run_algo != model_name:
                        print(f"!ERROR! {model_name} has algo {model_name}, while {run_algo} is required")
                    ###
                                    
                    front_artifact = api.artifact(f'{project_name}/run-{model["run_ids"][i]}-evalfront:latest')
                    local_path = f'./artifacts/{front_artifact.name}'
                    if not os.path.exists(local_path):
                        front_artifact.download(local_path)
                        
                    with open(f"{local_path}/eval/front.table.json", "r") as file:
                        fronts = json.load(file)['data']
                    groups = len(fronts[0])
                    if groups != nr_groups:
                        print(f"!ERROR! {objective['nr_groups']} nrgroups, {model_name} has {groups} groups, while {nr_groups} are required")

                    gini_index = gini(np.array(fronts))
                    total_efficiency = np.sum(fronts, axis=1)
                    avg_efficiency = np.mean(fronts, axis=1)
                    hv = hypervolume(ref_point, fronts)
                    # train_hv = run.summary['train/hypervolume']
                    nash_welfare = np.prod(fronts, axis=1)
                    
                    has_large_point = any(np.all(front >= eq_ref_point) for front in fronts)
                    if has_large_point:
                        print(f"WARNING - Point > equality reference point in {model_name}, {nr_groups} - {model['run_ids'][i]}")
                    else:
                        dist_to_eq_ref = euclidean_distance_to_equality_ref_point(eq_ref_point, np.array(fronts))
                        
                    results_by_objective[model_name]['fronts'] = fronts
                    results_by_objective[model_name]['total_efficiency'] = results_by_objective[model_name]['total_efficiency'] + total_efficiency.tolist()
                    results_by_objective[model_name]['avg_efficiency'] = results_by_objective[model_name]['avg_efficiency'] + avg_efficiency.tolist()
                    results_by_objective[model_name]['hv'] = results_by_objective[model_name]['hv'] + [hv]
                    # results_by_objective[model_name]['train_hv'] = results_by_objective[model_name].get('train_hv', []) + [train_hv]
                    results_by_objective[model_name]['gini'] = results_by_objective[model_name]['gini'] + gini_index.tolist()
                    if env_name != 'DST':
                        results_by_objective[model_name]['sen_welfare'] = results_by_objective[model_name]['sen_welfare'] + (total_efficiency * (1-gini_index)).tolist()
                    results_by_objective[model_name]['nash_welfare'] = results_by_objective[model_name]['nash_welfare'] + nash_welfare.tolist()
                    results_by_objective[model_name]['dist_to_eq_ref_point'] = results_by_objective[model_name]['dist_to_eq_ref_point'] + dist_to_eq_ref.tolist()
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
                means, lower_ci, upper_ci = average_per_step_with_ci(hvs_by_seed)
                # Store all three series in the DataFrame
                hv_over_time = pd.concat([
                    hv_over_time, 
                    pd.DataFrame({
                        f"{model_name_adj}_{nr_groups}_mean": means,
                        f"{model_name_adj}_{nr_groups}_lower": lower_ci,
                        f"{model_name_adj}_{nr_groups}_upper": upper_ci
                    })
                ])

            if len(eum_by_seed) > 0:
                means, lower_ci, upper_ci = average_per_step_with_ci(eum_by_seed)
                eum_over_time = pd.concat([
                    eum_over_time, 
                    pd.DataFrame({
                        f"{model_name_adj}_{nr_groups}_mean": means,
                        f"{model_name_adj}_{nr_groups}_lower": lower_ci,
                        f"{model_name_adj}_{nr_groups}_upper": upper_ci
                    })
                ])
                
            if len(sw_by_seed) > 0:
                means, lower_ci, upper_ci = average_per_step_with_ci(sw_by_seed)
                sw_over_time = pd.concat([
                    sw_over_time, 
                    pd.DataFrame({
                        f"{model_name_adj}_{nr_groups}_mean": means,
                        f"{model_name_adj}_{nr_groups}_lower": lower_ci,
                        f"{model_name_adj}_{nr_groups}_upper": upper_ci
                    })
                ])


                        
            # results_by_objective[model_name]['lambda'] = model['lambda'] if 'lambda' in model else ''
        # Quite a hacky way to get the results in a dataframe, but didn't have time to do it properly (thanks copilot)
        # Convert all_results to a dataframe, with columns 'model', 'metric', 'value', and each row is a different value and not a list
        # results_by_objective = pd.DataFrame([(name, metric, value) for name in results_by_objective.keys() for metric in results_by_objective[name].keys() for value in results_by_objective[name][metric]], columns=['model', 'metric', 'value'])
        # Convert all_results to a dataframe, with columns 'model', 'lambda; 'metric', 'value', and each row is a different value and not a list
        results_by_objective = pd.DataFrame([(name, model['lambda'] if 'lambda' in model else None, model['cd_threshold'] if 'cd_threshold' in model else 0.2, metric, value) for name in results_by_objective.keys() 
                                            for metric in results_by_objective[name].keys() 
                                            for value in results_by_objective[name][metric]], columns=['model', 'lambda', 'cd_threshold', 'metric', 'value'])
        results_by_objective['model'] = results_by_objective['model'].str.replace(f'-{env_name}', '')
        results_by_objective['nr_groups'] = nr_groups
        results_by_objective['lambda'] = results_by_objective[results_by_objective['model'].str.contains('Lambda')]['model'].str.split('-').str[-1].astype(float)
        results_by_objective['cd_threshold'] = results_by_objective[results_by_objective['model'].str.contains('tau')]['model'].str.split('-').str[-1].astype(float)
        results_by_objective.loc[results_by_objective['model'].str.contains('Lambda'), 'model'] = 'LCN_Lambda'
        all_results = pd.concat([all_results, results_by_objective])
        
    return all_results, hv_over_time, eum_over_time, sw_over_time

# ams_results, ams_hv_over_time, ams_eum_over_time, ams_sw_over_time = load_all_results_from_wadb(read_json('./result_ids_ams.txt'), 'Amsterdam')
# xian_results, xian_hv_over_time, xian_eum_over_time, xian_sw_over_time = load_all_results_from_wandb(read_json('./result_ids_xian.txt'), 'Xian')
xian_results, xian_hv_over_time, xian_eum_over_time, xian_sw_over_time = load_all_results_from_wandb(read_json('./result_ids_xian_test.txt'), 'Xian')

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
eum['value'] = eum.groupby('nr_groups')['value'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))
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
sen_welfare.groupby(['model', 'nr_groups']).agg({'value': ['mean', lambda x: np.std(x, ddof=1) / np.sqrt(len(x))]}).round(2)
# hv.groupby(['model', 'nr_groups']).agg({'value': ['mean', lambda x: np.std(x, ddof=1) / np.sqrt(len(x))]}).round(2)
# eff.groupby(['model', 'nr_groups']).agg({'value': ['mean', lambda x: np.std(x, ddof=1) / np.sqrt(len(x))]}).round(2)
# gini_.groupby(['model', 'nr_groups']).agg({'value': ['mean', lambda x: np.std(x, ddof=1) / np.sqrt(len(x))]}).round(2)
# eum.groupby(['model', 'nr_groups']).agg({'value': ['mean', lambda x: np.std(x, ddof=1) / np.sqrt(len(x))]}).round(2)
#%%
## Same but only hv and sen_welfare
fig, axs = plt.subplots(3, 1, figsize=(8, 10))
colors = ["#FFF2E5", "#BFBFBF", "#1A85FF"]
sns.set_palette(sns.color_palette(colors))
# plt.rcParams.update({'font.size': 18})
LINEWIDTH = 2

pcnvlcn = results_to_plot[results_to_plot['model'].isin(['GPI-LS', 'PCN', 'LCN-nondominated'])]
pcnvlcn.loc[pcnvlcn['model'] == 'LCN-nondominated', 'model'] = 'LCN'

hv = pcnvlcn[pcnvlcn['metric'] == 'hv']
hv['value'] = hv.groupby('nr_groups')['value'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))
hv['value'] = hv['value'].fillna(0)

hvboxplot = sns.boxplot(data=hv, x="nr_groups", y="value", hue="model", ax=axs[0], legend=False, linewidth=LINEWIDTH)
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
swboxplot = sns.boxplot(data=sen_welfare, x="nr_groups", y="value", hue="model", ax=axs[2], legend='brief', linewidth=LINEWIDTH)
axs[2].set_title('Normalized Sen Welfare')
axs[2].set_ylabel(None)
axs[2].set_xlabel('Number of Groups')
axs[2].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
handles, labels = swboxplot.get_legend_handles_labels()
swboxplot.get_legend().remove()

# dist_to_ref = pcnvlcn[pcnvlcn['metric'] == 'dist_to_eq_ref_point']
# dist_to_ref['value'] = dist_to_ref.groupby('nr_groups')['value'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))
# dtrboxplot = sns.boxplot(data=dist_to_ref, x="nr_groups", y="value", hue="model", ax=axs[1,1], legend='brief', linewidth=LINEWIDTH)
# # Necessary to have a legend for it to apepar below, but then we need to remove it.
# axs[1,1].set_title('Normalized Distance to Utopian Point (lower=better)')
# axs[1,1].set_ylabel(None)
# axs[1,1].set_xlabel('Number of Groups')
# axs[1,1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

# Move legend below the entire figure
fig.legend(handles, labels, fontsize=18, loc='center', bbox_to_anchor=(0.5, -0.02), ncol=3)

fig.tight_layout()
fig.savefig('figures/pcn_vs_lcn_2.png', bbox_inches='tight')
fig.savefig('figures/pcn_vs_lcn_2.pdf', bbox_inches='tight')

#%%
## Box plot of HV, Sen Welfare, EUM for all LCN models
fig, axs = plt.subplots(3, 1, figsize=(8, 10))
colors = ["#1A85FF", "#E66100", "#D41159", "#BFBFBF"]
sns.set_palette(sns.color_palette(colors))
# plt.rcParams.update({'font.size': 18})

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
swboxplot = sns.boxplot(data=sen_welfare, x="nr_groups", y="value", hue="model", ax=axs[2], legend='brief', linewidth=LINEWIDTH)
axs[2].set_title('Normalized Sen Welfare')
axs[2].set_ylabel(None)
axs[2].set_xlabel('Number of Groups')
handles, labels = swboxplot.get_legend_handles_labels()
swboxplot.get_legend().remove()

# dist_to_ref = all_lcn[all_lcn['metric'] == 'dist_to_eq_ref_point']
# dist_to_ref['value'] = dist_to_ref.groupby('nr_groups')['value'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))
# dtrboxplot = sns.boxplot(data=dist_to_ref, x="nr_groups", y="value", hue="model", ax=axs[1,1], legend='brief', linewidth=LINEWIDTH)
# # Necessary to have a legend for it to apepar below, but then we need to remove it.
# handles, labels = dtrboxplot.get_legend_handles_labels()
# dtrboxplot.get_legend().remove()
# axs[1,1].set_title('Normalized Distance to Utopian Point (lower=better)')
# axs[1,1].set_ylabel(None)
# axs[1,1].set_xlabel('Number of Groups')
# axs[1,1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

fig.legend(handles, labels, fontsize=18, loc='center', bbox_to_anchor=(0.5, -0.02), ncol=3)
fig.tight_layout()
fig.savefig('figures/all_lcn_results.png', bbox_inches='tight')
fig.savefig('figures/all_lcn_results.pdf', bbox_inches='tight')

#%%
## Boxplot but only sen_welfare and cardinality
# fig, axs = plt.subplots(2, 1, figsize=(7, 6))
# pcnvlcn = results_to_plot[results_to_plot['model'].isin(['PCN', 'LCN_ND'])]
# pcnvlcn.loc[pcnvlcn['model'] == 'LCN_ND', 'model'] = 'LCN'


# sen_welfare = pcnvlcn[pcnvlcn['metric'] == 'sen_welfare']
# sen_welfare['value'] = sen_welfare.groupby('nr_groups')['value'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))
# sns.boxplot(data=sen_welfare, x="nr_groups", y="value", hue="model", ax=axs[0], legend=False, linewidth=LINEWIDTH)
# axs[0].set_title('Normalized Sen Welfare')
# axs[0].set_xlabel('Number of Groups')
# axs[0].set_ylabel(None)


# cardinality = pcnvlcn[pcnvlcn['metric'] == 'cardinality']
# # cardinality['value'] = cardinality.groupby('nr_groups')['value'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))
# cardinality['value'] = cardinality['value'].fillna(0)

# hvboxplot = sns.boxplot(data=cardinality, x="nr_groups", y="value", hue="model", ax=axs[1], legend=True, linewidth=LINEWIDTH)
# hvboxplot.legend_.set_title(None)
# hvboxplot.legend(fontsize=14)
# axs[1].set_title('Cardinality')
# axs[1].set_xlabel(None)
# axs[1].set_ylabel(None)


# fig.tight_layout()

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

plt.rcParams.update({
    'axes.titlesize': 24,
    'axes.labelsize': 20,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 20,
    'font.family': 'Georgia',
})
def plot_over_time_results(ams_metric, xian_metric, groups, figsize, ylabel, nrows, ncols, linewidth=5):
    fig = plt.figure(constrained_layout=True, figsize=figsize)
    colors = ["#BFBFBF", "#1A85FF", "#E66100", "#D41159"]
    sns.set_palette(sns.color_palette(colors))
    LINEWIDTH = linewidth

    subfigs = fig.subfigures(nrows=2, ncols=1, wspace=0.4, hspace=0.1)  # Add spacing between subfigures
    subfigs[0].suptitle(f"Xi'an", fontweight='bold', fontsize=24)
    subfigs[1].suptitle(f"Amsterdam", fontweight='bold', fontsize=24)

    axs_xian = subfigs[0].subplots(nrows=nrows//2, ncols=ncols)
    axs_ams = subfigs[1].subplots(nrows=nrows//2, ncols=ncols)
    
    # WHen setting nrows = 2, the axs are only 1D arrays, so we need to convert them to 2D
    axs_xian = np.atleast_2d(axs_xian)
    axs_ams = np.atleast_2d(axs_ams)

    
    axs_xian[0, 0].set_ylabel(ylabel)
    axs_ams[0, 0].set_ylabel(ylabel)
    
    for i, group in enumerate(groups):
        row_xian, col_xian = divmod(i, ncols)
        row_ams, col_ams = divmod(i, ncols)
        
        axs_xian[row_xian, col_xian].plot(xian_metric[f'PCN_{group}_mean'], label='PCN', linewidth=LINEWIDTH)
        axs_xian[row_xian, col_xian].fill_between(xian_metric.index, xian_metric[f'PCN_{group}_lower'], xian_metric[f'PCN_{group}_upper'], alpha=0.2, label="_nolegend_")
        axs_xian[row_xian, col_xian].plot(xian_metric[f'LCN-nondominated_{group}_mean'], label='LCN_ND', linewidth=LINEWIDTH)
        axs_xian[row_xian, col_xian].fill_between(xian_metric.index, xian_metric[f'LCN-nondominated_{group}_lower'], xian_metric[f'LCN-nondominated_{group}_upper'], alpha=0.2, label="_nolegend_")
        axs_xian[row_xian, col_xian].plot(xian_metric[f'LCN-optimal_max_{group}_mean'], label='LCN_OPTMAX', linewidth=LINEWIDTH)
        axs_xian[row_xian, col_xian].fill_between(xian_metric.index, xian_metric[f'LCN-optimal_max_{group}_lower'], xian_metric[f'LCN-optimal_max_{group}_upper'], alpha=0.2, label="_nolegend_")
        axs_xian[row_xian, col_xian].plot(xian_metric[f'LCN-nondominated_mean_{group}_mean'], label='LCN_NDMEAN', linewidth=LINEWIDTH)
        axs_xian[row_xian, col_xian].fill_between(xian_metric.index, xian_metric[f'LCN-nondominated_mean_{group}_lower'], xian_metric[f'LCN-nondominated_mean_{group}_upper'], alpha=0.2, label="_nolegend_")
    
        axs_xian[row_xian, col_xian].set_xlabel('Step')
        axs_xian[row_xian, col_xian].set_title(f'{group} Objectives')
        axs_xian[row_xian, col_xian].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
        axs_ams[row_ams, col_ams].plot(ams_metric[f'PCN_{group}_mean'], label='PCN', linewidth=LINEWIDTH)
        axs_ams[row_ams, col_ams].fill_between(ams_metric.index, ams_metric[f'PCN_{group}_lower'], ams_metric[f'PCN_{group}_upper'], alpha=0.2, label="_nolegend_")
        axs_ams[row_ams, col_ams].plot(ams_metric[f'LCN-nondominated_{group}_mean'], label='LCN_ND', linewidth=LINEWIDTH)
        axs_ams[row_ams, col_ams].fill_between(ams_metric.index, ams_metric[f'LCN-nondominated_{group}_lower'], ams_metric[f'LCN-nondominated_{group}_upper'], alpha=0.2, label="_nolegend_")
        axs_ams[row_ams, col_ams].plot(ams_metric[f'LCN-optimal_max_{group}_mean'], label='LCN_OPTMAX', linewidth=LINEWIDTH)
        axs_ams[row_ams, col_ams].fill_between(ams_metric.index, ams_metric[f'LCN-optimal_max_{group}_lower'], ams_metric[f'LCN-optimal_max_{group}_upper'], alpha=0.2, label="_nolegend_")
        axs_ams[row_ams, col_ams].plot(ams_metric[f'LCN-nondominated_mean_{group}_mean'], label='LCN_NDMEAN', linewidth=LINEWIDTH)
        axs_ams[row_ams, col_ams].fill_between(ams_metric.index, ams_metric[f'LCN-nondominated_mean_{group}_lower'], ams_metric[f'LCN-nondominated_mean_{group}_upper'], alpha=0.2, label="_nolegend_")
    
        axs_ams[row_ams, col_ams].set_xlabel('Step')
        axs_ams[row_ams, col_ams].set_title(f'{group} Objectives')
        axs_ams[row_ams, col_ams].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        
        #add the y label only to the first column
        if col_xian == 0:
            axs_xian[row_xian, col_xian].set_ylabel(ylabel)
        if col_ams == 0:
            axs_ams[row_ams, col_ams].set_ylabel(ylabel)
        

    subfigs[1].legend(['PCN', 'LCN', 'LCN-Redist', 'LCN-Mean'], loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.1), fontsize=22)
    
    return fig


LINE_WIDTH_FOR_EUM_SW = 4

# Plot EUM for 3, 6, 9 objectives
eum_lines = plot_over_time_results(ams_eum_over_time, xian_eum_over_time, [3, 6, 10], (19.2, 12), 'EUM', 2, 3, linewidth=LINE_WIDTH_FOR_EUM_SW)
eum_lines.savefig('figures/eum_3_6_10.png', bbox_inches='tight')
eum_lines.savefig('figures/eum_3_6_10.pdf', bbox_inches='tight')
# Plot EUM for all objectives
eum_all = plot_over_time_results(ams_eum_over_time, xian_eum_over_time, range(2, 11), (19.2, 24), 'EUM', 6, 3, linewidth=LINE_WIDTH_FOR_EUM_SW)
eum_all.savefig('figures/eum_all.png', bbox_inches='tight')
eum_all.savefig('figures/eum_all.pdf', bbox_inches='tight')
# plot_over_time_results(ams_eum_over_time, xian_eum_over_time, [3, 10], (15, 8), 'EUM', linewidth=LINE_WIDTH_FOR_EUM_SW)

# Plot SW for 3, 6, 9 objectives
# plot_over_time_results(ams_sw_over_time, xian_sw_over_time, range(2, 11), (50, 15), 'Sen Welfare')
sw_lines = plot_over_time_results(ams_sw_over_time, xian_sw_over_time, [3, 6, 10], (19.2, 12), 'Sen Welfare', 2, 3, linewidth=LINE_WIDTH_FOR_EUM_SW)
sw_lines.savefig('figures/sw_3_6_10.png', bbox_inches='tight')
sw_lines.savefig('figures/sw_3_6_10.pdf', bbox_inches='tight')
sw_all = plot_over_time_results(ams_sw_over_time, xian_sw_over_time, range(2, 11), (19.2, 24), 'Sen Welfare', 6, 3, linewidth=LINE_WIDTH_FOR_EUM_SW)
sw_all.savefig('figures/sw_all.png', bbox_inches='tight')
sw_all.savefig('figures/sw_all.pdf', bbox_inches='tight')
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
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
import pandas as pd
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

#%%
lambdas = [0.0, 0.2, 0.4, 1.0]
lambda_lcn_results = results_to_plot[(results_to_plot['model'].isin(['LCN_Lambda'])) & (results_to_plot['lambda'].isin(lambdas))]
lambda_lcn_3 = lambda_lcn_results[lambda_lcn_results['nr_groups'] == 3]
lambda_lcn_6 = lambda_lcn_results[lambda_lcn_results['nr_groups'] == 6]

fig, axs = plt.subplots(2, 2, figsize=(12.8, 8))

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
    # ax.set_ylim(0e-3, 9e-3)  # Set the same y-limits for all subplots
    ax.get_legend().remove()  # Hide the legend

    # Remove x-axis labels for the top row
    if row == 0:
        ax.set_xticklabels([])

fig.tight_layout()
fig.savefig('figures/lambda_pc_xian.png', bbox_inches='tight')
fig.savefig('figures/lambda_pc_xian.pdf', bbox_inches='tight')

#%% Plot Total Efficiency, Gini Index, Sen Welfare for lambda-LCN (0.0-1.0) with CI
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
# Plot with confidence intervals
for idx, lambda_value in enumerate(lambdas):
    if lambda_value == 0.0:
        continue  # Skip lambda = 0.0 for this plot
    row = idx // 2
    col = idx % 2
    ax = axs[row, col]
    fronts = np.array(lambda_lcn_results[(lambda_lcn_results['lambda'] == lambda_value) & (lambda_lcn_results['metric'] == 'fronts') & (lambda_lcn_results['nr_groups'] == 3)]['value'].values.tolist())
    model_names = [f'λ={lambda_value}'] * len(fronts)
    
    # Plot parallel coordinates
    parallel_coordinates_plot(ax, fronts, ['Group 1', 'Group 2', 'Group 3'], model_names, plot_average=False, color_map={f'λ={lambda_value}': '#1A85FF'}, line_width=2, avg_line_width=6, opacity=1.0, xaxis_rotation=15)
    ax.set_title(f'λ={lambda_value}')
    # ax.set_ylim(0e-3, 9e-3)  # Set the same y-limits for all subplots
    ax.get_legend().remove()  # Hide the legend

    # Remove x-axis labels for the top row
    if row == 0:
        ax.set_xticklabels([])
    
# Add PCN lines
for ax in axs:
    ax.plot(pcn_hv['value'], label='PCN', linestyle='--', color='black', linewidth=LINEWIDTH)

fig.tight_layout()
fig.savefig('figures/lambda_pc_xian_ci.png', bbox_inches='tight')
fig.savefig('figures/lambda_pc_xian_ci.pdf', bbox_inches='tight')

# %%

# --- Prepare data ---
cardinality = (
    results_to_plot
    .query("metric == 'cardinality'")
    .assign(
        value=lambda x: pd.to_numeric(x['value'], errors='coerce'),
        nr_groups=lambda x: pd.to_numeric(x['nr_groups'], errors='coerce')
    )
    .dropna(subset=['value', 'nr_groups'])
    .assign(nr_groups=lambda x: x['nr_groups'].astype(int))
)

cardinality.loc[cardinality['model'] == 'LCN-nondominated', 'model'] = 'LCN'
cardinality.loc[cardinality['model'] == 'LCN-optimal_max', 'model'] = 'LCN-Redist'
cardinality.loc[cardinality['model'] == 'LCN-nondominated_mean', 'model'] = 'LCN-Mean'

models_to_show = ['PCN', 'LCN', 'LCN-Redist', 'LCN-Mean']
base_colors = {"PCN": "#BFBFBF", "LCN": "#1A85FF", "LCN-Redist": "#E66100", "LCN-Mean": "#D41159"}


# --- Aggregate stats ---
grp = (
    cardinality.groupby(['model', 'nr_groups'])['value']
    .agg(mean='mean', sem=lambda x: x.std() / np.sqrt(len(x)))
    .reset_index()
)

markers = ['o', 's', '^', 'D', 'v'][:len(models_to_show)]

# --- Plot ---
fig, ax = plt.subplots(figsize=(10, 6))
all_x = sorted(grp['nr_groups'].unique())

for label, marker in zip(cardinality[cardinality['model'].isin(models_to_show)]['model'].unique(), markers):
    sub = grp[grp['model'] == label]
    ax.errorbar(
        sub['nr_groups'], sub['mean'], yerr=sub['sem'],
        fmt=f'-{marker}', color=base_colors.get(label, "#848181"),
        ecolor=base_colors.get(label, "#848181"),
        mec='black', mfc=base_colors.get(label, "#848181"),
        capsize=8, linewidth=4, markersize=12, label=label
    )

ax.set(
    title=f"Cardinality of Non-Dominated Policies",
    xlabel="Number of Groups",
    ylabel="Cardinality"
)
ax.set_xticks(all_x)
ax.legend()
fig.tight_layout()
fig.savefig('figures/cardinality_plot.png', bbox_inches='tight')
fig.savefig('figures/cardinality_plot.pdf', bbox_inches='tight')

#%% Unified crowding penalty sensitivity (choose any two nr_groups)
def get_non_dominated_mask(points: np.ndarray) -> np.ndarray:
    """Pareto-efficient mask for MAXIMIZATION."""
    n = points.shape[0]
    efficient = np.ones(n, dtype=bool)
    for i in range(n):
        if not efficient[i]:
            continue
        dominates_i = np.all(points >= points[i], axis=1) & np.any(points > points[i], axis=1)
        dominates_i[i] = False
        if np.any(dominates_i):
            efficient[i] = False
    return efficient

def crowding_distance(points: np.ndarray) -> np.ndarray:
    """Crowding distance (larger = less crowded) for MAXIMIZATION."""
    if points.size == 0:
        return np.array([])
    norm = (points - points.min(axis=0)) / (points.ptp(axis=0) + 1e-12)
    n, m = norm.shape
    dist = np.zeros(n)
    for k in range(m):
        idx = np.argsort(norm[:, k])
        dist[idx[0]] = np.inf
        dist[idx[-1]] = np.inf
        rng = norm[idx[-1], k] - norm[idx[0], k]
        if rng == 0:
            continue
        for i in range(1, n - 1):
            dist[idx[i]] += (norm[idx[i + 1], k] - norm[idx[i - 1], k]) / rng
    return dist

def generate_returns(n_points: int, n_obj: int, corr: float = 0.3, lo: float = 0.0, hi: float = 1.0, seed: int = None):
    if seed is not None:
        np.random.seed(seed)
    r = np.random.uniform(lo, hi, size=(n_points, n_obj))
    for d in range(1, n_obj):
        r[:, d] += corr * r[:, d - 1]
    return np.clip(r, lo, hi)

def build_parallel_df(returns: np.ndarray, nd_mask: np.ndarray, max_dominated: int = 60, label_mean='LCN-Mean', label_redist='LCN-Redist'):
    n_obj = returns.shape[1]
    cols = [f'Obj{i+1}' for i in range(n_obj)]
    labels = np.where(nd_mask, 'Non-dominated', 'Dominated')
    df = pd.DataFrame(returns, columns=cols)
    df['Label'] = labels
    dom_df = df[df['Label'] == 'Dominated']
    if len(dom_df) > max_dominated:
        dom_df = dom_df.sample(max_dominated, random_state=42)
    nd_df = df[df['Label'] == 'Non-dominated']
    mean_vec = nd_df[cols].mean().values
    redist_vec = returns[returns.sum(axis=1).argmax()]
    extra = pd.DataFrame([mean_vec, redist_vec], columns=cols)
    extra['Label'] = [label_mean, label_redist]
    return pd.concat([dom_df, nd_df, extra], ignore_index=True)

def crowding_penalty_sensitivity(
    nr_groups_pair=(2, 8),
    n_points_pair=(100, 120),
    penalties=(1, 2, 5),
    threshold=0.2,
    seed=2314,
    corr_small=0.5,
    corr_large=0.3,
):
    """
    Creates one figure (2 rows x 3 cols) for two nr_groups values.
    Panels: (A,B,C) first nr_groups, (D,E,F) second nr_groups.
    Uses global rcParams font without resetting.
    """
    letters = ['(A)', '(B)', '(C)', '(D)', '(E)', '(F)']

    data_bundle = []
    for idx, (n_obj, n_pts) in enumerate(zip(nr_groups_pair, n_points_pair)):
        corr = corr_small if idx == 0 else corr_large
        returns = generate_returns(n_pts, n_obj, corr=corr, seed=seed + idx)
        nd_mask = get_non_dominated_mask(returns)
        nd_points = returns[nd_mask]

        dist_to_nd = np.min(np.linalg.norm(returns[:, None, :] - nd_points[None, :, :], axis=-1), axis=1)
        base_score = -dist_to_nd

        nd_indices = np.where(nd_mask)[0]
        _, unique_idx = np.unique(nd_points, axis=0, return_index=True)
        unique_global_idx = nd_indices[unique_idx]
        dup_mask = np.ones(len(base_score), dtype=bool)
        dup_mask[unique_global_idx] = False
        base_score[dup_mask] -= 1e-5

        crowd_dist = crowding_distance(returns)
        thresholds = np.linspace(0.0, threshold, 10)
        proportions = [(np.isfinite(crowd_dist) & (crowd_dist <= t)).mean() for t in thresholds]

        scores_by_penalty = {}
        penalize_mask = np.isfinite(crowd_dist) & (crowd_dist <= threshold)
        for p in penalties:
            s = base_score.copy()
            s[penalize_mask] *= p
            scores_by_penalty[p] = s

        data_bundle.append({
            'n_obj': n_obj,
            'returns': returns,
            'nd_mask': nd_mask,
            'nd_points': nd_points,
            'crowd_dist': crowd_dist,
            'thresholds': thresholds,
            'proportions': proportions,
            'scores_by_penalty': scores_by_penalty
        })

    fig, axs = plt.subplots(2, 3, figsize=(19.2, 10))

    for row, info in enumerate(data_bundle):
        n_obj = info['n_obj']
        returns = info['returns']
        nd_mask = info['nd_mask']
        nd_points = info['nd_points']
        thresholds = info['thresholds']
        proportions = info['proportions']
        scores_by_penalty = info['scores_by_penalty']

    

        # (A/D)
        ax = axs[row, 0]
        panel_idx = row * 3 + 0
        if n_obj == 2:
            dom = returns[~nd_mask]
            nd = nd_points
            ax.scatter(dom[:, 0], dom[:, 1], alpha=0.6, color='#BFBFBF', label='Dominated', s=80)
            ax.scatter(nd[:, 0], nd[:, 1], alpha=1.0, color='#1A85FF', edgecolors='k', s=110, label='Non-dominated')
            mean_vec = nd.mean(axis=0)
            redist_vec = returns[returns.sum(axis=1).argmax()]
            ax.scatter(mean_vec[0], mean_vec[1], color='#D41159', edgecolors='k', s=110, label='LCN-Mean')
            ax.scatter(redist_vec[0], redist_vec[1], color='#E66100', edgecolors='k', s=110, label='LCN-Redist')
            ax.set_xlabel("Objective 1")
            ax.set_ylabel("Objective 2")
            ax.legend(loc='upper left')
        else:
            pc_df = build_parallel_df(returns, nd_mask)
            parallel_coordinates(
                pc_df,
                'Label',
                color=['#BFBFBF', '#1A85FF', '#D41159', '#E66100'],
                ax=ax,
                linewidth=1.2,
                alpha=0.55
            )
            ax.set_xlabel("Objectives")
            ax.set_ylabel("Objective value")
            ax.tick_params(axis='x', rotation=25)
            # ax.legend(loc='upper left')
            ax.legend().remove()
        ax.set_title(f"{letters[panel_idx]} Sample Experiences ({n_obj} Objectives)")

        # (B/E)
        ax = axs[row, 1]
        panel_idx = row * 3 + 1
        ax.plot(thresholds, proportions, lw=3, color='#3845DC')
        ax.scatter(thresholds, proportions, s=50, color='#3845DC')
        ax.set_xlabel(r"$\tau_{cd}$")
        ax.set_ylabel("Penalized experiences (%)")
        ax.set_title(f"{letters[panel_idx]} Threshold Sensitivity")

        # (C/F)
        ax = axs[row, 2]
        panel_idx = row * 3 + 2
        colors_pen = sns.color_palette("muted", n_colors=len(penalties))
        for p, c in zip(penalties, colors_pen):
            data = scores_by_penalty[p]
            sns.kdeplot(data, fill=True, common_norm=False, alpha=0.35, color=c,
                        linewidth=2, ax=ax, label=r'$\rho_{pen}=$' + f'{p}')
            ax.axvline(np.median(data), color=c, linestyle='--', linewidth=1.3)
        ax.set_xlabel("Penalty-adjusted distance score")
        ax.set_ylabel("Density")
        ax.set_title(f"{letters[panel_idx]} Penalty Sensitivity ($\\tau_{{cd}}={threshold}$)")
        ax.legend()

    return fig

# Example usage
fig = crowding_penalty_sensitivity(nr_groups_pair=(2, 5))
fig.tight_layout()
fig.savefig(f'./figures/crowding_penalty_sensitivity.png', dpi=300)
fig.savefig(f'./figures/crowding_penalty_sensitivity.pdf', dpi=300)
#%%

def plot_threshold_lines(df, 
    nr_groups=2, 
    model_bases=['LCN-nondominated', 'LCN-optimal_max', 'LCN-nondominated_mean'], 
    display_names={
        'LCN-nondominated': 'LCN',
        'LCN-optimal_max': 'LCN-Redist',
        'LCN-nondominated_mean': 'LCN-Mean'
    }, 
    thresholds=[0.0, 0.05, 0.1, 0.15, 0.2], 
    colors={
        'LCN-nondominated': "#1A85FF",
        'LCN-optimal_max': "#E66100",
        'LCN-nondominated_mean': "#D41159"
    },
    ylabel=None):

    # Assign colors per model_base (generate if not provided)
    if colors is None:
        palette = sns.color_palette("tab10", len(model_bases))
        colors = dict(zip(model_bases, palette))

    alphas = np.linspace(0.35, 1.0, len(thresholds))
    widths = np.linspace(1.2, 4.0, len(thresholds))

    fig, axs = plt.subplots(1, len(model_bases), figsize=(18, 6), sharey=True)
    if len(model_bases) == 1:
        axs = [axs]

    legend_labels, legend_handles = [], []
    seen_thresholds = set()

    for ax, model_base in zip(axs, model_bases):
        base_color = colors.get(model_base, "#1A85FF")
        for thr in thresholds:
            thr_idx = thresholds.index(thr)
            model_names = [f"{model_base}-tau-{thr}"]

            for m in model_names:
                mean_col = f"{m}_{nr_groups}_mean"
                low_col = f"{m}_{nr_groups}_lower"
                up_col = f"{m}_{nr_groups}_upper"
                if mean_col not in df.columns:
                    continue

                x = df.index
                y = df[mean_col].astype(float)
                ax.plot(x, y, color=base_color, alpha=alphas[thr_idx], linewidth=widths[thr_idx])

                if low_col in df.columns and up_col in df.columns:
                    ax.fill_between(
                        x,
                        df[low_col].astype(float),
                        df[up_col].astype(float),
                        color=base_color,
                        alpha=alphas[thr_idx] * 0.25
                    )

            if thr not in seen_thresholds:
                # Use a neutral color for threshold legend for clarity
                legend_handles.append(
                    Line2D([0], [0], color="black", alpha=alphas[thr_idx], linewidth=widths[thr_idx])
                )
                legend_labels.append(r"$\tau$=" + str(thr))
                seen_thresholds.add(thr)

        ax.set_title(display_names.get(model_base, model_base))
        ax.set_xlabel("Step")
        
    if ylabel is not None:
        axs[0].set_ylabel(ylabel)

    fig.legend(
        legend_handles, legend_labels, title=r"$\tau_{cd}$",
        loc="lower center", ncol=min(8, len(legend_labels)),
        bbox_to_anchor=(0.5, -0.15)
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    return fig


fig = plot_threshold_lines(
    df=xian_hv_over_time[[c for c in xian_hv_over_time.columns if 'tau' in c]],
    ylabel='Hypervolume',
)

fig = plot_threshold_lines(
    df=xian_eum_over_time[[c for c in xian_eum_over_time.columns if 'tau' in c]],    
    ylabel='EUM',
)

fig = plot_threshold_lines(
    df=xian_sw_over_time[[c for c in xian_sw_over_time.columns if 'tau' in c]],    
    ylabel='Sen Welfare',
)
    

# %%
fig, axs = plt.subplots(2, 1, figsize=(12, 6))
tauplots = results_to_plot[(results_to_plot['model'].str.contains('tau')) & (results_to_plot['nr_groups'] == 2)]
tauplots.loc[tauplots['model'].str.contains('LCN-nondominated_mean'), 'model'] = 'LCN-Mean'
tauplots.loc[tauplots['model'].str.contains('LCN-nondominated'), 'model'] = 'LCN'
tauplots.loc[tauplots['model'].str.contains('LCN-optimal_max'), 'model'] = 'LCN-Redist'

palette = {
    0.0: "#f1eef6",
    0.05: "#bdc9e1",
    0.1: "#74a9cf",
    0.2: "#0570b0",
}

LINEWIDTH = 1.5

hv = tauplots[tauplots['metric'] == 'hv']
hv['value'] = hv.groupby(['nr_groups'])['value'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))
hv['value'] = hv['value'].fillna(0)

hvboxplot = sns.boxplot(data=hv, x="model", y="value", hue="cd_threshold", palette=palette, ax=axs[0], linewidth=LINEWIDTH)
hvboxplot.legend_.set_title(r'$\tau_{cd}$')
axs[0].legend(ncol=4)
# hvboxplot.legend(fontsize=12)

axs[0].set_title('Normalized Hypervolume')
axs[0].set_xlabel(None)
axs[0].set_ylabel(None)
axs[0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

sw = tauplots[tauplots['metric'] == 'sen_welfare']
sw['value'] = sw.groupby(['nr_groups'])['value'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))
sw['value'] = sw['value'].fillna(0)
swboxplot = sns.boxplot(data=sw, x="model", y="value", hue="cd_threshold", palette=palette, ax=axs[1], legend=None, linewidth=LINEWIDTH)
# swboxplot.legend_.set_title(None)
axs[1].set_title('Normalized Sen Welfare')
axs[1].set_xlabel(None)
axs[1].set_ylabel(None)
axs[1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

fig.tight_layout()