import os
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import yaml

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


general_path = 'data/model_weights'
models = sorted(os.listdir(general_path))



results = {}
for model in models:
    train_metrics = pd.read_csv(os.path.join(general_path, model, 'train_metrics.csv'), index_col=0)
    val_metrics = pd.read_csv(os.path.join(general_path, model, 'val_metrics.csv'), index_col=0)
    args =  yaml.safe_load(open(os.path.join(general_path, model,'checkpoint', 'args.yaml')))


    if os.path.exists(os.path.join(general_path, model, 'val_summary.csv')):
        val_summary = pd.read_csv(os.path.join(general_path, model, 'val_summary.csv'), header=None, names=['batch','length','edit_distance','WER'])
        val_summary['batch'] = val_summary['batch'].str.replace( 'checkpoint_batch_','').astype(int)
        val_summary = val_summary.sort_values('batch')
    else:
        val_summary = None
    results[model] = {
        'train': train_metrics,
        'val': val_metrics,
        'val_summary': val_summary,
        'args': args
    }


y_max_loss = 30
y_min_loss = 0
y_max_per = 0.2
y_min_per = 0.0


baseline_arg = results['baseline']['args']


# plot results
fig = plt.figure(figsize=(12, 18))
for model in models:
 
    train_metrics = results[model]['train']
    val_metrics = results[model]['val']

    plt.subplot(3, 1, 1)
    plt.plot(moving_average(train_metrics['train_losses'].tolist(),n=100), label=model)
    plt.title('Training Loss')
    plt.xlabel('batch')
    plt.ylabel('Loss')
    plt.ylim(y_min_loss, y_max_loss)
    plt.legend()

    plt.subplot(3, 1, 2)
    x = np.array(list(range(len(val_metrics['val_losses']))))*250
    plt.plot(x, val_metrics['val_losses'], label=model)
    plt.title('Validation Loss')
    plt.xlabel('batch')
    plt.ylabel('Loss')
    plt.ylim(y_min_loss, y_max_loss)
    plt.legend()

    # val per
    plt.subplot(3, 1, 3)
    plt.plot(val_metrics['val_PERs'], label=model)
    plt.title('Validation PER')
    plt.xlabel('batch')
    plt.ylabel('PER')
    plt.ylim(y_min_per, y_max_per)
    plt.legend()

    if model == 'baseline':
        continue



    # print differences in args
    model_arg = results[model]['args']
    print(f'Comparing {model} to baseline:')
    for key in baseline_arg.keys():
        if baseline_arg[key] != model_arg[key]:
            if key=='dataset':
                continue
            if key in ['output_dir', 'checkpoint_dir', 'dir_name', 'init_from_checkpoint','init_checkpoint_path']:
                continue
            print(f'  {key}: baseline={baseline_arg[key]} vs {model}={model_arg[key]}')

    model_arg = results[model]['args']['dataset']['data_transforms']
    for key in baseline_arg['dataset']['data_transforms'].keys():
        if baseline_arg['dataset']['data_transforms'][key] != model_arg[key]:
            print(f'  {key}: baseline={baseline_arg["dataset"]["data_transforms"][key]} vs {model}={model_arg[key]}')


    print('')
    


plt.savefig('results.png')

