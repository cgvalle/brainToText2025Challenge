import os
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


general_path = 'data/model_weights'
models = ['baseline','time_warp_001_010','time_warp_020_800', 'time_warp_010_2000']

results = {}
for model in models:
    train_metrics = pd.read_csv(os.path.join(general_path, model, 'train_metrics.csv'), index_col=0)
    val_metrics = pd.read_csv(os.path.join(general_path, model, 'val_metrics.csv'), index_col=0)
    
    if os.path.exists(os.path.join(general_path, model, 'val_summary.csv')):
        print(model)
        val_summary = pd.read_csv(os.path.join(general_path, model, 'val_summary.csv'), header=None, names=['batch','length','edit_distance','WER'])
        val_summary['batch'] = val_summary['batch'].str.replace( 'checkpoint_batch_','').astype(int)
        val_summary = val_summary.sort_values('batch')
        #if len(val_summary == 0):
        #    val_summary = None
    else:
        val_summary = None



    results[model] = {
        'train': train_metrics,
        'val': val_metrics,
        'val_summary': val_summary
    }
# plot results
fig = plt.figure(figsize=(12, 18))
for model in models:
    train_metrics = results[model]['train']
    val_metrics = results[model]['val']

    plt.subplot(3, 1, 1)
    plt.plot(moving_average(train_metrics['train_losses'].tolist(),n=1500), label=model)
    plt.title('Training Loss')
    plt.xlabel('batch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(3, 1, 2)
    x = np.array(list(range(len(val_metrics['val_losses']))))*250
    plt.plot(x, val_metrics['val_losses'], label=model)
    plt.title('Validation Loss')
    plt.xlabel('batch')
    plt.ylabel('Loss')
    plt.legend()

    # val per
    plt.subplot(3, 1, 3)
    plt.plot(val_metrics['val_PERs'], label=model)
    plt.title('Validation PER')
    plt.xlabel('batch')
    plt.ylabel('PER')
    plt.legend()

plt.savefig('results.png')


fig = plt.figure(figsize=(12, 6))
for model in models:
    val_summary = results[model]['val_summary']
    if val_summary is not None:
        print(val_summary['batch'])
        plt.plot(val_summary['batch'], val_summary['WER'], label=model)
        plt.title(f'Validation WER for {model}')
        plt.xlabel('batch')
        plt.ylabel('WER')
        plt.legend()
plt.savefig(f'results_WER.png')
plt.close()