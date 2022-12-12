import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import json
import os

DATA_KEYS = [
    'num_nodes',
    'num_classes',
    'num_programs',
    'solve_time',
    'obj_start',
    'obj_opt',
    'obj_ext',
]

def load_data(path):
    with open(path) as fp:
        ret = []
        for line in fp:
            ret.append(json.loads(line))
        return ret

def process_data(eval_data):
    # Keys: number of iterations of eq sat, we that the first 10 data points
    # values: Object with keys 'ilp' or 'lp'
    #         indicating which solver was used
    result = {}
    for data in eval_data:
        num_iter = int(data['num_iter'])
        if num_iter in result:
            key = data['algorithm']
            for data_key in DATA_KEYS:
                result[num_iter][key][data_key] = result[num_iter][key].get(data_key, []) + [data[data_key]]
        else:
            result[num_iter] = {
                'ilp': {},
                'lp': {},
            }
    return result

def aggregate_data(model_name, data, dst_file):
    if os.path.isfile(dst_file):
        aggregate_data = json.load(open(dst_file, 'r'))
    else:
        aggregate_data = {}
    aggregate_data[model_name] = data
    with open(dst_file, 'w') as fp:
        json.dump(aggregate_data, fp)


def get_processed(data_file):
    data = load_data(data_file)
    return process_data(data)
    
def aggregate_mode(models, datasets, dst_file):
    assert len(models) == len(datasets), "Unequal length of models and datasets"
    for model, dataset in zip(models, datasets):
        print("Processing model: {}, dataset: {}".format(model, dataset))
        data = get_processed(dataset)
        aggregate_data(model, data, dst_file)
        
def render_ilp_lp_comparison(data_file, counter_example=False):
    data = json.load(open(data_file))
    models = list(data.keys())
    # this is the counter-example
    if not counter_example:
        models.remove('resnext50')
        models.remove('bert')
    else:
        models = ['resnext50', 'bert']
    
    labels = models
    fig, ax = plt.subplots()
    width = 0.3
    
    lp_values = []
    lp_extracted_values = []
    ilp_values = []
    x_values = np.arange(len(labels))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    for model in labels:
        max_iter = max(data[model].keys())
        model_data = data[model][max_iter]
        lp_normalized = [x / y for x, y in zip(model_data['lp']['obj_opt'], model_data['lp']['obj_start'])]
        lp_ext_normalized = [x / y for x, y in zip(model_data['lp']['obj_ext'], model_data['lp']['obj_start'])]
        ilp_normalized = [x / y for x, y in zip(model_data['ilp']['obj_opt'], model_data['ilp']['obj_start'])]
        lp_values.append(np.mean(lp_normalized))
        lp_extracted_values.append(np.mean(lp_ext_normalized))
        ilp_values.append(np.mean(ilp_normalized))
    
    lp_bar = ax.bar(x_values - width, lp_values, width, label='LP Opt', zorder=1)
    ilp_bar = ax.bar(x_values, ilp_values, width, label='ILP Opt', zorder=2)
    ext_bar = ax.bar(x_values + width, lp_extracted_values, width, label='Rounded', zorder=3)
    ax.set_ylabel('Normalized Cost (by costs of inputs)')
    ax.set_xticks(x_values, labels)
    ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    ax.bar_label(lp_bar, padding=3, fmt='%.2f')
    ax.bar_label(ilp_bar, padding=3, fmt='%.2f')
    ax.bar_label(ext_bar, padding=3, fmt='%.2f')
    fig.tight_layout()
    plt.savefig(f'ilp_lp_comparison{"_ce" if counter_example else ""}.png')

def runtime_comparison(data_file):
    data = json.load(open(data_file))
    models = list(data.keys())
    def set_box_color(bp, color):
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)
    plt.figure()
    lp_runtimes = []
    ilp_runtimes = []
    for model in models:
        max_iter = max(data[model].keys())
        model_data = data[model][max_iter]
        lp_runtimes.append(model_data['lp']['solve_time'])
        ilp_runtimes.append(model_data['ilp']['solve_time'])
        
    lp_box = plt.boxplot(lp_runtimes, positions=np.arange(len(models)) * 2.0 - 0.4, sym='', widths=0.6)
    ilp_box = plt.boxplot(ilp_runtimes, positions=np.arange(len(models)) * 2.0 + 0.4, sym='', widths=0.6)
    set_box_color(lp_box, '#D7191C')
    set_box_color(ilp_box, '#2C7BB6')
    plt.plot([], c='#D7191C', label='LP Solver Time')
    plt.plot([], c='#2C7BB6', label='ILP Solver Time')
    plt.legend()
    plt.title('Solver Time Comparison')
    plt.ylabel('Solver Time (ms)')
    plt.yscale('symlog')

    plt.xticks(range(0, len(models) * 2, 2), models)
    # plt.xlim(-2, len(models)*2)
    # plt.ylim(0, 8)
    plt.tight_layout()
    plt.savefig('runtime_comparison.png')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str)
    parser.add_argument('--mode', default='render', type=str)
    parser.add_argument('--datasets', nargs='+', type=str)
    parser.add_argument('--models', nargs='+', type=str)
    parser.add_argument('--graph', type=str)
    parser.add_argument('--dst_file', type=str)
    args = parser.parse_args()
    if args.mode == 'aggregate':
        aggregate_mode(args.models, args.datasets, args.dst_file)
    elif args.mode == 'render':
        if args.graph == 'ilp_lp_comparison':
            render_ilp_lp_comparison(args.data_file)
            render_ilp_lp_comparison(args.data_file, counter_example=True)
        elif args.graph == 'runtime_comparison':
            runtime_comparison(args.data_file)
    