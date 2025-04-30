from utils import *
import os, sys
import pickle
from autoattack import AutoAttack
import pandas as pd
import shutil

def main(quant, prune):
    save_str = dict()
    for train_method in ['standard', 'robust']:
        adv = True if train_method == 'robust' else False
        for dataset in ['mnist', 'fashionmnist', 'svhn', 'cifar10', 'cifar100', 'tiny']:
            loader = DataLoad(dataset)
            for model_struct in ['convnet', 'wide_resnet50_2', 'vit_base_mci_224']:
                save_str['save_dir'] = f"models/{dataset}_models/{dataset}_models_{model_struct}"
                if not os.path.isdir(save_str['save_dir']): os.makedirs(save_str['save_dir'])
                for i in range(1):
                    print(f'{train_method} training, {dataset}, {model_struct}, experiment {i}: ')
                    save_str['model_name'] = f'{train_method}_{i}'
                    save_str['file_name'] = f"results_{save_str['model_name']}"
                    save_path = f"{save_str['save_dir']}/{save_str['model_name']}_none_none.pt"
                    model = ModelLoad(model_struct, loader.channel, loader.size, loader.classes).get_network()
                    Train(model, loader).train(num_epochs=10, adv=adv, save=save_path)
                    Compress(model, loader, quant, prune, save_str).execute()


def auto(quant, prune):
    for train_method in ['standard', 'robust']:
        for dataset in ['mnist', 'fashionmnist', 'svhn', 'cifar10', 'cifar100', 'tiny']:
            loader = DataLoad(dataset)
            x_test, y_test = torch.zeros(0), torch.zeros(0)
            for x, y in loader.test_loader:
                x_test = torch.cat((x_test, x), 0)
                y_test = torch.cat((y_test, y), 0)
            x_test, y_test = x_test.float(), y_test.long()
            for model_struct in ['convnet', 'wide_resnet50_2', 'vit_base_mci_224']:
                save_dir = f"models/{dataset}_models/{dataset}_models_{model_struct}"
                if not os.path.isdir(save_dir): os.mkdir(save_dir)
                for i in range(3):
                    results = {'model_name': [],
                               'compress_method': [],
                               'compress_level': [],
                               'test': [],
                               'robust': []}
                    model_name = f'{train_method}_{i}'
                    file_name = f'results_{model_name}'
                    for compressor in [quant, prune]:
                        for level in compressor['level']:
                            for method in compressor['method']:
                                dst = f'{save_dir}/{model_name}_{method}_{level}.pt'
                                model = torch.load(dst, map_location=device).eval()
                                adversary = AutoAttack(model=model, norm='Linf', eps=loader.epsilon, verbose=False, version='rand', device=device)
                                test = adversary.clean_accuracy(x_test, y_test)
                                _, auto = adversary.run_standard_evaluation(x_test, y_test)
                                print('{}, test accuracy: {:.2f}%, AUTO accuracy: {:.2f}%'.format(model_name, test, auto))
                                results['model_name'].append(model_name)
                                results['compress_method'].append(method)
                                results['compress_level'].append(level)
                                results['test'].append(test)
                                results['auto'].append(auto)
                                os.remove(dst)
                    results = pd.DataFrame(results)
                    results.to_pickle(f'{save_dir}/{file_name}_auto.pkl')


def read_results():
    final = dict()
    for train_method in ['standard', 'robust']:
        final[train_method] = dict()
        for dataset in ['mnist', 'fashionmnist', 'cifar10', 'cifar100', 'svhn', 'tiny']:
            final[train_method][dataset] = dict()
            for model_struct in ['convnet', 'wide_resnet50_2', 'vit_base_mci_224']:
                final[train_method][dataset][model_struct] = dict()
                for i in range(3):
                    try:
                        with open(f'models/{dataset}_models/{dataset}_models_{model_struct}/results_{train_method}_{i}.pkl', 'rb') as f:
                            results = pickle.load(f)
                        for k in range(len(results)):
                            s = str(results['compress_method'][k])+'_'+str(results['compress_level'][k])
                            try:
                                final[train_method][dataset][model_struct][s]
                            except KeyError:
                                final[train_method][dataset][model_struct][s] = dict()
                            try:
                                final[train_method][dataset][model_struct][s]['test'].append(results['test'][k])
                                final[train_method][dataset][model_struct][s]['robust'].append(results['robust'][k])
                            except KeyError:
                                final[train_method][dataset][model_struct][s]['test'] = [results['test'][k]]
                                final[train_method][dataset][model_struct][s]['robust'] = [results['robust'][k]]
                    except FileNotFoundError:
                        pass
                if train_method == 'standard':
                    for s in ['none_none', 'prune_std_0.9', 'prune_std_0.7', 'prune_std_0.5', 'prune_adv_0.9', 'prune_adv_0.7', 'prune_adv_0.5']: #'PTQ_std_int16', 'PTQ_std_int8', 'PTQ_std_int4', 'PTQ_adv_int16', 'PTQ_adv_int8', 'PTQ_adv_int4']: #, 'prune_std_0.9', 'prune_std_0.7', 'prune_std_0.5', 'prune_adv_0.9', 'prune_adv_0.7', 'prune_adv_0.5']:
                        print('{} {} {} {} test: {} / {}'.format(train_method, dataset, model_struct, s, np.mean(final[train_method][dataset][model_struct][s]['test']), np.std(final[train_method][dataset][model_struct][s]['test'])))
                        print('{} {} {} {} auto: {} / {}'.format(train_method, dataset, model_struct, s, np.mean(final[train_method][dataset][model_struct][s]['robust']), np.std(final[train_method][dataset][model_struct][s]['robust'])))
                else:
                    for s in ['none_none']: #['PTQ_adv_int16', 'PTQ_adv_int8', 'PTQ_adv_int4', 'prune_adv_0.9', 'prune_adv_0.7', 'prune_adv_0.5']:
                        print('{} {} {} {} test: {} / {}'.format(train_method, dataset, model_struct, s, np.mean(final[train_method][dataset][model_struct][s]['test']), np.std(final[train_method][dataset][model_struct][s]['test'])))
                        print('{} {} {} {} auto: {} / {}'.format(train_method, dataset, model_struct, s, np.mean(final[train_method][dataset][model_struct][s]['robust']), np.std(final[train_method][dataset][model_struct][s]['robust'])))

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    quant = {'level': ['int16', 'int8', 'int4'], #, 'int2', 'int1'],
             'method': ['PTQ_std', 'PTQ_adv']}  # , 'PTQ_none', 'QAT_none', 'QAT_std', 'QAT_adv']}
    prune = {'level': [0.9, 0.7, 0.5], #[0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
             'method': ['prune_std', 'prune_adv']} #, 'prune_none']}

    main(quant, prune)
    auto(quant, prune)
    read_results()