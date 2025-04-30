import numpy as np
import copy
import timm
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler, TensorDataset
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import LambdaLR

import nni
from nni.compression import TorchEvaluator
from nni.compression.utils import auto_set_denpendency_group_ids
from nni.compression.quantization import PtqQuantizer, QATQuantizer
from nni.compression.pruning import LevelPruner, L1NormPruner, MovementPruner
from nni.compression.speedup import ModelSpeedup


class DataLoad:
    def __init__(self, dataset, batch_size=64, num_workers=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

        if dataset == 'mnist':
            self.mean, self.std = (0.1307,), (0.3081,)
            self.channel, self.size, self.classes = 1, 28, 10
            self.epsilon = 0.1
        elif dataset == 'fashionmnist':
            self.mean, self.std = (0.2861,), (0.3530,)
            self.channel, self.size, self.classes = 1, 28, 10
            self.epsilon = 0.1
        elif dataset == 'svhn':
            self.mean, self.std = (0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)
            self.channel, self.size, self.classes = 3, 32, 10
            self.epsilon = 8 / 255
        elif dataset == 'cifar10':
            self.mean, self.std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            self.channel, self.size, self.classes = 3, 32, 10
            self.epsilon = 8 / 255
        elif dataset == 'cifar100':
            self.mean, self.std = (0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2762)
            self.channel, self.size, self.classes = 3, 32, 100
            self.epsilon = 8 / 255
        elif dataset == 'tiny':
            self.mean, self.std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            self.channel, self.size, self.classes = 3, 64, 200
            self.epsilon = 4 / 255

        self._transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

        self._prepare_data()

    def _prepare_data(self, val=False):
        if self.dataset == 'mnist':
            train_data = datasets.MNIST('data', train=True, download=True, transform=self._transform)
            test_data = datasets.MNIST('data', train=False, download=True, transform=self._transform)
        elif self.dataset == 'fashionmnist':
            train_data = datasets.FashionMNIST('data', train=True, download=True, transform=self._transform)
            test_data = datasets.FashionMNIST('data', train=False, download=True, transform=self._transform)
        elif self.dataset == 'svhn':
            train_data = datasets.SVHN('data', split='train', download=True, transform=self._transform)
            test_data = datasets.SVHN('data', split='test', download=True, transform=self._transform)
        elif self.dataset == 'cifar10':
            train_data = datasets.CIFAR10('data', train=True, download=True, transform=self._transform)
            test_data = datasets.CIFAR10('data', train=False, download=True, transform=self._transform)
        elif self.dataset == 'cifar100':
            train_data = datasets.CIFAR100('data', train=True, download=True, transform=self._transform)
            test_data = datasets.CIFAR100('data', train=False, download=True, transform=self._transform)
        elif self.dataset == 'tiny':
            self.data = torch.load('data/tinyimagenet.pt')
            images_train = self.data['images_train']
            labels_train = self.data['labels_train']
            images_train = images_train.detach().float() / 255.0
            labels_train = labels_train.detach()
            for c in range(3):
                images_train[:, c] = (images_train[:, c] - self.mean[c]) / self.std[c]
            train_data = TensorDataset(images_train, labels_train)

            images_test = self.data['images_val']
            labels_test = self.data['labels_val']
            images_test = images_test.detach().float() / 255.0
            labels_test = labels_test.detach()
            for c in range(3):
                images_test[:, c] = (images_test[:, c] - self.mean[c]) / self.std[c]
            test_data = TensorDataset(images_test, labels_test)

        if val:
            num_train = len(train_data)
            indices = list(range(num_train))
            np.random.shuffle(indices)
            split = int(np.floor(0.2 * num_train))
            train_idx, validation_idx = indices[split:], indices[:split]
            train_sampler = SubsetRandomSampler(train_idx)
            validation_sampler = SubsetRandomSampler(validation_idx)
            self.validation_loader = DataLoader(train_data, batch_size=self.batch_size, sampler=validation_sampler, num_workers=self.num_workers)
            self.train_loader = DataLoader(train_data, batch_size=self.batch_size, sampler=train_sampler, num_workers=self.num_workers)
        else:
            self.train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    # def get_classes(self):
    #     if self.dataset in ['mnist', 'svhn']:
    #         return ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    #     elif self.dataset == 'fashionmnist':
    #         return ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt',
    #                 'Sneaker', 'Bag', 'Ankle boot']
    #     elif self.dataset == 'cifar10':
    #         return ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    #     elif self.dataset == 'cifar100':
    #         return ['beaver', 'dolphin', 'otter', 'seal', 'whale', 'aquarium fish', 'flatfish', 'ray', 'shark', 'trout',
    #             'orchids', 'poppies', 'roses', 'sunflowers', 'tulips', 'bottles', 'bowls', 'cans', 'cups', 'plates',
    #             'apples', 'mushrooms', 'oranges', 'pears', 'sweet peppers', 'clock', 'computer keyboard', 'lamp',
    #             'telephone', 'television', 'bed', 'chair', 'couch', 'table', 'wardrobe', 'bee', 'beetle', 'butterfly',
    #             'caterpillar', 'cockroach', 'bear', 'leopard', 'lion', 'tiger', 'wolf', 'bridge', 'castle', 'house',
    #             'road', 'skyscraper', 'cloud', 'forest', 'mountain', 'plain', 'sea', 'camel', 'cattle', 'chimpanzee',
    #             'elephant', 'kangaroo', 'fox', 'porcupine', 'possum', 'raccoon', 'skunk', 'crab', 'lobster', 'snail',
    #             'spider', 'worm', 'baby', 'boy', 'girl', 'man', 'woman', 'crocodile', 'dinosaur', 'lizard', 'snake',
    #             'turtle', 'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel', 'maple', 'oak', 'palm', 'pine', 'willow',
    #             'bicycle', 'bus', 'motorcycle', 'pickup truck', 'train', 'lawn-mower', 'rocket', 'streetcar', 'tank',
    #             'tractor']
    #     elif self.dataset == 'tiny':
    #         return self.data['classes']


class ConvNet(nn.Module):
    def __init__(self, channel=3, classes=10):
        super(ConvNet, self).__init__()
        self.channel = channel

        self.conv1 = nn.Conv2d(channel, 32, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1, stride=2)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1, stride=2)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1, stride=2)
        self.relu6 = nn.ReLU()
        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.relu7 = nn.ReLU()
        self.fc2 = nn.Linear(128, classes)

    def forward(self, x):
        # breakpoint()
        x = nn.functional.relu(nn.Conv2d(x.shape[1], 32, 3, padding=1)(x))
        x = nn.functional.relu(nn.Conv2d(32, 32, 3, padding=1, stride=2)(x))
        x = nn.functional.relu(nn.Conv2d(32, 64, 3, padding=1)(x))
        x = nn.functional.relu(nn.Conv2d(64, 64, 3, padding=1, stride=2)(x))
        x = nn.functional.relu(nn.Conv2d(64, 128, 3, padding=1)(x))
        x = nn.functional.relu(nn.Conv2d(128, 128, 3, padding=1, stride=2)(x))
        x = x.view(x.shape[0], -1)
        x = nn.functional.relu(nn.Linear(128 * 4 * 4, 128)(x))
        x = nn.Linear(128, 10)(x)
        # x = nn.functional.relu(self.conv1(x))
        # x = nn.functional.relu(self.conv2(x))
        # x = nn.functional.relu(self.conv3(x))
        # x = nn.functional.relu(self.conv4(x))
        # x = nn.functional.relu(self.conv5(x))
        # x = nn.functional.relu(self.conv6(x))
        # x = x.view(x.shape[0], -1)
        # x = nn.functional.relu(self.fc1(x))
        # x = self.fc2(x)
        return x


class ModelLoad:
    def __init__(self, model_structure, channel=3, size=64, classes=10, pretrain=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_structure = model_structure
        self.channel = channel
        self.size = size
        self.classes = classes
        self.pretrain = pretrain

    def get_network(self):
        '''
        examples: 'resnet18', 'resnet34', 'resnet50', 'wide_resnet50_2', 'inception_v3', 'vit_base_mci_224'
        '''
        if self.model_structure == 'convnet':
            return ConvNet(self.channel, self.classes).to(self.device)
        elif 'resnet' in self.model_structure:
            return timm.create_model(self.model_structure, in_chans=self.channel, num_classes=self.classes, pretrained=self.pretrain).to(self.device)
        elif 'vit' in self.model_structure:
            return timm.create_model(self.model_structure, in_chans=self.channel, img_size=self.size, num_classes=self.classes, pretrained=self.pretrain).to(self.device)


class Attack:
    def __init__(self, model, epsilon):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.epsilon = epsilon

    def fgsm(self, x, y):
        delta = torch.zeros_like(x, requires_grad=True)
        loss = nn.CrossEntropyLoss()(self.model(x + delta), y)
        loss.backward()
        return self.epsilon * delta.grad.detach().sign()

    def pgd_linf(self, x, y, alpha=0.01, num_iter=50, randomize=False):
        if randomize:
            delta = torch.rand_like(x, requires_grad=True)
            delta.data = delta.data * 2 * self.epsilon - self.epsilon
        else:
            delta = torch.zeros_like(x, requires_grad=True)
        for t in range(num_iter):
            loss = nn.CrossEntropyLoss()(self.model(x + delta), y)
            loss.backward()
            delta.data = (delta + alpha * delta.grad.detach().sign()).clamp(-self.epsilon, self.epsilon)
            delta.grad.zero_()
        return delta.detach()


class Train:
    def __init__(self, model, loader, attack_method='pgd'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.loader = loader
        self.epsilon = loader.epsilon
        self.attack_method = attack_method
        self.attack = Attack(self.model, self.epsilon).fgsm if self.attack_method == 'fgsm' else Attack(self.model, self.epsilon).pgd_linf

    def epoch(self, opt=None, adv=False, **kwargs):
        if opt:
            self.model.train()
            current_loader = self.loader.train_loader
        else:
            self.model.eval()
            current_loader = self.loader.test_loader
        total_loss, total_err = 0., 0.
        for x, y in current_loader:
            x, y = x.to(self.device), y.to(self.device)
            if adv:
                delta = self.attack(x, y, **kwargs)
                yp = self.model(x + delta)
            else:
                yp = self. model(x)
            loss = nn.CrossEntropyLoss()(yp, y)
            if opt:
                opt.zero_grad()
                loss.backward()
                opt.step()
            if adv:
                total_err += ((yp.max(dim=1)[1] != y) * (self.model(x).max(dim=1)[1] != y)).sum().item()
            else:
                total_err += (yp.max(dim=1)[1] != y).sum().item()
            total_loss += loss.item() * x.shape[0]
        return (1.0 - (total_err / len(current_loader.dataset))) * 100, total_loss / len(current_loader.dataset)


    def train(self, num_epochs=20, adv=False, save=False, log=True):
        opt = optim.SGD(self.model.parameters(), lr=1e-1)
        for t in range(num_epochs):
            train_acc, train_loss = self.epoch(opt=opt, adv=adv)
            test_acc, test_loss = self.epoch(adv=False)
            adv_acc, adv_loss = self.epoch(adv=True)
            if t == 4:
                for param_group in opt.param_groups:
                    param_group["lr"] = 1e-2
            if log:
                print(f'epoch {t}: ')
                print('train accuracy: {:.2f}%, test accuracy: {:.2f}%, {} accuracy: {:.2f}%'.format(train_acc, test_acc, self.attack_method.upper(), adv_acc))
        if save:
            torch.save(self.model, save)


class Quantizer:
    def __init__(self, model, loader, bit, attack=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.loader = loader
        self.attack = attack
        self.adv = True if self.attack else False
        self.quantizer_config_list = [{
            'op_types': ['BatchNorm2d', 'Conv2d', 'Linear'],
            'target_names': ['_input_', 'weight', '_output_'],
            'quant_dtype': bit,
            'quant_scheme': 'symmetric',
            'granularity': 'default'
        }, {
            'op_types': ['ReLU'],
            'target_names': ['_output_'],
            'quant_dtype': bit,
            'quant_scheme': 'affine',
            'granularity': 'default'
        }]

    def train_step(self, batch, model, *args, **kwargs):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        if self.attack:
            delta = self.attack(x, y)
            yp = model(x + delta)
        else:
            yp = model(x)
        loss = nn.CrossEntropyLoss()(yp, y)
        return loss

    def train_model(self, model, optimizer, train_step, lr_scheduler=None, max_steps=None, max_epochs=None, *args, **kwargs):
        model.train()
        total_epochs = max_epochs if max_epochs else 10
        total_steps = max_steps if max_steps else 10 ** 9
        current_step = 0
        for t in range(total_epochs):
            for batch in self.loader.train_loader:
                loss = train_step(batch, model)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                current_step += 1
                if current_step >= total_steps:
                    return

    def eval_model(self, model, *args, **kwargs):
        trainer = Train(model, self.loader)
        accuracy, _ = trainer.epoch(adv=self.adv)
        return accuracy

    def fine_tune(self, model):
        trainer = Train(model, self.loader)
        trainer.train(num_epochs=3, adv=self.adv, log=False)

    def compr(self, qat=False, ft=False):
        traced_optimizer = nni.trace(optim.SGD)(self.model.parameters(), lr=0.01, momentum=0.9)
        evaluator = TorchEvaluator(training_func=self.train_model, optimizers=traced_optimizer, training_step=self.train_step, evaluating_func=self.eval_model)
        if qat:
            quantizer = QATQuantizer(self.model, self.quantizer_config_list, evaluator)
        else:
            quantizer = PtqQuantizer(self.model, self.quantizer_config_list, evaluator)
        self.model, _ = quantizer.compress(max_steps=None, max_epochs=3)
        if ft:
            self.fine_tune(self.model)
        return self.model


class Pruner:
    def __init__(self, model, loader, level, attack=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.loader = loader
        self.level = level
        self.dummy_input = torch.rand(1, self.loader.channel, self.loader.size, self.loader.size).to(self.device)
        self.attack = attack
        self.adv = True if self.attack else False
        if self.model.__class__.__name__ in ['ResNet', 'ConvNet']:
            self.config_list = [{
                'op_types': ['Conv2d'],
                'sparse_ratio': self.level
            }]
        elif self.model.__class__.__name__ == 'VisionTransformer':
            self.config_list = [{
                'op_types': ['Linear'],
                'sparse_ratio': self.level
            }]

    def train_step(self, batch, model, *args, **kwargs):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        if self.attack:
            delta = self.attack(x, y)
            yp = model(x + delta)
        else:
            yp = model(x)
        loss = nn.CrossEntropyLoss()(yp, y)
        return loss

    def train_model(self, model, optimizer, train_step, lr_scheduler=None, max_steps=None, max_epochs=None, *args, **kwargs):
        model.train()
        total_epochs = max_epochs if max_epochs else 10
        total_steps = max_steps if max_steps else 10 ** 9
        current_step = 0
        for t in range(total_epochs):
            for batch in self.loader.train_loader:
                loss = train_step(batch, model)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                current_step += 1
                if current_step >= total_steps:
                    return

    def eval_model(self, model):
        trainer = Train(model, self.loader)
        accuracy, _ = trainer.epoch(adv=self.adv)
        return accuracy

    def fine_tune(self, model):
        trainer = Train(model, self.loader)
        trainer.train(num_epochs=3, adv=self.adv, log=False)

    def compr(self, ft=False):
        # if self.model.__class__.__name__ in ['ResNet', 'ConvNet']:
        #     pruner = L1NormPruner(self.model, self.config_list)
        #     _, mask = pruner.compress()
        # elif self.model.__class__.__name__ == 'VisionTransformer':
        #     steps_per_epoch = len(self.loader.train_loader)
        #     total_epochs = 4
        #     total_steps = total_epochs * steps_per_epoch
        #     warmup_steps = 1 * steps_per_epoch
        #     cooldown_steps = 1 * steps_per_epoch
        #     traced_optimizer = nni.trace(optim.SGD)(self.model.parameters(), lr=0.01, momentum=0.9)
        #     evaluator = TorchEvaluator(training_func=self.train_model, optimizers=traced_optimizer, training_step=self.train_step, evaluating_func=self.eval_model)
        #     pruner = MovementPruner(model=self.model,
        #                             config_list=self.config_list,
        #                             evaluator=evaluator,
        #                             warmup_step=warmup_steps,
        #                             cooldown_begin_step=total_steps - cooldown_steps)
        #     _, mask = pruner.compress(max_steps=None, max_epochs=3)
        pruner = L1NormPruner(self.model, self.config_list)
        _, mask = pruner.compress()
        pruner.unwrap_model()
        self.model = ModelSpeedup(self.model, self.dummy_input, mask).speedup_model()
        # print('Pruned model parameter number: ', sum([param.numel() for param in self.model.parameters()]))
        if ft:
            self.fine_tune(self.model)
        return self.model


class Compress:
    def __init__(self, model, loader, quant, prune, save_str, attack_method='pgd'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.loader = loader
        self.epsilon = loader.epsilon
        self.attack_method = attack_method
        self.quant = quant
        self.prune = prune
        self.model_name = save_str['model_name']
        self.file_name = save_str['file_name']
        self.save_dir = save_str['save_dir']
        self.results = {'model_name': [],
                        'compress_method': [],
                        'compress_level': [],
                        'test': [],
                        'robust': []}

    def compress(self, compressor, config):
        for level in config['level']:
            for method in config['method']:
                if method.split('_')[1] == 'adv':
                    attack = Attack(self.model, self.epsilon)
                    attack = attack.fgsm if self.attack_method == 'fgsm' else attack.pgd_linf
                else:
                    attack = None
                model = copy.deepcopy(self.model).to(self.device)
                comp = compressor(model, self.loader, level, attack)
                ft = False if method.split('_')[1] == 'none' else True
                comp.compr(ft=ft)
                self.eval_save(model, level=level, method=method)

    def eval_save(self, model, level, method):
        trainer = Train(model, self.loader, self.attack_method)
        test, _ = trainer.epoch()
        robust, _ = trainer.epoch(adv=True)
        self.results['model_name'].append(self.model_name)
        self.results['compress_method'].append(method)
        self.results['compress_level'].append(level)
        self.results['test'].append(test)
        self.results['robust'].append(robust)
        torch.save(model, f'{self.save_dir}/{self.model_name}_{method}_{level}.pt')
        print('{}_{}_{}, test accuracy: {:.2f}%, {} accuracy: {:.2f}%'.format(self.model_name, method, level, test, self.attack_method.upper(), robust))

    def execute(self):
        self.eval_save(self.model, level="none", method="none")
        self.compress(Quantizer, self.quant)
        self.compress(Pruner, self.prune)
        self.results = pd.DataFrame(self.results)
        self.results.to_pickle(f'{self.save_dir}/{self.file_name}.pkl')