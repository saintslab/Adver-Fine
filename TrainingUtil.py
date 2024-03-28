import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import math



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def fgsm(model, X, y, epsilon=0.1):
    """ Construct FGSM adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    loss = nn.CrossEntropyLoss()(model(X + delta), y)
    loss.backward()
    return epsilon * delta.grad.detach().sign()

def get_pgd_linf(epsilon):
    '''Current fix for pgd_linf'''
    
    def pgd_linf(model, X, y, epsilon=epsilon, alpha=0.01, num_iter=20, randomize=False):
        """ Construct FGSM adversarial examples on the examples X
        Suggested epsilon: 0.1 for Fashion-MNIST, 8.0/255.0 for CIFAR10
        """
        if randomize:
            delta = torch.rand_like(X, requires_grad=True)
            delta.data = delta.data * 2 * epsilon - epsilon
        else:
            delta = torch.zeros_like(X, requires_grad=True)
            
        for t in range(num_iter):
            loss = nn.CrossEntropyLoss()(model(X + delta), y)
            loss.backward()
            delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
            delta.grad.zero_()
        return delta.detach()
    
    return pgd_linf




def epoch(loader, model, opt=None):
    """Standard training/evaluation epoch over the dataset"""
    if opt==None:
        model.eval()
    else: model.train()
    total_loss, total_err = 0.,0.
    for X,y in loader:
        X,y = X.to(device), y.to(device)
        yp = model(X)
        loss = nn.CrossEntropyLoss()(yp,y)
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]
    return 1.0 - (total_err / len(loader.dataset)), total_loss / len(loader.dataset)


def epoch_adversarial(loader, model, attack, opt=None, **kwargs):
    """Adversarial training/evaluation epoch over the dataset"""
    if opt==None:
        model.eval()
    else: model.train()
    total_loss, total_err = 0.,0.
    for X,y in loader:
        X,y = X.to(device), y.to(device)
        delta = attack(model, X, y, **kwargs)
        yp = model(X+delta)
        loss = nn.CrossEntropyLoss()(yp,y)
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]
    return 1.0 - (total_err / len(loader.dataset)), total_loss / len(loader.dataset)


def train_and_save_model(model, train_loader, test_loader, pgd_linf, lr_schedule=None, num_epochs=10, r_train=True, save_path="model.pt", project_name="compressed-robust"):
    wandb.init(project=project_name, name=save_path[:-3])

    
    # Define your optimizer
    # opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    opt = optim.SGD(model.parameters(), lr=1e-1)

    # Define your learning rate schedule
    def lr_lambda(epoch):
        if epoch < 5:
            return epoch / 5
        else:
            return 0.1 * (1 + math.cos(math.pi * epoch / 50))

    # scheduler = LambdaLR(opt, lr_lambda)
    
    test_err, test_loss = epoch(test_loader, model)
    adv_err, adv_loss = epoch_adversarial(test_loader, model, pgd_linf)

    wandb.log({"test_loss": test_loss, "test_error": test_err, "adv_error": adv_err})
    
    '''
    lr_schedule_type = "linear"
    s_factor = 1.0
    e_factor = 1.0
    # Define your learning rate schedule
    scheduler = LinearLR(opt,start_factor=s_factor, end_factor=e_factor,total_iters=num_epochs)
    ,config = {"lr_schedule_type":lr_schedule_type,"lr":lr,"momentum":momentum,"start factor":s_factor,"end_factor":e_factor}
    '''
    wandb.init(project=project_name, name=save_path[:-3])
    
    for t in range(num_epochs):
        current_lr = opt.param_groups[0]["lr"]

        if r_train:
            train_err, train_loss = epoch_adversarial(train_loader, model, pgd_linf, opt)
            test_err, test_loss = epoch(test_loader, model)
            adv_err, adv_loss = epoch_adversarial(test_loader, model, pgd_linf)
        else:
            train_err, train_loss = epoch(train_loader, model, opt)
            test_err, test_loss = epoch(test_loader, model)
            adv_err, adv_loss = epoch_adversarial(test_loader, model, pgd_linf)
        if t == 4:
            for param_group in opt.param_groups:
                param_group["lr"] = 1e-2

        wandb.log({"train_loss": train_loss, "train_error": train_err, "test_loss": test_loss, "test_error": test_err, "adv_error": adv_err, "adv_loss":adv_loss})

        print(*("{:.6f}".format(i) for i in (train_err, test_err, adv_err)), sep="\t")
    
    torch.save(model.state_dict(), save_path)
    wandb.finish()

def evaluate_model(model, test_loader, pgd_linf):

    test_err, test_loss = epoch(test_loader, model)
    adv_err, adv_loss = epoch_adversarial(test_loader, model, pgd_linf)
        
    return test_err, adv_err


