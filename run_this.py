from sklearn.base import is_classifier
from TrainingUtil import *
from model import *
from dataloader import *
import wandb
from torch.optim.lr_scheduler import StepLR
from torch.nn.parallel import DataParallel
from CompressEval import CompressEval
import TrainingUtil

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

#for Weights & Biases:
wandb_project_name = "model_compression_and_finetuning"

#set model name and filename for results file
results_file_name = 'results'
model_name = "normal"
save_directory = "models" #directory for saved models, remember to create directory

#epsilon for PGD l-INF
epsilon = 0.1 #Suggested: 0.1 for Fashion_MNIST, 8.0/255.0 for CIFAR10
pgd = TrainingUtil.get_pgd_linf(epsilon)

#define sparsity levels
quant_levels = ['int16','int8','int4','int2','int1']
quant_methods = ['ptq','ptqRFT','ptqNFT'] #QAT available with 'QATrobust'
prune_levels = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
prune_methods = ['prune']

# Load the dataset
data_loader = FASHIONMNISTDataLoader() #For cifar10 use CIFAR10DataLoader()
train_loader = data_loader.train_loader
test_loader = data_loader.test_loader
classes = data_loader.get_classes()


#repeat experiments three times
for i in range(3):
    #define model
    normal_model = EightLayerConv().to(device) #for ResNet-18 use get_resnet18().to(device)
    #normal_model.load_state_dict(torch.load('models/normal.pt',map_location=device))
    train_and_save_model(normal_model, train_loader, test_loader, pgd, num_epochs=20, r_train=False, save_path=f'{save_directory}/{model_name}.pt', project_name=wandb_project_name)

    #prune, quantize, save and evaluate model for all sparsity levels
    compressor = CompressEval(device,train_loader,test_loader,pgd,quant_levels, quant_methods,prune_levels,prune_methods,results_file_name,save_directory,wandb_project_name)
    compressor.multi_compress(normal_model,model_name)

###### Example wtih robust baseline model:






