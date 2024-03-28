import nni
from nni.compression.quantization import PtqQuantizer
from nni.compression.quantization import QATQuantizer
from nni.compression import TorchEvaluator
from torch.optim import SGD
import torch.nn as nn
import time
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import math
import wandb
import torch.nn.functional as F



class quantizer:
    def __init__(self, dataloader, test_loader, bit_width :str, device, attack, project_name):
        self.dataloader = dataloader
        self.device = device
        self.attack = attack
        self.project_name = project_name
        self.test_loader = test_loader
        self.bit_width = bit_width
        self.quantizer_config_list = [{
            'op_types': ['BatchNorm2d', 'Conv2d','Linear'],
            'target_names': ['_input_', 'weight', '_output_'],
            'quant_dtype': bit_width,
            'quant_scheme': 'symmetric',
            'granularity': 'default'
            },{
            'op_types': ['ReLU'],
            'target_names': ['_output_'],
            'quant_dtype': bit_width,
            'quant_scheme': 'affine',
            'granularity': 'default'
            }]

    #returns the train_step or adv_train_step function for the quantizer
    def getTrainStepFunc(self, adversarial):
        
        def training_step(batch,model):
            input_data, target = batch
            X, y = input_data.to(self.device), target.to(self.device)
            if adversarial:
                delta = self.attack(model, X, y)
                yp = model(X + delta)
            else:
                yp = model(X)

            loss = nn.CrossEntropyLoss()(yp,y)
            return loss

        return training_step

    def train_model(self, model, optimizer, training_step, lr_scheduler, max_steps, max_epochs,**kwargs):
        
        model.train()
        total_epochs = max_epochs if max_epochs else 0
        total_steps = max_steps if max_steps else 10**8
        current_step = 0
    
        loader = self.dataloader
        for e in range(total_epochs):
            total_loss = 0.0
            for batch in loader:
                loss = training_step(batch,model)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch[0].shape[0]

                current_step += 1
                if current_step >= total_steps:
                    print("max steps reached")
                    return
                
            train_loss = total_loss / len(loader.dataset)

            wandb.log({"train_loss":train_loss})


    def get_evaluation_func(self, robust,is_qat):

        def evaluation_func(model):
            model.eval()
            total_loss, total_err = 0.,0.
            for X,y in self.test_loader:
                X,y = X.to(self.device), y.to(self.device)
                if robust:
                    delta = self.attack(model, X, y)
                    yp = model(X+delta)
                else:
                    yp = model(X)
                loss = nn.CrossEntropyLoss()(yp,y)
                total_err += (yp.max(dim=1)[1] != y).sum().item()
                total_loss += loss.item() * X.shape[0]
            total_err = total_err / len(self.test_loader.dataset)
            total_loss = total_loss / len(self.test_loader.dataset)
            accuracy = 1.0-total_err
            return accuracy

        return evaluation_func

    
    def quantize(self, model, robust_model:bool, QAT,run_name, fine_tune = True):
        
        start = time.time()
        training_step = self.getTrainStepFunc(robust_model)
        
        traced_optimizer = nni.trace(SGD)(model.parameters(), lr=0.01,momentum=0.9) # 
        evaluator = TorchEvaluator(training_func=self.train_model, optimizers=traced_optimizer, training_step=training_step,evaluating_func=self.get_evaluation_func(robust_model,is_qat=QAT))  #, evaluating_func=self.get_evaluation_func(robust_model,is_qat=QAT) type: ignore
        if QAT==True:
            wandb.init(project=self.project_name, name=run_name)
            quantizer = QATQuantizer(model, self.quantizer_config_list, evaluator)
            _, self.calibration_config = quantizer.compress(max_steps=10**9, max_epochs=20)
            wandb.finish()
        elif QAT==False and fine_tune==True:
            wandb.init(project=self.project_name, name=run_name)
            quantizer = PtqQuantizer(model, self.quantizer_config_list, evaluator)
            _, self.calibration_config = quantizer.compress(max_steps=10**8, max_epochs=3)
            wandb.finish()
        else:
            quantizer = PtqQuantizer(model, self.quantizer_config_list, evaluator)
            _, self.calibration_config = quantizer.compress(max_steps=1, max_epochs=0)
        print(f'quantized in {time.time()-start:.1f} seconds, with QAT={QAT}, and fine_tune={fine_tune}')
        
        return model



