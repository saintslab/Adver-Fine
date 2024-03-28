import torch
import numpy as np
from nni.compression.pruning import LevelPruner, L1NormPruner
from nni.compression.speedup import ModelSpeedup
import torch.nn.utils.prune as prune

class ModelPruner:
    def __init__(self, model, config_list, device, dummy_input=torch.rand((1, 1, 28, 28))):
        self.model = model
        self.config_list = config_list
        self.device = device
        self.dummy_input = dummy_input.to(device)

    def prune(self):
        self.pruner = L1NormPruner(self.model, self.config_list)
        _, mask = self.pruner.compress()

        self.pruner.unwrap_model()
        # dummy_input = torch.rand((1, 1, 28, 28)).to(self.device)

        model_speedup = ModelSpeedup(self.model, self.dummy_input, mask).speedup_model()

        return model_speedup
    
    def pytorch_prune(self, amount=0.1, remove=False):
        parameters_to_prune = [
        (module, "weight") for module in filter(lambda m: type(m) == torch.nn.Conv2d, self.model.modules())
        
        ]
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=amount,
        )   
        if remove:
        #Freeze the pruned weights
            for module in filter(lambda m: type(m) == torch.nn.Conv2d, self.model.modules()):
                prune.remove(module, "weight")

        for name, param in self.model.named_parameters():
            if 'weight_orig' in name:  # Adjust based on your pruning method
                param.requires_grad = False
        return self.model

    def check_sparsity(self):
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                print(
                    "Sparsity in {}.weight: {:.2f}%".format(
                        name, 100. * float(torch.sum(module.weight == 0))
                        / float(module.weight.nelement())
                    )
                )
            elif isinstance(module, torch.nn.Linear):
                print(
                    "Sparsity in {}.weight: {:.2f}%".format(
                        name, 100. * float(torch.sum(module.weight == 0))
                        / float(module.weight.nelement())
                    )
                )