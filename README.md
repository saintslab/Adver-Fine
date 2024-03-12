# Adver-Fine
[![arXiv](https://img.shields.io/badge/arXiv-2210.06015-b31b1b.svg)](https://arxiv.org/abs/2210.06015)

## Abstract
As deep learning (DL) models are increasingly being integrated into our everyday lives, ensuring their safety by making them robust against adversarial attacks has become increasingly critical. DL models have been found to be susceptible to adversarial attacks which can be achieved by introducing small, targeted perturbations to disrupt the input data. Adversarial training has been presented as a mitigation strategy which can result in more robust models. This adversarial robustness comes with additional computational costs required to design adversarial attacks during training. The two objectives -- adversarial robustness and computational efficiency -- then appear to be in conflict of each other. In this work, we explore the effects of two different model compression methods -- structured weight pruning and quantization -- on adversarial robustness. We specifically explore the effects of fine-tuning compressed models, and present the trade-offs between normal fine-tuning and robust fine-tuning. Our results show that compression does not inherently lead to loss in model robustness and adversarial fine-tuning of a compressed models can yield large improvements to the robustness performance of models. We present experiments on two benchmark datasets showing that adversarial fine-tuning of compressed models can achieve robustness performance comparable to adversarially trained models, while also maximizing computational efficiency.

<p float="left">
  <img src="fig_1.png" width="600" height="" />
</p>

## Citation
Kindly use the following BibTeX entry if you use the code in your work.
```
@article{thorsteinsson2024adversarial,
 	title={Adversarial Fine-tuning of Compressed Neural Networks for Joint Improvement of Robustness and Efficiency},
	author={Hallgrimur Thorsteinsson and Valdemar J Henriksen and Tong Chen and Raghavendra Selvan},
 	journal={Arxiv},
	year={2024}}
```

