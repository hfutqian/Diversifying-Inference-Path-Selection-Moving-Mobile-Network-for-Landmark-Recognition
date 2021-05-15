# Diversifying Inference Path Selection: Moving-Mobile-Network for Landmark Recognition

## Environment

* python 2.7
* pytorch 0.3.1

## Datasets
We construct two landmark classification datasets: Landmark-420 and Landmark-732, which are available in link

![dataset samples](https://github.com/hfutqian/Diversifying-Inference-Path-Selection-Moving-Mobile-Network-for-Landmark-Recognition/blob/main/images/dataset_samples.png)

## Training process

In the pre-training phase, the policy network is first trained based on both the landmark images and geographic locations. Then the policy network and the pre-trained recognition network are jointly finetuned in the finetuning phase.

(1) Pre-training phase

Run the python file ./Landmark-732/pre-training phase/training_policy_network.py

(2) Finetuning phase

Run the python file ./Landmark-732/finetuning phase/finetune.py


## Citation

