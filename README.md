# The Repo of Frequency Domain Distillation for Data-Free Quantization of Vision Transformer
## The overview of our method
Coming soon...
## Environment
1. python 3.9.7
2. torch 1.12.1
3. torchvision 0.13.1
4. timm 0.5.4
5. pandas 1.3.4
6. tensorboard 2.9.1

We recommend install 'torch' and 'torchvision' from pytorch's official website [here](https://pytorch.org/get-started/locally/).
## Run
1. Modify [command_FDD.sh](command_FDD.sh)(ours) or [command.sh](command.sh)(PSAQ)
```
--model [swin_tiny, swin_small, deit_tiny, deit_small, deit_base]
--dataset [THE PATH TO YOUR IMAGENET]
--mode [0: Generate fake data
        1: Noise
        2: Real data
        3: Load from local(Only support command_FDD.sh)]
--w_bit [BIT FOR WEIGHT]
--a_bit [BIT FOR ACTIVATION]
```
2. Run the command
```bash
bash command_FDD.sh
bash command.sh
```
## Results
### swin_tiny (seed 666 epoch 100)
|method|bit(wa)|acc(top1)|epoch|
|:-:|:-:|:-:|:-:|
|Noise|48|1.058|--|
|Real|48|69.042|--|
|PSAQ|48|69.082|1000|
|Ours|48|**72.156**|**300**|
|Noise|88|1.608|--|
|Real|88|73.742|--|
|PSAQ|88|73.324|1000|
|Ours|88|**75.494**|**300**|
### swin_small (seed 444 epoch 200)
|method|bit(wa)|acc(top1)|epoch|
|:-:|:-:|:-:|:-:|
|Noise|48|0.736|--|
|Real|48|71.258|--|
|PSAQ|48|74.892|1000|
|Ours|48|**75.286**|**600**|
|Noise|88|0.796|--|
|Real|88|73.574|--|
|PSAQ|88|76.538|1000|
|Ours|88|**76.846**|**600**|
### deit_tiny (seed 666 epoch 100)
|method|bit(wa)|acc(top1)|epoch|
|:-:|:-:|:-:|:-:|
|Noise|48|14.758|--|
|Real|48|65.192|--|
|PSAQ|48|65.592|1000|
|Ours|48|**65.768**|**300**|
|Noise|88|17.710|--|
|Real|88|71.270|--|
|PSAQ|88|71.554|1000|
|Ours|88|**71.670**|**300**|
### deit_small (seed 444 epoch 300)
|method|bit(wa)|acc(top1)|epoch|
|:-:|:-:|:-:|:-:|
|Noise|48|7.570|--|
|Real|48|72.338|--|
|PSAQ|48|72.324|1000|
|Ours|48|**72.452**|**900**|
|Noise|88|11.858|--|
|Real|88|76.132|--|
|PSAQ|88|76.172|1000|
|Ours|88|**76.288**|**900**|
### deit_base (seed 444 555 epoch 300)
|method|bit(wa)|acc(top1)|epoch|
|:-:|:-:|:-:|:-:|
|Noise|48|12.142|--|
|Real|48|76.280|--|
|PSAQ|48|76.526|1000|
|Ours|48|**77.160**|**900**|
|Noise|88|15.732|--|
|Real|88|78.684|--|
|PSAQ|88|78.964|1000|
|Ours|88|**79.384**|**900**|
## Acknowledgement
Thanks to Zhikai for his excellent work [PSAQ](https://arxiv.org/abs/2203.02250) and [code](https://github.com/zkkli/PSAQ-ViT).