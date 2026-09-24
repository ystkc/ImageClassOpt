# ImageClassOpt

平台：腾讯CloudStudio PyTorch预设，GPU T4

python 3.11.1

安装（由于腾讯CloudStudio平台的PyTorch预设链pip check都不通过且积弊严重，禁止其检测依赖，注意python有多个，请使用完整路径）
```
/root/.pyenv/versions/3.11.1/bin/python -m pip install --no-deps "numpy>=1.26.4,<2.3" "scipy>=1.13,<1.17" "torchmetrics==1.9.0" "lightning-utilities==0.15.3" "torch==2.6.0" "torchvision==0.21.0" "transformers==5.17.0"
/root/.pyenv/versions/3.11.1/bin/python -m pip install omegaconf
```
修改参数在config.yaml中，运行直接运行run_experiment.py，训练脚本train.py，推理脚本test.py

