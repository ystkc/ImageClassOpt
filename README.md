# ImageClassOpt

训练和测试参数分别位于 `configs/train.yaml` 与 `configs/test.yaml`，由 OmegaConf 加载。

先安装配置依赖：

```powershell
pip install -r requirements.txt
```

## 实验调用壳

打开 `run_experiment.py`，修改文件顶部的两个硬编码值：

```python
SCRIPT = "train.py"
CONFIG = "configs/train.yaml"
```

运行时传入实验名称：

```powershell
python run_experiment.py resnet50_baseline
```

也可以省略名称，程序会在运行时询问。YAML 参数可在命令行临时覆盖：

```powershell
python run_experiment.py resnet50_lr_test train.epochs=10 train.learning_rate=1e-4
```

每次运行会创建 `experiments/<时间戳>_<实验名>/`，其中包括：

- `config.yaml`：合并命令行覆盖项后的完整配置备份；
- `source/`：本次运行时的全部 Python 源码备份；
- `stdout.log` 与 `stderr.log`：程序标准输出和标准错误；
- 训练模型、类别映射、指标记录，或测试生成的 CSV。

也可绕过调用壳直接运行：

```powershell
python train.py --config configs/train.yaml train.epochs=10
python test.py --config configs/test.yaml data.batch_size=64
```
