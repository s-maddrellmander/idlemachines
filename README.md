


```
conda create -n fp8 python=3.12 -y
conda activate fp8

python -m pip install -U pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
python -m pip install torchao --index-url https://download.pytorch.org/whl/cu130
python -m pip install matplotlib seaborn pandas
```