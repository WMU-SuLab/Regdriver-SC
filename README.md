# Regdriver-SC
Regdriver-SC can be used to distinguish driver mutations from non-driver mutations, identify true driver events, and integrates mutation impact scoring with mutation burden analysis.

1.Create the initial environment

```conda create -n Regdriver-SC python=3.8
conda activate Regdriver-SC
# (optional if you would like to use flash attention)
# install triton from source
git clone https://github.com/openai/triton.git
cd triton/python
pip install cmake       # 构建依赖
pip install -e .        # 从源码安装 Triton
python3 -m pip install -r requirements.txt
conda install -c huggingface transformers
conda install -c bioconda pyfaidx
conda install pytorch torchvision torchaudio cpuonly -c pytorch
conda config --add channels defaults
conda config --add channels bioconda
conda config --add channels conda-forge```

2.Versions
------
