# Regdriver-SC
Regdriver-SC can be used to distinguish driver mutations from non-driver mutations, identify true driver events, and integrates mutation impact scoring with mutation burden analysis.

## 1.Create the initial environment

```conda create -n Regdriver-SC python=3.8
conda activate Regdriver-SC
#(optional if you would like to use flash attention)
#install triton from source
git clone https://github.com/openai/triton.git
cd triton/python
pip install cmake       # 构建依赖
pip install -e .        # 从源码安装 Triton
#install required packages（在本地，后上传至github上）
python3 -m pip install -r requirements.txt
conda install -c huggingface transformers
conda install -c bioconda pyfaidx
conda install pytorch torchvision torchaudio cpuonly -c pytorch

conda config --add channels defaults
conda config --add channels bioconda
conda config --add channels conda-forge```

## 2.versions
  ```- python=3.8
  - samtools
  - pandas
  - bedtools
  - numpy
  - openjdk=8
  - tabix
  - htslib
  - pyfaidx
  - r-base=3.5.1
  - cudatoolkit=10.2
  - pytorch=1.7.1```
