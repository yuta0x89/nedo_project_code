

# Setup

~~~
conda create -n synthtext python=3.11 -y
conda activate synthtext

pip install vllm==0.4.2
pip install datasets==2.19.1

conda install nvidia/label/cuda-12.2.2::cuda-toolkit -y
#cuda toolkit 12.1をcondaで入れて､パスを通す

~~~

# Run
~~~
#1gpuでOK
conda activate synthtext
#パスを通す
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PATH=$CONDA_PREFIX/bin:$PATH
#export CUDA_VISIBLE_DEVICES=0
python 0530autogen.py # wikipediaの場合
~~~