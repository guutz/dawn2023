git config --global --add safe.directory /home/jovyan/work/dawn2023;
eval "$(ssh-agent -s)";
ssh-add /home/jovyan/work/.ssh/.ssh/id_rsa;
mamba create -n tf -c conda-forge cudatoolkit=11.8.0 tensorflow-gpu tensorflow-probability umap-learn bokeh astropy h5py scikit-learn matplotlib pandas jupyter ipykernel jupyter-server-proxy jupyter_bokeh;
conda activate tf;
pip install nvidia-cudnn-cu11==8.6.0.163;
CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"));
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$CUDNN_PATH/lib:$LD_LIBRARY_PATH;
ipython kernel install --user --name=tf;
jupyter serverextension enable --py jupyter_server_proxy

----

git config --global --add safe.directory /home/magutier_caltech_edu/erda_mount/dawn2023;
eval "$(ssh-agent -s)";
ssh-add /home/magutier_caltech_edu/erda_mount/.ssh/.ssh/id_rsa;
mamba create -n tf -c conda-forge cudatoolkit=11.8.0 tensorflow-gpu tensorflow-probability umap-learn bokeh astropy h5py scikit-learn matplotlib pandas jupyter ipykernel jupyter-server-proxy jupyter_bokeh;
conda activate tf;
pip install nvidia-cudnn-cu11==8.6.0.163;
CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"));
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$CUDNN_PATH/lib:$LD_LIBRARY_PATH;
ipython kernel install --user --name=tf;