git config --global --add safe.directory /home/jovyan/work/dawn2023;
eval "$(ssh-agent -s)";
ssh-add /home/jovyan/work/.ssh/.ssh/id_rsa;
mamba env create -f /home/jovyan/work/umap_erda.yml;
conda activate umap;
mkdir -p $CONDA_PREFIX/lib/nvvm/libdevice;
cp $CONDA_PREFIX/lib/libdevice.10.bc $CONDA_PREFIX/lib/nvvm/libdevice/;
jupyter lab build;
jupyter serverextension enable --py jupyter_server_proxy;
ipython kernel install --user --name=umap