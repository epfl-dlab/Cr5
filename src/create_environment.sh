conda config --add channels intel
conda create -n idp_full intelpython3_full cython python=3
source activate idp_full
export MKL_DYNAMIC=FALSE
export OMP_NUM_THREADS=1