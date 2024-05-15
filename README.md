1. Install anaconda: https://www.anaconda.com/pricing
2. Add anaconda to env: https://www.geeksforgeeks.org/how-to-setup-anaconda-path-to-environment-variable/
3. Open anaconda prompt and type: "conda create -n py310 python=3.10"
4. Activate environment: "conda activate py310"
5. Install cuda and cudnn: "conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0"
6. Install tensorflow: "pip install "tensorflow<2.11" 
7. Open VSCode, in Terminal type: "conda activate py310"
8. In Terminal, type: "python 2TDCNN2CNN3D_32.py" to run 2TDCNN2CNN3D_32.py