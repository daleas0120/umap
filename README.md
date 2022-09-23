# umap
Implements the UMAP algorithm from https://github.com/lmcinnes/umap in a jupyter notebook with the following features
1. Clustering with scikit-learn DBSCAN algorithm
2. Metrics on the clusters using scikit-learn metrics
3. Interactive plotting using Dash and Plotly

## Installation
1. Create a new `conda` environment 

`conda create -n umap`

2. Install the following packages:

`conda install pandas yaml tqdm numpy plotly jupyter-dash dash umap-learn matplotlib scikit-learn scikit-image kneed pillow jupyterlab jupyterlab-dash ipywidgets jupyter`

The package versions at this time are:
- python 3.10
- dash 2.6.1
- ipywidgets 8.0.2
- jupyter 1.0.0
- jupyter-dash 0.4.2
- jupyter-lab 3.4.7
- kneed 0.7.0
- matplotlib 3.6.0
- numpy 1.22.4
- pandas 1.5.0
- pillow 9.2.0
- plotly 5.10.0
- scikit-image 0.19.3
- scikit-learn 1.1.2
- tqdm 4.64.1
- umap-learn 0.5.3
- yaml 0.2.5


3. Launch Jupyter Lab in your browser

`jupyterlab`

4. Open the notebook `umap_analysis.ipynb` from the browser
5. Run the notebook
