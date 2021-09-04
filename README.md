This repository is for building deep learning-based nGA diagnostic and prognostic models. GradCAM technique is applied to visualize the saliency map. The performance of the GradCAM output in biomarker identification is evaluated.


## Installation

```bash
pip install -r requirements.txt
```

## Description
There are 5 folders in the root.    
**models**: Deep learning model training.   
**script**: Bash scripts for spell job submission (on model training, cross-validation, etc.)
**datasets**:  Jupyter notebook to split the dataset and load the dataset.   
**analysis**: Model's performance evaluation, GradCAM analysis.  
**pytorch_grad_cam_update**: This is cloned from pytorch_grad_cam package. Small modificiation was added, which remove the automated scaling of the GradCAM output. 

## Usage

### Model Training
You may need to change parameter settings in the bash script.
```bash
bash scripts/run_spell_nGA_3D.sh
```

### Grid Search
You may need to change parameter settings in the bash script.
```bash
bash scripts/run_grid_search_3d.sh
```

### Cross-Validation
You may need to change parameter settings in the bash script.
```bash
bash scripts/run_grid_search_2d_cv.sh
```

### Analysis
```bash
cd analysis
```
You need set up the input settings (including input data, trained model, output path, etc.) in the individual python script or jupyter notebooks.

#### Evaluate Model's Performance
```bash
python test_3d_nga_dianostic_model.py
```

#### Generate GradCAM output
```bash
python generate_gradcam_output.py
```