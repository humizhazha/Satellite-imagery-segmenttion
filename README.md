# Satellite imagery segmention


## Main software for training neural networks
* Python 3.6
* Keras 1.2.2
* Theano 0.9.0

## Requirements

```bash
h5py==2.6.0
Keras==1.2.2
matplotlib==1.5.3
numba==0.30.1
numpy==1.12.0
pandas==0.18.1
scikit_image==0.12.3
tifffile==0.10.0
tqdm==4.11.2
Theano==0.9.0
tensorflow
opencv-python
shapely
```
1. Install required OS and Python
2. Install packages with pip install -r requirements.txt

## File Structure
### Data Structure
```bash
data / theree_band / *
     / sixteen_band / *
    grid_sizes.csv
    train_wkt_v4.csv
```
### Overall Structure
```bash
Satellite-imagery-segmention 
     / data/ *
     / src / *

```
## Prepare data for training
1. Run python get_3_band_shapes.py
2. Run cache_train.py

## Train model
Each class in our solution has separate neural network, so it requires running of several distinct models one by one (or in parallel if there are enough computing resources)

1. Run python unet_buidings.py
2. Run python unet_structures.py
3. Run python unet_road.py
4. Run python unet_track.py
5. Run python unet_trees.py
6. Run python unet_crops.py

For water predictions we used different method and it can be created by running:

1. Run python fast_water.py
2. Run python slow_water.py

Trained weights and model architectures are saved in cache directory and can be used by prediction scripts (see the next section).