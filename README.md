# On the Usefulness of Deep Ensemble Diversity for Out-of-Distribution Detection

This repository contains code for our paper 

## Requirements
The main requirements for this repository are:
```
python
pytorch (+torchvision)
numpy
pandas
matplotlib
seaborn
scikit-learn
tqdm
opencv
pillow
```


## Datasets and Setup
The datasets used in our work can be obtained by following the links and/or instructions below.
- [ImageNet-200, Near-ImageNet-200 and Caltech-45](https://github.com/daintlab/unknown-detection-benchmarks): download the [ImageNet-based benchmark](https://docs.google.com/uc?export=download&id=1gapHov_B-DZ9bKOffg2DFx7lLPOe1T7l)
- [ImageNet](https://www.image-net.org/)
- [OpenImage-O](https://github.com/haoqiwang/vim): download the [test set](https://github.com/cvdfoundation/open-images-dataset) and place the datalist file `utils/openimages_datalist.txt` the level above the directory containing the images.
- [iNaturalist](https://github.com/deeplearning-wisc/large_scale_ood)
- [Textures](https://www.robots.ox.ac.uk/~vgg/data/dtd/): download the dataset and place `utils/textures_datalist.txt` the level above the directory containing the images.
- [Colonoscopy](http://www.depeca.uah.es/colonoscopy_dataset/): run `python utils/extract_frames.py <path/to/data_dir>` to download the videos and extract their frames to the specified directory.
- [Colorectal](https://zenodo.org/record/53169#.Yr21hXbMJ3j)
- Noise: run `python utils/generate_gaussian_noise.py <path/to/data_dir>` to generate the data and save it to the specified directory.
- [ImageNet-O](https://github.com/hendrycks/natural-adv-examples)

After obtaining the data edit `experiment_configs/change_paths.py` such that the dictionary `data_paths` is updated with the correct paths to all the datasets and `RESULTS` points to the directory where you want results (plots and `.csv` files) to be saved. Then run the script to update the configuration `.json` files.
```bash
cd experiment_configs
python change_paths.py
cd ..
```
## Training
To train and test 5 models from scratch run:
```bash
cd models
mkdir saved_models
cd ..
cd experiment_scripts
chmod +x *
./resnet50_imagenet200.sh
cd ..

``` 
You can specify which GPU to use within the files. The training script allows multigpu, but this must be specified in the `experiment_config/resnet50_imagenet200.json` configuration file rather than a command line argument. For example, you could edit the field `"gpu_id": [0,1,2]` to use 3 GPUs. The field `"data_parallel"` also needs to be set to `true`.
## Testing
We also provide the [weights](https://drive.google.com/uc?export=download&id=1nXE-nXDvhZZrQcwGnfE2xQhN2kIQ2ZQy
) for our trained models. Once you have `models.zip`, place it in `models/` and `unzip` it.
```bash
cd models
mkdir saved_models
unzip models.zip -d saved_models
cd ..
```
To test the trained models run 
```bash
cd experiment_scripts
chmod +x *
./resnet50_imagenet200.sh -t 1
cd ..
```
The python script `test.py` runs inference and saves logits/features, so it only needs to be run once. If you want to re-run evaluation then you can run `resnet50_imagenet200_from_logits.sh` to obtain results `.csv` files. 

To obtain tables and plots from our paper after testing all networks run 
```bash
cd experiment_scripts
./present_results.sh
cd ..
```
 The `--latex` command line argument for `ens_table.py` controls whether the tables are printed in TeX or not. Set it to `1` inside `present_eval.sh` if you want to render the tables like in the paper.
