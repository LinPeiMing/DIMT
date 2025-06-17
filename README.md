# DIMT
official code for "Degradation Intensity Modulation Transformer for Blind Stereo Image Super-Resolution"

:star: If DIMT is helpful to your images or projects, please help star this repo. Thanks! :hugs:


### 📌 TODO
- [   ] release the pretrained models
- [   ] release the training code
- [   ] release the training and testing data

## ⚙️ Dependencies and Installation
The codes are based on [BasicSR](https://github.com/xinntao/BasicSR).
```
## git clone this repository
git clone https://github.com/LinPeiMing/DIMT.git
cd DIMT

# create an environment with python >= 3.7
conda create -n dimt python=3.7
conda activate dimt
pip install -r requirements.txt
python setup.py develop
```
## 🚀 Quick Inference
#### Step 1: Download the pretrained models
- Download the pretrained models.(TODO)
  
You can put the models into `experiment/pretrained_models`.

#### Step 2: Prepare testing data
You can put the testing images in the `datasets/test_datasets`.

#### Step 3: Modify setting
Modify the test set path and pre-training model path in `options/test/DIMT-GAN.yml`.

#### Step 4: Running testing command
```
bash test.sh
```

## 🌈 Train 

#### Step1: Prepare training data
Modify the training set and validation set path in `options/train/DIMT.yml`,`options/train/DIMT-GAN.yml`,


#### Step2. Begin to train
For DIMT, run this command: 
```
bash run.sh
```
For DIMT-GAN, change the pretrained model path in `options/train/DIMT-GAN.yml` to the pretrained model path obtained before. We apply two-stage training by commenting out the code in run.sh for different training configs.
Then run this command:
```
bash run.sh
```


## 🌈 PS: 
To modify the Real-SHGD model, you can change the degradation pool.

Take the blur kernel as an example:
```
kernel_paths = glob.glob(os.path.join(self.opt['kernel_path'], '*/*_kernel_x{scale}.mat'))
kernel_num = len(kernel_paths)
kernel_path = kernel_paths[np.random.randint(0, kernel_num)]
mat = loadmat(kernel_path)
k = np.array([mat_L['Kernel']]).squeeze()
img_lq = imresize(img_gt, scale_factor=1.0 / scale, kernel=k)
```
Take the noise as an example:
```
from basicsr.data.noise_dataset import noiseDataset
noise_patches = noiseDataset(opt['noise_pool_path'], opt['gt_size']/opt['scale'])
noise_patch = noise_patches[np.random.randint(0, len(noise_patches))]
img_lq = torch.clamp(img_lq + noise_patch, 0, 1)
```
