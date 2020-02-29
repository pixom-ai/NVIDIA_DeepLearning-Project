Instructions on setting up your environment can be found in the main Readme.

## Step 1: Data augmentation
```
th data_augmentatiuon.lua

Options:
    --data                 (default train.t7b)            data file name
    --datapath             (default ./stl-10/)            input path
    --saveto               (default ./stl-10_augmented/)  output path
    --crop                 (default 0)                    random crop
    --translate            (default 0)                    random translate
    --rotate               (default 0)                    random rotation
    --scale                (default 0)                    false / integer N (NxN)
    --hflip                (default 0)                    horizontal flip [0, 1]
    --vflip                (default 0)                    vertical flip [0, 1]
```

## Step 2: k-means centroids for initialization
Please see Coates et al. (2011), An Analysis of Single-Layer Networks in Unsupervised Feature Learning, for more details.

```
th kmeans_train.lua

Options:
    -savepath             (default ./centroids/)         path to save output
    -modelname            (default nonamemodel)          model name to save centroids
    -surrogate            (default 10)                   number of base images
    -n_patch              (default 10)                   number of patches to select from each image
    -patchsize            (default 10)                   size of patch
    -grad                 (default 0)                    use gradient proportional sampling
    -clusters             (default 64)                   number of clusters for k-means
    -k_iter               (default 3000)                 number of iterations in k-means
```

## Step 3: Training an image classifier with Supervised Learning
```
th train.lua

Options:
    -s,--save                  (default "logs/log")      subdirectory to save logs
    --temp_model               (default "logs/temp")     subdirectory to save temp model
    --best_model               (default "logs/best")     subdirectory to save best model
    -b,--batchSize             (default 64)              batch size
    -r,--learningRate          (default 1)               learning rate
    --learningRateDecay        (default 1e-7)            learning rate decay
    --weightDecay              (default 0.0005)          weightDecay
    -m,--momentum              (default 0.9)             momentum
    --epoch_step               (default 25)              epoch step
    --model                    (default vgg_bn_drop)     model name
    --max_epoch                (default 130)             maximum number of iterations
    --backend                  (default nn)              backend
    --data                     (default train.t7b)       dataset
    --name                     (default noname)          name
    --cont                     (default 0)               continue training 
```

## Step 4: Labeling the unlabeled data
In this step, we put labels to the extra data using the classifier trained in Step 3.
```
th put_confidence.lua
th pull_confident_images.lua
```

## Step 5: Training a model with all data
Repeate Step 3 with the original training data + the labeled extra data.
