# Lab 2

Instructions on setting up your environment can be found in the main Readme.

This set of code is modified from Sergey Zagoruyko's cifar.torch tutorial: [cifar.torch](https://github.com/szagoruyko/cifar.torch/blob/master/README.md).

Data processing:

```bash
th -i provider.lua
```

```lua
provider = Provider()
provider:normalize()
torch.save('provider.t7',provider)
```

Training:
```bash
th train.lua --model vgg_bn_drop -s logs/vgg
```

Note: cross-validation is not required for this assignment.
But please keep in mind that if by any chance you tend to make formal comparison with published results in the future, reading carefully the standard testing protocol is where you should start with.


#### Acknowledgement                     
We thank Xiang Zhang for providing a preprocessed STL-10 dataset.
