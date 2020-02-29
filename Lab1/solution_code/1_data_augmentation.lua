
require 'torch'   -- torch
require 'image'   -- for color transforms
require 'nn'      -- provides a normalization operator
require 'util'    -- data transformation etc.

function loadData(data_path, size, val_ratio)

    print '==> downloading dataset'
    local train_file = paths.concat(data_path, 'train_32x32.t7')
    local test_file = paths.concat(data_path, 'test_32x32.t7')
    local valRatio = val_ratio or 0.166666666666 
    local trsize = nil
    local tesize = nil
    if size == 'full' then
        print '==> using regular, full training data'
        trsize = 60000
        tesize = 10000
    elseif size == 'small' then
        print '==> using reduced training data, for fast experiments'
        trsize = 6000
        tesize = 1000
    end
    local vlsize = torch.ceil(trsize*(valRatio))

    print(vlsize.." "..trsize)
    trsize = trsize - vlsize 

    print '==> loading dataset'
    local loaded = torch.load(train_file, 'ascii')
    local trainData = {
        data = loaded.data[{{1,trsize},{},{},{}}],
        labels = loaded.labels[{{1,trsize}}],
        size = function() return trsize end
    }
    local validData = {
        data = loaded.data[{{trsize+1,trsize+vlsize},{},{},{}}],
        labels = loaded.labels[{{trsize+1,trsize+vlsize}}],
        size = function() return vlsize end
    }
    local loaded = torch.load(test_file,'ascii')
    local testData = {
        data = loaded.data,
        labels = loaded.labels,
        size = function() return tesize end
    }

    print '==> preprocessing data'
    trainData.data = trainData.data:float()
    validData.data = validData.data:float()
    testData.data = testData.data:float()

    print '==> transform images'
    local inputSize = 32
    local multiple = opt.augm or 1
    local transformed_train = torch.Tensor((#trainData.data)[1]*multiple,1,inputSize,inputSize) 
    local newlabels_train = torch.Tensor((#trainData.data)[1]*multiple)
    local newidx = 1
    for i = 1, (#trainData.data)[1] do
        for j = 1, multiple do        
            transformed_train[newidx] = transform_3(trainData.data[i])
            newlabels_train[newidx] = trainData.labels[i]
            newidx = newidx + 1
        end
    end
    trainData.data = transformed_train
    trainData.labels = newlabels_train

    print '==> preprocessing data: scale globally [0,1]'
    local max = trainData.data:max()
    local min = trainData.data:min()
    trainData.data[{ {},1,{},{} }]:add(-min)
    trainData.data[{ {},1,{},{} }]:mul(1/(max-min))
    validData.data[{ {},1,{},{} }]:add(-min)
    validData.data[{ {},1,{},{} }]:mul(1/(max-min))
    testData.data[{ {},1,{},{} }]:add(-min)
    testData.data[{ {},1,{},{} }]:mul(1/(max-min))

    print '==> verify statistics'
    local trainMax = trainData.data[{ {},1 }]:max() -- :mean()
    local trainMin = trainData.data[{ {},1 }]:min()  -- :std()
    local validMax = validData.data[{ {},1 }]:max()
    local validMin = validData.data[{ {},1 }]:min()
    local testMax = testData.data[{ {},1 }]:max()
    local testMin = testData.data[{ {},1 }]:min()

    print('training data max: ' .. trainMax)
    print('training data min: ' .. trainMin)
    print('valid data max: ' .. validMax)
    print('valid data min: ' .. validMin)
    print('test data max: ' .. testMax)
    print('test data min: ' .. testMin)

    return trainData, validData, testData

end
