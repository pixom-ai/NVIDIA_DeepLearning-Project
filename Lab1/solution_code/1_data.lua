
require 'torch'   -- torch
require 'image'   -- for color transforms
require 'nn'      -- provides a normalization operator

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

    print '==> preprocessing data: normalize globally'
    local mean = trainData.data[{ {},1,{},{} }]:mean()
    local std = trainData.data[{ {},1,{},{} }]:std()
    trainData.data[{ {},1,{},{} }]:add(-mean)
    trainData.data[{ {},1,{},{} }]:div(std)
    validData.data[{ {},1,{},{} }]:add(-mean)
    validData.data[{ {},1,{},{} }]:div(std)
    testData.data[{ {},1,{},{} }]:add(-mean)
    testData.data[{ {},1,{},{} }]:div(std)

    print '==> preprocessing data: normalize locally'
    local neighborhood = image.gaussian1D(7)
    local normalization = nn.SpatialContrastiveNormalization(1, neighborhood, 1):float()
    for i = 1,trainData:size() do
       trainData.data[{ i,{1},{},{} }] = normalization:forward(trainData.data[{ i,{1},{},{} }])
    end
    for i = 1,validData:size() do
       validData.data[{ i,{1},{},{} }] = normalization:forward(validData.data[{ i,{1},{},{} }])
    end
    for i = 1,testData:size() do
       testData.data[{ i,{1},{},{} }] = normalization:forward(testData.data[{ i,{1},{},{} }])
    end

    print '==> verify statistics'
    local trainMean = trainData.data[{ {},1 }]:mean()
    local trainStd = trainData.data[{ {},1 }]:std()
    local validMean = validData.data[{ {},1 }]:mean()
    local validStd = validData.data[{ {},1 }]:std()
    local testMean = testData.data[{ {},1 }]:mean()
    local testStd = testData.data[{ {},1 }]:std()

    print('training data mean: ' .. trainMean)
    print('training data standard deviation: ' .. trainStd)
    print('valid data mean: ' .. validMean)
    print('valid data standard deviation: ' .. validStd)
    print('test data mean: ' .. testMean)
    print('test data standard deviation: ' .. testStd)

    return trainData, validData, testData

end
