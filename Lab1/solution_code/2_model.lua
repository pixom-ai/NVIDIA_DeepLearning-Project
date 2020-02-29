
require 'torch'
require 'image'
require 'nn'

function generateModel(model_type, cpu_type)
    print('model type '..model_type)
    print '==> define parameters'
    local model = nil


    local noutputs = 10
    local nfeats = 0
    local width = 0
    local height = 0
    local ninputs = 0

    if opt.augm == 1 then
        nfeats = 1
        width = 32
        height = 32
        ninputs = nfeats*width*height
    else
        nfeats = 1
        width = 20
        height = 20
        ninputs = nfeats*width*height
    end

    local nhiddens = ninputs / 2

    local nstates = {64,64,128}
    local filtsize = 5
    local poolsize = 2
    local normkernel = image.gaussian1D(7)


    print '==> construct model'

    if model_type == 'linear' then


       print('generating '..model_type..' model') 
       model = nn.Sequential()
       model:add(nn.Reshape(ninputs))
       model:add(nn.Linear(ninputs,noutputs))

    elseif model_type == 'mlp' then


       print('generating '..model_type..' model')
       model = nn.Sequential()
       model:add(nn.Reshape(ninputs))
       model:add(nn.Linear(ninputs,nhiddens))
       model:add(nn.Tanh())
       model:add(nn.Linear(nhiddens,noutputs))

    elseif model_type == 'model1' then
       print('generating '..model_type..'model')

       model = nn.Sequential()

       -- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
       model:add(nn.SpatialConvolutionMM(nfeats, nstates[1], filtsize, filtsize))
       model:add(nn.Tanh())
       model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))

       -- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
       model:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsize, filtsize))
       model:add(nn.Tanh())
       model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))

       -- stage 3 : standard 2-layer neural network
       model:add(nn.View(nstates[2]*filtsize*filtsize))
       model:add(nn.Linear(nstates[2]*filtsize*filtsize, nstates[3]))
       model:add(nn.Tanh())
       model:add(nn.Linear(nstates[3], noutputs))

    elseif model_type == 'model2' then
       print('generating '..model_type..'model')

       model = nn.Sequential()

       -- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
       model:add(nn.SpatialConvolutionMM(nfeats, nstates[1], filtsize, filtsize))
       model:add(nn.Tanh())
       model:add(nn.SpatialLPPooling(nstates[1],2,poolsize,poolsize,poolsize,poolsize))
       model:add(nn.SpatialSubtractiveNormalization(nstates[1], normkernel))

       -- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
       model:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsize, filtsize))
       model:add(nn.Tanh())
       model:add(nn.SpatialLPPooling(nstates[2],2,poolsize,poolsize,poolsize,poolsize))
       model:add(nn.SpatialSubtractiveNormalization(nstates[2], normkernel))

       -- stage 3 : standard 2-layer neural network
       model:add(nn.Reshape(nstates[2]*filtsize*filtsize))
       model:add(nn.Linear(nstates[2]*filtsize*filtsize, nstates[3]))
       model:add(nn.Tanh())
       model:add(nn.Linear(nstates[3], noutputs))

    elseif model_type == 'model3' then
       print('generating '..model_type..'model')
       -- model2 + dropout with 0.5

       model = nn.Sequential()

       -- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
       model:add(nn.SpatialConvolutionMM(nfeats, nstates[1], filtsize, filtsize))
       model:add(nn.Tanh())
       model:add(nn.SpatialLPPooling(nstates[1],2,poolsize,poolsize,poolsize,poolsize))
       model:add(nn.SpatialSubtractiveNormalization(nstates[1], normkernel))

       -- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
       model:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsize, filtsize))
       model:add(nn.Tanh())
       model:add(nn.SpatialLPPooling(nstates[2],2,poolsize,poolsize,poolsize,poolsize))
       model:add(nn.SpatialSubtractiveNormalization(nstates[2], normkernel))

       -- stage 3 : standard 2-layer neural network
       model:add(nn.Reshape(nstates[2]*filtsize*filtsize))
       model:add(nn.Dropout(0.5))
       model:add(nn.Linear(nstates[2]*filtsize*filtsize, nstates[3]))
       model:add(nn.Tanh())
       model:add(nn.Linear(nstates[3], noutputs))

    elseif model_type == 'model4' then
       print('generating '..model_type..'model')
       -- conv + relu + max pooling
       -- fast model

       model = nn.Sequential()

       -- stage 1 : filter bank -> squashing -> max pooling
       model:add(nn.SpatialConvolutionMM(nfeats, nstates[1], filtsize, filtsize))
       model:add(nn.ReLU())
       model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))

       -- stage 2 : filter bank -> squashing -> max pooling
       model:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsize, filtsize))
       model:add(nn.ReLU())
       model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))

       -- stage 3 : standard 2-layer neural network
       model:add(nn.Reshape(nstates[2]*filtsize*filtsize))
       model:add(nn.Dropout(0.5))
       model:add(nn.Linear(nstates[2]*filtsize*filtsize, nstates[3]))
       model:add(nn.ReLU())
       model:add(nn.Linear(nstates[3], noutputs))
        
    elseif model_type == 'model5' then
       print('generating '..model_type..'model')
       -- Multi column convNet

       -- feature extructor
       local features = nn.ConcatTable() -- 5 columns

       -- branch 1
       local branch1 = nn.Sequential()
       branch1:add(nn.SpatialConvolution(1,32,5,5)) -- 32 -> 28
       branch1:add(nn.ReLU(true))
       branch1:add(nn.SpatialMaxPooling(2,2,2,2)) -- 28 -> 14
       branch1:add(nn.SpatialConvolution(32,64,6,6)) -- 14 -> 9
       branch1:add(nn.ReLU(true))
       branch1:add(nn.SpatialMaxPooling(3,3,3,3)) -- 9 -> 3

       -- branch 2
       local branch2 = branch1:clone()
       for k,v in ipairs(branch2:findModules('nn.SpatialConvolution')) do
           v:reset()
       end

       -- branch 3
       local branch3 = branch1:clone()
       for k,v in ipairs(branch3:findModules('nn.SpatialConvolution')) do
           v:reset()
       end

       -- branch 4
       local branch4 = branch1:clone()
       for k,v in ipairs(branch4:findModules('nn.SpatialConvolution')) do
           v:reset()
       end

       -- branch 5
       local branch5 = branch1:clone()
       for k,v in ipairs(branch5:findModules('nn.SpatialConvolution')) do
           v:reset()
       end

       -- store 5 branches in ConcatTable
       features:add(branch1)
       features:add(branch2)
       features:add(branch3)
       features:add(branch4)
       features:add(branch5)

       -- dense layers
       local classifier = nn.Sequential()
       classifier:add(nn.View(64*3*3))
       classifier:add(nn.Dropout(0.5))
       classifier:add(nn.Linear(64*3*3, 150))
       classifier:add(nn.ReLU(true))
       classifier:add(nn.Linear(150, noutputs))

       -- aggregate columns and feed the output to dense layer
       model = nn.Sequential():add(features):add(nn.CAddTable()):add(classifier) 

    elseif model_type == 'model6' then
       print('generating '..model_type..'model')
       -- model2 + dropout with 0.5

       model = nn.Sequential()

       -- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
       model:add(nn.SpatialConvolutionMM(nfeats, nstates[1], filtsize, filtsize))
       model:add(nn.ReLU(true))
       model:add(nn.SpatialLPPooling(nstates[1],2,poolsize,poolsize,poolsize,poolsize))
       model:add(nn.SpatialSubtractiveNormalization(nstates[1], normkernel))

       -- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
       model:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsize, filtsize))
       model:add(nn.ReLU(true))
       model:add(nn.SpatialLPPooling(nstates[2],2,poolsize,poolsize,poolsize,poolsize))
       model:add(nn.SpatialSubtractiveNormalization(nstates[2], normkernel))

       -- stage 3 : standard 2-layer neural network
       model:add(nn.Reshape(nstates[2]*filtsize*filtsize))
       model:add(nn.Dropout(0.5))
       model:add(nn.Linear(nstates[2]*filtsize*filtsize, nstates[3]))
       model:add(nn.ReLU(true))
       model:add(nn.Linear(nstates[3], noutputs))
    
    elseif model_type == 'model7' then
       print('generating '..model_type..'model')
       -- model2 + dropout with 0.5

       model = nn.Sequential()

       -- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
       model:add(nn.SpatialConvolutionMM(nfeats, nstates[1], filtsize, filtsize))
       model:add(nn.ReLU(true))
       model:add(nn.SpatialLPPooling(nstates[1],2,poolsize,poolsize,poolsize,poolsize))
       model:add(nn.SpatialSubtractiveNormalization(nstates[1], normkernel))

       -- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
       model:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsize, filtsize))
       model:add(nn.ReLU(true))
       model:add(nn.SpatialLPPooling(nstates[2],2,poolsize,poolsize,poolsize,poolsize))
       model:add(nn.SpatialSubtractiveNormalization(nstates[2], normkernel))

       -- stage 3 : standard 2-layer neural network
       model:add(nn.Reshape(nstates[2]*filtsize*filtsize))
       model:add(nn.Dropout(0.5))
       model:add(nn.Linear(nstates[2]*filtsize*filtsize, nstates[3]))
       model:add(nn.ReLU(true))
       model:add(nn.Dropout(0.5))
       model:add(nn.Linear(nstates[3], noutputs))

    elseif model_type == 'model8' then
       print('generating '..model_type..'model')
       -- model2 + dropout with 0.5 + overlapping stride in pooling (L2-pooling)

       model = nn.Sequential()

       -- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
       model:add(nn.SpatialConvolutionMM(1, 64, 5, 5))
       model:add(nn.ReLU(true))
       model:add(nn.SpatialLPPooling(64,2,2,2,1,1))
       model:add(nn.SpatialSubtractiveNormalization(nstates[1], normkernel))

       -- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
       model:add(nn.SpatialConvolutionMM(64, 64, 5, 5))
       model:add(nn.ReLU(true))
       model:add(nn.SpatialLPPooling(64, 2, 5, 5, 3, 3))
       model:add(nn.SpatialSubtractiveNormalization(64, normkernel))

       -- stage 3 : standard 2-layer neural network
       model:add(nn.Reshape(64*7*7))
       model:add(nn.Dropout(0.5))
       model:add(nn.Linear(64*7*7, 128))
       model:add(nn.ReLU(true))
       model:add(nn.Dropout(0.5))
       model:add(nn.Linear(128, noutputs))

    elseif model_type == 'model9' then
       print('generating '..model_type..'model')
       -- model2 + dropout with 0.5 + overlapping stride in pooling (Max-pooling)

       model = nn.Sequential()

       -- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
       model:add(nn.SpatialConvolutionMM(1, 64, 5, 5))
       model:add(nn.ReLU(true))
       model:add(nn.SpatialMaxPooling(2,2,1,1))
       model:add(nn.SpatialSubtractiveNormalization(64, normkernel))

       -- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
       model:add(nn.SpatialConvolutionMM(64, 64, 5, 5))
       model:add(nn.ReLU(true))
       model:add(nn.SpatialMaxPooling(5, 5, 3, 3))
       model:add(nn.SpatialSubtractiveNormalization(64, normkernel))

       -- stage 3 : standard 2-layer neural network
       model:add(nn.Reshape(64*7*7))
       model:add(nn.Dropout(0.5))
       model:add(nn.Linear(64*7*7, 128))
       model:add(nn.ReLU(true))
       model:add(nn.Dropout(0.5))
       model:add(nn.Linear(128, noutputs))

    elseif model_type == 'model10' then
       print('generating '..model_type..'model')
       -- model3 + 128 feature maps in stage 2

       model = nn.Sequential()

       -- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
       model:add(nn.SpatialConvolutionMM(nfeats, nstates[1], filtsize, filtsize))
       model:add(nn.Tanh())
       model:add(nn.SpatialLPPooling(nstates[1],2,poolsize,poolsize,poolsize,poolsize))
       model:add(nn.SpatialSubtractiveNormalization(nstates[1], normkernel))

       -- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
       model:add(nn.SpatialConvolutionMM(nstates[1], nstates[3], filtsize, filtsize))
       model:add(nn.Tanh())
       model:add(nn.SpatialLPPooling(nstates[3],2,poolsize,poolsize,poolsize,poolsize))
       model:add(nn.SpatialSubtractiveNormalization(nstates[3], normkernel))

       -- stage 3 : standard 2-layer neural network
       model:add(nn.Reshape(nstates[3]*filtsize*filtsize))
       model:add(nn.Dropout(0.5))
       model:add(nn.Linear(nstates[3]*filtsize*filtsize, nstates[3]))
       model:add(nn.Tanh())
       model:add(nn.Linear(nstates[3], noutputs))

    elseif model_type == 'model11' then
       print('generating '..model_type..'model')
       -- model3 + 128 feature maps & dropout in stage 2 

       model = nn.Sequential()

       -- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
       model:add(nn.SpatialConvolutionMM(nfeats, nstates[1], filtsize, filtsize))
       model:add(nn.Tanh())
       model:add(nn.SpatialLPPooling(nstates[1],2,poolsize,poolsize,poolsize,poolsize))
       model:add(nn.SpatialSubtractiveNormalization(nstates[1], normkernel))

       -- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
       model:add(nn.SpatialConvolutionMM(nstates[1], nstates[3], filtsize, filtsize))
       model:add(nn.Tanh())
       model:add(nn.SpatialLPPooling(nstates[3],2,poolsize,poolsize,poolsize,poolsize))
       model:add(nn.SpatialSubtractiveNormalization(nstates[3], normkernel))
       model:add(nn.Dropout(0.25))

       -- stage 3 : standard 2-layer neural network
       model:add(nn.Reshape(nstates[3]*filtsize*filtsize))
       model:add(nn.Dropout(0.5))
       model:add(nn.Linear(nstates[3]*filtsize*filtsize, nstates[3]))
       model:add(nn.Tanh())
       model:add(nn.Linear(nstates[3], noutputs))  
        
    elseif model_type == 'model12' then
       print('generating '..model_type..'model')
       -- model3 + 128/256 feature maps in stage 1/2

       model = nn.Sequential()

       -- stage 1 : filter bank -> relu -> max pooling
       model:add(nn.SpatialConvolutionMM(1, 128, 5, 5))   -- 32 -> 28 
       model:add(nn.ReLU(true))
       model:add(nn.SpatialMaxPooling(2,2,2,2))           -- 28 -> 14    

       -- stage 2 : filter bank -> relu -> max pooling 
       model:add(nn.SpatialConvolutionMM(128, 256, 5, 5))  -- 14 -> 10
       model:add(nn.ReLU(true))
       model:add(nn.SpatialMaxPooling(2,2,2,2))  -- 10 -> 5

       -- stage 3 : standard 2-layer neural network
       model:add(nn.Reshape(256*5*5))
       model:add(nn.Dropout(0.5))
       model:add(nn.Linear(256*5*5, 256))
       model:add(nn.ReLU(true))
       model:add(nn.Dropout(0.5))
       model:add(nn.Linear(256, noutputs))    

    elseif model_type == 'model13' then
       print('generating '..model_type..'model')
       -- model2 + 2 dropout layer with 0.5

       model = nn.Sequential()

       -- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
       model:add(nn.SpatialConvolutionMM(nfeats, nstates[1], filtsize, filtsize))
       model:add(nn.Tanh())
       model:add(nn.SpatialLPPooling(nstates[1],2,poolsize,poolsize,poolsize,poolsize))
       model:add(nn.SpatialSubtractiveNormalization(nstates[1], normkernel))

       -- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
       model:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsize, filtsize))
       model:add(nn.Tanh())
       model:add(nn.SpatialLPPooling(nstates[2],2,poolsize,poolsize,poolsize,poolsize))
       model:add(nn.SpatialSubtractiveNormalization(nstates[2], normkernel))

       -- stage 3 : standard 2-layer neural network
       model:add(nn.Reshape(nstates[2]*filtsize*filtsize))
       model:add(nn.Dropout(0.5))
       model:add(nn.Linear(nstates[2]*filtsize*filtsize, nstates[3]))
       model:add(nn.Tanh())
       model:add(nn.Dropout(0.5))
       model:add(nn.Linear(nstates[3], noutputs))

    elseif model_type == 'model14' then
       print('generating '..model_type..'model')
       -- model3 + more nodes

       model = nn.Sequential()

       -- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
       model:add(nn.SpatialConvolutionMM(nfeats, nstates[1], filtsize, filtsize))
       model:add(nn.Tanh())
       model:add(nn.SpatialLPPooling(nstates[1],2,poolsize,poolsize,poolsize,poolsize))
       model:add(nn.SpatialSubtractiveNormalization(nstates[1], normkernel))

       -- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
       model:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsize, filtsize))
       model:add(nn.Tanh())
       model:add(nn.SpatialLPPooling(nstates[2],2,poolsize,poolsize,poolsize,poolsize))
       model:add(nn.SpatialSubtractiveNormalization(nstates[2], normkernel))

       -- stage 3 : standard 2-layer neural network
       model:add(nn.Reshape(nstates[2]*filtsize*filtsize))
       model:add(nn.Dropout(0.5))
       model:add(nn.Linear(nstates[2]*filtsize*filtsize, 256))
       model:add(nn.Tanh())
       -- model:add(nn.Dropout(0.5))
       model:add(nn.Linear(256, noutputs))

    elseif model_type == 'model15' then
       print('generating '..model_type..'model')
       -- model3 + 2 dropout layers in FC layers + more nodes

       model = nn.Sequential()

       -- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
       model:add(nn.SpatialConvolutionMM(nfeats, nstates[1], filtsize, filtsize))
       model:add(nn.Tanh())
       model:add(nn.SpatialLPPooling(nstates[1],2,poolsize,poolsize,poolsize,poolsize))
       model:add(nn.SpatialSubtractiveNormalization(nstates[1], normkernel))

       -- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
       model:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsize, filtsize))
       model:add(nn.Tanh())
       model:add(nn.SpatialLPPooling(nstates[2],2,poolsize,poolsize,poolsize,poolsize))
       model:add(nn.SpatialSubtractiveNormalization(nstates[2], normkernel))

       -- stage 3 : standard 2-layer neural network
       model:add(nn.Reshape(nstates[2]*filtsize*filtsize))
       model:add(nn.Dropout(0.5))
       model:add(nn.Linear(nstates[2]*filtsize*filtsize, 256))
       model:add(nn.Tanh())
       model:add(nn.Dropout(0.5))
       model:add(nn.Linear(256, noutputs))
        
    elseif model_type == 'model16' then
       print('generating '..model_type..'model')
       -- Two column convNet

       -- feature extructor
       local features = nn.Concat(1) -- 5 columns

       -- branch 1
       local branch1 = nn.Sequential()

       -- same as model3
       -- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
       branch1:add(nn.SpatialConvolution(1, 64, 5, 5))    -- 32 -> 28
       branch1:add(nn.Tanh())
       branch1:add(nn.SpatialLPPooling(64, 2, 2, 2, 2, 2)) -- 28 -> 14
       branch1:add(nn.SpatialSubtractiveNormalization(64, normkernel))

       -- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
       branch1:add(nn.SpatialConvolution(64, 64, 5, 5)) -- 14 -> 10
       branch1:add(nn.Tanh())
       branch1:add(nn.SpatialLPPooling(64, 2, 2, 2, 2, 2)) -- 10 -> 5
       branch1:add(nn.SpatialSubtractiveNormalization(64, normkernel))

       -- branch 2
       local branch2 = branch1:clone()
       for k,v in ipairs(branch2:findModules('nn.SpatialConvolution')) do
           v:reset()
       end

       -- store 5 branches in ConcatTable
       features:add(branch1)
       features:add(branch2)  
        
       -- output here is size 64x10x5

       -- fc layers
       local classifier = nn.Sequential()
       classifier:add(nn.View(64*10*5))
       classifier:add(nn.Dropout(0.5))
       classifier:add(nn.Linear(64*10*5, 256))
       classifier:add(nn.Tanh())
       classifier:add(nn.Linear(256, 10))

       -- aggregate columns and feed the output to dense layer
       model = nn.Sequential():add(features):add(classifier) 

    elseif model_type == 'model17' then
       print('generating '..model_type..'model')
       -- 3 conv/maxpooling layers + relu + dropout

       model = nn.Sequential()

       -- stage 1 : filter bank -> squashing -> max pooling
       model:add(nn.SpatialConvolutionMM(1, 64, 3, 3)) -- 32 -> 30
       model:add(nn.ReLU(true))
       model:add(nn.SpatialMaxPooling(3, 3, 1, 1)) -- 30 -> 28   

       -- stage 2 : filter bank -> squashing -> max pooling
       model:add(nn.SpatialConvolutionMM(64, 64, 3, 3)) -- 28 -> 26 
       model:add(nn.ReLU(true))
       model:add(nn.SpatialMaxPooling(2, 2, 2, 2)) -- 26 -> 13

       -- stage 3 : filter bank -> squashing -> max pooling
       model:add(nn.SpatialConvolutionMM(64, 64, 3, 3, 1, 1, 1, 1)) -- 13 -> 12 
       model:add(nn.ReLU(true))
       model:add(nn.SpatialMaxPooling(2, 2, 2, 2)) -- 12 -> 6 
        
       -- stage 4 : standard 2-layer neural network
       model:add(nn.Reshape(64*6*6))
       model:add(nn.Dropout(0.5))
       model:add(nn.Linear(64*6*6, 128))
       model:add(nn.ReLU(true))
       model:add(nn.Linear(128, noutputs))
        
    else

       error('unknown -model')

    end

    ----------------------------------------------------------------------
    print '==> here is the model:'
    print(model)
    
    return model
    
end


