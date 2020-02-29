require 'torch'
require 'nn'
require 'xlua'
require 'optim'
require 'cunn'
require 'image'
dofile './provider.lua'
local c = require 'trepl.colorize'

opt = lapp[[
   -s,--save                  (default "logs/log")      subdirectory to save logs
   --temp_model               (default "logs/temp")     subdirectory to save temp model
   --best_model               (default "logs/best")     subdirectory to save best model
   -b,--batchSize             (default 64)          batch size
   -r,--learningRate          (default 1)        learning rate
   --learningRateDecay        (default 1e-7)      learning rate decay
   --weightDecay              (default 0.0005)      weightDecay
   -m,--momentum              (default 0.9)         momentum
   --epoch_step               (default 25)          epoch step
   --model                    (default vgg_bn_drop)     model name
   --max_epoch                (default 130)           maximum number of iterations
   --backend                  (default nn)            backend
   --data                     (default train.t7b)    dataset
   --name                     (default noname)       name
   --cont                     (default 0)            continue training 
]]

print(opt)

do -- data augmentation module
  local BatchFlip,parent = torch.class('nn.BatchFlip', 'nn.Module')

  function BatchFlip:__init()
    parent.__init(self)
    self.train = true
  end

  function BatchFlip:updateOutput(input)
    if self.train then
      local bs = input:size(1)
      local flip_mask = torch.randperm(bs):le(bs/2)
      for i=1,input:size(1) do
        if flip_mask[i] == 1 then image.hflip(input[i], input[i]) end
      end
    end
    self.output:set(input)
    return self.output
  end
end

print(c.blue '==>' ..' configuring model')


local model = nn.Sequential()

if opt.cont == 0 then
    model:add(nn.BatchFlip():float())
    model:add(nn.Copy('torch.FloatTensor','torch.CudaTensor'):cuda())
    model:add(dofile('models/'..opt.model..'.lua'):cuda())
    model:get(2).updateGradInput = function(input) return end
else 
    print ' = continue training..'
    model:add(nn.BatchFlip():float())
    model:add(nn.Copy('torch.FloatTensor','torch.CudaTensor'):cuda())
    model:add(torch.load('./logs/temp/' ..opt.model):cuda())
    model:get(2).updateGradInput = function(input) return end
end


-- intialize the first layer 
if opt.backend == 'cudnn' then
   require 'cudnn'
   cudnn.convert(model:get(3), cudnn)
end

print(model)
--[[
print(c.blue '==>' ..' init weights')

cent = torch.load('./centroids/test.t7')
print(torch.type(cent))
first_layer = model:get(3):get(1)
first_layer:reset(0)
first_layer.weights = cent--:reshape(cent:size(1),cent:size(2),cent:size(3),cent:size(4))
]]--

print(c.blue '==>' ..' loading data')

--[[
provider = torch.load 'provider.t7'
provider.trainData.data = provider.trainData.data:float()
provider.valData.data = provider.valData.data:float()
]]--
local valfile = nil
 
if opt.data == 'train.t7b' then 
    valfile = 'val.t7b'
elseif opt.data == 'train_4500.t7b' then 
    valfile = 'val_500.t7b'
elseif opt.data == 'train_4500_flip.t7b' then 
    valfile = 'val_500.t7b' 
else 
    print 'Assume data split was 4000:1000'
    valfile = 'val.t7b' 
end

trainData = torch.load('./stl-10_t7/' .. opt.data)
--valData = torch.load('./stl-10_t7/' .. valfile)

--trainData.data = trainData.data:cuda()
--trainData.labels = trainData.labels:cuda()
--valData.data = valData.data:cuda()
--valData.labels = valData.labels:cuda()


print(trainData)
--print(valData)

confusion = optim.ConfusionMatrix(10)


print('Will save at '..opt.save)
paths.mkdir(opt.save)
valLogger = optim.Logger(paths.concat(opt.save, (opt.name .. '_val.log')))
valLogger:setNames{'% mean class accuracy (train set)', '% mean class accuracy (val set)'}
valLogger.showPlot = false

parameters,gradParameters = model:getParameters()


print(c.blue'==>' ..' setting criterion')
criterion = nn.CrossEntropyCriterion():cuda()


print(c.blue'==>' ..' configuring optimizer')
optimState = {
  learningRate = opt.learningRate,
  weightDecay = opt.weightDecay,
  momentum = opt.momentum,
  learningRateDecay = opt.learningRateDecay,
}


function train()
  model:training()
  epoch = epoch or 1

  -- drop learning rate every "epoch_step" epochs
  if epoch % opt.epoch_step == 0 then optimState.learningRate = optimState.learningRate/2 end
  
  print(c.blue '==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')

  local targets = torch.CudaTensor(opt.batchSize)
  local indices = torch.randperm(trainData.data:size(1)):long():split(opt.batchSize)
  -- remove last element so that all the batches have equal size
  indices[#indices] = nil

  local tic = torch.tic()
  for t,v in ipairs(indices) do
    xlua.progress(t, #indices)

    local inputs = trainData.data:index(1,v)
    targets:copy(trainData.labels:index(1,v))

    local feval = function(x)
      if x ~= parameters then parameters:copy(x) end
      gradParameters:zero()
      
      local outputs = model:forward(inputs)
      local f = criterion:forward(outputs, targets)
      local df_do = criterion:backward(outputs, targets)
      model:backward(inputs, df_do)

      confusion:batchAdd(outputs, targets)

      return f,gradParameters
    end
    optim.sgd(feval, parameters, optimState)
  end

  confusion:updateValids()
  print(('Train accuracy: '..c.cyan'%.2f'..' %%\t time: %.2f s'):format(
        confusion.totalValid * 100, torch.toc(tic)))

  train_acc = confusion.totalValid * 100

  epoch = epoch + 1

-- save model every 5 epochs
  if epoch % 5 == 0 then
    local filename = paths.concat(opt.temp_model, opt.name .. '_model.net')
    print('==> saving model to '..filename)
    torch.save(filename, model:get(3))
  end
  
  confusion:zero()
end


function val()
  -- disable flips, dropouts and batch normalization
  model:evaluate()
  print(c.blue '==>'.." valing")
  local bs = 25
  for i=1,valData.data:size(1),bs do
    local outputs = model:forward(valData.data:narrow(1,i,bs))
    confusion:batchAdd(outputs, valData.labels:narrow(1,i,bs))
  end

  confusion:updateValids()
  print('val accuracy:', confusion.totalValid * 100)
  print('bad count:', bad_count)
    
  local val_acc = confusion.totalValid * 100
  
  if valLogger then
    paths.mkdir(opt.save)
    valLogger:add{train_acc, confusion.totalValid * 100}
    valLogger:style{'-','-'}
    valLogger:plot()

    local base64im
    do
      os.execute(('convert -density 200 %s/val.log.eps %s/val.png'):format(opt.save,opt.save))
      os.execute(('openssl base64 -in %s/val.png -out %s/val.base64'):format(opt.save,opt.save))
      local f = io.open(opt.save..'/val.base64')
      if f then base64im = f:read'*all' end
    end

    local file = io.open(opt.save..'/report.html','w')
    file:write(([[
    <!DOCTYPE html>
    <html>
    <body>
    <title>%s - %s</title>
    <img src="data:image/png;base64,%s">
    <h4>optimState:</h4>
    <table>
    ]]):format(opt.save,epoch,base64im))
    for k,v in pairs(optimState) do
      if torch.type(v) == 'number' then
        file:write('<tr><td>'..k..'</td><td>'..v..'</td></tr>\n')
      end
    end
    file:write'</table><pre>\n'
    file:write(tostring(confusion)..'\n')
    file:write(tostring(model)..'\n')
    file:write'</pre></body></html>'
    file:close()
  end

  

-- save model every 5 epochs
  if epoch % 5 == 0 then
    local filename = paths.concat(opt.temp_model, opt.name .. '_model.net')
    print('==> saving model to '..filename)
    torch.save(filename, model:get(3))
  end
    
  if val_acc > best_val_acc then
    best_val_acc = val_acc
    local filename = paths.concat(opt.best_model, opt.name .. '_best_model.net')
    print('==> saving model to '..filename)
    torch.save(filename, model:get(3))
    bad_count = 0 
  else
    bad_count = bad_count + 1
  end
  confusion:zero()
end

best_val_acc = 0
bad_count = 0
for i=1,opt.max_epoch do
  train()
  --val()
end


