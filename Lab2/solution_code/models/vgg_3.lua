require 'nn'

local net = nn.Sequential()

net:add(nn.SpatialConvolutionMM(3,64, 3,3, 1,1, 1,1))          -- 96 -> 96 
net:add(nn.SpatialBatchNormalization(64,1e-3))
net:add(nn.ReLU(true))
net:add(nn.SpatialConvolutionMM(64,64, 3,3, 1,1, 1,1))         -- 96 -> 96 
net:add(nn.SpatialBatchNormalization(64,1e-3))
net:add(nn.ReLU(true))
net:add(nn.SpatialMaxPooling(4,4,4,4):ceil())                  -- 96 -> 24

net:add(nn.SpatialConvolutionMM(64,128, 3,3, 1,1, 1,1))        -- 24 -> 24 
net:add(nn.SpatialBatchNormalization(128,1e-3))
net:add(nn.ReLU(true))
net:add(nn.SpatialConvolutionMM(128,128, 3,3, 1,1, 1,1))       -- 24 -> 24 
net:add(nn.SpatialBatchNormalization(128,1e-3))
net:add(nn.ReLU(true))
net:add(nn.SpatialMaxPooling(4,4,4,4):ceil())                  -- 24 -> 6


net:add(nn.SpatialConvolutionMM(128,256, 3,3, 1,1, 1,1))       -- 6 -> 6 
net:add(nn.SpatialBatchNormalization(256,1e-3))
net:add(nn.ReLU(true))
net:add(nn.SpatialConvolutionMM(256,256, 3,3, 1,1, 1,1))       -- 6 -> 6 
net:add(nn.SpatialBatchNormalization(256,1e-3))
net:add(nn.ReLU(true))
net:add(nn.SpatialMaxPooling(3,3,3,3):ceil())                  -- 6 -> 2


net:add(nn.View(1024))   -- 256*2*2
classifier = nn.Sequential()
classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(1024,256))
classifier:add(nn.BatchNormalization(256))
classifier:add(nn.ReLU(true))
classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(256,10))
net:add(classifier)


-- initialization from MSR
local function MSRinit(net)
  local function init(name)
    for k,v in pairs(net:findModules(name)) do
      local n = v.kW*v.kH*v.nOutputPlane
      v.weight:normal(0,math.sqrt(2/n))
      v.bias:zero()
    end
  end
  -- have to do for both backends
  init'nn.SpatialConvolutionMM'
end

MSRinit(net)


return net
