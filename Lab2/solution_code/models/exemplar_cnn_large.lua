require 'nn'

-- Exemplar CNN Large
-- 92c5-256c5-512c5-1024f

local ex = nn.Sequential()

ex:add(nn.SpatialConvolutionMM(3,92, 5,5, 1,1))     -- 96 -> 92
ex:add(nn.ReLU(true))
ex:add(nn.SpatialMaxPooling(2,2,2,2))               -- 92 -> 46

ex:add(nn.SpatialConvolutionMM(92,256, 5,5, 1,1))   -- 46 -> 42
ex:add(nn.ReLU(true))
ex:add(nn.SpatialMaxPooling(2,2,2,2))               -- 42 -> 21

ex:add(nn.SpatialConvolutionMM(256,512, 5,5, 1,1 )) -- 21 -> 17
ex:add(nn.ReLU(true))
ex:add(nn.SpatialMaxPooling(2,2,2,2):ceil())        -- 17 -> 9

ex:add(nn.View(512*9*9))
ex:add(nn.Dropout(0.5))
ex:add(nn.Linear(512*9*9,1024))
ex:add(nn.ReLU(true))
ex:add(nn.Linear(1024,10))

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

MSRinit(ex)

-- check that we can propagate forward without errors
-- should get 16x10 tensor
--print(#vgg:cuda():forward(torch.CudaTensor(16,3,32,32)))

return ex
