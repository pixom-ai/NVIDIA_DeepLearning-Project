require 'nn'

-- DeepCNet / Kaggle CIFAR-10 Competition Winning Architecture (modified)
-- l = 5, k = 320
-- Imagesize dwon from 96 to 1
-- Leaky Relu with alpha=3

local ben = nn.Sequential()
local alpha = 1/3

ben:add(nn.SpatialConvolutionMM(3,320, 2,2, 1,1))         -- 96 -> 95
ben:add(nn.LeakyReLU(alpha,true))
ben:add(nn.SpatialConvolutionMM(320,320, 2,2, 1,1))       -- 95 -> 94
ben:add(nn.LeakyReLU(alpha,true))
ben:add(nn.SpatialMaxPooling(2,2,2,2))                    -- 94 -> 47

ben:add(nn.SpatialConvolutionMM(320,640, 2,2, 1,1))       -- 47 -> 46
ben:add(nn.LeakyReLU(alpha,true))
ben:add(nn.Dropout(0.2))
ben:add(nn.SpatialConvolutionMM(640,640, 2,2, 1,1))       -- 46 -> 45
ben:add(nn.LeakyReLU(alpha,true))
ben:add(nn.Dropout(0.2))
ben:add(nn.SpatialMaxPooling(2,2,2,2):floor())            -- 45 -> 22

ben:add(nn.SpatialConvolutionMM(640,960, 2,2, 1,1))       -- 22 -> 21
ben:add(nn.LeakyReLU(alpha,true))
ben:add(nn.Dropout(0.3))
ben:add(nn.SpatialConvolutionMM(960,960, 2,2, 1,1))       -- 21 -> 20
ben:add(nn.LeakyReLU(alpha,true))
ben:add(nn.Dropout(0.3))
ben:add(nn.SpatialMaxPooling(2,2,2,2))                    -- 20 -> 10

ben:add(nn.SpatialConvolutionMM(960,1280, 2,2, 1,1))      -- 10 -> 9
ben:add(nn.LeakyReLU(alpha,true))
ben:add(nn.Dropout(0.4))
ben:add(nn.SpatialConvolutionMM(1280,1280, 2,2, 1,1))     -- 9 -> 8
ben:add(nn.LeakyReLU(alpha,true))
ben:add(nn.Dropout(0.4))
ben:add(nn.SpatialMaxPooling(2,2,2,2))                    -- 8 -> 4

ben:add(nn.SpatialConvolutionMM(1280,1600, 2,2, 1,1))     -- 4 -> 3
ben:add(nn.LeakyReLU(alpha,true))
ben:add(nn.Dropout(0.5))
ben:add(nn.SpatialConvolutionMM(1600,1600, 2,2, 1,1))     -- 3 -> 2
ben:add(nn.LeakyReLU(alpha,true))
ben:add(nn.Dropout(0.5))
ben:add(nn.SpatialMaxPooling(2,2,2,2))                    -- 2 -> 1

ben:add(nn.View(1600))
ben:add(nn.Linear(1600,10))


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

MSRinit(ben)


return ben
