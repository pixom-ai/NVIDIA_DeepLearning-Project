require 'nn'

-- DeepCNet
-- l = 5, k = 320
-- Imagesize dwon from 96 to 1
-- Leaky Relu with alpha=3

local cnet = nn.Sequential()
local alpha = 1/3

cnet:add(nn.SpatialConvolutionMM(3,320, 2,2, 1,1))         -- 96 -> 95 
cnet:add(nn.LeakyReLU(alpha,true))
cnet:add(nn.SpatialMaxPooling(2,2,2,2):ceil())             -- 95 -> 48

cnet:add(nn.SpatialConvolutionMM(320,640, 2,2, 1,1))       -- 48 -> 47 
cnet:add(nn.LeakyReLU(alpha,true))
cnet:add(nn.SpatialMaxPooling(2,2,2,2):ceil())             -- 47 -> 24

cnet:add(nn.SpatialConvolutionMM(640,960, 2,2, 1,1))       -- 24 -> 23
cnet:add(nn.LeakyReLU(alpha,true))
cnet:add(nn.SpatialMaxPooling(2,2,2,2):ceil())             -- 23 -> 12

cnet:add(nn.SpatialConvolutionMM(960,1280, 2,2, 1,1))      -- 12 -> 11
cnet:add(nn.LeakyReLU(alpha,true))
cnet:add(nn.SpatialMaxPooling(2,2,2,2):ceil())             -- 11 -> 6

cnet:add(nn.SpatialConvolutionMM(1280,1600, 2,2, 1,1))     --  6 -> 5
cnet:add(nn.LeakyReLU(alpha,true))
cnet:add(nn.SpatialMaxPooling(2,2,2,2):floor())            --  5 -> 2

cnet:add(nn.SpatialConvolutionMM(1600,1920, 2,2, 1,1))     --  2 -> 1

cnet:add(nn.View(1920))   -- 1920*1*1
cnet:add(nn.Linear(1920,10))  -- 1920*1*1, 10


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

MSRinit(cnet)


return cnet
