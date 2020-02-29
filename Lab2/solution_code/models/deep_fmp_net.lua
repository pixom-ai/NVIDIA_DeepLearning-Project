require 'nn'

-- Deep FMP Net
-- (160nC2 - FMP2^(1/3))*12 - C2 - C1 - output
-- Leaky Relu with alpha=3

local alpha = 1/3
local fmp = nn.Sequential()

-- 1
fmp:add(nn.SpatialConvolutionMM(3,160, 2,2, 2,2)) 
fmp:add(nn.LeakyReLU(alpha,true)
fmp:add(nn.SpatialFractionalMaxPooling(, , outW, outH))
-- 2
fmp:add(nn.SpatialConvolutionMM(160,320, 2,2, 2,2))
fmp:add(nn.LeakyReLU(alpha,true))
fmp:add(nn.SpatialFractionalMaxPooling(kW, kH, outW, outH))
-- 3
fmp:add(nn.SpatialConvolutionMM(320,480, 2,2, 2,2))
fmp:add(nn.LeakyReLU(alpha,true))
fmp:add(nn.SpatialFractionalMaxPooling(kW, kH, outW, outH))
-- 4
fmp:add(nn.SpatialConvolutionMM(480,640, 2,2, 2,2))
fmp:add(nn.LeakyReLU(alpha,true))
fmp:add(nn.SpatialFractionalMaxPooling(kW, kH, outW, outH))
-- 5
fmp:add(nn.SpatialConvolutionMM(640,800, 2,2, 2,2))
fmp:add(nn.LeakyReLU(alpha,true))
fmp:add(nn.SpatialFractionalMaxPooling(kW, kH, outW, outH))
-- 6
fmp:add(nn.SpatialConvolutionMM(800,960, 2,2, 2,2))
fmp:add(nn.LeakyReLU(alpha,true))
fmp:add(nn.SpatialFractionalMaxPooling(kW, kH, outW, outH))
-- 7
fmp:add(nn.SpatialConvolutionMM(960,1120, 2,2, 2,2))
fmp:add(nn.LeakyReLU(alpha,true))
fmp:add(nn.SpatialFractionalMaxPooling(kW, kH, outW, outH))
-- 8
fmp:add(nn.SpatialConvolutionMM(1120,1280, 2,2, 2,2))
fmp:add(nn.LeakyReLU(alpha,true))
fmp:add(nn.SpatialFractionalMaxPooling(kW, kH, outW, outH))
-- 9
fmp:add(nn.SpatialConvolutionMM(1280,1440, 2,2, 2,2))
fmp:add(nn.LeakyReLU(alpha,true))
fmp:add(nn.SpatialFractionalMaxPooling(kW, kH, outW, outH))
-- 10
fmp:add(nn.SpatialConvolutionMM(1440,1600, 2,2, 2,2))
fmp:add(nn.LeakyReLU(alpha,true))
fmp:add(nn.SpatialFractionalMaxPooling(kW, kH, outW, outH))
-- 11
fmp:add(nn.SpatialConvolutionMM(1600,1760, 2,2, 2,2))
fmp:add(nn.LeakyReLU(alpha,true))
fmp:add(nn.SpatialFractionalMaxPooling(kW, kH, outW, outH))
-- 12
fmp:add(nn.SpatialConvolutionMM(1760,1920, 2,2, 2,2))
fmp:add(nn.LeakyReLU(alpha,true))
fmp:add(nn.SpatialFractionalMaxPooling(kW, kH, outW, outH))

fmp:add(nn.SpatialConvolutionMM(1920,2080, 2,2, 2,2))
fmp:add(nn.SpatialConvolutionMM(2080,2240, 1,1, 1,1))




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

MSRinit(fmp)


return fmp
