require 'torch'
require 'nn'
require 'image'
require 'xlua'
require 'unsup'

require 'data_loader'
--require 'provider.lua'

-------------------------------------------------------------------------------
-- A script for data augmentation
--
-------------------------------------------------------------------------------

opt = lapp[[
   --data                 (default train.t7b)            data file name
   --datapath             (default ./stl-10/)            input path
   --saveto               (default ./stl-10_t7/)  output path
]]

print(opt)

-- concat datasets
function concat_data(data1, data2)
    if data1.labels ~= nil then
        assert(data2.lablels == nil)
        data1.data = torch.cat(data1.data, data2.data, 1)
        data1.labels = torch.cat(data1.labels, data2.labels, 1)
        local newsize = data1:size() + data2:size()
        data1.size = function() return newsize end 
    else
        assert(data2.lablels ~= nil)
        data1.data = torch.cat(data1.data, data2.data, 1) 
        local newsize = data1:size() + data2:size()
        data1.size = function() return newsize end 
    end
    return data1
end 

--[[ Data Augmentation ]]------------------------------------------------------

-- load data

--local data = load_train(paths.concat(opt.datapath, opt.data))
-- hard code
local train = torch.load('./stl-10_t7/train.t7b')
local val = torch.load('./stl-10_t7/val.t7b')

print '==> concat data'
data = concat_data(train, val)
filename = 'train_val.t7b'

-- save data
print '==> save data'
print(data)

print('==> save augmented data: ' .. filename)
torch.save(paths.concat(opt.saveto, filename .. '.t7b'), data)


collectgarbage()
