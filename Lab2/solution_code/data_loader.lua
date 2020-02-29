require 'torch'
require 'nn'
require 'image'

function load_train()
    local n_samples = 4000
    local n_channels = 3
    local height = 96
    local width = 96
    local raw_data = torch.load('./stl-10/train.t7b')
    
    local dataset = {data = torch.Tensor(),
                  labels = torch.Tensor(),
                  size = function() return n_samples end}
    
    dataset.data, dataset.labels = parseDataLabel(raw_data.data, n_samples, n_channels, height, width)
    dataset.data = dataset.data:float()
    dataset.labels = dataset.labels:float()
   
    collectgarbage()
    
    print '==> train data'
    print(dataset)
   
    return dataset
end

function load_val()
    local n_samples = 1000
    local n_channels = 3
    local height = 96
    local width = 96
    local raw_data = torch.load('./stl-10/val.t7b')
    
    local dataset = {data = torch.Tensor(),
                     labels = torch.Tensor(),
                     size = function() return n_samples end}

    dataset.data, dataset.labels = parseDataLabel(raw_data.data, n_samples, n_channels, height, width)
    dataset.data = dataset.data:float()
    dataset.labels = dataset.labels:float()
   
    collectgarbage()
    
    print '==> val data'
    print(dataset)
   
    return dataset
end

function load_test()
    local n_samples = 8000
    local n_channels = 3
    local height = 96
    local width = 96
    local raw_data = torch.load('./stl-10/test.t7b')
    
    local dataset = {data = torch.Tensor(),
                     labels = torch.Tensor(),
                     size = function() return n_samples end}
    
    dataset.data, dataset.labels = parseDataLabel(raw_data.data, n_samples, n_channels, height, width)
    dataset.data = dataset.data:float()
    dataset.labels = dataset.labels:float()

    collectgarbage()
    
    print '==> test data'
    print(dataset)
   
    return dataset
end

function load_extra(n_samples)
    local n_samples = 100000
    local n_channels = 3
    local height = 96
    local width = 96
    local raw_data = torch.load('./stl-10/extra.t7b')
    
    local dataset = {data = torch.Tensor(),
                     size = function() return n_samples end}
    
    dataset.data, _ = parseDataLabel(raw_data.data, n_samples, n_channels, height, width)
    dataset.data = dataset.data:float()
    
    if n_samples then
        dataset.data = dataset.data[{{1,n_samples},{},{},{}}]
    end
    collectgarbage()
    
    print '==> extra data'
    print(dataset)
   
    return dataset
end

function parseDataLabel(d, numSamples, numChannels, height, width)
   local t = torch.ByteTensor(numSamples, numChannels, height, width)
   local l = torch.ByteTensor(numSamples)
   local idx = 1
   for i = 1, #d do
        local this_d = d[i]
        for j = 1, #this_d do
            t[idx]:copy(this_d[j])
            l[idx] = i
            idx = idx + 1
        end
   end
   assert(idx == numSamples+1)
   return t, l
end

function parseData(d, numSamples, numChannels, height, width)
   local t = torch.ByteTensor(numSamples, numChannels, height, width)
   local idx = 1
   print(d)
   for i = 1, #d do
       local this_d = d[i]
       for j = 1, #this_d do
           t[idx]:copy(this_d[j])
           idx = idx + 1
       end
   end
   print(idx)
   print(numSamples+1)
   assert(idx == numSamples+1)
   return t
end
