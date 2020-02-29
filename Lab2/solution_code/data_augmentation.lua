require 'torch'
require 'nn'
require 'image'
require 'xlua'
require 'unsup'

require 'data_loader'
require 'provider.lua'

-------------------------------------------------------------------------------
-- A script for data augmentation
--
-------------------------------------------------------------------------------

opt = lapp[[
   --data                 (default train.t7b)            data file name
   --datapath             (default ./stl-10/)            input path
   --saveto               (default ./stl-10_augmented/)  output path
   --crop                 (default 0)                    random crop
   --translate            (default 0)                    random translate
   --rotate               (default 0)                    random rotation
   --scale                (default 0)                    false / integer N (NxN)
   --hflip                (default 0)                    horizontal flip [0, 1]
   --vflip                (default 0)                    vertical flip [0, 1]
]]

print(opt)


--[[ Helper Functions ]]-------------------------------------------------------

-- RGB to HSV
local function rgb2hsv(img)
    return image.rgb2hsv(img)
end

-- HSV to RGB
local function hsv2rgb(img)
    return image.hsv2rgb(img)
end

-- image gradient
local function image_gradient(img, axis)

    local lx = nil
    local ly = nil

    if axis == 'x' then 

        -- filter
        local dx = torch.Tensor({-1, 0, 1}):reshape(1, 3)

        -- convolution
        lx = torch.conv2(img[{1,{},{}}],dx,'V')

        -- padding
        local padx = nn.Sequential()
        padx:add(nn.Padding(2,1,2,lx:mean()))
        padx:add(nn.Padding(2,-1,2,lx:mean()))
        lx = padx:forward(lx)

        return lx

    elseif axis == 'y' then

        -- filter
        local dy = torch.Tensor({-1, 0, 1}):reshape(3, 1)

        -- convolution    
        ly = torch.conv2(img[{1,{},{}}],dy,'V')

        -- padding    
        local pady = nn.Sequential()
        pady:add(nn.Padding(1,1,2,ly:mean()))
        pady:add(nn.Padding(1,-1,2,ly:mean()))
        ly = pady:forward(ly)

        return ly

    else
        -- invalid arg for 'axis'    
        return nil
    end
end

-- mean squared gradient magnitude
local function msgm(img_grad)
    return torch.pow(img_grad,2):sum() / #img_grad
end

-- computer probability proportional to mean squared gradient
local function prob(patch)

    -- compute probability of each patch over the image
    p = torch.Tensor(61,61):zero() -- cut down 96x96 -> 92x92

    -- (x,y) is the location of the left top pixel of a patch 
    for x=1,61 do 
        for y=1,61 do
            p[{x, y}] = p(test[{{x, x + 31},{y, y + 31}}])
        end
    end

    local sum = p:sum()
    p:div(sum)
    assert(p:sum() == 1)
    return p
end

-- concat datasets
local function concat_data(data1, data2)
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


--[[ Transformation ]]---------------------------------------------------------

-- random crop
local function crop(dataset, width, height, multiple)

    local n_samples = dataset:size() * multiple
    local cropped = nil
    local height = height or width
    local x = nil
    local y = nil
    local x_limit = dataset.data:size(3) - width
    local y_limit = dataset.data:size(4) - height
    local idx = 1

    if dataset.labels ~= nil then
        cropped = {data = torch.Tensor(n_samples, 
                                       dataset.data:size(2),
                                       height,
                                       width),
                   labels = torch.Tensor(n_samples),
                   size = function() return n_samples end}
        for i=1,n_samples do
            for j=1,multiple do
                x = torch.random(1, x_limit)
                y = torch.random(1, y_limit)
                cropped.data[{idx,{},{},{}}] = image.crop(dataset.data[{i,{},{},{}}], 
                                                          x,
                                                          y,
                                                          x + width,
                                                          y + height)
                cropped.data[idx] = dataset.labels[i]
                idx = idx + 1
            end
        end
        assert(n_samples == idx - 1)
    else
        cropped = {data = torch.Tensor(n_samples, 
                                       dataset.data:size(2),
                                       height,
                                       width),
                   size = function() return n_samples end}
        for i=1,n_samples do
            for j=1,multiple do
                x = torch.random(1, x_limit)
                y = torch.random(1, y_limit)
                cropped.data[i] = image.crop(dataset.data[i], 
                                             x,
                                             y,
                                             x + width,
                                             y + height)
            end
        end
        assert(n_samples == idx - 1)
    end
    return cropped
end

-- random translation
local function translate(dataset, multiple)

    local n_samples = dataset:size() * multiple
    local translated = nil
    local x = nil
    local y = nil
    local x_limit = torch.floor(dataset.data:size(3) * 0.2)
    local y_limit = torch.floor(dataset.data:size(4) * 0.2)
    local idx = 1

    if dataset.labels ~= nil then
        translated = {data = torch.Tensor(n_samples, 
                                          dataset.data:size(2),
                                          dataset.data:size(3),
                                          dataset.data:size(4)),
                      labels = torch.Tensor(n_samples),
                   size = function() return n_samples end}
        for i=1,n_samples do
            for j=1,multiple do
                x = torch.random(1, x_limit)
                y = torch.random(1, y_limit)
                translated.data[idx] = image.translate(dataset.data[i], x, y)
                translated.data[idx] = dataset.labels[i]
                idx = idx + 1
            end
        end
        assert(n_samples == idx - 1)
    else
        translated = {data = torch.Tensor(n_samples, 
                                          dataset.data:size(2),
                                          dataset.data:size(3),
                                          dataset.data:size(4)),
                      size = function() return n_samples end}
        for i=1,n_samples do
            for j=1,multiple do
                x = torch.random(1, x_limit)
                y = torch.random(1, y_limit)
                translated.data[i] = image.translate(dataset.data[i], x, y)
            end
        end
        assert(n_samples == idx - 1)
    end
    return translated
end

-- random rotation
local function rotate(dataset, angle, multiple)

    local n_samples = dataset:size() * multiple
    local rotated = nil
    local c = nil
    local angle = angle or math.pi/9
    local theta = nil
    local idx = 1

    if dataset.labels ~= nil then
        rotated = {data = torch.Tensor(n_samples, 
                                       dataset.data:size(2),
                                       dataset.data:size(3),
                                       dataset.data:size(4)),
                   labels = torch.Tensor(n_samples),
                   size = function() return n_samples end}
        for i=1,n_samples do
            for j=1,multiple do
                c = torch.rand(1)
                theta = c * range - angle
                rotated.data[idx] = image.rotate(dataset.data[i], theta)
                rotated.data[idx] = dataset.labels[i]
                idx = idx + 1
            end
        end
        assert(n_samples == idx - 1)
    else
        rotated = {data = torch.Tensor(n_samples, 
                                       dataset.data:size(2),
                                       dataset.data:size(3),
                                       dataset.data:size(4)),
                   size = function() return n_samples end}
        for i=1,n_samples do
            for j=1,multiple do
                c = torch.rand(1)
                theta = c * range - angle
                rotated.data[i] = image.rotate(dataset.data[i], theta)
            end
        end
        assert(n_samples == idx - 1)
    end
    return rotated
end

-- scaling
local function scale(dataset, width, height)

    local n_samples = dataset:size()
    local scaled = nil
    local height = height or width

    if dataset.labels ~= nil then
        -- labeled data
        scaled = {data = torch.Tensor(n_samples, 
                                       dataset.data:size(2),
                                       dataset.data:size(3),
                                       dataset.data:size(4)),
                   labels = torch.Tensor(n_samples),
                   size = function() return n_samples end}
        for i=1,n_samples do 
            flipped.data[i] = image.scale(dataset.data[i], width, height)
            flipped.labels[i] = dataset.labels[i]
        end
    else
        -- unlabeled data
        scaled = {data = torch.Tensor(n_samples, 
                                      dataset.data:size(2),
                                      dataset.data:size(3),
                                      dataset.data:size(4)),
                  size = function() return n_samples end}
        for i=1,n_samples do 
            flipped.data[i] = image.scale(dataset.data[i], width, height)
        end
    end
    return scaled
end

-- hirizontal flip
local function horizontal_flip(dataset)

    local n_samples = dataset:size(1)
    local flipped = nil

    if dataset.labels ~= nil then
        -- labeled data
        flipped = {data = torch.Tensor(n_samples, 
                                       dataset.data:size(2),
                                       dataset.data:size(3),
                                       dataset.data:size(4)),
                   labels = torch.Tensor(n_samples),
                   size = function() return n_samples end}
        for i=1,n_samples do 
            flipped.data[i] = image.hflip(dataset.data[i])
            flipped.labels[i] = dataset.labels[i]
        end
    else
        -- unlabeled data
        flipped = {data = torch.Tensor(n_samples, 
                                       dataset.data:size(2),
                                       dataset.data:size(3),
                                       dataset.data:size(4)),
                   size = function() return n_samples end}
        for i=1,n_samples do 
            flipped.data[i] = image.hflip(dataset.data[i])
        end
    end
    return flipped
end

-- vertical flip
local function vertical_flip(dataset)

    local n_samples = dataset:size(1)
    local flipped = nil

    if dataset.labels ~= nil then
        -- labeled data
        flipped = {data = torch.Tensor(n_samples, 
                                       dataset.data:size(2),
                                       dataset.data:size(3),
                                       dataset.data:size(4)),
                   labels = torch.Tensor(n_samples),
                   size = function() return n_samples end}
        for i=1,n_samples do 
            flipped.data[i] = image.vflip(dataset.data[i])
            flipped.labels[i] = dataset.labels[i]
        end
    else
        -- unlabeled data
        flipped = {data = torch.Tensor(n_samples, 
                                       dataset.data:size(2),
                                       dataset.data:size(3),
                                       dataset.data:size(4)),
                   size = function() return n_samples end}
        for i=1,n_samples do 
            flipped.data[i] = image.vflip(dataset.data[i])
        end
    end
    return flipped
end

--[[ Data Augmentation ]]------------------------------------------------------

-- load data

local data = load_train(paths.concat(opt.datapath, opt.data))


-- transform data

local filename = paths.basename(opt.data, 't7b')
local augmentation_flag = 0

if opt.translate ~= 0 then
    
    print '==> translate'

    local multiple = opt.translate

    local translated = translate(data, multiple)
    data = concat_data(data, translated) 
    filename = filename .. '_trs'
    augmentation_flag = 1

end

if opt.rotate ~= 0 then

    print '==> rotate'

    local angle = math.pi/9
    local multiple = opt.rotate

    local rotated = rotate(data, angle, multiple)
    data = concat_data(data, rotated)
    filename = filename .. '_rot'    
    augmentation_flag = 1

end

if opt.hflip == 1 then
    
    print '==> hflip'

    local hflipped = horizontal_flip(data)
    print(hflipped)
    data = concat_data(data, hflipped)
    filename = filename .. '_hfl'
    augmentation_flag = 1

end

if opt.vflip == 1 then

    print '==> vflipped'

    local vflipped = vertical_flip(data)
    print(vflipped)
    data = concat_data(data, vflipped)
    filename = filename .. '_vfl'
    augmentation_flag = 1

end 

if opt.crop ~= 0 then -- NOTE: crop all samples

    print '==> crop'

    local width  = 32
    local height = 32
    local multiple = opt.crop

    local data = crop(data, width, height, multiple)
    filename = filename .. '_crp'
    augmentation_flag = 1

end

if opt.scale ~= 0 then -- NOTE: scale all samples

    print '==> scale'

    local width  = opt.scale
    local height = opt.scale

    data = scale(data, width, height)
    filename = filename .. '_scl'
    augmentation_flag = 1

end

-- save data
print(data)
if augmentation_flag == 1 then
    print('==> save augmented data: ' .. filename)
    torch.save(paths.concat(opt.saveto, filename .. '.t7b'), data)
end

collectgarbage()
