--[[ HELPER FUNCTIONS ]]-----------------------------------
-- Count the number of samples for each class.
--
-- Args:
--     arr: Userdata, 1-d array, expect labels        
-- Returns:
--     freq: Table, dictionary like, class:frequency
--
-----------------------------------------------------------
function count(arr) 
    local freq = {}
    
    for i = 1, (#arr)[1] do 
        if freq[arr[i]] ~= nil then
           freq[arr[i]] = freq[arr[i]] + 1         
        else
           freq[arr[i]] = 1
        end
    end 
    return freq    
end

--[[ DATA AUGMENTATION ]]----------------------------------
-- Preprocess single image with the specified instructions.
-- Scale to 28x28 -> crop to 24x24 -> rotate ± π/4 -> scale
-- to 20x20.
--
-- Args:
--     x_: Userdata, single image data (32x32)     
-- Returns:
--     x: Userdata, transformed image data (20x20)
--
-----------------------------------------------------------
function transform_1(x_)
    -- assume input is 32x32
    local x = x_
    x = image.scale(x,28,28)
    local x_loc = torch.randperm(4)[1]
    local y_loc = torch.randperm(4)[1]
    x = image.crop(x, x_loc, y_loc, x_loc+24, y_loc+24)
    local range = 2 * math.pi/4
    local c = torch.rand(1)
    local theta = c * range - math.pi/4
    x = image.rotate(x,theta[1])
    x = image.scale(x,20,20)
    local pad = nn.Sequential()
    pad:add(nn.Padding(2,-6))
    pad:add(nn.Padding(3,-6))
    pad:add(nn.Padding(2,6))
    pad:add(nn.Padding(3,6))
    return pad:forward(x)
end

-----------------------------------------------------------
-- Preprocess single image with the specified instructions.
-- Scale to 28x28 -> crop to 24x24 -> rotate ± π/4
--
-- Args:
--     x_: Userdata, single image data        
-- Returns:
--     x: Userdata, transformed image data
--
-----------------------------------------------------------
function transform_2(x_)
    -- assume input is 32x32
    local x = x_
    x = image.scale(x,28,28)
    local x_loc = torch.randperm(4)[1]
    local y_loc = torch.randperm(4)[1]
    x = image.crop(x, x_loc, y_loc, x_loc+24, y_loc+24)
    local range = 2 * math.pi/4
    local c = torch.rand(1)
    local theta = c * range - math.pi/4
    x = image.rotate(x,theta[1])
    local pad = nn.Sequential()
    pad:add(nn.Padding(2,-4))
    pad:add(nn.Padding(3,-4))
    pad:add(nn.Padding(2,4))
    pad:add(nn.Padding(3,4))
    return pad:forward(x)
end

-----------------------------------------------------------
-- Preprocess single image with the specified instructions.
-- Scale to 28x28 -> crop to 24x24 -> rotate ± π/4
--
-- Args:
--     x_: Userdata, single image data        
-- Returns:
--     x: Userdata, transformed image data
--
-----------------------------------------------------------
function transform_3(x_)
    -- assume input is 32x32
    local x = x_
    local x_loc = torch.randperm(4)[1]
    local y_loc = torch.randperm(4)[1]
    x = image.crop(x, x_loc, y_loc, x_loc+28, y_loc+28)
    local range = 2 * math.pi/9
    local c = torch.rand(1)
    local theta = c * range - math.pi/9
    x = image.rotate(x,theta[1])
    local pad = nn.Sequential()
    pad:add(nn.Padding(2,-2))
    pad:add(nn.Padding(3,-2))
    pad:add(nn.Padding(2,2))
    pad:add(nn.Padding(3,2))
    return pad:forward(x)
end
