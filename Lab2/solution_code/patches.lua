require 'torch'
require 'nn'
require 'image'
require 'xlua'
require 'unsup'

function whiten(data)

    -- Input:
    --      data           _.data only
    -- Output:
    --      whiten_data
    
    local whiten_data = nil
    whiten_data, _, _, _ = unsup.zca_whiten(data.data[i])

    return whiten_data
end


function tensor_normalize()
    return nil
end


function patch(data, patch_size, n_patches, grad)
    
    local grad_patch = grad or false
    local n_samples = data:size(1)
    local total_patches = n_patches*n_samples
    local dataset = {data = torch.Tensor(total_patches, 3, patch_size, patch_size),
                     size = function() return total_patches end}
    local patch_idx = 1
    
    if grad_patch==1 then 
        
        for i=1, data:size(1) do
            local img = data[i]
            local img_grad = image_gradient(img, 'x')

            -- compute probability proportional to image grad
            local mag_dim = 93 - patch_size
            local mag = torch.Tensor(mag_dim,mag_dim):zero()

            for x=1,mag_dim do 
                for y=1,mag_dim do
                    mag[{x, y}] = magm(img_grad[{{x, x + patch_size - 1},{y, y + patch_size - 1}}])
                end
            end
            local sum = mag:sum()
            mag:div(sum)
                        
            -- sampling 
            for j=1,n_patches do
                xlua.progress(patch_idx, total_patches)
                local trial = torch.rand(1)
                local x, y = 1, 1
                repeat
                    repeat
                        if y > mag_dim then 
                            y = 1
                            break 
                        end
                        trial = trial - mag[{x, y}]
                        y = y + 1
                    until(trial[1] < 0 or x > mag_dim)
                    x = x + 1
                until(trial[1] < 0)
                local loc = {x-1, y-1}  
                dataset.data[patch_idx] = crop(img, loc, patch_size, patch_size)
                patch_idx = patch_idx + 1
            end        
        end    
    else
        for i=1, data:size(1) do
            for j=1,n_patches do
                -- sampling 
                xlua.progress(patch_idx, total_patches)
                local x = torch.random(1, 96 - patch_size)
                local y = torch.random(1, 96 - patch_size)
                loc = {x, y}  
                dataset.data[patch_idx] = crop(data[{i,{},{},{}}], loc, patch_size, patch_size)
                patch_idx = patch_idx + 1
            end
        end
    end
    
    return dataset
end



-- image gradient
function image_gradient(img, axis)
    
    local lx = nil
    local ly = nil
    
    if axis == 'x' then 
        
        -- filter
        local dx = torch.Tensor({-1, 0, 1}):reshape(1, 3):float()
        
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
        local dy = torch.Tensor({-1, 0, 1}):reshape(3, 1):float()

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
    
    
-- mean abs gradient magnitude
function magm(img_grad)
    return img_grad:abs():mean()
end

-- crop
function crop(img, offsets, width, height)
    height = height or width
    return image.crop(img, 
                      offsets[1], 
                      offsets[2], 
                      offsets[1] + width, 
                      offsets[2] + height)
end
