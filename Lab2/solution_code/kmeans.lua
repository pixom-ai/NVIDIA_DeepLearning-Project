require 'unsup'

function kmeans(patches, K, num_iterations)
    local N_data = patches.data:size(1)
    local channels = patches.data:size(2)
    local patch_h = patches.data:size(3)
    local patch_w = patches.data:size(4)
    local dimension = channels*patch_h*patch_w

    
    -- reshape data for input, should be (n_data, representation_dim)
    local patch_data = torch.Tensor(N_data,dimension)
    for i=1, patches.data:size(1) do
        patch_data[{{i},{}}] = patches.data[{{i},{},{},{}}]:reshape(dimension)
    end
    
    centroids = unsup.kmeans(patch_data, K, num_iterations)
    print(centroids:size())
    centroids = centroids:reshape(K, channels,patch_h, patch_w)
    return centroids
end