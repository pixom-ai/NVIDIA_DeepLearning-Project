
print '==> processing options'

cmd = torch.CmdLine()
cmd:text()
cmd:option('-savepath','./centroids/','path to save output')
cmd:option('-modelname','nonamemodel','model name to save centroids')
cmd:option('-surrogate',10,'number of base images')
cmd:option('-n_patch',10,'number of patches to select from each image')
cmd:option('-patchsize',10,'size of patch')
cmd:option('-grad',0,'use gradient proportional sampling')
cmd:option('-clusters',64,'number of clusters for k-means')
cmd:option('-k_iter',3000,'number of iterations in k-means')
cmd:text()
opt = cmd:parse(arg or {})


dofile 'data_loader.lua'
dofile 'patches.lua'
dofile 'kmeans.lua'

print('loading surrogate data...')
training_data = load_extra()
local dataset = {data = torch.Tensor(
                 opt.surrogate,
                 3,
                 96,
                 96
                 ),
                 size = function() return opt.surrogates end}
shuffle_index = torch.randperm(100000)
for i = 1,opt.surrogate,1 do
    dataset[{{i},{},{},{}}] = training_data.data[shuffle_index[i]]
end

print ('generating patches...')
if opt.grad=='no' then
    opt.grad = false
else
    opt.grad = true
end
patch_data = patch(dataset.data, opt.patchsize, opt.n_patch, opt.grad)

print('performing k-means')
centroids = kmeans(patch_data, opt.clusters, opt.k_iter)

print('saving centroids')
torch.save(opt.savepath..opt.modelname..'.t7', centroids)


