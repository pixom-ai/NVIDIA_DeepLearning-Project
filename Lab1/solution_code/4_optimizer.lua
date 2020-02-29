
require 'torch'
require 'xlua'
require 'optim'

function setupOptimizer(cpu_type, optim_options, save_path, model_name)
    if cpu_type == 'cuda' then
       model:cuda()
       criterion:cuda()
    end

    print '==> defining some tools'
    local trsize = data.trainData.size()
    local model_name = model_name or 'NoName'
    local savePath = save_path or "."
    trainLogger = optim.Logger(paths.concat(savePath, 'log' , model_name..'train.log'))
    testLogger = optim.Logger(paths.concat(savePath, 'log', model_name..'test.log'))
    local optiMethod = nil
    local optimState = nil

    print '==> configuring optimizer'
    if optim_options.method == 'CG' then
       optimState = {
          maxIter = optim_options.maxIter
       }
       optimMethod = optim.cg
    elseif optim_options.method == 'LBFGS' then
       optimState = {
          learningRate = optim_options.learningRate,
          maxIter = optim_options.maxIter,
          nCorrection = 10
       }
       optimMethod = optim.lbfgs
    elseif optim_options.method == 'SGD' then
       optimState = {
          learningRate = optim_options.learningRate,
          weightDecay = optim_options.weightDecay,
          momentum = optim_options.momentum,
          learningRateDecay = 1e-7
       }
       optimMethod = optim.sgd
    elseif optim_options.method == 'ASGD' then
       optimState = {
          eta0 = optim_options.learningRate,
          t0 = trsize * t0
       }
       optimMethod = optim.asgd
    elseif optim_options.method == 'adadelta' then
       optimState = {
          rho = optim_options.adadelta_rho,
          eps = optim_options.adadelta_eps
       }
       optimMethod = optim.adadelta
    else
       error('unknown optimization method')
    end
    return optimMethod, optimState
end
