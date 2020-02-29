function trainModel(cpu_type, optimMethod, optimState, optim_options)
    local time = sys.clock()
    local classes = {'1','2','3','4','5','6','7','8','9','0'}
    confusion = optim.ConfusionMatrix(classes)
    parameters, gradParameters = model:getParameters()
    model:training()
    local trsize = data.trainData.data:size(1)
    local shuffle = torch.randperm(trsize)

    print(trsize)
    print('==> doing epoch on training data:')
    print("==> online epoch # " .. epoch .. ' [batchSize = ' .. optim_options.batchSize .. ']')
    for t = 1,trsize,optim_options.batchSize do
        xlua.progress(t, trsize)
        local inputs = {}
        local targets = {}
        for i = t,math.min(t+optim_options.batchSize-1,trsize) do
            local input = data.trainData.data[shuffle[i]]
            local target = data.trainData.labels[shuffle[i]]
            if cpu_type == 'double' then input = input:double()
            elseif cpu_type == 'cuda' then input = input:cuda() end
            table.insert(inputs, input)
            table.insert(targets, target)
        end
        local feval = function(x)
            if x ~= parameters then
                parameters:copy(x)
            end
            gradParameters:zero()
            local f = 0
            for i = 1,#inputs do
                local output = model:forward(inputs[i])
                local err = criterion:forward(output, targets[i])
                f = f + err
                local df_do = criterion:backward(output, targets[i])
                model:backward(inputs[i], df_do)
                output:resize(10)
                confusion:add(output, targets[i])
            end
            gradParameters:div(#inputs)
            f = f/#inputs
            return f,gradParameters
        end
        if optimMethod == optim.asgd then
            _,_,average = optimMethod(feval, parameters, optimState)
        else
            optimMethod(feval, parameters, optimState)
        end
    end
    time = sys.clock() - time
    time = time / data.trainData:size()
    print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')
    print(confusion)
    trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
    confusion:zero()
end
