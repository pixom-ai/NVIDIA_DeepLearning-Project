require 'torch'
require 'nn'

function defineLoss(model, loss_type)
    print '==> define loss'
    local criterion = nil
    if loss_type == 'margin' then
       criterion = nn.MultiMarginCriterion()
    elseif loss_type == 'nll' then
       model:add(nn.LogSoftMax())
       criterion = nn.ClassNLLCriterion()
    else
       error('unknown -loss')
    end
    print '==> here is the loss function:'
    print(criterion)
    return criterion
end
