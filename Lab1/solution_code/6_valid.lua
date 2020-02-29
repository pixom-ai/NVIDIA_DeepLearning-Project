
require 'torch'
require 'xlua'
require 'optim'
require 'string'

print '==> defining test procedure'

function validModel(valid_dataset, cpu_type)
   local time = sys.clock()
   model:evaluate()
   confusion:zero()

   print('==> testing on validation set:')
   for t = 1,valid_dataset:size() do
      xlua.progress(t, valid_dataset:size())
      local input = valid_dataset.data[t]
      if cpu_type == 'double' then input = input:double() 
      elseif cpu_type == 'cuda' then input = input:cuda()  end
      local target = valid_dataset.labels[t]
      local pred = model:forward(input)
      pred:resize(10)
      confusion:add(pred, target)
   end
   time = sys.clock() - time
   time = time / valid_dataset:size()
   print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')
   print(confusion)
   testLogger:add{['% mean class accuracy (valid set)'] = confusion.totalValid * 100}
   return confusion.totalValid * 100
end
