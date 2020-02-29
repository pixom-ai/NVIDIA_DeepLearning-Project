
require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods
require 'string'
require 'nn'
--require 'cunn'

print '==== Define slice function'
function table.slice(tbl, first, last, step)
  print(first)
  print(last)
  local sliced = torch.Tensor((last-first)+1,3,96,96)
  print(tbl[1]:size())
  for i = first or 1, last or #tbl, step or 1 do
    print(i)
    sliced[i] = tbl[i]
    tbl[i] = nil
    print('done')
  end
  collectgarbage()
  collectgarbage()
  return sliced
end


print '==== Define parseData function'
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
print '==== Define test procedure'
-- test function
local function testModel(test_dataset)
    
    -- local vars
    local time = sys.clock()
    local results = nil

    model:evaluate()
    
    print('==>'.." testing")
    local bs = 10
    local t = 1
    local softmax_out = nil 
    for i=1,test_dataset:size(1),bs do
        local inputs = test_dataset:narrow(1,i,bs)
        print(inputs:size())
        local outputs = model:forward(inputs)
        softmax_out = nn.SoftMax():cuda():forward(outputs):float()
        --softmax_out = nn.SoftMax():float():forward(outputs):float()
        prob, idx = torch.max(softmax_out, 2)
        -- write the prediction in the tensor
        for j=1,bs do 
            probas[{iter,1}] = prob[j][1]
            probas[{iter,2}] = idx[j][1]
            t = t + 1
        end
        
    end  
    
    -- timing
    time = sys.clock() - time
    time = time / test_dataset:size()
    print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

end


-- load model -----------------------------------------------------------------
print '==> load model'

model = torch.load('./YOUR_MODEL.net')


-- load dataset ---------------------------------------------------------------
print '==> load unlabeled'
raw_data = torch.load('./stl-10/extra.t7b').data[1]
chunk_size = 2000
N_chunks = 100000/chunk_size

-- container to hold psuedo-labels
probas = torch.Tensor(100000,2)

iter = 1
for j=1,N_chunks do
   print('run '..j..' of '..N_chunks)
   local chunk_input = table.slice(raw_data,((j-1)*chunk_size)+1,(j)*chunk_size,1)
   print(type(chunk_input))
   print(type(chunk_input[1]))
   print(torch.Tensor(chunk_input):size())
   --local testData.data = chunk:cuda()
   local testData = chunk_input:cuda()
   print('scoring chunk'..j)
   test_acc = testModel(testData)
end

print('saving scores')
torch.save('./scored_extra.t7',probas)
print '==> Done!'


