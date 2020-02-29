
print '==> processing options'

cmd = torch.CmdLine()
cmd:text()
cmd:option('-savepath','./results')
cmd:option('-datasize','full')
cmd:option('-modelname','NoNameModel')
cmd:option('-datapath','./mnist.t7')
cmd:option('-valratio',0.166666666666)
cmd:option('-modeltype','linear')
cmd:option('-losstype','nll')
cmd:option('-optimization', 'SGD', 'optimization method: SGD | ASGD | CG | LBFGS | adadelta')
cmd:option('-learningRate', 1e-3, 'learning rate at t=0')
cmd:option('-batchSize', 1, 'mini-batch size (1 = pure stochastic)')
cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
cmd:option('-momentum', 0, 'momentum (SGD only)')
cmd:option('-t0', 1, 'start averaging at t0 (ASGD only), in nb of epochs')
cmd:option('-maxIter', 2, 'maximum nb of iterations for CG and LBFGS')
cmd:option('-adadelta_rho', 0.9,'rho parameter for AdaDelta')
cmd:option('-adadelta_eps', 1e-6,'eps parameter for AdaDelta')
cmd:option('-maxEpochs',30,'maximum nb of epochs to run')
cmd:option('-augm', 1 , 'how many times expand data: integer')
cmd:option('-patience', 10 , 'how many epochs we wait with no improvement')
cmd:option('-loadmodel', 'no' , 'use saved model to continue training')
cmd:text()
opt = cmd:parse(arg or {})

-- load functions -------------------------------------------------------------
if opt.augm <= 1 then dofile '1_data.lua' 
else dofile '1_data_augmentation.lua' end
dofile '2_model.lua'
dofile '3_loss.lua'
dofile '4_optimizer.lua'
dofile '5_train.lua'
dofile '6_valid.lua'

-- load dataset ---------------------------------------------------------------
train,valid,test = loadData(opt.datapath,opt.datasize)
print(train)
print(valid)
print(test)


-- load model -----------------------------------------------------------------
model = nil
if opt.loadmodel == 'no' then  
    model = generateModel(opt.modeltype,"double")
else
    print('==> load model from '.. opt.modeltype)
    model = torch.load(paths.concat(opt.modeltype))
end
print('model before loss')
print(model)

-- load criterion -------------------------------------------------------------
criterion = defineLoss(model, "nll")
print(criteria)
print('model after loss')
print(model)

-- define params --------------------------------------------------------------
data = {trainData = train,
        validData = valid,
        testData = test}
opts = {method = opt.optimization,
        maxIter = opt.maxIter,
        learningRate = opt.learningRate,
        weightDecay = opt.weightDecay,
        momentum = opt.momentum,
        batchSize = opt.batchSize,
	adadelta_rho = opt.adadelta_rho,
	adadelra_eps = opt.adadelta_eps}

-- load optimizer -------------------------------------------------------------
optimMethod,optimState = setupOptimizer("double",opts, opt.savepath, opt.modelname)
print(optimState)

-- params for early stopping --------------------------------------------------
local patience = opt.patience
local bad_count = 0
local best_acc = 0
local acc = nil
local test_acc = nil
local continue_loop = true
epoch = 1

print('patience = '..patience)

-- train/validation -----------------------------------------------------------
while continue_loop do
   trainModel("double", optimMethod, optimState, opts)
   acc = validModel(data.validData,"double")
   if acc > best_acc then
       best_acc = acc
       bad_count = 0
       torch.save(opt.savepath..'/models/'..opt.modelname..'.t7', model)
   else
       bad_count = bad_count + 1
   end
   if bad_count == patience or epoch == opt.maxEpochs then
       continue_loop = false
       print("early stop after " .. epoch .. " epochs")
       trainLogger:style{['% mean class accuracy (train set)'] = '-'}
       trainLogger:plot()
       testLogger:style{['% mean class accuracy (valid set)'] = '-'}
       testLogger:plot()
   end
   epoch = epoch + 1
   print("bad_count = " .. bad_count)
end
