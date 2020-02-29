------------------------------------------------------------
-- Deep Learning Teaching Kit - Lab4 Q1
------------------------------------------------------------

require 'nn';
require 'nngraph';

------------------------------------------------------------
-- initial params
------------------------------------------------------------

local W_x = torch.Tensor{{1,1,1,1},{1,1,1,1}}
local W_y = torch.Tensor{{1,1,1,1,1},{1,1,1,1,1}}

local b_1 = torch.Tensor{{1},{1}}
local b_2 = torch.Tensor{{1},{1}}

------------------------------------------------------------
-- build nngraph
------------------------------------------------------------

print '==> build nngraph'

-- inputs
local x = nn.Identity()()
local y = nn.Identity()()
local z = nn.Identity()() 

-- create a computational graph
local h_1 = nn.Linear(4,2)({x}) -- Wx + b
local h_2 = nn.Linear(5,2)({y}) -- Wy + b

-- non-linear
local fh_1 = nn.Tanh()({h_1})
local fh_2 = nn.Sigmoid()({h_2})

-- square
local sfh_1 = nn.Square()({fh_1})
local sfh_2 = nn.Square()({fh_2})

-- elem-wise mul
local left = nn.CMulTable()({sfh_1, sfh_2})
local a = nn.CAddTable()({left, z})

--[[
-- init params 
h_1.data.module.weight = W_x
h_1.data.module.bias = b_1
h_2.data.module.weight = W_y
h_2.data.module.bias = b_2
]]--

-- eval
local output = nn.gModule({x, y, z}, {a})

-- draw diagram
--graph.dot(output.fg, 'output','outputBasename')

------------------------------------------------------------
-- inputs & initial params
------------------------------------------------------------
local i = 1
local j = 1
local k = 1
local x_ = torch.Tensor{i,i,i,i}
local y_ = torch.Tensor{j,j,j,j,j}
local z_ = torch.Tensor{k,k}

local grad_out = torch.Tensor{{1},{1}}

print '==> inputs'
print 'x ='
print(x_)
print 'y ='
print(y_)
print 'z ='
print(z_)

------------------------------------------------------------
-- demo
------------------------------------------------------------

print '==> result of forward'
print(output:forward({x_, y_, z_}))

print '==> result of backward'
print '==> w.r.t. x'
print(output:backward({x_, y_, z_}, grad_out)[1])

print '==> w.r.t. y'
print(output:backward({x_, y_, z_}, grad_out)[2])

print '==> w.r.t. z'
print(output:backward({x_, y_, z_}, grad_out)[3])
