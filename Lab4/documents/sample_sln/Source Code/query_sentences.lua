------------------------------------------------------
-- Deep Learning Teaching Kit - Lab4 Q2 Sampler
------------------------------------------------------

gpu = false
if gpu then
    require 'cunn'
    print("Running on GPU") 
    
else
    require 'nn'
    print("Running on CPU")
end

stringx = require('pl.stringx')
require 'io'
require('nngraph')
require('base')
require('string')
--require('main_lstm_gru')

------------------------------------------------------
-- configs
------------------------------------------------------

local params = {
                batch_size=20, -- minibatch
                seq_length=20, -- unroll length
                layers=2,
                decay=1.2,
                rnn_size=200, -- hidden unit size
                dropout=0.5, 
                init_weight=0.05, -- random weight initialization limits, default 0.1
                lr=1, -- learning rate
                vocab_size=10000, -- limit on the vocabulary size
                max_epoch=6,  -- when to start decaying learning rate, default 4
                max_max_epoch=39, -- final epoch
                max_grad_norm=5, -- clip when gradients exceed this norm value
                patience=39, -- early stopping
                encoder='lstm',
                model_name='mid1',
                wordchoice='multinom'
               }

------------------------------------------------------
-- load model
------------------------------------------------------

print '==> load model'

model = torch.load('lstm_mid_1_best_model.net')

------------------------------------------------------
-- create dict & inv_dict
------------------------------------------------------

print '==> create dict & inv_dict'

local stringx = require('pl.stringx')
local file = require('pl.file')
local ptb_path = "./data/"
local trainfn = ptb_path .. "ptb.train.txt"

local function create_dict(fname)
    local data = file.read(fname)
    local vocab_idx = 0
    local vocab_map = {}
    data = stringx.replace(data, '\n', '<eos>')
    data = stringx.split(data)
    --print(string.format("Loading %s, size of data = %d", fname, #data))
    local x = torch.zeros(#data)
    for i = 1, #data do
        if vocab_map[data[i]] == nil then
            vocab_idx = vocab_idx + 1
            vocab_map[data[i]] = vocab_idx
        end
    end
    return vocab_map
end

local function inverse_dict(dict)
    local inv_dict = {}
    for k, v in pairs(dict) do
        inv_dict[v] = k
    end
    return inv_dict
end

w2id = create_dict(trainfn) -- dictionary
id2w = inverse_dict(w2id) -- inverse dictionary

------------------------------------------------------
-- generate predictions
------------------------------------------------------

print '==> generate predictions'

-- some utils
local function word2batch(w, batch_size)
    return torch.Tensor(batch_size):fill(w)
end

local function top_word(pred) 
    _, pred_idx = pred:max(1)
    return pred_idx[1]
end

local function multinom(pred)
    return torch.multinomial(torch.exp(pred), 1, true)[1]
end

local function reset_state()
    if model ~= nil and model.start_s ~= nil then
        for d = 1, 2 * params.layers do
            model.start_s[d]:zero()
        end
    end
end

local function is_in(dict, key)
    return dict[key] ~= nil
end

-- generate words given inputs
local function sampler(sequence, num_word)
    reset_state()
    g_disable_dropout(model.rnns)
    local batch_size = 20 -- TODO 
    -- temp
    if not sequence then sequence = {'i', 'like', 'to', 'eat'} end
    if not num_word then num_word = 3 end

    local len = #sequence + num_word
    local next_word_idx = nil
    local results = {}
    g_replace_table(model.s[0], model.start_s)
    for i = 1, len do
        local x = nil
        if i <= #sequence then
            local word_idx = nil
            if is_in(w2id, sequence[i]) then 
                word_idx = w2id[sequence[i]]
            else
                word_idx = w2id['<unk>']
            end
            x = word2batch(word_idx, batch_size)
        else 
            x = word2batch(next_word_idx, batch_size)
        end
        local y = x
        _, model.s[1], pred = unpack(model.rnns[1]:forward({x, y, model.s[0]}))
        if params.wordchoice == 'top' then
            next_word_idx = top_word(pred[1])
        elseif params.wordchoice == 'multinom' then
            next_word_idx = multinom(pred[1])
        else
            next_word_idx = top_word(pred[1])
        end
        if i <= #sequence then 
            results[i] = id2w[x[1]]
        else
            results[i] = id2w[next_word_idx]
        end
        g_replace_table(model.s[0], model.s[1])
    end
    g_enable_dropout(model.rnns)
    return results
end

------------------------------------------------------
-- get user inputs
------------------------------------------------------

local function readline()
  local line = io.read("*line")
  local num = nil
  local string = {}
  if line == nil then error({code="EOF"}) end
  line = stringx.split(line)
  if tonumber(line[1]) == nil then error({code="init"}) end
  num = line[1]
  for i = 2,#line do
    string[i-1] = line[i]:lower()
  end
  return {tonumber(num), string}
end

------------------------------------------------------
-- run
------------------------------------------------------

while true do
  print("Sampler: len word1 word2 etc")
  local ok, line = pcall(readline)
  if not ok then
    if line.code == "EOF" then
      break -- end loop
    elseif line.code == "init" then
      print("Start with a number")
    else
      print(line)
      print("Failed, try again")
    end
  else
    print("sentence: ")
    num_word = line[1]
    sequence = line[2]
    results = sampler(sequence, num_word)
    for i = 1, #results do
        io.write(results[i]..' ')
    end
    io.write('\n')
    print(" ")
  end
end
