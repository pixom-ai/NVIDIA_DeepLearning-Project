dofile 'data_loader.lua'


function setContains(set, key)
    return set[key] ~= nil
end

function tablelength(T)
  local count = 0
  for _ in pairs(T) do count = count + 1 end
  return count
end


--load extra
print('loading extra dataset')
extra = load_extra()

--load psuedo
print('loading psuedo-data')
psuedo = torch.load('./scored_extra.t7')
val,idx = torch.sort(psuedo[{{},{3}}],1)
valf = torch.Tensor(val:size())
idxf = torch.Tensor(idx:size())
--flip: high values on top
for i=1,val:size(1) do
    valf[val:size(1)-i+1] = val[i]
    idxf[idxf:size(1)-i+1] = idx[i]
end

print('generating index sets')
sets = {10000}
classes = {1,2,3,4,5,6,7,8,9,10}
set = {set}
-- counting number for each class
counter = {}
idxf_class = {}
for i,c in ipairs(classes) do
   idxf_class[c] = {}
   for j,s in ipairs(sets) do
      idxf_class[c][s] = {}  
   end
end

-- generate a by-class indexer
for n=1,extra:size(1) do
    id = idxf[n][1]
--    print(id)
    class = psuedo[id][1]
--    print(class)

       for i,s in ipairs(sets) do
           if tablelength(idxf_class[class][s]) < s/10 then
              idxf_class[class][s][id] = 1
           end       
       end
 
end    

-- loop through obs
for i,s in ipairs(sets) do
    print('Setting up dataset of size '..s)
    labels_extra = torch.Tensor(s/10)
    data_extra = torch.Tensor(s/10,3,96,96)
    ticker = 1
    
    
    for n=1,extra:size(1) do
        class = psuedo[n][1]
        if idxf_class[class][s][n] == 1 then
        if class == 1 then
            print('True ')
            labels_extra[ticker] = class
            data_extra[ticker] = extra.data[n]
            ticker = ticker + 1 
        end
        end
    end
    dataset = {data = data_extra,
               labels = labels_extra ,
               size = function() return s/10  end
               } 
    
    print('saving dataset of size')
    print(dataset)
    name = s/10
    torch.save('./extra_data/extra_class1_'..name..'.t7',dataset)
    
    dataset = nil
    collectgarbage()
    collectgarbage()
end
