-- grab unlabeled class/confidence tensor
-- loop through data files and append

probas = torch.load('./labeled.t7b')
print(probas:size())
additional = 10000

-- extract top-K
max,idx = torch.sort(probas[{{},1}])

dataset = {
           data=torch.tensor(additional,3,96,96),
	   labels=torch.tensor(additional}
          }

for i==1,10 do
   local testData = torch.load('./chunks/'..i..'.t7')
   for j=1,1000 do
   
   end
end

