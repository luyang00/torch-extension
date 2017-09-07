require 'torch'
require 'nn'
require 'cunn'
require 'nnex'

torch.setdefaulttensortype('torch.FloatTensor')

local net = nn.HadamardProduct():cuda()
for i = 1, 10 do
	local input1 = torch.rand(64, 3, 128, 128):cuda()
	local input2 = torch.rand(64, 3, 128, 128):cuda()

	local output = net:forward({ input1, input2 })
	local expected = torch.cmul(input1, input2)

	print(torch.sum(output - expected), '<-- should be 0.0')

	net:backward({ input1, input2 }, output)
end

print('switching to DataParallelTable')

-- the DataParallelTable does not like not having any learnable parameters, this example hence adds a SpatialConvolution that does not have any effect
local dummy = nn.SpatialConvolution(3, 3, 1, 1, 1, 1, 0, 0):noBias()
dummy.weight:zero()
dummy.weight[{ { 1 }, { 1 }, {}, {} }] = 1.0
dummy.weight[{ { 2 }, { 2 }, {}, {} }] = 1.0
dummy.weight[{ { 3 }, { 3 }, {}, {} }] = 1.0

local net = nn.DataParallelTable(1):add(nn.Sequential():add(nn.HadamardProduct()):add(dummy), torch.range(1, cutorch.getDeviceCount()):totable()):cuda()
for i = 1, 10 do
	local input1 = torch.rand(64, 3, 128, 128):cuda()
	local input2 = torch.rand(64, 3, 128, 128):cuda()

	local output = net:forward({ input1, input2 })
	local expected = torch.cmul(input1, input2)

	print(torch.sum(output - expected), '<-- should be 0.0')

	net:backward({ input1, input2 }, output)
end