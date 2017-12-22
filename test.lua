require 'torch'
require 'nn'
require 'cunn'
require 'nnex'
require 'optim'

torch.setdefaulttensortype('torch.FloatTensor')

----------------------------------------------------------

-- the gradcheck requires learnable parameters, this example hence adds a dummy SpatialConvolution that does not have any effect

dummy = nn.SpatialConvolution(3, 3, 1, 1, 1, 1, 0, 0):noBias()

dummy.weight:zero()
dummy.weight[{ { 1 }, { 1 }, {}, {} }] = 1.0
dummy.weight[{ { 2 }, { 2 }, {}, {} }] = 1.0
dummy.weight[{ { 3 }, { 3 }, {}, {} }] = 1.0

----------------------------------------------------------

local net = nn.Sequential():add(nn.HadamardProduct()):add(dummy):cuda()
local criterion = nn.AbsCriterion():cuda()
local params, gradients = net:getParameters()
for i = 1, 3 do
	local input1 = torch.rand(2, 3, 8, 8):cuda()
	local input2 = torch.rand(2, 3, 8, 8):cuda()
	local truth = torch.rand(2, 3, 8, 8):cuda()

	local output = net:forward({ input1, input2 })
	local expected = torch.cmul(input1, input2)

	print(torch.sum(output - expected), '<-- should be 0.0')
	print(optim.checkgrad(function(params)
		gradients:zero()

		local output = net:forward({ input1, input2 })
		local loss = criterion:forward(output, truth)

		net:backward({ input1, input2 }, criterion:backward(output, truth))

		return loss, gradients
	end, params, 0.001), '<-- should be small')
end

print('switching to DataParallelTable')

local net = nn.DataParallelTable(1):add(nn.Sequential():add(nn.HadamardProduct()):add(dummy), torch.range(1, cutorch.getDeviceCount()):totable()):cuda()
local criterion = nn.AbsCriterion():cuda()
local params, gradients = net:getParameters()
for i = 1, 3 do
	local input1 = torch.rand(2, 3, 8, 8):cuda()
	local input2 = torch.rand(2, 3, 8, 8):cuda()
	local truth = torch.rand(2, 3, 8, 8):cuda()

	local output = net:forward({ input1, input2 })
	local expected = torch.cmul(input1, input2)

	print(torch.sum(output - expected), '<-- should be 0.0')
	print(optim.checkgrad(function(params)
		gradients:zero()

		local output = net:forward({ input1, input2 })
		local loss = criterion:forward(output, truth)

		net:backward({ input1, input2 }, criterion:backward(output, truth))

		return loss, gradients
	end, params, 0.001), '<-- should be small')
end