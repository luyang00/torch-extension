local HadamardProduct, parent = torch.class('nn.HadamardProduct', 'nn.Module')

function HadamardProduct:__init()
	parent.__init(self)

	if true then
		self.output = torch.Tensor()
		self.gradInput = {}
		self.gradInput[1] = torch.Tensor()
		self.gradInput[2] = torch.Tensor()
	end
end

function HadamardProduct:updateOutput(input)
	assert(input[1]:isContiguous() == true)
	assert(input[2]:isContiguous() == true)

	self.output:resizeAs(input[1]):zero()

	if torch.typename(input[1]):find('torch.Cuda') ~= nil then
		input[1].nn.HadamardProduct_cuda_forward(self, input[1], input[2], self.output)

	elseif torch.typename(input[1]):find('torch.Cuda') == nil then
		assert(false) -- CPU VERSION NOT IMPLEMENTED

	end

	return self.output
end

function HadamardProduct:updateGradInput(input, gradOutput)
	assert(input[1]:isContiguous() == true)
	assert(input[2]:isContiguous() == true)
	assert(gradOutput:isContiguous() == true)

	self.gradInput[1]:resizeAs(input[1]):zero()
	self.gradInput[2]:resizeAs(input[2]):zero()

	if torch.typename(input[1]):find('torch.Cuda') ~= nil then
		input[1].nn.HadamardProduct_cuda_backward(self, input[1], input[2], gradOutput, self.gradInput[1], self.gradInput[2])

	elseif torch.typename(input[1]):find('torch.Cuda') == nil then
		assert(false) -- CPU VERSION NOT IMPLEMENTED

	end

	return self.gradInput
end

function HadamardProduct:clearState()
	self.output:set()
	self.gradInput[1]:set()
	self.gradInput[2]:set()

	return self
end