# torch-extension
This is an example of a CUDA extension for Torch which computes the Hadamard product of two tensors.

For the PyTorch version of this example extension, please see: https://github.com/sniklaus/pytorch-extension

To build the extension, run `bash install.bash` and make sure that the `CUDA_HOME` environment variable is set. After succesfully building the extension, run `th test.lua` to test it. Should you receive an error message regarding an invalid device function when making use of the extension, configure the CUDA architecture within `CMakeLists.txt` to something your graphics card supports.

```lua
require 'torch'
require 'nn'
require 'cunn'
require 'nnex'

torch.setdefaulttensortype('torch.FloatTensor')

local net = nn.HadamardProduct():cuda()

local input1 = torch.rand(64, 3, 128, 128):cuda()
local input2 = torch.rand(64, 3, 128, 128):cuda()

local output = net:forward({ input1, input2 })
local expected = torch.cmul(input1, input2)

print(torch.sum(output - expected), '<-- should be 0.0')
```