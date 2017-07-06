#include <THC.h>
#include <THCGeneral.h>

#include "HadamardProduct_kernel.h"

int HadamardProduct_cuda_forward(lua_State* state) {
	HadamardProduct_kernel_forward(
		getTorchState(state),
		(THCudaTensor*) luaT_checkudata(state, 2, "torch.CudaTensor"),
		(THCudaTensor*) luaT_checkudata(state, 3, "torch.CudaTensor"),
		(THCudaTensor*) luaT_checkudata(state, 4, "torch.CudaTensor")
	);

	return 1;
}

int HadamardProduct_cuda_backward(lua_State* state) {
	HadamardProduct_kernel_backward(
		getTorchState(state),
		(THCudaTensor*) luaT_checkudata(state, 2, "torch.CudaTensor"),
		(THCudaTensor*) luaT_checkudata(state, 3, "torch.CudaTensor"),
		(THCudaTensor*) luaT_checkudata(state, 4, "torch.CudaTensor"),
		(THCudaTensor*) luaT_checkudata(state, 5, "torch.CudaTensor"),
		(THCudaTensor*) luaT_checkudata(state, 6, "torch.CudaTensor")
	);

	return 1;
}

const struct luaL_Reg HadamardProduct_cuda_register[] = {
	{ "HadamardProduct_cuda_forward", HadamardProduct_cuda_forward },
	{ "HadamardProduct_cuda_backward", HadamardProduct_cuda_backward },
	{ NULL, NULL }
};

void HadamardProduct_cuda_init(lua_State* state) {
	luaT_pushmetatable(state, "torch.CudaTensor");
	luaT_registeratname(state, HadamardProduct_cuda_register, "nn");
	lua_pop(state, 1);
}