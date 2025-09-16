/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_BACKWARD_H_INCLUDED
#define CUDA_RASTERIZER_BACKWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include "auxiliary.h"


namespace BACKWARD
{
	void render(

		const dim3 grid, const dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float* bg_color,
		const float2* means2D,
		const float4* conic_opacity,
		const float* colors,
		const float* depths,
		const float* final_Ts,
		const uint32_t* n_contrib,

		//输入
		const float* dL_dpixels,
		const float* dL_dpixels_2,

		//输出
		float* dL_dcolors,
		float* dL_dcolors_2,
		
		float2* dL_dmean2Ds,
		float3* dL_dmean2Ds_2,

		float3* dL_dconic2Ds,

		float3x3* dL_dconic2Ds_2,
		float2x3* dL_dmean2D_dconic2Ds,
		float3x2* dL_dconic2D_dmean2Ds

		// float* dL_dopacity
		
		);

	void pos_solve(
		
		int points_num,
		int deg, int max_coeffs, const float3* means, glm::vec3* campos, const float* shs, const bool* clamped,

		
		const int* radii,
		const float* cov3Ds,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		const float* view_matrix,
		const float* proj_matrix,

		float* dL_dcolors,
		float* dL_dcolors_2,
		
		float2* dL_dmean2Ds,
		float3* dL_dmean2Ds_2,

		float3* dL_dconic2Ds,

		float3x3* dL_dconic2Ds_2,
		float2x3* dL_dmean2D_dconic2Ds,
		float3x2* dL_dconic2D_dmean2Ds,

		
		//output
		float3* __restrict__ dL_dmean3Ds,
		float2x3* __restrict__ dL_dmean3Ds_2
		
		);
}

#endif
