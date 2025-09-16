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

#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

__device__ __forceinline__ float sq(float x) { return x * x; }

__device__ inline float3 elementwiseMul(const glm::vec3& a, const glm::vec3& b) {
	return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}



__global__ void pos_solve_sh(){
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	// Compute intermediate values, as it is done during forward
	glm::vec3 pos = means[idx];
	glm::vec3 dir_orig = pos - campos;

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;

	// Use PyTorch rule for clamping: if clamping was applied,
	// gradient becomes 0.
	glm::vec3 dpixel_dRGB;
	dpixel_dRGB.x = clamped[3 * idx + 0] ? 0 : dpixel_dcolors[idx];
	dpixel_dRGB.y = clamped[3 * idx + 1] ? 0 : dpixel_dcolors[idx];
	dpixel_dRGB.z = clamped[3 * idx + 2] ? 0 : dpixel_dcolors[idx];


	x = dir_orig.x, y = dir_orig.y, z = dir_orig.z;
	xx = x * x, yy = y * y, zz = z * z;
	xy = x * y, yz = y * z, xz = x * z;

	float xyz = xy * z;
	float xxxx = xx * xx, yyyy = yy * yy, zzzz = zz * zz;
	float yyzz = yy * zz;


	float length = xx + yy + zz;
	float length1div2 = sqrtf(length);
	float length3div2 = length * length1div2;
	float length2 = length * length;
	float length5div2 = length2 * length1div2;
	float length3 = length2 * length;
	float length7div2 = length3 * length1div2;

	float inv_length3div2 = 1.0 / (length3div2 + 0.0000001f);
	float inv_length2 = 1.0 / (length2 + 0.0000001f);
	float inv_length5div2 = 1.0 / (length5div2 + 0.0000001f);
	float inv_length3 = 1.0 / (length3 + 0.0000001f);
	float inv_length7div2 = 1.0 / (length7div2 + 0.0000001f);


	glm::vec3 dRGBdx(0, 0, 0);
	glm::vec3 dRGBdy(0, 0, 0);
	glm::vec3 dRGBdz(0, 0, 0);


	if (deg > 0)
	{
		dRGBdx = (
			SH_C1 * sh[1] * xy
			-SH_C1 * sh[2] * xz
			-SH_C1 * sh[3] * (yy + zz)
		) * inv_length3div2;
		dRGBdy = (
			-SH_C1 * sh[1] * (xx + zz)
			-SH_C1 * sh[2] * yz
			+SH_C1 * sh[3] * xy
		) * inv_length3div2;
		dRGBdz = (
			SH_C1 * sh[1] * yz
			+SH_C1 * sh[2] * (xx + yy)
			+SH_C1 * sh[3] * xz
		) * inv_length3div2;
		if (deg > 1)
		{
			dRGBdx += (
				SH_C2[0] * sh[4] * y * (-xx + yy + zz)
				-SH_C2[1] * sh[5] * 2.0f * xyz
				-6.0f * SH_C2[2] * sh[6] * x * zz
				+SH_C2[3] * sh[7] * z * (-xx + yy + zz)
				+2.0f * SH_C2[4] * sh[8] * x * (2.0f * yy + zz)

			) * inv_length2;

			dRGBdy += (
				SH_C2[0] * sh[4] * x * (xx - yy + zz)
				+SH_C2[1] * sh[5] * z * (xx - yy + zz)
				-6.0f * SH_C2[2] * sh[6] * y * zz
				-2.0f * SH_C2[3] * sh[7] * xyz
				-2.0f * SH_C2[4] * sh[8] * y * (2.0f * xx + zz)

			) * inv_length2;

			dRGBdz += (
				-SH_C2[0] * sh[4] * 2.f * xyz
				+SH_C2[1] * sh[5] * y * (xx + yy - zz)
				+6.0f * SH_C2[2] * sh[6] * z * (xx + yy)
				+SH_C2[3] * sh[7] * x * (xx + yy - zz)
				+2.0f * SH_C2[4] * sh[8] * z * (yy - xx)

			) * inv_length2;


			if (deg > 2)
			{
				dRGBdx += (
					3.0f * SH_C3[0] * sh[9] * y * (-x*xx + 3.0f*x*yy + 2.0f*x*zz)
					+SH_C3[1] * sh[10] * yz * (-2.0f*xx + yy + zz)
					+SH_C3[2] * sh[11] * xy * (xx + yy - 14.0f*zz)
					+3.0f * SH_C3[3] * sh[12] * xz * ((xx + yy) - 4.0f * zz)
					-SH_C3[4] * sh[13] * (xx*(yy + 11.0f*zz) + yyyy - 3.0f*yyzz - 4.0f*zzzz)
					+SH_C3[5] * sh[14] * z * (-(x * xx) + 5.0f * x * yy + 2.0f * x * zz)
					+3.0f * SH_C3[6] * sh[15] * (xx * (3.0f * yy + zz) - yy * (yy + zz))

				) * inv_length5div2;

				dRGBdy += (
					3.0f * SH_C3[0] * sh[9] * (xxxx + xx*(zz - 3.0f*yy) - yyzz)
					+SH_C3[1] * sh[10] * xz * (xx - 2.0f*yy + zz)
					-SH_C3[2] * sh[11] * (xxxx + xx*(yy - 3.0f*zz) + 11.0f*yy*zz - 4.0f*zzzz)
					+3.0f * SH_C3[3] * sh[12] * yz * ((xx + yy) - 4.0f * zz)
					+SH_C3[4] * sh[13] * xy * (xx + yy - 14.0f*zz)
					+SH_C3[5] * sh[14] * yz * (-5.0f * xx + yy - 2.0f * zz)
					-3.0f * SH_C3[6] * sh[15] * xy * (3.0f * xx - yy + 2.0f * zz)

				) * inv_length5div2;


				dRGBdz += (
					3.0f * SH_C3[0] * sh[9] * yz * (yy - 3.0f*xx)
					+SH_C3[1] * sh[10] * xy * (xx + yy - 2.0f*zz)
					+SH_C3[2] * sh[11] * yz * (11.0f*xx + 11.0f*yy - 4.0f*zz)
					-3.0f * SH_C3[3] * sh[12] * (xx + yy) * ((xx + yy) - 4.0f * zz)
					+SH_C3[4] * sh[13] * xz * (11.0f*xx + 11.0f*yy - 4.0f*zz)
					+SH_C3[5] * sh[14] * (xx - yy) * (xx + yy - 2.0f * zz)
					-3.0f * SH_C3[6] * sh[15] * xz * (xx - 3.0f * yy)

				) * inv_length5div2;
			}
		}
	}
	
	float3 dpixel_dmean_x = elementwiseMul(dRGBdx, dpixel_dRGB);
	float3 dpixel_dmean_y = elementwiseMul(dRGBdy, dpixel_dRGB);
	float3 dpixel_dmean_z = elementwiseMul(dRGBdz, dpixel_dRGB);


	//here,xyz denote means3D
	glm::vec3 dRGBdxx(0, 0, 0);
	glm::vec3 dRGBdxy(0, 0, 0);
	glm::vec3 dRGBdxz(0, 0, 0);
	glm::vec3 dRGBdyy(0, 0, 0);
	glm::vec3 dRGBdyz(0, 0, 0);
	glm::vec3 dRGBdzz(0, 0, 0);

	if (deg > 0)
	{
		dRGBdxx = (
			SH_C1 * sh[1] * y * (-2.f * xx + yy + zz)
			-SH_C1 * sh[2] * z * (-2.f * xx + yy + zz)
			+3.f * SH_C1 * sh[3] * x * (yy + zz)

		) * inv_length5div2;
		
		dRGBdxy = (
			SH_C1 * sh[1] * x * (xx - 2.f * yy + zz)
			+3.f * SH_C1 * sh[2] * xyz
			+SH_C1 * sh[3] * y * (-2.f * xx + yy + zz)

		) * inv_length5div2;
		
		dRGBdxz = (
			-3.f * SH_C1 * sh[1] * xyz
			-SH_C1 * sh[2] * x * (xx + yy - 2.f * zz)
			+SH_C1 * sh[3] * z * (-2.f * xx + yy + zz)

		) * inv_length5div2;
		
		
		dRGBdyy = (
			3.f * SH_C1 * sh[1] * y * (xx + zz)
			-SH_C1 * sh[2] * z * (xx - 2.f * yy + zz)
			+SH_C1 * sh[3] * x * (xx - 2.f * yy + zz)

		) * inv_length5div2;
		
		dRGBdyz = (
			SH_C1 * sh[1] * z * (xx - 2.f * yy + zz)
			-SH_C1 * sh[2] * y * (xx + yy - 2.f * zz)
			-3.f * SH_C1 * sh[3] * xyz

		) * inv_length5div2;
		
		dRGBdzz = (
			SH_C1 * sh[1] * y * (xx + yy - 2.f * zz)
			-3.f * SH_C1 * sh[2] * z * (xx + yy)
			+SH_C1 * sh[3] * x * (xx + yy - 2.f * zz)

		) * inv_length5div2;

		if (deg > 1)
		{
			dRGBdxx += (
				SH_C2[0] * sh[4] * 2.f * xy * (xx - 3.f * (yy + zz))
				-SH_C2[1] * sh[5] * 2.f * yz * (-3.f * xx + yy + zz)
				-SH_C2[2] * sh[6] * 6.f * zz * (-3.f * xx + yy + zz)
				+SH_C2[3] * sh[7] * 2.f * xz * (xx - 3.f * (yy + zz))
				-SH_C2[4] * sh[8] * 2.f * (2.f*yy + zz) * (3.f*xx - yy - zz)

			) * inv_length3;
			dRGBdxy += (
				-SH_C2[0] * sh[4] * (xxxx - 6.f * xx * yy + yyyy - zzzz)
				-SH_C2[1] * sh[5] * 2.f * xz * (xx - 3.f * yy + zz)
				+SH_C2[2] * sh[6] * 24.f * xy * zz
				-SH_C2[3] * sh[7] * 2.f * yz * (-3.f * xx + yy + zz)
				+SH_C2[4] * sh[8] * 8.f * xy * (xx - yy)

			) * inv_length3;
			dRGBdxz += (
				-SH_C2[0] * sh[4] * 2.f * yz * (-3.f * xx + yy + zz)
				-SH_C2[1] * sh[5] * 2.f * xy * (xx + yy - 3.f * zz)
				-SH_C2[2] * sh[6] * 12.f * xz * (xx + yy - zz)
				-SH_C2[3] * sh[7] * (xxxx - 6.f * xx * zz - yyyy + zzzz)
				+SH_C2[4] * sh[8] * 4.f * xz * (xx - 3.f*yy - zz)

			) * inv_length3;
			dRGBdyy += (
				-SH_C2[0] * sh[4] * 2.f * xy * (3.f * xx - yy + 3.f * zz)
				+SH_C2[1] * sh[5] * 2.f * yz * (-3.f * xx + yy - 3.f * zz)
				-SH_C2[2] * sh[6] * 6.f * zz * (xx - 3.f * yy + zz)
				-SH_C2[3] * sh[7] * 2.f * xz * (xx - 3.f * yy + zz)
				-SH_C2[4] * sh[8] * 2.f * (2.f*xx + zz) * (xx - 3.f*yy + zz)

			) * inv_length3;
			dRGBdyz += (
				-SH_C2[0] * sh[4] * 2.f * xz * (xx - 3.f * yy + zz)
				+SH_C2[1] * sh[5] * (xxxx - yyyy + 6.f * yyzz - zzzz)
				-SH_C2[2] * sh[6] * 12.f * yz * (xx + yy - zz)
				-SH_C2[3] * sh[7] * 2.f * xy * (xx + yy - 3.f * zz)
				+SH_C2[4] * sh[8] * 4.f * yz * (3.f*xx - yy + zz)

			) * inv_length3;
			dRGBdzz += (
				-SH_C2[0] * sh[4] * 2.f * xy * (xx + yy - 3.f * zz)
				-SH_C2[1] * sh[5] * 2.f * yz * (3.f * xx + 3.f * yy - zz)
				+SH_C2[2] * sh[6] * 6.f * (xx + yy) * (xx + yy - 3.f * zz)
				-SH_C2[3] * sh[7] * 2.f * xz * (3.f * xx + 3.f * yy - zz)
				-SH_C2[4] * sh[8] * 2.f * (xx - yy) * (xx + yy - 3.f*zz)

			) * inv_length3;

			if (deg > 2)
			{

				dRGBdxx += (
					3.f * SH_C3[0] * sh[9] * y * (2.f * xxxx - xx * (15.f * yy + 11.f * zz) + 3.f * yyyy + 5.f * yyzz + 2.f * zzzz)
					-3.f * SH_C3[1] * sh[10] * yz * (3.f * x * (yy + zz) - 2.f * x * xx)
					+SH_C3[2] * sh[11] * y * (-2.f * xxxx - xx * (yy - 59.f * zz) + yyyy - 13.f * yyzz - 14.f * zzzz)
					-3.f * SH_C3[3] * sh[12] * z * (2.f * xxxx + xx * (yy - 19.f * zz) - yyyy + 3.f * yyzz + 4.f * zzzz)
					+3.f * SH_C3[4] * sh[13] * x * (xx * (yy + 11.f * zz) + yyyy - 13.f * yyzz - 14.f * zzzz)
					+SH_C3[5] * sh[14] * z * (2.f * xxxx - xx * (23.f * yy + 11.f * zz) + 5.f * yyyy + 7.f * yyzz + 2.f * zzzz)
					-3.f * SH_C3[6] * sh[15] * x * (3.f * xx * (3.f * yy + zz) - 11.f * yyyy - 13.f * yyzz - 2.f * zzzz)

				) * inv_length7div2;
		
				dRGBdxy += (
					-3.f * SH_C3[0] * sh[9] * x * (xxxx - xx * (13.f * yy + zz) + 6.f * yyyy - yyzz - 2.f * zzzz)
					+SH_C3[1] * sh[10] * z * (-2.f * xxxx + xx * (11.f * yy - zz) - 2.f * yyyy - yyzz + zzzz)
					+SH_C3[2] * sh[11] * x * (xxxx - xx * (yy + 13.f * zz) - 2.f * yyyy + 59.f * yyzz - 14.f * zzzz)
					-3.f * SH_C3[3] * sh[12] * xyz * (3.f * xx + 3.f * yy - 22.f * zz)
					-SH_C3[4] * sh[13] * y * (2 * xxxx + xx * (yy - 59.f * zz) - yyyy + 13.f * yyzz + 14.f * zzzz)
					+15.f * SH_C3[5] * sh[14] * xyz * (xx - yy)
					+3.f * SH_C3[6] * sh[15] * y * (6.f * xxxx - xx * (13.f * yy + zz) + yyyy - yyzz - 2.f * zzzz)

				) * inv_length7div2;
		
				dRGBdxz += (
					3.f * SH_C3[0] * sh[9] * xyz * (9.f * xx - 11.f * yy - 6.f * zz)
					+SH_C3[1] * sh[10] * y * (-2.f * xxxx - xx * (yy - 11.f * zz) + yyyy - yyzz - 2.f * zzzz)
					-3.f * SH_C3[2] * sh[11] * xyz * (11.f * xx + 11.f * yy - 14.f * zz)
					+3.f * SH_C3[3] * sh[12] * x * (xxxx + 2.f * xx * (yy - 8.f * zz) + yyyy - 16.f * yyzz + 8.f * zzzz)
					-SH_C3[4] * sh[13] * z * (22.f * xxxx + xx * (11.f * yy - 49.f * zz) - 11.f * yyyy - 7.f * yyzz + 4.f * zzzz)
					-SH_C3[5] * sh[14] * x * (xxxx - 2.f * xx * (2.f * yy + 5 * zz) - 5.f * yyyy + 14.f * yyzz + 4.f * zzzz)
					+3.f * SH_C3[6] * sh[15] * z * (2.f * xxxx - 3.f * xx * (5.f * yy + zz) + 3.f * yy * (yy + zz))

				) * inv_length7div2;
		
		
				dRGBdyy += (
					-3.f * SH_C3[0] * sh[9] * y * (11.f * xxxx + xx * (13.f * zz - 9.f * yy) - 3.f * yyzz + 2.f * zzzz)
					-3.f * SH_C3[1] * sh[10] * xyz * (3.f * xx - 2.f * yy + 3.f * zz)
					+3.f * SH_C3[2] * sh[11] * y * (xxxx + xx * (yy - 13.f * zz) + 11.f * yyzz - 14.f * zzzz)
					-3.f * SH_C3[3] * sh[12] * z * (-xxxx + xx * (yy + 3.f * zz) + 2.f * yyyy - 19.f * yyzz + 4.f * zzzz)
					+SH_C3[4] * sh[13] * x * (xxxx - xx * (yy + 13.f * zz) - 2.f * yyyy + 59.f * yyzz - 14.f * zzzz)
					-SH_C3[5] * sh[14] * z * (5.f * xxxx + xx * (7.f * zz - 23.f * yy) + 2.f * yyyy - 11.f * yyzz + 2.f * zzzz)
					-3.f * SH_C3[6] * sh[15] * x * (3.f * xxxx + 5.f * xx * (zz - 3.f * yy) + 2.f * yyyy - 11.f * yyzz + 2.f * zzzz)

				) * inv_length7div2;
		
				dRGBdyz += (
					-3.f * SH_C3[0] * sh[9] * z * (3.f * xxxx + 3.f * xx * (zz - 5.f * yy) + 2.f * yyyy - 3.f * yyzz)
					+SH_C3[1] * sh[10] * x * (xxxx - xx * (yy + zz) - 2.f * yyyy + 11.f * yyzz - 2.f * zzzz)
					+SH_C3[2] * sh[11] * z * (11.f * xxxx + xx * (7.f * zz - 11.f * yy) - 22.f * yyyy + 49.f * yyzz - 4.f * zzzz)
					+3.f * SH_C3[3] * sh[12] * y * (xxxx + 2.f * xx * (yy - 8.f * zz) + yyyy - 16.f * yyzz + 8.f * zzzz)
					-3.f * SH_C3[4] * sh[13] * xyz * (11.f * xx + 11.f * yy - 14.f * zz)
					+SH_C3[5] * sh[14] * y * (-5.f * xxxx + xx * (14.f * zz - 4.f * yy) + yyyy - 10.f * yyzz + 4.f * zzzz)
					+3.f * SH_C3[6] * sh[15] * xyz * (11.f * xx - 9.f * yy + 6.f * zz)

				) * inv_length7div2;
		
				dRGBdzz += (
					3.f * SH_C3[0] * sh[9] * y * (yy - 3.f * xx) * (xx + yy - 4.f * zz)
					-3.f * SH_C3[1] * sh[10] * xyz * (3.f * xx + 3.f * yy - 2.f * zz)
					+SH_C3[2] * sh[11] * y * (11.f * xxxx + xx * (22.f * yy - 56.f * zz) + 11.f * yyyy - 56.f * yyzz + 8.f * zzzz)
					+3.f * SH_C3[3] * sh[12] * z * (xx + yy) * (13.f * xx + 13.f * yy - 12.f * zz)
					+SH_C3[4] * sh[13] * x * (11 * xxxx + xx * (22.f * yy - 56.f * zz) + 11.f * yyyy - 56.f * yyzz + 8.f * zzzz)
					-3.f * SH_C3[5] * sh[14] * z * (xx - yy) * (3.f * xx + 3.f * yy - 2.f * zz)
					-3.f * SH_C3[6] * sh[15] * x * (xx - 3.f * yy) * (xx + yy - 4.f * zz)

				) * inv_length7div2;
			}
		}
	}


	// [a,b,c]
	// [d,e,f]
	// [g,h,i]
	float3 dpixel_dmean_2_a = elementwiseMul(dRGBdxx, dpixel_dRGB);
	float3 dpixel_dmean_2_b = elementwiseMul(dRGBdxy, dpixel_dRGB);
	float3 dpixel_dmean_2_c = elementwiseMul(dRGBdxz, dpixel_dRGB);
	float3 dpixel_dmean_2_e = elementwiseMul(dRGBdyy, dpixel_dRGB);
	float3 dpixel_dmean_2_f = elementwiseMul(dRGBdyz, dpixel_dRGB);
	float3 dpixel_dmean_2_i = elementwiseMul(dRGBdzz, dpixel_dRGB);



}

__global__ void pos_solve_proj() {
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	float3 m = means[idx];

	// Taking care of gradients from the screenspace points
	float4 m_clip = transformPoint4x4(m, proj);
	float inv_m_clip_w = 1.0f / (m_clip.w + 0.0000001f);
	float inv_m_clip_w_2 = inv_m_clip_w * inv_m_clip_w;


	// Compute loss gradient w.r.t. 3D means due to gradients of 2D means
	// from rendering procedure
	float3 dG_dmean;
	float mul1 = (proj[0] * m.x + proj[4] * m.y + proj[8] * m.z + proj[12]);
	float mul2 = (proj[1] * m.x + proj[5] * m.y + proj[9] * m.z + proj[13]);
	float mul1_2 = mul1 * inv_m_clip_w_2;
	float mul2_2 = mul2 * inv_m_clip_w_2;
	float mul1_3 = mul1_2 * inv_m_clip_w;
	float mul2_3 = mul2_2 * inv_m_clip_w;


	// [a,b,c]
	// [d,e,f]
	float2x3 dmean2D_dmean3D;

	dmean2D_dmean3D[0].x = proj[0] * inv_m_clip_w - proj[3] * mul1_2;
	dmean2D_dmean3D[0].y = proj[4] * inv_m_clip_w - proj[7] * mul1_2;
	dmean2D_dmean3D[0].z = proj[8] * inv_m_clip_w - proj[11] * mul1_2;

	dmean2D_dmean3D[1].x = proj[1] * inv_m_clip_w - proj[3] * mul2_2;
	dmean2D_dmean3D[1].y = proj[5] * inv_m_clip_w - proj[7] * mul2_2;
	dmean2D_dmean3D[1].z = proj[9] * inv_m_clip_w - proj[11] * mul2_2;



	dG_dmean.x = (dmean2D_dmean3D[0].x) * dG_dmean2D[idx].x + (dmean2D_dmean3D[1].x) * dG_dmean2D[idx].y;
	dG_dmean.y = (dmean2D_dmean3D[0].y) * dG_dmean2D[idx].x + (dmean2D_dmean3D[1].y) * dG_dmean2D[idx].y;
	dG_dmean.z = (dmean2D_dmean3D[0].z) * dG_dmean2D[idx].x + (dmean2D_dmean3D[1].z) * dG_dmean2D[idx].y;


	// [a,b,c]
	// [d,e,f]
	// [g,h,i]

	float2x3 dG_dmean_2;


	dG_dmean_2[0].x = dG_dmean2D[idx].x * (-2.f * proj[0] * proj[3] * inv_m_clip_w_2 + 2.f * proj[3] * proj[3] * mul1_3) + dG_dmean2D[idx].y * (-2.f * proj[1] * proj[3] * inv_m_clip_w_2 + 2.f * proj[3] * proj[3] * mul2_3);
	dG_dmean_2[0].y = dG_dmean2D[idx].x * (2.f * proj[3] * proj[7] * mul1_3 + (- proj[0] * proj[7] - proj[3] * proj[4]) * inv_m_clip_w_2) + dG_dmean2D[idx].y * (2.f * proj[3] * proj[7] * mul2_3 + (- proj[1] * proj[7] - proj[3] * proj[5]) * inv_m_clip_w_2);
	dG_dmean_2[0].z = dG_dmean2D[idx].x * (2.f * proj[11] * proj[3] * mul1_3 - (proj[0] * proj[11] + proj[3] * proj[8]) * inv_m_clip_w_2) + dG_dmean2D[idx].y * (2.f * proj[11] * proj[3] * mul2_3 - (proj[1] * proj[11] + proj[3] * proj[9]) * inv_m_clip_w_2);
	dG_dmean_2[1].x = dG_dmean2D[idx].x * (2.f * proj[7] * proj[7] * mul1_3 - 2.f * proj[4] * proj[7] * inv_m_clip_w_2) + dG_dmean2D[idx].y * (2.f * proj[7] * proj[7] * mul2_3 - 2.f * proj[5] * proj[7] * inv_m_clip_w_2);
	dG_dmean_2[1].y = dG_dmean2D[idx].x * (2.f * proj[11] * proj[7] * mul1_3 - (proj[11] * proj[4] + proj[7] * proj[8]) * inv_m_clip_w_2) + dG_dmean2D[idx].y * (2.f * proj[11] * proj[7] * mul2_3 - (proj[11] * proj[5] + proj[7] * proj[9]) * inv_m_clip_w_2);
	dG_dmean_2[1].z = dG_dmean2D[idx].x * (2.f * proj[11] * (proj[11] * mul1_3 - proj[8] * inv_m_clip_w_2)) + dG_dmean2D[idx].y * (2.f * proj[11] * (proj[11] * mul2_3 - proj[9] * inv_m_clip_w_2));

	float mul3_1_1 = dmean2D_dmean3D[0].x * dG_dmean2D_2.x + dmean2D_dmean3D[1].x * dG_dmean2D_2.y;
	float mul3_1_2 = dmean2D_dmean3D[0].x * dG_dmean2D_2.y + dmean2D_dmean3D[1].x * dG_dmean2D_2.z;
	float mul3_2_1 = dmean2D_dmean3D[0].y * dG_dmean2D_2.x + dmean2D_dmean3D[1].y * dG_dmean2D_2.y;
	float mul3_2_2 = dmean2D_dmean3D[0].y * dG_dmean2D_2.y + dmean2D_dmean3D[1].y * dG_dmean2D_2.z;
	float mul3_3_1 = dmean2D_dmean3D[0].z * dG_dmean2D_2.x + dmean2D_dmean3D[1].z * dG_dmean2D_2.y;
	float mul3_3_2 = dmean2D_dmean3D[0].z * dG_dmean2D_2.y + dmean2D_dmean3D[1].z * dG_dmean2D_2.z;

	dG_dmean_2[0].x += dmean2D_dmean3D[0].x * mul3_1_1 + dmean2D_dmean3D[1].x * mul3_1_2;
	dG_dmean_2[0].y += dmean2D_dmean3D[0].y * mul3_1_1 + dmean2D_dmean3D[1].y * mul3_1_2;
	dG_dmean_2[0].z += dmean2D_dmean3D[0].z * mul3_1_1 + dmean2D_dmean3D[1].z * mul3_1_2;
	dG_dmean_2[1].x += dmean2D_dmean3D[0].y * mul3_2_1 + dmean2D_dmean3D[1].y * mul3_2_2;
	dG_dmean_2[1].y += dmean2D_dmean3D[0].z * mul3_2_1 + dmean2D_dmean3D[1].z * mul3_2_2;
	dG_dmean_2[1].z += dmean2D_dmean3D[0].z * mul3_3_1 + dmean2D_dmean3D[1].z * mul3_3_2;









	const float* cov3D = cov3Ds + 6 * idx;

	float3 mean = means[idx];
	float3 t = transformPoint4x3(mean, view_matrix);
	

	const float t_z_inv = 1.0f / t.z;
	const float t_z_2_inv = 1.0f / (t.z * t.z);
	const float t_z_3_inv = 1.0f / (t_z_inv * t_z_2_inv);
	const float t_z_4_inv = 1.0f / (t_z_2_inv * t_z_2_inv);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x * t_z_inv;
	const float tytz = t.y * t_z_inv;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;
	
	const float x_grad_mul = txtz < -limx || txtz > limx ? 0 : 1;
	const float y_grad_mul = tytz < -limy || tytz > limy ? 0 : 1;


	glm::mat3 J = glm::mat3(h_x * t_z_inv, 0.0f, 0.0f,
		0.0f, h_y * t_z_inv, 0.0f,
		-(h_x * t.x) * t_z_2_inv, -(h_y * t.y) * t_z_2_inv, 0.0f);


	glm::mat3 W = glm::mat3(
		view_matrix[0], view_matrix[1], view_matrix[2],
		view_matrix[4], view_matrix[5], view_matrix[6],
		view_matrix[8], view_matrix[9], view_matrix[10]);

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);



	glm::mat3 term1 = W * Vrk * glm::transpose(W);
	glm::mat3 term2 = term1 * glm::transpose(J);
	glm::mat3 term3 = J * term1;

	glm::mat3 cov2D = J * term2;

	float c_xx = cov2D[0].x;
	float c_xy = cov2D[0].y;
	float c_yy = cov2D[1].y;
	
	constexpr float h_var = 0.3f;

	c_xx += h_var;
	c_yy += h_var;


	const float P_c_xy_2 = c_xy * c_xy;
	const float P_c_xy_3 = P_c_xy_2 * c_xy;
	const float P_c_xy_4 = P_c_xy_2 * P_c_xy_2;

	const float P_c_xx_2 = c_xx * c_xx;
	const float P_c_xx_3 = P_c_xx_2 * c_xx;
	
	const float P_c_yy_2 = c_yy * c_yy;
	const float P_c_yy_3 = P_c_yy_2 * c_yy;

	const float P_c_xx_xy = c_xx * c_xy;
	const float P_c_xx_yy = c_xx * c_yy;
	const float P_c_xy_yy = c_xy * c_yy;
	const float P_c_xx_xy_yy = P_c_xx_xy * c_yy;

	float denom = P_c_xy_2 - P_c_xx_yy;

	float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);
	float denom3inv = 1.0f / ((denom * denom * denom) + 0.0000001f);

	float3x3 dconic2D_dcov2D;

	dconic2D_dcov2D[0].x = -P_c_yy_2 * denom2inv;
	dconic2D_dcov2D[0].y = P_c_xy_yy * denom2inv;
	dconic2D_dcov2D[0].z = -P_c_xy_2 * denom2inv;

	dconic2D_dcov2D[1].x = dconic2D_dcov2D[0].y;
	//(in cuda) dconic2D_dcov2D[1].y = (in math) dconic2D_dcov2D[1,2,1,2] + dconic2D_dcov2D[1,2,2,1] = (in math) dconic2D_dcov2D[2,1,1,2] + dconic2D_dcov2D[2,1,2,1]
	dconic2D_dcov2D[1].y = -(P_c_xx_yy + P_c_xy_2) * denom2inv;
	dconic2D_dcov2D[1].z = P_c_xx_xy * denom2inv;

	dconic2D_dcov2D[2].x = dconic2D_dcov2D[0].z;
	dconic2D_dcov2D[2].y = dconic2D_dcov2D[1].z;
	dconic2D_dcov2D[2].z = -P_c_xx_2 * denom2inv;



	float dG_dc_xx = 0;
	float dG_dc_xy = 0;
	float dG_dc_yy = 0;


	if (denom2inv != 0)
	{
		dG_dc_xx = denom2inv * (-P_c_xy_2 * dG_dconic2D.x + 2 * P_c_xy_yy * dG_dconic2D.y + (- P_c_xy_2) * dG_dconic2D.z);
		dG_dc_xy = denom2inv * (P_c_xy_yy * dG_dconic2D.x - (denom + 2 * P_c_xy_2) * dG_dconic2D.y + P_c_xx_xy * dG_dconic2D.z);
		dG_dc_yy = denom2inv * (-P_c_xy_2 * dG_dconic2D.z + 2 * P_c_xx_xy * dG_dconic2D.y + (- P_c_xy_2) * dG_dconic2D.x);
	}





	float3x3 dG_dcov2D_2;
	dG_dcov2D_2[0].x = denom3inv * (
		2.0f * P_c_xy_3 * c_yy * dG_dconic2D_dcov2D[1].x 
		- P_c_xy_4 * dG_dconic2D_dcov2D[2].x 
		+ P_c_yy_3 * (c_xx * dG_dconic2D_dcov2D[0].x - 2.0f * dG_dconic2D.x) 
		- 2.0f * c_xy * P_c_yy_2 * (c_xx * dG_dconic2D_dcov2D[1].x - 2.0f * dG_dconic2D.y) 
		+ P_c_xy_2 * c_yy * (-c_yy * dG_dconic2D_dcov2D[0].x + c_xx * dG_dconic2D_dcov2D[2].x - 2.0f * dG_dconic2D.z)
	);
	dG_dcov2D_2[0].y = denom3inv * (
		P_c_xy_2 * c_yy * (c_xx * dG_dconic2D_dcov2D[2].y - c_yy * dG_dconic2D_dcov2D[0].y - 3.0f * dG_dconic2D.y) +
		P_c_xy_yy * (c_xx * (dG_dconic2D.z - 2.0f * c_yy * dG_dconic2D_dcov2D[1].y) + 2.0f * c_yy * dG_dconic2D.x) +
		c_xx * P_c_yy_2 * (c_yy * dG_dconic2D_dcov2D[0].y - dG_dconic2D.y) +
		(-P_c_xy_4 * dG_dconic2D_dcov2D[2].y) +
		P_c_xy_3 * (2.0f * c_yy * dG_dconic2D_dcov2D[1].y + dG_dconic2D.z)
	);
	dG_dcov2D_2[0].z = denom3inv * (
		-P_c_xy_2 * (-P_c_xx_yy * dG_dconic2D_dcov2D[2].z + 2.0f * c_xx * dG_dconic2D.z + P_c_yy_2 * dG_dconic2D_dcov2D[0].z + 2.0f * c_yy * dG_dconic2D.x) +
		2.0f * P_c_xx_xy_yy * (dG_dconic2D.y - c_yy * dG_dconic2D_dcov2D[1].z) +
		c_xx * P_c_yy_3 * dG_dconic2D_dcov2D[0].z +
		(-P_c_xy_4 * dG_dconic2D_dcov2D[2].z) +
		2.0f * P_c_xy_3 * (c_yy * dG_dconic2D_dcov2D[1].z + dG_dconic2D.y)
	);

	dG_dcov2D_2[1].x = denom3inv * (
		P_c_xy_yy * (-P_c_xx_2 * dG_dconic2D_dcov2D[2].x + c_xx * (dG_dconic2D.z - c_yy * dG_dconic2D_dcov2D[0].x) + 2.0f * c_yy * dG_dconic2D.x) +
		P_c_xy_3 * (c_xx * dG_dconic2D_dcov2D[2].x + c_yy * dG_dconic2D_dcov2D[0].x + dG_dconic2D.z) +
		c_xx * P_c_yy_2 * (c_xx * dG_dconic2D_dcov2D[1].x - dG_dconic2D.y) +
		(-P_c_xy_4 * dG_dconic2D_dcov2D[1].x) +
		(-3.0f * P_c_xy_2 * c_yy * dG_dconic2D.y)
	);
	//(in cuda) dG_dcov2D_2[1].y = (in math) dG_dcov2D_2[1,2,1,2]+dG_dcov2D_2[1,2,2,1] = (in math) dG_dcov2D_2[2,1,1,2]+dG_dcov2D_2[2,1,2,1]
	dG_dcov2D_2[1].y = 
		denom3inv * (
		P_c_xx_2 * P_c_yy_2 * dG_dconic2D_dcov2D[1].y +
		P_c_xy_3 * (c_xx * dG_dconic2D_dcov2D[2].y + c_yy * dG_dconic2D_dcov2D[0].y + 2.0f * dG_dconic2D.y) +
		(-2.0f * P_c_xy_2 * (c_xx * dG_dconic2D.z + c_yy * dG_dconic2D.x)) +
		(-P_c_xx_xy_yy * (c_xx * dG_dconic2D_dcov2D[2].y + c_yy * dG_dconic2D_dcov2D[0].y - 2.0f * dG_dconic2D.y)) +
		(-P_c_xy_4 * dG_dconic2D_dcov2D[1].y)
		+ denom3inv * (
			P_c_xy_3 * (c_xx * dG_dconic2D_dcov2D[2].y + c_yy * dG_dconic2D_dcov2D[0].y) +
			(-P_c_xy_2 * (c_xx * dG_dconic2D.z + c_yy * dG_dconic2D.x)) +
			(-P_c_xx_xy_yy * (c_xx * dG_dconic2D_dcov2D[2].y + c_yy * dG_dconic2D_dcov2D[0].y - 4.0f * dG_dconic2D.y)) +
			P_c_xx_yy * (P_c_xx_yy * dG_dconic2D_dcov2D[1].y - c_xx * dG_dconic2D.z - c_yy * dG_dconic2D.x) +
			(-P_c_xy_4 * dG_dconic2D_dcov2D[1].y)
		);
	);
	dG_dcov2D_2[1].z = denom3inv * (
		P_c_xx_2 * c_yy * (c_yy * dG_dconic2D_dcov2D[1].z - dG_dconic2D.y) +
		P_c_xy_3 * (c_xx * dG_dconic2D_dcov2D[2].z + c_yy * dG_dconic2D_dcov2D[0].z + dG_dconic2D.x) +
		(-3.0f * c_xx * P_c_xy_2 * dG_dconic2D.y) +
		P_c_xx_xy * (c_yy * (dG_dconic2D.x - c_xx * dG_dconic2D_dcov2D[2].z) + 2.0f * c_xx * dG_dconic2D.z - P_c_yy_2 * dG_dconic2D_dcov2D[0].z) +
		(-P_c_xy_4 * dG_dconic2D_dcov2D[1].z)
	);

	dG_dcov2D_2[2].x = denom3inv * (
		(c_xx * c_xx * c_xx) * c_yy * dG_dconic2D_dcov2D[2].x +
		(-P_c_xy_2 * (P_c_xx_2 * dG_dconic2D_dcov2D[2].x - P_c_xx_yy * dG_dconic2D_dcov2D[0].x + 2.0f * c_xx * dG_dconic2D.z + 2.0f * c_yy * dG_dconic2D.x)) +
		2.0f * P_c_xy_3 * (c_xx * dG_dconic2D_dcov2D[1].x + dG_dconic2D.y) +
		2.0f * P_c_xx_xy_yy * (dG_dconic2D.y - c_xx * dG_dconic2D_dcov2D[1].x) +
		(-P_c_xy_4 * dG_dconic2D_dcov2D[0].x)
	);
	dG_dcov2D_2[2].y = denom3inv * (
		P_c_xx_2 * c_yy * (c_xx * dG_dconic2D_dcov2D[2].y - dG_dconic2D.y) +
		P_c_xy_3 * (2.0f * c_xx * dG_dconic2D_dcov2D[1].y + dG_dconic2D.x) +
		c_xx * P_c_xy_2 * (-(c_xx) * dG_dconic2D_dcov2D[2].y + c_yy * dG_dconic2D_dcov2D[0].y - 3.0f * dG_dconic2D.y) +
		P_c_xx_xy * (-2.0f * P_c_xx_yy * dG_dconic2D_dcov2D[1].y + 2.0f * c_xx * dG_dconic2D.z + c_yy * dG_dconic2D.x) +
		(-P_c_xy_4 * dG_dconic2D_dcov2D[0].y)
	);
	dG_dcov2D_2[2].z = denom3inv * (
		(c_xx * c_xx * c_xx) * (c_yy * dG_dconic2D_dcov2D[2].z - 2.0f * dG_dconic2D.z) +
		(-2.0f * P_c_xx_2 * c_xy * (c_yy * dG_dconic2D_dcov2D[1].z - 2.0f * dG_dconic2D.y)) +
		2.0f * c_xx * P_c_xy_3 * dG_dconic2D_dcov2D[1].z +
		c_xx * P_c_xy_2 * (-(c_xx) * dG_dconic2D_dcov2D[2].z + c_yy * dG_dconic2D_dcov2D[0].z - 2.0f * dG_dconic2D.x) +
		(-P_c_xy_4 * dG_dconic2D_dcov2D[0].z)
	);






	float4x3 dJ_dmean;
	

	const float h_x_div_t_z_2 = J[0].x * t_z_inv;
	const float h_y_div_t_z_2 = J[1].y * t_z_inv;
	const float h_x_t_x_div_t_z_3 = -J[2].x * t_z_inv;
	const float h_y_t_y_div_t_z_3 = -J[2].y * t_z_inv;

	dJ_dmean[0].x = -h_x_div_t_z_2 * view_matrix[2];
	dJ_dmean[0].y = -h_x_div_t_z_2 * view_matrix[6];
	dJ_dmean[0].z = -h_x_div_t_z_2 * view_matrix[10];

	dJ_dmean[1].x = -h_x_div_t_z_2 * view_matrix[0] + 2.0f * h_x_t_x_div_t_z_3 * view_matrix[2];
	dJ_dmean[1].y = -h_x_div_t_z_2 * view_matrix[4] + 2.0f * h_x_t_x_div_t_z_3 * view_matrix[6];
	dJ_dmean[1].z = -h_x_div_t_z_2 * view_matrix[8] + 2.0f * h_x_t_x_div_t_z_3 * view_matrix[10];

	dJ_dmean[2].x = -h_y_div_t_z_2 * view_matrix[2];
	dJ_dmean[2].y = -h_y_div_t_z_2 * view_matrix[6];
	dJ_dmean[2].z = -h_y_div_t_z_2 * view_matrix[10];

	dJ_dmean[3].x = -h_y_div_t_z_2 * view_matrix[1] + 2.0f * h_y_t_y_div_t_z_3 * view_matrix[2];
	dJ_dmean[3].y = -h_y_div_t_z_2 * view_matrix[5] + 2.0f * h_y_t_y_div_t_z_3 * view_matrix[6];
	dJ_dmean[3].z = -h_y_div_t_z_2 * view_matrix[9] + 2.0f * h_y_t_y_div_t_z_3 * view_matrix[10];



	float8x3 dJ_dmean_2;

	
	const float pre_c1 = 2.0f * t_z_3_inv;
	const float pre_c2 = 4.0f * t_z_3_inv;
	const float pre_c3 = 6.0f * t_z_4_inv;

	const float v2_sq = view_matrix[2] * view_matrix[2];
	const float v6_sq = view_matrix[6] * view_matrix[6];
	const float v10_sq = view_matrix[10] * view_matrix[10];
	const float v2_v6 = view_matrix[2] * view_matrix[6];
	const float v10_v2 = view_matrix[10] * view_matrix[2];
	const float v10_v6 = view_matrix[10] * view_matrix[6];

	const float pre_factor_a = pre_c1 * h_x;
	const float pre_factor_e = pre_c1 * h_y;
	const float pre_factor_c_1 = pre_factor_a;
	const float pre_factor_c_2 = pre_c2 * h_x;
	const float pre_factor_c_3 = pre_c3 * h_x * t.x;
	const float pre_factor_f_1 = pre_factor_e;
	const float pre_factor_f_2 = pre_c2 * h_y;
	const float pre_factor_f_3 = pre_c3 * h_y * t.y;

	dJ_dmean_2[0].x = pre_factor_a * v2_sq;
	dJ_dmean_2[0].y = pre_factor_a * v2_v6;
	dJ_dmean_2[0].z = pre_factor_a * v10_v2;
	dJ_dmean_2[1].x = pre_factor_a * v6_sq;
	dJ_dmean_2[1].y = pre_factor_a * v10_v6;
	dJ_dmean_2[1].z = pre_factor_a * v10_sq;

	dJ_dmean_2[2].x = pre_factor_c_2 * view_matrix[0] * view_matrix[2] - pre_factor_c_3 * v2_sq;
	dJ_dmean_2[2].y = -pre_factor_c_3 * v2_v6 + pre_factor_c_1 * (view_matrix[0] * view_matrix[6] + view_matrix[2] * view_matrix[4]);
	dJ_dmean_2[2].z = -pre_factor_c_3 * v10_v2 + pre_factor_c_1 * (view_matrix[0] * view_matrix[10] + view_matrix[2] * view_matrix[8]);
	dJ_dmean_2[3].x = pre_factor_c_2 * view_matrix[4] * view_matrix[6] - pre_factor_c_3 * v6_sq;
	dJ_dmean_2[3].y = -pre_factor_c_3 * v10_v6 + pre_factor_c_1 * (view_matrix[10] * view_matrix[4] + view_matrix[6] * view_matrix[8]);
	dJ_dmean_2[3].x = pre_factor_c_2 * view_matrix[10] * view_matrix[8] - pre_factor_c_3 * v10_sq;

	dJ_dmean_2[4].x = pre_factor_e * v2_sq;
	dJ_dmean_2[4].y = pre_factor_e * v2_v6;
	dJ_dmean_2[4].z = pre_factor_e * v10_v2;
	dJ_dmean_2[5].x = pre_factor_e * v6_sq;
	dJ_dmean_2[5].y = pre_factor_e * v10_v6;
	dJ_dmean_2[5].z = pre_factor_e * v10_sq;

	dJ_dmean_2[6].x = pre_factor_f_2 * view_matrix[1] * view_matrix[2] - pre_factor_f_3 * v2_sq;
	dJ_dmean_2[6].y = -pre_factor_f_3 * v2_v6 + pre_factor_f_1 * (view_matrix[1] * view_matrix[6] + view_matrix[2] * view_matrix[5]);
	dJ_dmean_2[6].z = -pre_factor_f_3 * v10_v2 + pre_factor_f_1 * (view_matrix[1] * view_matrix[10] + view_matrix[2] * view_matrix[9]);
	dJ_dmean_2[7].x = pre_factor_f_2 * view_matrix[5] * view_matrix[6] - pre_factor_f_3 * v6_sq;
	dJ_dmean_2[7].y = -pre_factor_f_3 * v10_v6 + pre_factor_f_1 * (view_matrix[10] * view_matrix[5] + view_matrix[6] * view_matrix[9]);
	dJ_dmean_2[7].z = pre_factor_f_2 * view_matrix[10] * view_matrix[9] - pre_factor_f_3 * v10_sq;





	float3x3 dcov2D_dmean;

	dcov2D_dmean[0].x = dJ_dmean[0].x * term2[0].x + dJ_dmean[1].x * term2[0].z + dJ_dmean[0].x * term3[0].x + dJ_dmean[1].x * term3[2].x;
	dcov2D_dmean[0].y = dJ_dmean[0].y * term2[0].x + dJ_dmean[1].y * term2[0].z + dJ_dmean[0].y * term3[0].x + dJ_dmean[1].y * term3[2].x;
	dcov2D_dmean[0].z = dJ_dmean[0].z * term2[0].x + dJ_dmean[1].z * term2[0].z + dJ_dmean[0].z * term3[0].x + dJ_dmean[1].z * term3[2].x;

	dcov2D_dmean[1].x = dJ_dmean[0].x * term2[1].x + dJ_dmean[1].x * term2[1].z + dJ_dmean[2].x * term3[1].x + dJ_dmean[3].x * term3[2].x;
	dcov2D_dmean[1].y = dJ_dmean[0].y * term2[1].x + dJ_dmean[1].y * term2[1].z + dJ_dmean[2].y * term3[1].x + dJ_dmean[3].y * term3[2].x;
	dcov2D_dmean[1].z = dJ_dmean[0].z * term2[1].x + dJ_dmean[1].z * term2[1].z + dJ_dmean[2].z * term3[1].x + dJ_dmean[3].z * term3[2].x;

	dcov2D_dmean[2].x = dJ_dmean[2].x * term2[1].y + dJ_dmean[3].x * term2[1].z + dJ_dmean[2].x * term3[1].y + dJ_dmean[3].x * term3[2].y;
	dcov2D_dmean[2].y = dJ_dmean[2].y * term2[1].y + dJ_dmean[3].y * term2[1].z + dJ_dmean[2].y * term3[1].y + dJ_dmean[3].y * term3[2].y;
	dcov2D_dmean[2].z = dJ_dmean[2].z * term2[1].y + dJ_dmean[3].z * term2[1].z + dJ_dmean[2].z * term3[1].y + dJ_dmean[3].z * term3[2].y;



	float6x3 dcov2D_dmean_2;


	pre_t1 = term2[0].x + term3[0].x;
	pre_t2 = term2[0].z + term3[2].x;
	pre_t3 = term2[1].y + term3[1].y;
	pre_t4 = term2[1].z + term3[2].y;

	dcov2D_dmean_2[0].x = 2.0f * dJ_dmean[0].x * dJ_dmean[0].x * term1[0].x 
		+ 4.0f * dJ_dmean[0].x * dJ_dmean[1].x * term1[2].x
		+ 2.0f * dJ_dmean[1].x * dJ_dmean[1].x * term1[2].z
		+ dJ_dmean_2[0].x * pre_t1 + dJ_dmean_2[2].x * pre_t2;

	dcov2D_dmean_2[0].y = 2.0f * dJ_dmean[0].x * dJ_dmean[0].y * term1[0].x
		+ 2.0f * dJ_dmean[0].x * dJ_dmean[1].y * term1[2].x
		+ 2.0f * dJ_dmean[0].y * dJ_dmean[1].x * term1[2].x
		+ 2.0f * dJ_dmean[1].x * dJ_dmean[1].y * term1[2].z
		+ dJ_dmean_2[0].y * pre_t1 + dJ_dmean_2[2].y * pre_t2;

	dcov2D_dmean_2[0].z = 2.0f * dJ_dmean[0].x * dJ_dmean[0].z * term1[0].x
		+ 2.0f * dJ_dmean[0].x * dJ_dmean[1].z * term1[2].x
		+ 2.0f * dJ_dmean[0].z * dJ_dmean[1].x * term1[2].x
		+ 2.0f * dJ_dmean[1].x * dJ_dmean[1].z * term1[2].z
		+ dJ_dmean_2[0].z * pre_t1 + dJ_dmean_2[2].z * pre_t2;

	dcov2D_dmean_2[1].x = 2.0f * dJ_dmean[0].y * dJ_dmean[0].y * term1[0].x
		+ 4.0f * dJ_dmean[0].y * dJ_dmean[1].y * term1[2].x
		+ 2.0f * dJ_dmean[1].y * dJ_dmean[1].y * term1[2].z
		+ dJ_dmean_2[1].x * pre_t1 + dJ_dmean_2[3].x * pre_t2;

	dcov2D_dmean_2[1].y = 2.0f * dJ_dmean[0].y * dJ_dmean[0].z * term1[0].x
		+ 2.0f * dJ_dmean[0].y * dJ_dmean[1].z * term1[2].x
		+ 2.0f * dJ_dmean[0].z * dJ_dmean[1].y * term1[2].x
		+ 2.0f * dJ_dmean[1].y * dJ_dmean[1].z * term1[2].z
		+ dJ_dmean_2[1].y * pre_t1 + dJ_dmean_2[3].y * pre_t2;
	
	dcov2D_dmean_2[1].z = 2.0f * dJ_dmean[0].z * dJ_dmean[0].z * term1[0].x
		+ 4.0f * dJ_dmean[0].z * dJ_dmean[1].z * term1[2].x
		+ 2.0f * dJ_dmean[1].z * dJ_dmean[1].z * term1[2].z
		+ dJ_dmean_2[1].z * pre_t1 + dJ_dmean_2[3].z * pre_t2;
	


	dcov2D_dmean_2[2].x = 2.0f * dJ_dmean[0].x * dJ_dmean[2].x * term1[1].x
		+ 2.0f * dJ_dmean[0].x * dJ_dmean[3].x * term1[2].x
		+ 2.0f * dJ_dmean[1].x * dJ_dmean[2].x * term1[2].y
		+ 2.0f * dJ_dmean[1].x * dJ_dmean[3].x * term1[2].z
		+ dJ_dmean_2[0].x * term2[1].x
		+ dJ_dmean_2[2].x * term2[1].z
		+ dJ_dmean_2[4].x * term3[1].x
		+ dJ_dmean_2[6].x * term3[2].x;

	dcov2D_dmean_2[2].y = dJ_dmean[0].x * dJ_dmean[2].y * term1[1].x
		+ dJ_dmean[0].x * dJ_dmean[3].y * term1[2].x
		+ dJ_dmean[0].y * dJ_dmean[2].x * term1[1].x
		+ dJ_dmean[0].y * dJ_dmean[3].x * term1[2].x
		+ dJ_dmean[1].x * dJ_dmean[2].y * term1[2].y
		+ dJ_dmean[1].x * dJ_dmean[3].y * term1[2].z
		+ dJ_dmean[1].y * dJ_dmean[2].x * term1[2].y
		+ dJ_dmean[1].y * dJ_dmean[3].x * term1[2].z
		+ dJ_dmean_2[0].y * term2[1].x
		+ dJ_dmean_2[2].y * term2[1].z
		+ dJ_dmean_2[4].y * term3[1].x
		+ dJ_dmean_2[6].y * term3[2].x;

	dcov2D_dmean_2[2].z = dJ_dmean[0].x * dJ_dmean[2].z * term1[1].x
		+ dJ_dmean[0].x * dJ_dmean[3].z * term1[2].x
		+ dJ_dmean[0].z * dJ_dmean[2].x * term1[1].x
		+ dJ_dmean[0].z * dJ_dmean[3].x * term1[2].x
		+ dJ_dmean[1].x * dJ_dmean[2].z * term1[2].y
		+ dJ_dmean[1].x * dJ_dmean[3].z * term1[2].z
		+ dJ_dmean[1].z * dJ_dmean[2].x * term1[2].y
		+ dJ_dmean[1].z * dJ_dmean[3].x * term1[2].z
		+ dJ_dmean_2[0].z * term2[1].x
		+ dJ_dmean_2[2].z * term2[1].z
		+ dJ_dmean_2[4].z * term3[1].x
		+ dJ_dmean_2[6].z * term3[2].x;

	dcov2D_dmean_2[3].x = 2.0f * dJ_dmean[0].y * dJ_dmean[2].y * term1[1].x
		+ 2.0f * dJ_dmean[0].y * dJ_dmean[3].y * term1[2].x
		+ 2.0f * dJ_dmean[1].y * dJ_dmean[2].y * term1[2].y
		+ 2.0f * dJ_dmean[1].y * dJ_dmean[3].y * term1[2].z
		+ dJ_dmean_2[1].x * term2[1].x
		+ dJ_dmean_2[3].x * term2[1].z
		+ dJ_dmean_2[5].x * term3[1].x
		+ dJ_dmean_2[7].x * term3[2].x;

	dcov2D_dmean_2[3].y = dJ_dmean[0].y * dJ_dmean[2].z * term1[1].x
		+ dJ_dmean[0].y * dJ_dmean[3].z * term1[2].x
		+ dJ_dmean[0].z * dJ_dmean[2].y * term1[1].x
		+ dJ_dmean[0].z * dJ_dmean[3].y * term1[2].x
		+ dJ_dmean[1].y * dJ_dmean[2].z * term1[2].y
		+ dJ_dmean[1].y * dJ_dmean[3].z * term1[2].z
		+ dJ_dmean[1].z * dJ_dmean[2].y * term1[2].y
		+ dJ_dmean[1].z * dJ_dmean[3].y * term1[2].z
		+ dJ_dmean_2[1].y * term2[1].x
		+ dJ_dmean_2[3].y * term2[1].z
		+ dJ_dmean_2[5].y * term3[1].x
		+ dJ_dmean_2[7].y * term3[2].x;

	dcov2D_dmean_2[3].z = 2.0f * dJ_dmean[0].z * dJ_dmean[2].z * term1[1].x
		+ 2.0f * dJ_dmean[0].z * dJ_dmean[3].z * term1[2].x
		+ 2.0f * dJ_dmean[1].z * dJ_dmean[2].z * term1[2].y
		+ 2.0f * dJ_dmean[1].z * dJ_dmean[3].z * term1[2].z
		+ dJ_dmean_2[1].z * term2[1].x
		+ dJ_dmean_2[3].z * term2[1].z
		+ dJ_dmean_2[5].z * term3[1].x
		+ dJ_dmean_2[7].z * term3[2].x;




	dcov2D_dmean_2[4].x = 2.0f * dJ_dmean[2].x * dJ_dmean[2].x * term1[1].y
		+ 4.0f * dJ_dmean[2].x * dJ_dmean[3].x * term1[2].y
		+ 2.0f * dJ_dmean[3].x * dJ_dmean[3].x * term1[2].z
		+ dJ_dmean_2[4].x * pre_t3 + dJ_dmean_2[6].x * pre_t4;

	dcov2D_dmean_2[4].y = 2.0f * dJ_dmean[2].x * dJ_dmean[2].y * term1[1].y
		+ 2.0f * dJ_dmean[2].x * dJ_dmean[3].y * term1[2].y
		+ 2.0f * dJ_dmean[2].y * dJ_dmean[3].x * term1[2].y
		+ 2.0f * dJ_dmean[3].x * dJ_dmean[3].y * term1[2].z
		+ dJ_dmean_2[4].y * pre_t3 + dJ_dmean_2[6].y * pre_t4;

	dcov2D_dmean_2[4].z = 2.0f * dJ_dmean[2].x * dJ_dmean[2].z * term1[1].y
		+ 2.0f * dJ_dmean[2].x * dJ_dmean[3].z * term1[2].y
		+ 2.0f * dJ_dmean[2].z * dJ_dmean[3].x * term1[2].y
		+ 2.0f * dJ_dmean[3].x * dJ_dmean[3].z * term1[2].z
		+ dJ_dmean_2[4].z * pre_t3 + dJ_dmean_2[6].z * pre_t4;

	dcov2D_dmean_2[5].x = 2.0f * dJ_dmean[2].y * dJ_dmean[2].y * term1[1].y
		+ 4.0f * dJ_dmean[2].y * dJ_dmean[3].y * term1[2].y
		+ 2.0f * dJ_dmean[3].y * dJ_dmean[3].y * term1[2].z
		+ dJ_dmean_2[5].x * pre_t3 + dJ_dmean_2[7].x * pre_t4;

	dcov2D_dmean_2[5].y = 2.0f * dJ_dmean[2].y * dJ_dmean[2].z * term1[1].y
		+ 2.0f * dJ_dmean[2].y * dJ_dmean[3].z * term1[2].y
		+ 2.0f * dJ_dmean[2].z * dJ_dmean[3].y * term1[2].y
		+ 2.0f * dJ_dmean[3].y * dJ_dmean[3].z * term1[2].z
		+ dJ_dmean_2[5].y * pre_t3 + dJ_dmean_2[7].y * pre_t4;

	dcov2D_dmean_2[5].z = 2.0f * dJ_dmean[2].z * dJ_dmean[2].z * term1[1].y
		+ 4.0f * dJ_dmean[2].z * dJ_dmean[3].z * term1[2].y
		+ 2.0f * dJ_dmean[3].z * dJ_dmean[3].z * term1[2].z
		+ dJ_dmean_2[5].z * pre_t3 + dJ_dmean_2[7].z * pre_t4;




	dG_dmean.x += dcov2D_dmean[0].x * dG_dc_xx + 2.0f * dcov2D_dmean[1].x * dG_dc_xy + dcov2D_dmean[2].x * dG_dc_yy;
	dG_dmean.y += dcov2D_dmean[0].y * dG_dc_xx + 2.0f * dcov2D_dmean[1].y * dG_dc_xy + dcov2D_dmean[2].y * dG_dc_yy;
	dG_dmean.z += dcov2D_dmean[0].z * dG_dc_xx + 2.0f * dcov2D_dmean[1].z * dG_dc_xy + dcov2D_dmean[2].z * dG_dc_yy;


	dG_dmean_2[0].x += dG_dc_xx * dcov2D_dmean_2[0].x + 2.0f * dG_dc_xy * dcov2D_dmean_2[2].x + dG_dc_yy * dcov2D_dmean_2[4].x;
	dG_dmean_2[0].y += dG_dc_xx * dcov2D_dmean_2[0].y + 2.0f * dG_dc_xy * dcov2D_dmean_2[2].y + dG_dc_yy * dcov2D_dmean_2[4].y;
	dG_dmean_2[0].z += dG_dc_xx * dcov2D_dmean_2[0].z + 2.0f * dG_dc_xy * dcov2D_dmean_2[2].z + dG_dc_yy * dcov2D_dmean_2[4].z;

	dG_dmean_2[1].x += dG_dc_xx * dcov2D_dmean_2[1].x + 2.0f * dG_dc_xy * dcov2D_dmean_2[3].x + dG_dc_yy * dcov2D_dmean_2[5].x;
	dG_dmean_2[1].y += dG_dc_xx * dcov2D_dmean_2[1].y + 2.0f * dG_dc_xy * dcov2D_dmean_2[3].y + dG_dc_yy * dcov2D_dmean_2[5].y;
	dG_dmean_2[1].z += dG_dc_xx * dcov2D_dmean_2[1].z + 2.0f * dG_dc_xy * dcov2D_dmean_2[3].z + dG_dc_yy * dcov2D_dmean_2[5].z;





	dG_dmean_2[0].x += dcov2D_dmean[0].x * dcov2D_dmean[0].x * dG_dcov2D_2[0].x +
		2.0f * dcov2D_dmean[0].x * dcov2D_dmean[1].x * (dG_dcov2D_2[0].y + dG_dcov2D_2[1].x) + 
		2.0f * dcov2D_dmean[1].x * dcov2D_dmean[1].x * dG_dcov2D_2[1].y + 
		dcov2D_dmean[0].x * dcov2D_dmean[2].x * (dG_dcov2D_2[0].z + dG_dcov2D_2[2].x) + 
		2.0f * dcov2D_dmean[1].x * dcov2D_dmean[2].x * (dG_dcov2D_2[1].z + dG_dcov2D_2[2].y) + 
		dcov2D_dmean[2].x * dcov2D_dmean[2].x * dG_dcov2D_2[2].z;

	dG_dmean_2[0].y += dcov2D_dmean[0].x * (dcov2D_dmean[0].y * dG_dcov2D_2[0].x + 2.0f * dcov2D_dmean[1].y * dG_dcov2D_2[0].y + dcov2D_dmean[2].y * dG_dcov2D_2[0].z) +
		2.0f * dcov2D_dmean[0].y * dcov2D_dmean[1].x * dG_dcov2D_2[1].x +
		dcov2D_dmean[0].y * dcov2D_dmean[2].x * dG_dcov2D_2[2].x +
		2.0f * dcov2D_dmean[1].x * dcov2D_dmean[1].y * dG_dcov2D_2[1].y +
		2.0f * dcov2D_dmean[1].x * dcov2D_dmean[2].y * dG_dcov2D_2[1].z +
		2.0f * dcov2D_dmean[1].y * dcov2D_dmean[2].x * dG_dcov2D_2[2].y +
		dcov2D_dmean[2].x * dcov2D_dmean[2].y * dG_dcov2D_2[2].z;

	dG_dmean_2[0].z += dcov2D_dmean[0].x * (dcov2D_dmean[0].z * dG_dcov2D_2[0].x + 2.0f * dcov2D_dmean[1].z * dG_dcov2D_2[0].y + dcov2D_dmean[2].z * dG_dcov2D_2[0].z) +
		2.0f * dcov2D_dmean[0].z * dcov2D_dmean[1].x * dG_dcov2D_2[1].x +
		dcov2D_dmean[0].z * dcov2D_dmean[2].x * dG_dcov2D_2[2].x +
		2.0f * dcov2D_dmean[1].x * dcov2D_dmean[1].z * dG_dcov2D_2[1].y +
		2.0f * dcov2D_dmean[1].x * dcov2D_dmean[2].z * dG_dcov2D_2[1].z +
		2.0f * dcov2D_dmean[1].z * dcov2D_dmean[2].x * dG_dcov2D_2[2].y +
		dcov2D_dmean[2].x * dcov2D_dmean[2].z * dG_dcov2D_2[2].z;


	dG_dmean_2[1].x += dcov2D_dmean[0].y * dcov2D_dmean[0].y * dG_dcov2D_2[0].x +
		2.0f * dcov2D_dmean[0].y * dcov2D_dmean[1].y * (dG_dcov2D_2[0].y + dG_dcov2D_2[1].x) +
		2.0f * dcov2D_dmean[1].y * dcov2D_dmean[1].y * dG_dcov2D_2[1].y +
		dcov2D_dmean[0].y * dcov2D_dmean[2].y * (dG_dcov2D_2[0].z + dG_dcov2D_2[2].x) +
		2.0f * dcov2D_dmean[1].y * dcov2D_dmean[2].y * (dG_dcov2D_2[1].z + dG_dcov2D_2[2].y) +
		dcov2D_dmean[2].y * dcov2D_dmean[2].y * dG_dcov2D_2[2].z;

	dG_dmean_2[1].y += dcov2D_dmean[0].y * (dcov2D_dmean[0].z * dG_dcov2D_2[0].x + 2.0f * dcov2D_dmean[1].z * dG_dcov2D_2[0].y + dcov2D_dmean[2].z * dG_dcov2D_2[0].z) +
		2.0f * dcov2D_dmean[0].z * dcov2D_dmean[1].y * dG_dcov2D_2[1].x +
		dcov2D_dmean[0].z * dcov2D_dmean[2].y * dG_dcov2D_2[2].x +
		2.0f * dcov2D_dmean[1].y * dcov2D_dmean[1].z * dG_dcov2D_2[1].y +
		2.0f * dcov2D_dmean[1].y * dcov2D_dmean[2].z * dG_dcov2D_2[1].z +
		2.0f * dcov2D_dmean[1].z * dcov2D_dmean[2].y * dG_dcov2D_2[2].y +
		dcov2D_dmean[2].y * dcov2D_dmean[2].z * dG_dcov2D_2[2].z;

	dG_dmean_2[1].z += dcov2D_dmean[0].z * dcov2D_dmean[0].z * dG_dcov2D_2[0].x +
		2.0f * dcov2D_dmean[0].z * dcov2D_dmean[1].z * (dG_dcov2D_2[0].y + dG_dcov2D_2[1].x) +
		2.0f * dcov2D_dmean[1].z * dcov2D_dmean[1].z * dG_dcov2D_2[1].y +
		dcov2D_dmean[0].z * dcov2D_dmean[2].z * (dG_dcov2D_2[0].z + dG_dcov2D_2[2].x) +
		2.0f * dcov2D_dmean[1].z * dcov2D_dmean[2].z * (dG_dcov2D_2[1].z + dG_dcov2D_2[2].y) +
		dcov2D_dmean[2].z * dcov2D_dmean[2].z * dG_dcov2D_2[2].z;


	

	float3x2 dG_dcov2D_dmean2D;

	dG_dcov2D_dmean2D[0].x = (
		dG_dconic2D_dmean2D[0].x * dconic2D_dcov2D[0].x
		+ 2.0f * dG_dconic2D_dmean2D[1].x * dconic2D_dcov2D[1].x
		+ dG_dconic2D_dmean2D[2].x * dconic2D_dcov2D[2].x
	);
	dG_dcov2D_dmean2D[0].y = (
		dG_dconic2D_dmean2D[0].y * dconic2D_dcov2D[0].x
		+ 2.0f * dG_dconic2D_dmean2D[1].y * dconic2D_dcov2D[1].x
		+ dG_dconic2D_dmean2D[2].y * dconic2D_dcov2D[2].x
	);

	dG_dcov2D_dmean2D[1].x = (
		dG_dconic2D_dmean2D[0].x * dconic2D_dcov2D[0].y
		+ dG_dconic2D_dmean2D[1].x * dconic2D_dcov2D[1].y
		+ dG_dconic2D_dmean2D[2].x * dconic2D_dcov2D[2].y
	);
	dG_dcov2D_dmean2D[1].y = (
		dG_dconic2D_dmean2D[0].y * dconic2D_dcov2D[0].y
		+ dG_dconic2D_dmean2D[1].y * dconic2D_dcov2D[1].y
		+ dG_dconic2D_dmean2D[2].y * dconic2D_dcov2D[2].y
	);
		
	dG_dcov2D_dmean2D[2].x = (
		dG_dconic2D_dmean2D[0].x * dconic2D_dcov2D[0].z
		+ 2.0f * dG_dconic2D_dmean2D[1].x * dconic2D_dcov2D[1].z
		+ dG_dconic2D_dmean2D[2].x * dconic2D_dcov2D[2].z
	);
	dG_dcov2D_dmean2D[2].y = (
		dG_dconic2D_dmean2D[0].y * dconic2D_dcov2D[0].z
		+ 2.0f * dG_dconic2D_dmean2D[1].y * dconic2D_dcov2D[1].z
		+ dG_dconic2D_dmean2D[2].y * dconic2D_dcov2D[2].z
	);





	float2x3 dG_dmean2D_dcov2D;

	dG_dmean2D_dcov2D[0].x = (
		dconic2D_dcov2D[0].x * dG_dmean2D_dconic2D[0].x 
		+ 2.0f * dconic2D_dcov2D[1].x * dG_dmean2D_dconic2D[0].y 
		+ dconic2D_dcov2D[2].x * dG_dmean2D_dconic2D[0].z
	);
	dG_dmean2D_dcov2D[0].y = (
		dconic2D_dcov2D[0].y * dG_dmean2D_dconic2D[0].x 
		+ 2.0f * dconic2D_dcov2D[1].y * dG_dmean2D_dconic2D[0].y 
		+ dconic2D_dcov2D[2].y * dG_dmean2D_dconic2D[0].z
	);
	dG_dmean2D_dcov2D[0].z = (
		dconic2D_dcov2D[0].z * dG_dmean2D_dconic2D[0].x 
		+ 2.0f * dconic2D_dcov2D[1].z * dG_dmean2D_dconic2D[0].y 
		+ dconic2D_dcov2D[2].z * dG_dmean2D_dconic2D[0].z
	);

	dG_dmean2D_dcov2D[1].x = (
		dconic2D_dcov2D[0].x * dG_dmean2D_dconic2D[1].x 
		+ 2.0f * dconic2D_dcov2D[1].x * dG_dmean2D_dconic2D[1].y 
		+ dconic2D_dcov2D[2].x * dG_dmean2D_dconic2D[1].z
	);
	dG_dmean2D_dcov2D[1].y = (
		dconic2D_dcov2D[0].y * dG_dmean2D_dconic2D[1].x 
		+ 2.0f * dconic2D_dcov2D[1].y * dG_dmean2D_dconic2D[1].y 
		+ dconic2D_dcov2D[2].y * dG_dmean2D_dconic2D[1].z
	);
	dG_dmean2D_dcov2D[1].z = (
		dconic2D_dcov2D[0].z * dG_dmean2D_dconic2D[1].x 
		+ 2.0f * dconic2D_dcov2D[1].z * dG_dmean2D_dconic2D[1].y 
		+ dconic2D_dcov2D[2].z * dG_dmean2D_dconic2D[1].z
	);





	const float term_1 = dmean2D_dmean3D[0].z * dG_dcov2D_dmean2D[0].x + dmean2D_dmean3D[1].z * dG_dcov2D_dmean2D[0].y;
	const float term_2 = dmean2D_dmean3D[0].z * dG_dcov2D_dmean2D[1].x + dmean2D_dmean3D[1].z * dG_dcov2D_dmean2D[1].y;
	const float term_3 = dmean2D_dmean3D[0].z * dG_dcov2D_dmean2D[2].x + dmean2D_dmean3D[1].z * dG_dcov2D_dmean2D[2].y;
	const float term_4 = dmean2D_dmean3D[0].y * dG_dcov2D_dmean2D[0].x + dmean2D_dmean3D[1].y * dG_dcov2D_dmean2D[0].y;
	const float term_5 = dmean2D_dmean3D[0].y * dG_dcov2D_dmean2D[1].x + dmean2D_dmean3D[1].y * dG_dcov2D_dmean2D[1].y;
	const float term_6 = dmean2D_dmean3D[0].y * dG_dcov2D_dmean2D[2].x + dmean2D_dmean3D[1].y * dG_dcov2D_dmean2D[2].y;

	dG_dmean_2[0].x += (
		dcov2D_dmean[0].x * (dmean2D_dmean3D[0].x * dG_dcov2D_dmean2D[0].x + dmean2D_dmean3D[1].x * dG_dcov2D_dmean2D[0].y)
		+ 2.0f * dcov2D_dmean[1].x * (dmean2D_dmean3D[0].x * dG_dcov2D_dmean2D[1].x + dmean2D_dmean3D[1].x * dG_dcov2D_dmean2D[1].y)
		+ dcov2D_dmean[2].x * (dmean2D_dmean3D[0].x * dG_dcov2D_dmean2D[2].x + dmean2D_dmean3D[1].x * dG_dcov2D_dmean2D[2].y)
	);
	dG_dmean_2[0].y += (
		dcov2D_dmean[0].x * (term_4)
		+ 2.0f * dcov2D_dmean[1].x * (term_5)
		+ dcov2D_dmean[2].x * (term_6)
	);

	dG_dmean_2[0].z += (
		dcov2D_dmean[0].x * (term_1)
		+ 2.0f * dcov2D_dmean[1].x * (term_2)
		+ dcov2D_dmean[2].x * (term_3)
	);

	dG_dmean_2[1].x += (
		dcov2D_dmean[0].y * (term_4)
		+ 2.0f * dcov2D_dmean[1].y * (term_5)
		+ dcov2D_dmean[2].y * (term_6)
	);
	dG_dmean_2[1].y += (
		dcov2D_dmean[0].y * (term_1)
		+ 2.0f * dcov2D_dmean[1].y * (term_2)
		+ dcov2D_dmean[2].y * (term_3)
	);
	dG_dmean_2[1].z += (
		dcov2D_dmean[0].z * (term_1)
		+ 2.0f * dcov2D_dmean[1].z * (term_2)
		+ dcov2D_dmean[2].z * (term_3)
	);

	const float term_7 = dmean2D_dmean3D[0].x * dG_dmean2D_dcov2D[0].x + dmean2D_dmean3D[1].x * dG_dmean2D_dcov2D[1].x;
	const float term_8 = dmean2D_dmean3D[0].x * dG_dmean2D_dcov2D[0].y + dmean2D_dmean3D[1].x * dG_dmean2D_dcov2D[1].y;
	const float term_9 = dmean2D_dmean3D[0].x * dG_dmean2D_dcov2D[0].z + dmean2D_dmean3D[1].x * dG_dmean2D_dcov2D[1].z;
	const float term_10 = dmean2D_dmean3D[0].y * dG_dmean2D_dcov2D[0].x + dmean2D_dmean3D[1].y * dG_dmean2D_dcov2D[1].x;
	const float term_11 = dmean2D_dmean3D[0].y * dG_dmean2D_dcov2D[0].y + dmean2D_dmean3D[1].y * dG_dmean2D_dcov2D[1].y;
	const float term_12 = dmean2D_dmean3D[0].y * dG_dmean2D_dcov2D[0].z + dmean2D_dmean3D[1].y * dG_dmean2D_dcov2D[1].z;



	dG_dmean_2[0].x += (
		dcov2D_dmean[0].x * (term_7)
		+ 2.0f * dcov2D_dmean[1].x * (term_8)
		+ dcov2D_dmean[2].x * (term_9)
	);
	dG_dmean_2[0].y += (
		dcov2D_dmean[0].y * (term_7)
		+ 2.0f * dcov2D_dmean[1].y * (term_8)
		+ dcov2D_dmean[2].y * (term_9)
	);
	dG_dmean_2[0].z += (
		dcov2D_dmean[0].z * (term_7)
		+ 2.0f * dcov2D_dmean[1].z * (term_8)
		+ dcov2D_dmean[2].z * (term_9)
	);

	dG_dmean_2[1].x += (
		dcov2D_dmean[0].y * (term_10)
		+ 2.0f * dcov2D_dmean[1].y * (term_11)
		+ dcov2D_dmean[2].y * (term_12)
	);

	dG_dmean_2[1].y += (
		dcov2D_dmean[0].z * (term_10)
		+ 2.0f * dcov2D_dmean[1].z * (term_11)
		+ dcov2D_dmean[2].z * (term_12)
	);

	dG_dmean_2[1].z += (
		dcov2D_dmean[0].z * (dmean2D_dmean3D[0].z * dG_dmean2D_dcov2D[0].x + dmean2D_dmean3D[1].z * dG_dmean2D_dcov2D[1].x)
		+ 2.0f * dcov2D_dmean[1].z * (dmean2D_dmean3D[0].z * dG_dmean2D_dcov2D[0].y + dmean2D_dmean3D[1].z * dG_dmean2D_dcov2D[1].y)
		+ dcov2D_dmean[2].z * (dmean2D_dmean3D[0].z * dG_dmean2D_dcov2D[0].z + dmean2D_dmean3D[1].z * dG_dmean2D_dcov2D[1].z)
	);





// Backward version of the rendering procedure.
template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float* __restrict__ bg_color,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ conic_opacity,
	const float* __restrict__ colors,
	const float* __restrict__ depths,
	const float* __restrict__ final_Ts,
	const uint32_t* __restrict__ n_contrib,



	float* __restrict__ dpixel_dcolors,
	float3* __restrict__ dpixel_dG,
	

	float2* __restrict__ dG_dmean2D,
	float3* __restrict__ dG_dmean2D_2,

	float3* __restrict__ dG_dconic2D,

	float3x3* __restrict__ dG_dconic2D_dcov2D,
	float2x3* __restrict__ dG_dmean2D_dconic2D,
	float3X2* __restrict__ dG_dconic2D_dmean2D,

	// for dG_dconic2D_2, a 2,2,2,2 hessian matrix looks like:
	// [ab][bd]
	// [bc][de]
	// [bd][ce]
	// [de][ef]

	// for dG_dmean2D_dconic2D, a 2*2*2 hessian matrix looks like:
	// [a][b]
	// [b][c]

	// [d][e]
	// [e][f]

	// for dG_dconic2D_dcov2D, a 2*2*2*2 hessian matrix looks like:
	// [a b][d e]
	// [b c][e f]
	
	// [d e][g h]
	// [e f][h i]

	float* __restrict__ dpixel_dopacity,

	// float3* __restrict__ dG_dconic2D_2_abc,
	// float3* __restrict__ dG_dconic2D_2_def,
)
{
	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();

	//水平方向block个数
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;

	//block负责的像素块左下角像素的坐标
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	//block负责的像素块右上角像素的坐标
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };

	//thread负责的像素的坐标
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	//thread负责的像素idx
	const uint32_t pix_id = W * pix.y + pix.x;

	//i * H * W + pix_id 代表 thread负责的像素在第i个通道的内存idx
	//block.thread_rank() 代表 thread在block中的局部idx

	const float2 pixf = { (float)pix.x, (float)pix.y };

	const bool inside = pix.x < W&& pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float collected_colors[C * BLOCK_SIZE];
	__shared__ float collected_depths[BLOCK_SIZE];


	// In the forward, we stored the final value for T, the
	// product of all (1 - alpha) factors. 
	const float T_final = inside ? final_Ts[pix_id] : 0;
	float T = T_final;

	// We start from the back. The ID of the last contributing
	// Gaussian is known from each pixel from the forward.
	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;

	float accum_rec[C] = { 0 };

	float last_alpha = 0;
	float last_color[C] = { 0 };


	// Traverse all Gaussians
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Load auxiliary data into shared memory, start in the BACK
		// and load them in revers order.
		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			for (int i = 0; i < C; i++)
				collected_colors[i * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + i];

			if(dL_invdepths)
			collected_depths[block.thread_rank()] = depths[coll_id];
		}
		block.sync();

		// Iterate over Gaussians
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current Gaussian ID. Skip, if this one
			// is behind the last contributor for this pixel.
			contributor--;
			if (contributor >= last_contributor)
				continue;

			const int global_id = collected_id[j];

			// Compute blending values, as before.
			const float2 xy = collected_xy[j];
			const float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			const float4 con_o = collected_conic_opacity[j];
			const float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			const float G = exp(power);
			const float alpha = min(0.99f, con_o.w * G);
			if (alpha < 1.0f / 255.0f)
				continue;

			T = T / (1.f - alpha);

			// Propagate gradients to per-Gaussian colors and keep
			// gradients w.r.t. alpha (blending factor for a Gaussian/pixel
			// pair).
			float3 dpixel_dalpha = 0.0f;
			float* dpixel_dalpha_ptr = (float*)&dpixel_dalpha;

			

			for (int ch = 0; ch < C; ch++)
			{
				const float c = collected_colors[ch * BLOCK_SIZE + j];
				// Update last color (to be used in the next iteration)
				accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
				last_color[ch] = c;

				dpixel_dalpha_ptr[ch] = (c - accum_rec[ch]);
				// Update the gradients w.r.t. color of the Gaussian. 
				// Atomic, since this pixel is just one of potentially
				// many that were affected by this Gaussian.
			}
			atomicAdd(&(dpixel_dcolors[global_id]), alpha * T);
			dpixel_dalpha *= T;

			// Update last alpha (to be used in the next iteration)
			last_alpha = alpha;

			for (int i = 0; i < C; i++)
				dpixel_dalpha_ptr[i] += (-T_final / (1.f - alpha)) * bg_color[i];
			

			

			const float gdx = G * d.x;
			const float gdy = G * d.y;

			float* dpixel_dG_ptr = (float*)&dpixel_dG[global_id];
			for (int i = 0; i < C; ++i){
				dpixel_dG_ptr[i] = con_o.w * dpixel_dalpha_ptr[i]
			}
			

			const float3 dx_2 = {d.x * d.x, d.x * d.y, d.y * d.y};


			const float W_div_2 = 0.5f * W;
			const float H_div_2 = 0.5f * H;

			// Update gradients w.r.t. 2D mean position of the Gaussian
			atomicAdd(&dG_dmean2D[global_id].x, (-gdx * con_o.x - gdy * con_o.y) * W_div_2);
			atomicAdd(&dG_dmean2D[global_id].y, (-gdy * con_o.z - gdx * con_o.y) * H_div_2);


			const float conic_y_2 = con_o.y * con_o.y;
			const float term1 = (-1.f + 2.f * con_o.y * dx_2.y);
			atomicAdd(&dG_dmean2D_2[global_id].x, G * (con_o.x * con_o.x * dx_2.x + conic_y_2 * dx_2.z + con_o.x * term1) * W_div_2 * W_div_2);
			atomicAdd(&dG_dmean2D_2[global_id].y, G * (conic_y_2 * dx_2.y + con_o.x * con_o.z * dx_2.y + con_o.y * (-1.f + con_o.x * dx_2.x + con_o.z * dx_2.z)) * W_div_2 * H_div_2);
			atomicAdd(&dG_dmean2D_2[global_id].z, G * (conic_y_2 * dx_2.x + con_o.z * con_o.z * dx_2.z + con_o.z * term1) * H_div_2 * H_div_2);


			// Update gradients w.r.t. 2D covariance (2x2 matrix, symmetric)
			atomicAdd(&dG_dconic2D[global_id].x, -0.5f * gdx * d.x);
			atomicAdd(&dG_dconic2D[global_id].y, -0.5f * gdx * d.y);
			atomicAdd(&dG_dconic2D[global_id].z, -0.5f * gdy * d.y);
			
			const float term2 = 0.25f * G * dx_2.x;
			const float term3 = 0.25f * G * dx_2.z;
			
			// atomicAdd(&dG_dconic2D_2_abc[global_id].x, term2 * dx_2.x);
			// atomicAdd(&dG_dconic2D_2_abc[global_id].y, term2 * dx_2.y);
			// atomicAdd(&dG_dconic2D_2_abc[global_id].z, term2 * dx_2.z);
			// atomicAdd(&dG_dconic2D_2_def[global_id].x, term2 * dx_2.z);
			// atomicAdd(&dG_dconic2D_2_def[global_id].y, term3 * dx_2.y);
			// atomicAdd(&dG_dconic2D_2_def[global_id].z, term3 * dx_2.z);


			const float term4 = con_o.x * dx_2.x + con_o.y * dx_2.y;
			const float term5 = con_o.y * dx_2.y + con_o.z * dx_2.z;
			const float term6 = G * d.x * 0.5f;
			const float term7 = G * d.y * 0.5f;


			const float common_factor = -0.25f * exp(-0.5f * (term4 + term5));
			const float p1 = con_o.x * d.x + con_o.y * d.y;
			const float p2 = con_o.y * d.x + con_o.z * d.y;

			const float p1_sq = p1 * p1;
			const float p2_sq = p2 * p2;
			const float p1_p2 = p1 * p2;
		
			atomicAdd(&dG_dmean2D_dconic2D[global_id][0].x, term6 * (-2.f + term4));
			atomicAdd(&dG_dmean2D_dconic2D[global_id][0].y, term7 * (-1.f + term4));
			atomicAdd(&dG_dmean2D_dconic2D[global_id][0].z, term3 * 2.f * p1);
			atomicAdd(&dG_dmean2D_dconic2D[global_id][1].x, term2 * 2.f * p2);
			atomicAdd(&dG_dmean2D_dconic2D[global_id][1].y, term6 * (-1.f + term5));
			atomicAdd(&dG_dmean2D_dconic2D[global_id][1].z, term7 * (-2.f + term5));




			atomicAdd(dG_dconic2D_dcov2D[global_id][0].x, common_factor * p1_sq * dx_2.x);
			atomicAdd(dG_dconic2D_dcov2D[global_id][0].y, common_factor * p1_p2 * dx_2.x);
			atomicAdd(dG_dconic2D_dcov2D[global_id][0].z, common_factor * p2_sq * dx_2.x);

			atomicAdd(dG_dconic2D_dcov2D[global_id][1].x, common_factor * p1_sq * dx_2.y);
			atomicAdd(dG_dconic2D_dcov2D[global_id][1].y, common_factor * p1_p2 * dx_2.y);
			atomicAdd(dG_dconic2D_dcov2D[global_id][1].z, common_factor * p2_sq * dx_2.y);

			atomicAdd(dG_dconic2D_dcov2D[global_id][2].x, common_factor * p1_sq * dx_2.z);
			atomicAdd(dG_dconic2D_dcov2D[global_id][2].y, common_factor * p1_p2 * dx_2.z);
			atomicAdd(dG_dconic2D_dcov2D[global_id][2].z, common_factor * p2_sq * dx_2.z);



			atomicAdd(&dG_dconic2D_dmean2D[global_id][0].x, term6 * W_div_2 * (-2.f + term4));
			atomicAdd(&dG_dconic2D_dmean2D[global_id][0].y, term2 * H * p2);

			atomicAdd(&dG_dconic2D_dmean2D[global_id][1].x, term7 * W_div_2 * (-1.f + term4));
			atomicAdd(&dG_dconic2D_dmean2D[global_id][1].y, term6 * H_div_2 * (-1.f + term5));

			atomicAdd(&dG_dconic2D_dmean2D[global_id][2].x, term3 * W * p1);
			atomicAdd(&dG_dconic2D_dmean2D[global_id][2].y, term7 * H_div_2 * (-2.f + term5));




			// Update gradients w.r.t. opacity of the Gaussian
			atomicAdd(&(dL_dopacity[global_id]), G * dL_dalpha);
		}
	}
}

void BACKWARD::preprocess(
	int P, int D, int M,
	const float3* means3D,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const float* opacities,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* cov3Ds,
	const float* viewmatrix,
	const float* projmatrix,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	const glm::vec3* campos,
	const float3* dL_dmean2D,
	const float* dL_dconic,
	const float* dL_dinvdepth,
	float* dL_dopacity,
	glm::vec3* dL_dmean3D,
	float* dL_dcolor,
	float* dL_dcov3D,
	float* dL_dsh,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot,
	bool antialiasing)
{
	// Propagate gradients for the path of 2D conic matrix computation. 
	// Somewhat long, thus it is its own kernel rather than being part of 
	// "preprocess". When done, loss gradient w.r.t. 3D means has been
	// modified and gradient w.r.t. 3D covariance matrix has been computed.	
	computeCov2DCUDA << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		radii,
		cov3Ds,
		focal_x,
		focal_y,
		tan_fovx,
		tan_fovy,
		viewmatrix,
		opacities,
		dL_dconic,
		dL_dopacity,
		dL_dinvdepth,
		(float3*)dL_dmean3D,
		dL_dcov3D,
		antialiasing);

	// Propagate gradients for remaining steps: finish 3D mean gradients,
	// propagate color gradients to SH (if desireD), propagate 3D covariance
	// matrix gradients to scale and rotation.
	preprocessCUDA<NUM_CHANNELS> << < (P + 255) / 256, 256 >> > (
		P, D, M,
		(float3*)means3D,
		radii,
		shs,
		clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		projmatrix,
		campos,
		(float3*)dL_dmean2D,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_dcov3D,
		dL_dsh,
		dL_dscale,
		dL_drot,
		dL_dopacity);
}

void BACKWARD::render(
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
	const float* dL_dpixels,
	const float* dL_invdepths,
	float3* dL_dmean2D,
	float4* dL_dconic2D,
	float* dL_dopacity,
	float* dL_dcolors,
	float* dL_dinvdepths)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> >(
		ranges,
		point_list,
		W, H,
		bg_color,
		means2D,
		conic_opacity,
		colors,
		depths,
		final_Ts,
		n_contrib,
		dL_dpixels,
		dL_invdepths,
		dL_dmean2D,
		dL_dconic2D,
		dL_dopacity,
		dL_dcolors,
		dL_dinvdepths
		);
}
