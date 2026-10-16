	.amdgcn_target "amdgcn-amd-amdhsa--gfx1151"
	.amdhsa_code_object_version 6
	.text
	.protected	_Z12gates_kernelPKfS0_S0_S0_PfS1_ii ; -- Begin function _Z12gates_kernelPKfS0_S0_S0_PfS1_ii
	.globl	_Z12gates_kernelPKfS0_S0_S0_PfS1_ii
	.p2align	8
	.type	_Z12gates_kernelPKfS0_S0_S0_PfS1_ii,@function
_Z12gates_kernelPKfS0_S0_S0_PfS1_ii:    ; @_Z12gates_kernelPKfS0_S0_S0_PfS1_ii
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_b32 s12, s[0:1], 0x34
	s_ashr_i32 s3, s2, 31
	s_load_b256 s[4:11], s[0:1], 0x0
	s_lshl_b64 s[2:3], s[2:3], 4
	v_lshlrev_b32_e32 v5, 2, v0
	s_waitcnt lgkmcnt(0)
	s_cmp_lg_u32 s12, 1
	s_cselect_b32 s2, s2, 0
	s_cselect_b32 s3, s3, 0
	v_add_co_u32 v1, s2, s2, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v2, null, s3, 0, s2
	s_load_b128 s[0:3], s[0:1], 0x20
	v_lshlrev_b64 v[1:2], 2, v[1:2]
	v_add_co_u32 v3, vcc_lo, s4, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v4, null, s5, v2, vcc_lo
	s_mov_b32 s4, exec_lo
	global_load_b32 v3, v[3:4], off
	global_load_b32 v4, v5, s[10:11]
	s_waitcnt vmcnt(0)
	v_add_f32_e32 v3, v3, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_nlt_f32_e32 0x41a00000, v3
	s_cbranch_execz .LBB0_2
; %bb.1:
	v_mul_f32_e32 v4, 0x3fb8aa3b, v3
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v3
	s_mov_b32 s5, 0x3e9b6dac
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_rndne_f32_e32 v5, v4
	v_fma_f32 v6, 0x3fb8aa3b, v3, -v4
	v_sub_f32_e32 v4, v4, v5
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fmamk_f32 v6, v3, 0x32a5705f, v6
	v_cvt_i32_f32_e32 v5, v5
	v_add_f32_e32 v4, v4, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v4, v4
	v_ldexp_f32 v4, v4, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v4, 0, v4, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v3
	v_cndmask_b32_e32 v3, 0x7f800000, v4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v4, 1.0, v3
	v_frexp_mant_f32_e32 v5, v4
	v_frexp_exp_i32_f32_e32 v6, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_gt_f32_e32 vcc_lo, 0x3f2aaaab, v5
	v_subrev_co_ci_u32_e64 v5, null, 0, v6, vcc_lo
	v_add_f32_e32 v6, -1.0, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_sub_nc_u32_e32 v7, 0, v5
	v_cvt_f32_i32_e32 v5, v5
	v_sub_f32_e32 v8, v6, v4
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_ldexp_f32 v4, v4, v7
	v_sub_f32_e32 v6, v3, v6
	v_add_f32_e32 v8, 1.0, v8
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_f32_e32 v9, 1.0, v4
	v_add_f32_e32 v6, v6, v8
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_f32_e32 v8, -1.0, v9
	v_cmp_neq_f32_e32 vcc_lo, 0x7f800000, v3
	v_ldexp_f32 v6, v6, v7
	v_add_f32_e32 v7, -1.0, v4
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_sub_f32_e32 v8, v4, v8
	v_add_f32_e32 v10, 1.0, v7
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_f32_e32 v8, v6, v8
	v_sub_f32_e32 v4, v4, v10
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_f32_e32 v10, v9, v8
	v_add_f32_e32 v4, v6, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rcp_f32_e32 v6, v10
	v_sub_f32_e32 v9, v9, v10
	v_dual_add_f32 v11, v7, v4 :: v_dual_add_f32 v8, v8, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_sub_f32_e32 v7, v7, v11
	v_mul_f32_e32 v12, v11, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_add_f32 v4, v4, v7 :: v_dual_mul_f32 v13, v10, v12
	v_fma_f32 v9, v12, v10, -v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v9, v12, v8
	v_add_f32_e32 v14, v13, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_sub_f32_e32 v15, v11, v14
	v_sub_f32_e32 v7, v14, v13
	v_sub_f32_e32 v11, v11, v15
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_sub_f32_e32 v7, v7, v9
	v_sub_f32_e32 v11, v11, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v4, v4, v11
	v_add_f32_e32 v4, v7, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v7, v15, v4
	v_mul_f32_e32 v9, v6, v7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_sub_f32 v14, v15, v7 :: v_dual_mul_f32 v11, v10, v9
	v_add_f32_e32 v4, v4, v14
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v10, v9, v10, -v11
	v_fmac_f32_e32 v10, v9, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v8, v11, v10
	v_sub_f32_e32 v13, v7, v8
	v_sub_f32_e32 v11, v8, v11
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_sub_f32_e32 v7, v7, v13
	v_sub_f32_e32 v7, v7, v8
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_sub_f32_e32 v8, v11, v10
	v_add_f32_e32 v4, v4, v7
	v_add_f32_e32 v7, v12, v9
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_f32_e32 v4, v8, v4
	v_sub_f32_e32 v8, v7, v12
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_f32_e32 v4, v13, v4
	v_sub_f32_e32 v8, v9, v8
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v4, v6, v4
	v_add_f32_e32 v4, v8, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v6, v7, v4
	v_mul_f32_e32 v8, v6, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fmaak_f32 v9, s5, v8, 0x3ecc95a3
	v_mul_f32_e32 v10, v6, v8
	v_dual_fmaak_f32 v8, v8, v9, 0x3f2aaada :: v_dual_add_f32 v9, v6, v6
	v_sub_f32_e32 v6, v6, v7
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v8, v10, v8
	v_mul_f32_e32 v10, 0x3f317218, v5
	v_dual_sub_f32 v4, v4, v6 :: v_dual_add_f32 v7, v9, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_f32_e32 v4, v4, v4
	v_sub_f32_e32 v6, v7, v9
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v9, 0x3f317218, v5, -v10
	v_dual_sub_f32 v6, v8, v6 :: v_dual_fmamk_f32 v5, v5, 0xb102e308, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_f32_e32 v4, v4, v6
	v_add_f32_e32 v6, v10, v5
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_f32_e32 v8, v7, v4
	v_sub_f32_e32 v10, v6, v10
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_f32_e32 v9, v6, v8
	v_sub_f32_e32 v7, v8, v7
	v_sub_f32_e32 v5, v5, v10
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_sub_f32 v11, v9, v6 :: v_dual_sub_f32 v4, v4, v7
	v_sub_f32_e32 v12, v9, v11
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_sub_f32 v7, v8, v11 :: v_dual_add_f32 v8, v5, v4
	v_sub_f32_e32 v6, v6, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_add_f32 v6, v7, v6 :: v_dual_sub_f32 v7, v8, v5
	v_add_f32_e32 v6, v8, v6
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_sub_f32_e32 v8, v8, v7
	v_sub_f32_e32 v4, v4, v7
	v_add_f32_e32 v10, v9, v6
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_sub_f32_e32 v5, v5, v8
	v_sub_f32_e32 v7, v10, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_add_f32 v4, v4, v5 :: v_dual_sub_f32 v5, v6, v7
	v_add_f32_e32 v4, v4, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v4, v10, v4
	v_cndmask_b32_e32 v4, 0x7f800000, v4, vcc_lo
	v_cmp_gt_f32_e64 vcc_lo, 0x33800000, |v3|
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v3, v4, v3, vcc_lo
.LBB0_2:
	s_or_b32 exec_lo, exec_lo, s4
	v_lshlrev_b32_e32 v0, 2, v0
	global_load_b32 v0, v0, s[8:9]
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v4, 0x3fb8aa3b, v0
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2ce8ed0, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fma_f32 v5, 0x3fb8aa3b, v0, -v4
	v_rndne_f32_e32 v6, v4
	v_fmamk_f32 v5, v0, 0x32a5705f, v5
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_sub_f32_e32 v4, v4, v6
	v_add_f32_e32 v4, v4, v5
	v_cvt_i32_f32_e32 v5, v6
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v4, v4
	v_ldexp_f32 v4, v4, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v4, 0, v4, vcc_lo
	v_cmp_nlt_f32_e32 vcc_lo, 0x42b17218, v0
	v_cndmask_b32_e32 v0, 0x7f800000, v4, vcc_lo
	s_waitcnt lgkmcnt(0)
	v_add_co_u32 v4, vcc_lo, s0, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v5, null, s1, v2, vcc_lo
	v_add_co_u32 v6, vcc_lo, s6, v1
	v_mul_f32_e64 v0, v3, -v0
	v_add_co_ci_u32_e64 v7, null, s7, v2, vcc_lo
	global_store_b32 v[4:5], v0, off
	global_load_b32 v0, v[6:7], off
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v3, 0xbfb8aa3b, v0
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v4, 0xbfb8aa3b, v0, -v3
	v_rndne_f32_e32 v5, v3
	v_dual_fmamk_f32 v4, v0, 0xb2a5705f, v4 :: v_dual_sub_f32 v3, v3, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v3, v3, v4
	v_cvt_i32_f32_e32 v4, v5
	v_exp_f32_e32 v3, v3
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ldexp_f32 v3, v3, v4
	v_cndmask_b32_e32 v3, 0, v3, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v0, 0x7f800000, v3, vcc_lo
	v_add_f32_e32 v0, 1.0, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_scale_f32 v3, null, v0, v0, 1.0
	v_div_scale_f32 v6, vcc_lo, 1.0, v0, 1.0
	v_rcp_f32_e32 v4, v3
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v5, -v3, v4, 1.0
	v_fmac_f32_e32 v4, v5, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v5, v6, v4
	v_fma_f32 v7, -v3, v5, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v5, v7, v4
	v_fma_f32 v3, -v3, v5, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v3, v3, v4, v5
	v_div_fixup_f32 v3, v3, v0, 1.0
	v_add_co_u32 v0, vcc_lo, s2, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v1, null, s3, v2, vcc_lo
	global_store_b32 v[0:1], v3, off
	s_endpgm
.Lfunc_end0:
	.size	_Z12gates_kernelPKfS0_S0_S0_PfS1_ii, .Lfunc_end0-_Z12gates_kernelPKfS0_S0_S0_PfS1_ii
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z12gates_kernelPKfS0_S0_S0_PfS1_ii
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 56
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 16
		.amdhsa_next_free_sgpr 13
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_shared_vgpr_count 0
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end0-_Z12gates_kernelPKfS0_S0_S0_PfS1_ii)<<4)&1008)>>4
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
                                        ; -- End function
	.set .L_Z12gates_kernelPKfS0_S0_S0_PfS1_ii.num_vgpr, 16
	.set .L_Z12gates_kernelPKfS0_S0_S0_PfS1_ii.num_agpr, 0
	.set .L_Z12gates_kernelPKfS0_S0_S0_PfS1_ii.numbered_sgpr, 13
	.set .L_Z12gates_kernelPKfS0_S0_S0_PfS1_ii.num_named_barrier, 0
	.set .L_Z12gates_kernelPKfS0_S0_S0_PfS1_ii.private_seg_size, 0
	.set .L_Z12gates_kernelPKfS0_S0_S0_PfS1_ii.uses_vcc, 1
	.set .L_Z12gates_kernelPKfS0_S0_S0_PfS1_ii.uses_flat_scratch, 0
	.set .L_Z12gates_kernelPKfS0_S0_S0_PfS1_ii.has_dyn_sized_stack, 0
	.set .L_Z12gates_kernelPKfS0_S0_S0_PfS1_ii.has_recursion, 0
	.set .L_Z12gates_kernelPKfS0_S0_S0_PfS1_ii.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 1324
; TotalNumSgprs: 15
; NumVgprs: 16
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 1
; NumSGPRsForWavesPerEU: 15
; NumVGPRsForWavesPerEU: 16
; Occupancy: 16
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	_Z11conv_kernelPKfS0_S0_PfS1_ii ; -- Begin function _Z11conv_kernelPKfS0_S0_PfS1_ii
	.globl	_Z11conv_kernelPKfS0_S0_PfS1_ii
	.p2align	8
	.type	_Z11conv_kernelPKfS0_S0_PfS1_ii,@function
_Z11conv_kernelPKfS0_S0_PfS1_ii:        ; @_Z11conv_kernelPKfS0_S0_PfS1_ii
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_mov_b32 s12, s3
	s_load_b32 s3, s[0:1], 0x3c
	s_waitcnt lgkmcnt(0)
	s_and_b32 s3, s3, 0xffff
	s_delay_alu instid0(SALU_CYCLE_1)
	v_mad_u64_u32 v[5:6], null, s2, s3, v[0:1]
	s_mov_b32 s2, exec_lo
	v_cmpx_gt_i32_e32 0x1800, v5
	s_cbranch_execz .LBB1_124
; %bb.1:
	s_clause 0x1
	s_load_b128 s[8:11], s[0:1], 0x20
	s_load_b256 s[0:7], s[0:1], 0x0
	v_ashrrev_i32_e32 v6, 31, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b64 v[3:4], 2, v[5:6]
	s_waitcnt lgkmcnt(0)
	v_mad_i64_i32 v[0:1], null, v5, s10, 0
	s_cmp_lg_u32 s11, 4
	s_cselect_b32 s16, -1, 0
	s_cmp_lg_u32 s11, 1
	s_cselect_b32 s17, -1, 0
	s_add_i32 s14, s12, -3
	v_lshlrev_b64 v[6:7], 2, v[0:1]
	v_add_co_u32 v0, vcc_lo, s0, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v1, null, s1, v4, vcc_lo
	s_cmp_gt_i32 s12, 2
	v_add_co_u32 v10, vcc_lo, s0, v6
	v_add_co_ci_u32_e64 v12, null, s1, v7, vcc_lo
	s_cbranch_scc0 .LBB1_4
; %bb.2:
	s_and_b32 vcc_lo, exec_lo, s17
	s_mov_b32 s15, 0
	s_cbranch_vccz .LBB1_5
; %bb.3:
	v_mad_u64_u32 v[6:7], null, 0x6000, s14, v[0:1]
	s_mov_b32 s0, s15
	global_load_b32 v11, v[6:7], off
	s_and_not1_b32 vcc_lo, exec_lo, s0
	s_cbranch_vccz .LBB1_6
	s_branch .LBB1_7
.LBB1_4:
	s_mov_b32 s1, -1
                                        ; implicit-def: $vgpr11
	s_branch .LBB1_8
.LBB1_5:
                                        ; implicit-def: $vgpr11
.LBB1_6:
	s_lshl_b64 s[0:1], s[14:15], 2
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v6, vcc_lo, v10, s0
	v_add_co_ci_u32_e64 v7, null, s1, v12, vcc_lo
	global_load_b32 v11, v[6:7], off
.LBB1_7:                                ; %Flow142
	s_mov_b32 s1, 0
.LBB1_8:                                ; %Flow144
	v_cndmask_b32_e64 v9, 0, 1, s16
	v_add_nc_u32_e32 v2, 0x4800, v5
	s_and_not1_b32 vcc_lo, exec_lo, s1
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_ne_u32_e64 s0, 1, v9
	s_cbranch_vccnz .LBB1_11
; %bb.9:
	s_waitcnt vmcnt(0)
	v_mov_b32_e32 v11, 0
	s_and_b32 vcc_lo, exec_lo, s0
	s_cbranch_vccnz .LBB1_11
; %bb.10:
	v_mad_u64_u32 v[6:7], null, 0x1800, s14, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v7, 31, v6
	v_lshlrev_b64 v[6:7], 2, v[6:7]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v6, vcc_lo, s2, v6
	v_add_co_ci_u32_e64 v7, null, s3, v7, vcc_lo
	global_load_b32 v11, v[6:7], off
.LBB1_11:                               ; %_Z8conv_tapPKfS0_iib.exit.i
	v_lshlrev_b32_e32 v5, 2, v5
	v_cndmask_b32_e64 v14, 0, 1, s17
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v6, 31, v5
	v_lshlrev_b64 v[5:6], 2, v[5:6]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v5, vcc_lo, s4, v5
	v_add_co_ci_u32_e64 v6, null, s5, v6, vcc_lo
	s_add_i32 s4, s12, -2
	s_cmp_lt_i32 s12, 2
	v_cmp_ne_u32_e32 vcc_lo, 1, v14
	global_load_b32 v7, v[5:6], off
	s_cbranch_scc1 .LBB1_14
; %bb.12:
	s_and_b32 vcc_lo, exec_lo, vcc_lo
	s_mov_b32 s5, 0
	s_cbranch_vccnz .LBB1_15
; %bb.13:
	v_mad_u64_u32 v[15:16], null, 0x6000, s4, v[0:1]
	s_mov_b32 s0, s5
	global_load_b32 v13, v[15:16], off
	s_and_not1_b32 vcc_lo, exec_lo, s0
	s_cbranch_vccz .LBB1_16
	s_branch .LBB1_17
.LBB1_14:
                                        ; implicit-def: $vgpr13
	v_cmp_ne_u32_e64 s0, 1, v9
	s_branch .LBB1_18
.LBB1_15:
                                        ; implicit-def: $vgpr13
.LBB1_16:
	s_lshl_b64 s[0:1], s[4:5], 2
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v15, vcc_lo, v10, s0
	v_add_co_ci_u32_e64 v16, null, s1, v12, vcc_lo
	global_load_b32 v13, v[15:16], off
.LBB1_17:                               ; %Flow137
	v_cmp_ne_u32_e64 s0, 1, v9
	s_cbranch_execnz .LBB1_20
.LBB1_18:
	s_waitcnt vmcnt(0)
	v_mov_b32_e32 v13, 0
	s_delay_alu instid0(VALU_DEP_2)
	s_and_b32 vcc_lo, exec_lo, s0
	s_cbranch_vccnz .LBB1_20
; %bb.19:
	v_mad_u64_u32 v[15:16], null, 0x1800, s4, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v16, 31, v15
	v_lshlrev_b64 v[15:16], 2, v[15:16]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v15, vcc_lo, s2, v15
	v_add_co_ci_u32_e64 v16, null, s3, v16, vcc_lo
	global_load_b32 v13, v[15:16], off
.LBB1_20:                               ; %_Z8conv_tapPKfS0_iib.exit.1.i
	global_load_b32 v8, v[5:6], off offset:4
	s_add_i32 s4, s12, -1
	s_cmp_lt_i32 s12, 1
	s_mov_b32 s1, -1
	v_cmp_ne_u32_e32 vcc_lo, 1, v14
                                        ; implicit-def: $vgpr14
	s_cbranch_scc1 .LBB1_26
; %bb.21:
	s_and_b32 vcc_lo, exec_lo, vcc_lo
	s_mov_b32 s5, 0
	s_cbranch_vccnz .LBB1_23
; %bb.22:
	v_mad_u64_u32 v[14:15], null, 0x6000, s4, v[0:1]
	s_mov_b32 s0, s5
	global_load_b32 v14, v[14:15], off
	s_and_not1_b32 vcc_lo, exec_lo, s0
	s_cbranch_vccz .LBB1_24
	s_branch .LBB1_25
.LBB1_23:
                                        ; implicit-def: $vgpr14
.LBB1_24:
	s_lshl_b64 s[0:1], s[4:5], 2
	s_waitcnt vmcnt(0)
	v_add_co_u32 v14, vcc_lo, v10, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v15, null, s1, v12, vcc_lo
	global_load_b32 v14, v[14:15], off
.LBB1_25:                               ; %Flow132
	s_mov_b32 s1, 0
.LBB1_26:                               ; %Flow135
	v_cmp_ne_u32_e64 s0, 1, v9
	s_and_not1_b32 vcc_lo, exec_lo, s1
	s_cbranch_vccnz .LBB1_29
; %bb.27:
	s_waitcnt vmcnt(0)
	v_mov_b32_e32 v14, 0
	s_and_b32 vcc_lo, exec_lo, s0
	s_cbranch_vccnz .LBB1_29
; %bb.28:
	v_mad_u64_u32 v[14:15], null, 0x1800, s4, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v15, 31, v14
	v_lshlrev_b64 v[14:15], 2, v[14:15]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v14, vcc_lo, s2, v14
	v_add_co_ci_u32_e64 v15, null, s3, v15, vcc_lo
	global_load_b32 v14, v[14:15], off
.LBB1_29:                               ; %_Z8conv_tapPKfS0_iib.exit.2.i
	global_load_b32 v9, v[5:6], off offset:8
	s_cmp_lt_i32 s12, 0
	s_mov_b32 s13, 0
	s_cbranch_scc1 .LBB1_32
; %bb.30:
	s_and_not1_b32 vcc_lo, exec_lo, s17
	s_cbranch_vccnz .LBB1_33
; %bb.31:
	v_mad_u64_u32 v[15:16], null, 0x6000, s12, v[0:1]
	global_load_b32 v15, v[15:16], off
	s_cbranch_execz .LBB1_34
	s_branch .LBB1_35
.LBB1_32:
                                        ; implicit-def: $vgpr15
	s_branch .LBB1_36
.LBB1_33:
                                        ; implicit-def: $vgpr15
.LBB1_34:
	s_lshl_b64 s[0:1], s[12:13], 2
	s_waitcnt vmcnt(0)
	v_add_co_u32 v15, vcc_lo, v10, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v16, null, s1, v12, vcc_lo
	global_load_b32 v15, v[15:16], off
.LBB1_35:                               ; %Flow126
	s_cbranch_execnz .LBB1_38
.LBB1_36:
	s_waitcnt vmcnt(0)
	v_mov_b32_e32 v15, 0
	s_and_not1_b32 vcc_lo, exec_lo, s16
	s_cbranch_vccnz .LBB1_38
; %bb.37:
	v_mad_u64_u32 v[15:16], null, 0x1800, s12, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v16, 31, v15
	v_lshlrev_b64 v[15:16], 2, v[15:16]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v15, vcc_lo, s2, v15
	v_add_co_ci_u32_e64 v16, null, s3, v16, vcc_lo
	global_load_b32 v15, v[15:16], off
.LBB1_38:                               ; %_Z8conv_tapPKfS0_iib.exit.3.i
	global_load_b32 v10, v[5:6], off offset:12
	s_waitcnt vmcnt(3)
	v_fma_f32 v5, v11, v7, 0
	s_waitcnt vmcnt(2)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v5, v13, v8
	s_waitcnt vmcnt(1)
	v_fmac_f32_e32 v5, v14, v9
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v5, v15, v10
	v_and_b32_e32 v6, 0x7f800000, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v6
                                        ; implicit-def: $vgpr6
	s_and_saveexec_b32 s0, vcc_lo
	s_xor_b32 s0, exec_lo, s0
; %bb.39:
	v_bfe_u32 v6, v5, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v6, v5, v6, 0x7fff
                                        ; implicit-def: $vgpr5
; %bb.40:                               ; %Flow124
	s_and_not1_saveexec_b32 s0, s0
; %bb.41:
	v_and_b32_e32 v6, 0xffff, v5
	v_or_b32_e32 v11, 0x10000, v5
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v6
	v_cndmask_b32_e32 v6, v11, v5, vcc_lo
; %bb.42:                               ; %_Z10bf16_roundf.exit.i
	s_or_b32 exec_lo, exec_lo, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v5, 0xffff0000, v6
	v_mul_f32_e32 v6, 0xbfb8aa3b, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v11, 0xbfb8aa3b, v5, -v6
	v_rndne_f32_e32 v12, v6
	v_sub_f32_e32 v6, v6, v12
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v5
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmamk_f32 v11, v5, 0xb2a5705f, v11
	v_add_f32_e32 v6, v6, v11
	v_cvt_i32_f32_e32 v11, v12
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v6, v6
	v_ldexp_f32 v6, v6, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v6, 0, v6, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v5
	v_cndmask_b32_e32 v6, 0x7f800000, v6, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v6, 1.0, v6
	v_div_scale_f32 v11, null, v6, v6, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v12, v11
	v_fma_f32 v13, -v11, v12, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v12, v13, v12
	v_div_scale_f32 v13, vcc_lo, v5, v6, v5
	v_mul_f32_e32 v14, v13, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v15, -v11, v14, v13
	v_fmac_f32_e32 v14, v15, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v11, -v11, v14, v13
	v_div_fmas_f32 v11, v11, v12, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v6, v11, v6, v5
	v_and_b32_e32 v5, 0x7f800000, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v5
                                        ; implicit-def: $vgpr5
	s_and_saveexec_b32 s0, vcc_lo
	s_xor_b32 s0, exec_lo, s0
; %bb.43:
	v_bfe_u32 v5, v6, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v5, v6, v5, 0x7fff
                                        ; implicit-def: $vgpr6
; %bb.44:                               ; %Flow123
	s_and_not1_saveexec_b32 s0, s0
; %bb.45:
	v_and_b32_e32 v5, 0xffff, v6
	v_or_b32_e32 v11, 0x10000, v6
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v5
	v_cndmask_b32_e32 v5, v11, v6, vcc_lo
; %bb.46:                               ; %_Z10conv_valuePKfS0_S0_iiibb.exit
	s_or_b32 exec_lo, exec_lo, s0
	s_mul_i32 s0, s12, 0x6000
	s_mul_hi_i32 s1, s12, 0x6000
	s_add_u32 s0, s6, s0
	s_addc_u32 s1, s7, s1
	s_cmp_eq_u32 s11, 2
	v_and_b32_e32 v11, 0xffff0000, v5
	v_add_co_u32 v5, vcc_lo, s0, v3
	s_subb_u32 s4, s10, 0
	v_add_co_ci_u32_e64 v6, null, s1, v4, vcc_lo
	s_add_i32 s0, s4, -3
	s_mov_b32 s1, 0
	s_cmp_gt_i32 s0, -1
	global_store_b32 v[5:6], v11, off
	s_cbranch_scc0 .LBB1_48
; %bb.47:
	v_mad_u64_u32 v[5:6], null, 0x6000, s0, v[0:1]
	s_and_not1_b32 vcc_lo, exec_lo, s1
	s_cbranch_vccz .LBB1_49
	s_branch .LBB1_50
.LBB1_48:
                                        ; implicit-def: $vgpr5_vgpr6
.LBB1_49:
	v_mad_u64_u32 v[5:6], null, 0x1800, s0, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v6, 31, v5
	v_lshlrev_b64 v[5:6], 2, v[5:6]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v5, vcc_lo, s2, v5
	v_add_co_ci_u32_e64 v6, null, s3, v6, vcc_lo
.LBB1_50:                               ; %_Z8conv_tapPKfS0_iib.exit
	global_load_b32 v5, v[5:6], off
	s_cmp_eq_u32 s11, 3
	s_cselect_b32 s4, -1, 0
	s_cmp_lg_u32 s11, 3
	s_cselect_b32 s1, -1, 0
	s_cmp_lt_i32 s0, 0
	s_cselect_b32 s5, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_or_b32 s1, s1, s5
	s_and_b32 vcc_lo, exec_lo, s1
	s_cbranch_vccnz .LBB1_72
; %bb.51:
	v_sub_co_u32 v11, s1, s0, 3
	s_and_not1_b32 vcc_lo, exec_lo, s1
	s_cbranch_vccz .LBB1_53
; %bb.52:
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[5:6], null, 0x6000, v11, v[0:1]
	s_cbranch_execz .LBB1_54
	s_branch .LBB1_55
.LBB1_53:
                                        ; implicit-def: $vgpr5_vgpr6
.LBB1_54:
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[5:6], null, 0x1800, v11, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v6, 31, v5
	v_lshlrev_b64 v[5:6], 2, v[5:6]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v5, vcc_lo, s2, v5
	v_add_co_ci_u32_e64 v6, null, s3, v6, vcc_lo
.LBB1_55:                               ; %_Z8conv_tapPKfS0_iib.exit.i45
	s_waitcnt vmcnt(0)
	global_load_b32 v11, v[5:6], off
	v_sub_co_u32 v12, s1, s0, 2
	s_and_b32 vcc_lo, exec_lo, s1
	s_cbranch_vccnz .LBB1_57
; %bb.56:
	v_mad_u64_u32 v[5:6], null, 0x6000, v12, v[0:1]
	s_cbranch_execz .LBB1_58
	s_branch .LBB1_59
.LBB1_57:
                                        ; implicit-def: $vgpr5_vgpr6
.LBB1_58:
	v_mad_u64_u32 v[5:6], null, 0x1800, v12, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v6, 31, v5
	v_lshlrev_b64 v[5:6], 2, v[5:6]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v5, vcc_lo, s2, v5
	v_add_co_ci_u32_e64 v6, null, s3, v6, vcc_lo
.LBB1_59:                               ; %_Z8conv_tapPKfS0_iib.exit.1.i48
	global_load_b32 v12, v[5:6], off
	v_sub_co_u32 v13, s1, s0, 1
	s_and_b32 vcc_lo, exec_lo, s1
	s_cbranch_vccnz .LBB1_61
; %bb.60:
	v_mad_u64_u32 v[5:6], null, 0x6000, v13, v[0:1]
	s_cbranch_execz .LBB1_62
	s_branch .LBB1_63
.LBB1_61:
                                        ; implicit-def: $vgpr5_vgpr6
.LBB1_62:
	v_mad_u64_u32 v[5:6], null, 0x1800, v13, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v6, 31, v5
	v_lshlrev_b64 v[5:6], 2, v[5:6]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v5, vcc_lo, s2, v5
	v_add_co_ci_u32_e64 v6, null, s3, v6, vcc_lo
.LBB1_63:                               ; %_Z8conv_tapPKfS0_iib.exit.2.i51
	v_mad_u64_u32 v[13:14], null, 0x6000, s0, v[0:1]
	global_load_b32 v6, v[5:6], off
	global_load_b32 v13, v[13:14], off
	s_waitcnt vmcnt(3)
	v_fma_f32 v5, v7, v11, 0
	s_waitcnt vmcnt(2)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v5, v8, v12
	s_waitcnt vmcnt(1)
	v_fmac_f32_e32 v5, v9, v6
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v5, v10, v13
	v_and_b32_e32 v6, 0x7f800000, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v6
                                        ; implicit-def: $vgpr6
	s_and_saveexec_b32 s1, vcc_lo
	s_xor_b32 s1, exec_lo, s1
; %bb.64:
	v_bfe_u32 v6, v5, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v6, v5, v6, 0x7fff
                                        ; implicit-def: $vgpr5
; %bb.65:                               ; %Flow117
	s_and_not1_saveexec_b32 s1, s1
; %bb.66:
	v_and_b32_e32 v6, 0xffff, v5
	v_or_b32_e32 v11, 0x10000, v5
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v6
	v_cndmask_b32_e32 v6, v11, v5, vcc_lo
; %bb.67:                               ; %_Z10bf16_roundf.exit.i56
	s_or_b32 exec_lo, exec_lo, s1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v5, 0xffff0000, v6
	v_mul_f32_e32 v6, 0xbfb8aa3b, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v11, 0xbfb8aa3b, v5, -v6
	v_rndne_f32_e32 v12, v6
	v_sub_f32_e32 v6, v6, v12
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v5
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmamk_f32 v11, v5, 0xb2a5705f, v11
	v_add_f32_e32 v6, v6, v11
	v_cvt_i32_f32_e32 v11, v12
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v6, v6
	v_ldexp_f32 v6, v6, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v6, 0, v6, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v5
	v_cndmask_b32_e32 v6, 0x7f800000, v6, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v6, 1.0, v6
	v_div_scale_f32 v11, null, v6, v6, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v12, v11
	v_fma_f32 v13, -v11, v12, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v12, v13, v12
	v_div_scale_f32 v13, vcc_lo, v5, v6, v5
	v_mul_f32_e32 v14, v13, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v15, -v11, v14, v13
	v_fmac_f32_e32 v14, v15, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v11, -v11, v14, v13
	v_div_fmas_f32 v11, v11, v12, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v5, v11, v6, v5
	v_and_b32_e32 v6, 0x7f800000, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v6
                                        ; implicit-def: $vgpr6
	s_and_saveexec_b32 s1, vcc_lo
	s_xor_b32 s1, exec_lo, s1
; %bb.68:
	v_bfe_u32 v6, v5, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v6, v5, v6, 0x7fff
                                        ; implicit-def: $vgpr5
; %bb.69:                               ; %Flow116
	s_and_not1_saveexec_b32 s1, s1
; %bb.70:
	v_and_b32_e32 v6, 0xffff, v5
	v_or_b32_e32 v11, 0x10000, v5
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v6
	v_cndmask_b32_e32 v6, v11, v5, vcc_lo
; %bb.71:                               ; %_Z10conv_valuePKfS0_S0_iiibb.exit62
	s_or_b32 exec_lo, exec_lo, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v5, 0xffff0000, v6
.LBB1_72:
	v_add_co_u32 v3, vcc_lo, s8, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v4, null, s9, v4, vcc_lo
	s_add_i32 s1, s0, 1
	s_mov_b32 s5, 0
	s_cmp_lt_i32 s1, 0
	s_waitcnt vmcnt(0)
	global_store_b32 v[3:4], v5, off
	s_cbranch_scc1 .LBB1_74
; %bb.73:
	v_mad_u64_u32 v[5:6], null, 0x6000, s1, v[0:1]
	s_and_not1_b32 vcc_lo, exec_lo, s5
	s_cbranch_vccz .LBB1_75
	s_branch .LBB1_76
.LBB1_74:
                                        ; implicit-def: $vgpr5_vgpr6
.LBB1_75:
	v_mad_u64_u32 v[5:6], null, 0x1800, s1, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v6, 31, v5
	v_lshlrev_b64 v[5:6], 2, v[5:6]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v5, vcc_lo, s2, v5
	v_add_co_ci_u32_e64 v6, null, s3, v6, vcc_lo
.LBB1_76:                               ; %_Z8conv_tapPKfS0_iib.exit.1
	global_load_b32 v5, v[5:6], off
	s_cmp_lt_i32 s1, 0
	s_mov_b32 s6, -1
	s_cselect_b32 s5, -1, 0
	s_xor_b32 s4, s4, -1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_or_b32 s5, s4, s5
	s_and_b32 vcc_lo, exec_lo, s5
	s_cbranch_vccnz .LBB1_95
; %bb.77:
	s_add_i32 s5, s0, -2
	s_cmp_lt_u32 s1, 3
                                        ; implicit-def: $vgpr5_vgpr6
	s_cbranch_scc1 .LBB1_79
; %bb.78:
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[5:6], null, 0x6000, s5, v[0:1]
	s_mov_b32 s6, 0
.LBB1_79:                               ; %Flow113
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s6
	s_cbranch_vccnz .LBB1_81
; %bb.80:
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[5:6], null, 0x1800, s5, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v6, 31, v5
	v_lshlrev_b64 v[5:6], 2, v[5:6]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v5, vcc_lo, s2, v5
	v_add_co_ci_u32_e64 v6, null, s3, v6, vcc_lo
.LBB1_81:                               ; %_Z8conv_tapPKfS0_iib.exit.i45.1
	s_waitcnt vmcnt(0)
	global_load_b32 v11, v[5:6], off
	s_add_i32 s5, s0, -1
	s_cmp_lt_u32 s1, 2
	s_mov_b32 s6, -1
                                        ; implicit-def: $vgpr5_vgpr6
	s_cbranch_scc0 .LBB1_104
; %bb.82:                               ; %Flow112
	s_and_not1_b32 vcc_lo, exec_lo, s6
	s_cbranch_vccz .LBB1_105
.LBB1_83:                               ; %_Z8conv_tapPKfS0_iib.exit.1.i48.1
	global_load_b32 v12, v[5:6], off
	s_cmp_eq_u32 s1, 0
	s_mov_b32 s5, 0
	s_cbranch_scc1 .LBB1_106
.LBB1_84:
	v_mad_u64_u32 v[5:6], null, 0x6000, s0, v[0:1]
	s_and_not1_b32 vcc_lo, exec_lo, s5
	s_cbranch_vccnz .LBB1_86
.LBB1_85:
	v_mad_u64_u32 v[5:6], null, 0x1800, s0, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v6, 31, v5
	v_lshlrev_b64 v[5:6], 2, v[5:6]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v5, vcc_lo, s2, v5
	v_add_co_ci_u32_e64 v6, null, s3, v6, vcc_lo
.LBB1_86:                               ; %_Z8conv_tapPKfS0_iib.exit.2.i51.1
	v_mad_u64_u32 v[13:14], null, 0x6000, s1, v[0:1]
	global_load_b32 v6, v[5:6], off
	global_load_b32 v13, v[13:14], off
	s_waitcnt vmcnt(3)
	v_fma_f32 v5, v7, v11, 0
	s_waitcnt vmcnt(2)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v5, v8, v12
	s_waitcnt vmcnt(1)
	v_fmac_f32_e32 v5, v9, v6
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v5, v10, v13
	v_and_b32_e32 v6, 0x7f800000, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v6
                                        ; implicit-def: $vgpr6
	s_and_saveexec_b32 s5, vcc_lo
	s_xor_b32 s5, exec_lo, s5
; %bb.87:
	v_bfe_u32 v6, v5, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v6, v5, v6, 0x7fff
                                        ; implicit-def: $vgpr5
; %bb.88:                               ; %Flow110
	s_and_not1_saveexec_b32 s5, s5
; %bb.89:
	v_and_b32_e32 v6, 0xffff, v5
	v_or_b32_e32 v11, 0x10000, v5
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v6
	v_cndmask_b32_e32 v6, v11, v5, vcc_lo
; %bb.90:                               ; %_Z10bf16_roundf.exit.i56.1
	s_or_b32 exec_lo, exec_lo, s5
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v5, 0xffff0000, v6
	v_mul_f32_e32 v6, 0xbfb8aa3b, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v11, 0xbfb8aa3b, v5, -v6
	v_rndne_f32_e32 v12, v6
	v_sub_f32_e32 v6, v6, v12
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v5
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmamk_f32 v11, v5, 0xb2a5705f, v11
	v_add_f32_e32 v6, v6, v11
	v_cvt_i32_f32_e32 v11, v12
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v6, v6
	v_ldexp_f32 v6, v6, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v6, 0, v6, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v5
	v_cndmask_b32_e32 v6, 0x7f800000, v6, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v6, 1.0, v6
	v_div_scale_f32 v11, null, v6, v6, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v12, v11
	v_fma_f32 v13, -v11, v12, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v12, v13, v12
	v_div_scale_f32 v13, vcc_lo, v5, v6, v5
	v_mul_f32_e32 v14, v13, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v15, -v11, v14, v13
	v_fmac_f32_e32 v14, v15, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v11, -v11, v14, v13
	v_div_fmas_f32 v11, v11, v12, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v5, v11, v6, v5
	v_and_b32_e32 v6, 0x7f800000, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v6
                                        ; implicit-def: $vgpr6
	s_and_saveexec_b32 s5, vcc_lo
	s_xor_b32 s5, exec_lo, s5
; %bb.91:
	v_bfe_u32 v6, v5, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v6, v5, v6, 0x7fff
                                        ; implicit-def: $vgpr5
; %bb.92:                               ; %Flow109
	s_and_not1_saveexec_b32 s5, s5
; %bb.93:
	v_and_b32_e32 v6, 0xffff, v5
	v_or_b32_e32 v11, 0x10000, v5
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v6
	v_cndmask_b32_e32 v6, v11, v5, vcc_lo
; %bb.94:                               ; %_Z10conv_valuePKfS0_S0_iiibb.exit62.1
	s_or_b32 exec_lo, exec_lo, s5
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v5, 0xffff0000, v6
.LBB1_95:
	v_add_co_u32 v11, vcc_lo, 0x6000, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v12, null, 0, v4, vcc_lo
	s_add_i32 s5, s0, 2
	s_mov_b32 s6, 0
	s_cmp_lt_i32 s5, 0
	s_waitcnt vmcnt(0)
	global_store_b32 v[11:12], v5, off
	s_cbranch_scc1 .LBB1_97
; %bb.96:
	v_mad_u64_u32 v[5:6], null, 0x6000, s5, v[0:1]
	s_and_not1_b32 vcc_lo, exec_lo, s6
	s_cbranch_vccz .LBB1_98
	s_branch .LBB1_99
.LBB1_97:
                                        ; implicit-def: $vgpr5_vgpr6
.LBB1_98:
	v_mad_u64_u32 v[5:6], null, 0x1800, s5, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v6, 31, v5
	v_lshlrev_b64 v[5:6], 2, v[5:6]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v5, vcc_lo, s2, v5
	v_add_co_ci_u32_e64 v6, null, s3, v6, vcc_lo
.LBB1_99:                               ; %_Z8conv_tapPKfS0_iib.exit.2
	global_load_b32 v5, v[5:6], off
	s_cmp_lt_i32 s5, 0
	s_cselect_b32 s6, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 s4, s4, s6
	s_mov_b32 s6, -1
	s_and_b32 vcc_lo, exec_lo, s4
	s_cbranch_vccnz .LBB1_123
; %bb.100:
	s_add_i32 s4, s0, -1
	s_cmp_lt_u32 s5, 3
                                        ; implicit-def: $vgpr5_vgpr6
	s_cbranch_scc0 .LBB1_107
; %bb.101:                              ; %Flow106
	s_and_not1_b32 vcc_lo, exec_lo, s6
	s_cbranch_vccz .LBB1_108
.LBB1_102:                              ; %_Z8conv_tapPKfS0_iib.exit.i45.2
	s_waitcnt vmcnt(0)
	global_load_b32 v11, v[5:6], off
	s_cmp_gt_u32 s0, -3
	s_cbranch_scc1 .LBB1_109
.LBB1_103:
	v_mad_u64_u32 v[5:6], null, 0x6000, s0, v[0:1]
	s_cbranch_execz .LBB1_110
	s_branch .LBB1_111
.LBB1_104:
	v_mad_u64_u32 v[5:6], null, 0x6000, s5, v[0:1]
	s_cbranch_execnz .LBB1_83
.LBB1_105:
	v_mad_u64_u32 v[5:6], null, 0x1800, s5, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v6, 31, v5
	v_lshlrev_b64 v[5:6], 2, v[5:6]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v5, vcc_lo, s2, v5
	v_add_co_ci_u32_e64 v6, null, s3, v6, vcc_lo
	global_load_b32 v12, v[5:6], off
	s_cmp_eq_u32 s1, 0
	s_mov_b32 s5, 0
	s_cbranch_scc0 .LBB1_84
.LBB1_106:
                                        ; implicit-def: $vgpr5_vgpr6
	s_branch .LBB1_85
.LBB1_107:
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[5:6], null, 0x6000, s4, v[0:1]
	s_cbranch_execnz .LBB1_102
.LBB1_108:
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[5:6], null, 0x1800, s4, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v6, 31, v5
	v_lshlrev_b64 v[5:6], 2, v[5:6]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v5, vcc_lo, s2, v5
	v_add_co_ci_u32_e64 v6, null, s3, v6, vcc_lo
	global_load_b32 v11, v[5:6], off
	s_cmp_gt_u32 s0, -3
	s_cbranch_scc0 .LBB1_103
.LBB1_109:
                                        ; implicit-def: $vgpr5_vgpr6
.LBB1_110:
	v_mad_u64_u32 v[5:6], null, 0x1800, s0, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v6, 31, v5
	v_lshlrev_b64 v[5:6], 2, v[5:6]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v5, vcc_lo, s2, v5
	v_add_co_ci_u32_e64 v6, null, s3, v6, vcc_lo
.LBB1_111:                              ; %_Z8conv_tapPKfS0_iib.exit.1.i48.2
	global_load_b32 v12, v[5:6], off
	s_cmp_eq_u32 s5, 0
	s_mov_b32 s0, 0
	s_cbranch_scc1 .LBB1_125
; %bb.112:
	v_mad_u64_u32 v[5:6], null, 0x6000, s1, v[0:1]
	s_and_not1_b32 vcc_lo, exec_lo, s0
	s_cbranch_vccnz .LBB1_114
.LBB1_113:
	v_mad_u64_u32 v[5:6], null, 0x1800, s1, v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v6, 31, v5
	v_lshlrev_b64 v[5:6], 2, v[5:6]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v5, vcc_lo, s2, v5
	v_add_co_ci_u32_e64 v6, null, s3, v6, vcc_lo
.LBB1_114:                              ; %_Z8conv_tapPKfS0_iib.exit.2.i51.2
	v_mad_u64_u32 v[0:1], null, 0x6000, s5, v[0:1]
	global_load_b32 v2, v[5:6], off
	global_load_b32 v1, v[0:1], off
	s_waitcnt vmcnt(3)
	v_fma_f32 v0, v7, v11, 0
	s_waitcnt vmcnt(2)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v0, v8, v12
	s_waitcnt vmcnt(1)
	v_fmac_f32_e32 v0, v9, v2
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v0, v10, v1
	v_and_b32_e32 v1, 0x7f800000, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v1
                                        ; implicit-def: $vgpr1
	s_and_saveexec_b32 s0, vcc_lo
	s_xor_b32 s0, exec_lo, s0
; %bb.115:
	v_bfe_u32 v1, v0, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v1, v0, v1, 0x7fff
                                        ; implicit-def: $vgpr0
; %bb.116:                              ; %Flow103
	s_and_not1_saveexec_b32 s0, s0
; %bb.117:
	v_and_b32_e32 v1, 0xffff, v0
	v_or_b32_e32 v2, 0x10000, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v1
	v_cndmask_b32_e32 v1, v2, v0, vcc_lo
; %bb.118:                              ; %_Z10bf16_roundf.exit.i56.2
	s_or_b32 exec_lo, exec_lo, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v0, 0xffff0000, v1
	v_mul_f32_e32 v1, 0xbfb8aa3b, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v2, 0xbfb8aa3b, v0, -v1
	v_rndne_f32_e32 v5, v1
	v_dual_fmamk_f32 v2, v0, 0xb2a5705f, v2 :: v_dual_sub_f32 v1, v1, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_add_f32_e32 v1, v1, v2
	v_cvt_i32_f32_e32 v2, v5
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v0
	v_exp_f32_e32 v1, v1
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ldexp_f32 v1, v1, v2
	v_cndmask_b32_e32 v1, 0, v1, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v1, 0x7f800000, v1, vcc_lo
	v_add_f32_e32 v1, 1.0, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_scale_f32 v2, null, v1, v1, v0
	v_rcp_f32_e32 v5, v2
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v6, -v2, v5, 1.0
	v_fmac_f32_e32 v5, v6, v5
	v_div_scale_f32 v6, vcc_lo, v0, v1, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v7, v6, v5
	v_fma_f32 v8, -v2, v7, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v7, v8, v5
	v_fma_f32 v2, -v2, v7, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v2, v2, v5, v7
	v_div_fixup_f32 v0, v2, v1, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v1, 0x7f800000, v0
	v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v1
                                        ; implicit-def: $vgpr1
	s_and_saveexec_b32 s0, vcc_lo
	s_delay_alu instid0(SALU_CYCLE_1)
	s_xor_b32 s0, exec_lo, s0
; %bb.119:
	v_bfe_u32 v1, v0, 16, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_add3_u32 v1, v0, v1, 0x7fff
                                        ; implicit-def: $vgpr0
; %bb.120:                              ; %Flow
	s_and_not1_saveexec_b32 s0, s0
; %bb.121:
	v_and_b32_e32 v1, 0xffff, v0
	v_or_b32_e32 v2, 0x10000, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 0, v1
	v_cndmask_b32_e32 v1, v2, v0, vcc_lo
; %bb.122:                              ; %_Z10conv_valuePKfS0_S0_iiibb.exit62.2
	s_or_b32 exec_lo, exec_lo, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v5, 0xffff0000, v1
.LBB1_123:                              ; %.loopexit.loopexit
	v_add_co_u32 v0, vcc_lo, 0xc000, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v1, null, 0, v4, vcc_lo
	s_waitcnt vmcnt(0)
	global_store_b32 v[0:1], v5, off
.LBB1_124:                              ; %.loopexit
	s_endpgm
.LBB1_125:
                                        ; implicit-def: $vgpr5_vgpr6
	s_branch .LBB1_113
.Lfunc_end1:
	.size	_Z11conv_kernelPKfS0_S0_PfS1_ii, .Lfunc_end1-_Z11conv_kernelPKfS0_S0_PfS1_ii
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z11conv_kernelPKfS0_S0_PfS1_ii
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 304
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 17
		.amdhsa_next_free_sgpr 18
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_shared_vgpr_count 0
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end1-_Z11conv_kernelPKfS0_S0_PfS1_ii)<<4)&1008)>>4
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
                                        ; -- End function
	.set .L_Z11conv_kernelPKfS0_S0_PfS1_ii.num_vgpr, 17
	.set .L_Z11conv_kernelPKfS0_S0_PfS1_ii.num_agpr, 0
	.set .L_Z11conv_kernelPKfS0_S0_PfS1_ii.numbered_sgpr, 18
	.set .L_Z11conv_kernelPKfS0_S0_PfS1_ii.num_named_barrier, 0
	.set .L_Z11conv_kernelPKfS0_S0_PfS1_ii.private_seg_size, 0
	.set .L_Z11conv_kernelPKfS0_S0_PfS1_ii.uses_vcc, 1
	.set .L_Z11conv_kernelPKfS0_S0_PfS1_ii.uses_flat_scratch, 0
	.set .L_Z11conv_kernelPKfS0_S0_PfS1_ii.has_dyn_sized_stack, 0
	.set .L_Z11conv_kernelPKfS0_S0_PfS1_ii.has_recursion, 0
	.set .L_Z11conv_kernelPKfS0_S0_PfS1_ii.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 4292
; TotalNumSgprs: 20
; NumVgprs: 17
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 2
; NumSGPRsForWavesPerEU: 20
; NumVGPRsForWavesPerEU: 17
; Occupancy: 16
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	_Z10rec_kernelPKfS0_S0_S0_S0_PfS1_i ; -- Begin function _Z10rec_kernelPKfS0_S0_S0_S0_PfS1_i
	.globl	_Z10rec_kernelPKfS0_S0_S0_S0_PfS1_i
	.p2align	8
	.type	_Z10rec_kernelPKfS0_S0_S0_S0_PfS1_i,@function
_Z10rec_kernelPKfS0_S0_S0_S0_PfS1_i:    ; @_Z10rec_kernelPKfS0_S0_S0_S0_PfS1_i
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_mov_b32 s4, -1
	s_mov_b32 s3, 0
	s_cmp_eq_u32 exec_lo, -1
	s_mov_b32 s21, 0
	s_cbranch_scc0 .LBB2_24
.LBB2_1:                                ; %Flow
	v_readfirstlane_b32 s20, v0
	s_and_not1_b32 vcc_lo, exec_lo, s4
	s_cbranch_vccnz .LBB2_15
; %bb.2:                                ; %_Z16__shfl_down_syncIyfET0_T_S0_ji.exit.i
	s_load_b256 s[4:11], s[0:1], 0x0
	v_lshl_add_u32 v2, s2, 7, v0
	s_load_b256 s[12:19], s[0:1], 0x20
	s_cmp_lg_u32 exec_lo, -1
	s_mov_b32 s3, -1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshlrev_b64 v[4:5], 2, v[2:3]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v6, vcc_lo, s6, v4
	v_add_co_ci_u32_e64 v7, null, s7, v5, vcc_lo
	global_load_b32 v6, v[6:7], off
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v7, v6, v6
	ds_swizzle_b32 v8, v7 offset:swizzle(BITMASK_PERM,"1pppp")
	s_cbranch_scc1 .LBB2_52
; %bb.3:                                ; %Flow178
	s_and_not1_b32 vcc_lo, exec_lo, s3
	s_cbranch_vccnz .LBB2_53
.LBB2_4:                                ; %_Z16__shfl_down_syncIyfET0_T_S0_ji.exit28.i
	v_mbcnt_lo_u32_b32 v1, -1, 0
	s_mov_b32 s3, 0
	s_cmp_lg_u32 exec_lo, -1
	s_mov_b32 s6, -1
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v8, v7, v8
	v_cmp_gt_u32_e32 vcc_lo, 24, v1
	v_cndmask_b32_e64 v9, 0, 8, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v9, v9, v1
	v_lshlrev_b32_e32 v7, 2, v9
	ds_bpermute_b32 v9, v7, v8
	s_cbranch_scc1 .LBB2_58
; %bb.5:                                ; %Flow180
	s_and_not1_b32 vcc_lo, exec_lo, s6
	s_cbranch_vccnz .LBB2_54
.LBB2_6:                                ; %_Z16__shfl_down_syncIyfET0_T_S0_ji.exit30.i
	v_cmp_gt_u32_e32 vcc_lo, 28, v1
	s_cmp_lg_u32 exec_lo, -1
	s_mov_b32 s6, -1
	v_cndmask_b32_e64 v10, 0, 4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v10, v10, v1
	s_waitcnt lgkmcnt(0)
	v_dual_add_f32 v9, v8, v9 :: v_dual_lshlrev_b32 v8, 2, v10
	ds_bpermute_b32 v10, v8, v9
	s_cbranch_scc1 .LBB2_59
; %bb.7:                                ; %Flow182
	s_and_not1_b32 vcc_lo, exec_lo, s6
	s_cbranch_vccnz .LBB2_54
.LBB2_8:                                ; %_Z16__shfl_down_syncIyfET0_T_S0_ji.exit32.i
	v_cmp_gt_u32_e32 vcc_lo, 30, v1
	s_cmp_lg_u32 exec_lo, -1
	s_mov_b32 s6, -1
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v10, v9, v10
	v_cndmask_b32_e64 v11, 0, 2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v11, v11, v1
	v_lshlrev_b32_e32 v9, 2, v11
	ds_bpermute_b32 v11, v9, v10
	s_cbranch_scc1 .LBB2_60
; %bb.9:                                ; %Flow184
	s_and_not1_b32 vcc_lo, exec_lo, s6
	s_cbranch_vccnz .LBB2_54
.LBB2_10:                               ; %_Z16__shfl_down_syncIyfET0_T_S0_ji.exit34.i
	v_add_co_u32 v4, vcc_lo, s4, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v5, null, s5, v5, vcc_lo
	v_cmp_ne_u32_e32 vcc_lo, 31, v1
	v_and_b32_e32 v12, 31, v0
	global_load_b32 v4, v[4:5], off
	v_add_co_ci_u32_e64 v1, null, 0, v1, vcc_lo
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v10, v10, v11
	v_cmp_eq_u32_e64 s3, 0, v12
	s_delay_alu instid0(VALU_DEP_3)
	v_lshlrev_b32_e32 v5, 2, v1
	ds_bpermute_b32 v11, v5, v10
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB2_12
; %bb.11:
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v10, v10, v11
	v_lshrrev_b32_e32 v11, 3, v0
	ds_store_b32 v11, v10 offset:1024
.LBB2_12:
	s_or_b32 exec_lo, exec_lo, s4
	s_load_b32 s0, s[0:1], 0x4c
	v_mov_b32_e32 v10, 0
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	buffer_gl0_inv
	s_bfe_u32 s0, s0, 0xb0005
	s_delay_alu instid0(SALU_CYCLE_1)
	v_cmp_gt_u32_e64 s1, s0, v0
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB2_25
; %bb.13:
	v_lshlrev_b32_e32 v10, 2, v0
	ds_load_b32 v10, v10 offset:1024
	s_or_b32 exec_lo, exec_lo, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_eq_u32 exec_lo, -1
	s_mov_b32 s0, -1
	s_cbranch_scc1 .LBB2_26
.LBB2_14:
	s_mov_b32 s21, -1
	s_trap 2
	s_cbranch_execz .LBB2_27
	s_branch .LBB2_65
.LBB2_15:
                                        ; implicit-def: $sgpr22
                                        ; implicit-def: $vgpr5
                                        ; implicit-def: $vgpr7
                                        ; implicit-def: $vgpr8
                                        ; implicit-def: $sgpr0
                                        ; implicit-def: $vgpr1
                                        ; implicit-def: $vgpr4
                                        ; implicit-def: $vgpr6
                                        ; implicit-def: $vgpr2_vgpr3
                                        ; implicit-def: $vgpr0
                                        ; implicit-def: $sgpr4_sgpr5_sgpr6_sgpr7_sgpr8_sgpr9_sgpr10_sgpr11_sgpr12_sgpr13_sgpr14_sgpr15_sgpr16_sgpr17_sgpr18_sgpr19
	s_and_b32 vcc_lo, exec_lo, s21
	s_cbranch_vccnz .LBB2_56
.LBB2_16:                               ; %Flow215
	s_and_not1_b32 vcc_lo, exec_lo, s3
	s_cbranch_vccnz .LBB2_57
.LBB2_17:                               ; %_Z16__shfl_down_syncIyfET0_T_S0_ji.exit44.i77
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v5, v7, v5
	v_lshlrev_b32_e32 v1, 2, v1
	s_barrier
	buffer_gl0_inv
	ds_bpermute_b32 v1, v1, v5
	s_and_saveexec_b32 s1, s22
	s_cbranch_execz .LBB2_19
; %bb.18:
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v1, v5, v1
	v_mov_b32_e32 v5, 0
	ds_store_b32 v5, v1 offset:1024
.LBB2_19:                               ; %_Z9block_sumfPf.exit78
	s_or_b32 exec_lo, exec_lo, s1
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v1, 0
	s_barrier
	buffer_gl0_inv
	v_mul_f32_e32 v9, 0x45800000, v8
	ds_load_b32 v5, v1 offset:1024
	s_ashr_i32 s21, s20, 31
	s_mov_b32 s5, 0
	s_lshl_b64 s[2:3], s[20:21], 2
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v5, 0x358637bd, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v7, 0x4b800000, v5
	v_cmp_gt_f32_e32 vcc_lo, 0x800000, v5
	v_cndmask_b32_e32 v5, v5, v7, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rsq_f32_e32 v5, v5
	v_mul_f32_e32 v7, 0x45800000, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v5, v5, v7, vcc_lo
	v_cndmask_b32_e64 v7, v8, v9, s0
	s_add_u32 s0, s10, s2
	s_addc_u32 s1, s11, s3
	v_dual_mul_f32 v4, v4, v5 :: v_dual_lshlrev_b32 v5, 2, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v6, v6, v7
	v_mul_f32_e32 v4, 0x3db504f3, v4
	s_barrier
	buffer_gl0_inv
	s_barrier
	buffer_gl0_inv
	ds_store_2addr_stride64_b32 v5, v6, v4 offset1:2
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_gl0_inv
	global_load_b32 v7, v1, s[0:1]
	s_add_u32 s0, s12, s2
	s_addc_u32 s1, s13, s3
	v_lshlrev_b64 v[5:6], 2, v[0:1]
	global_load_b32 v4, v1, s[0:1]
	s_lshl_b64 s[0:1], s[20:21], 16
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	s_add_u32 s2, s14, s0
	s_addc_u32 s3, s15, s1
	v_add_co_u32 v5, vcc_lo, s2, v5
	v_add_co_ci_u32_e64 v6, null, s3, v6, vcc_lo
	s_mov_b64 s[2:3], 0
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s4, v7
.LBB2_20:                               ; =>This Inner Loop Header: Depth=1
	v_add_co_u32 v7, vcc_lo, v5, s2
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v8, null, s3, v6, vcc_lo
	v_mov_b32_e32 v31, s5
	s_add_u32 s2, s2, 0x2000
	s_addc_u32 s3, s3, 0
	s_clause 0x7
	global_load_b32 v15, v[7:8], off
	global_load_b32 v16, v[7:8], off offset:512
	global_load_b32 v17, v[7:8], off offset:1024
	global_load_b32 v18, v[7:8], off offset:1536
	global_load_b32 v19, v[7:8], off offset:2048
	global_load_b32 v20, v[7:8], off offset:2560
	global_load_b32 v21, v[7:8], off offset:3072
	global_load_b32 v22, v[7:8], off offset:3584
	v_add_co_u32 v7, vcc_lo, 0x1000, v7
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v8, null, 0, v8, vcc_lo
	s_add_i32 s5, s5, 64
	s_cmp_eq_u32 s2, 0x10000
	s_clause 0x7
	global_load_b32 v23, v[7:8], off
	global_load_b32 v24, v[7:8], off offset:512
	global_load_b32 v25, v[7:8], off offset:1024
	global_load_b32 v26, v[7:8], off offset:1536
	global_load_b32 v27, v[7:8], off offset:2048
	global_load_b32 v28, v[7:8], off offset:2560
	global_load_b32 v29, v[7:8], off offset:3072
	global_load_b32 v30, v[7:8], off offset:3584
	ds_load_b128 v[7:10], v31
	ds_load_b128 v[11:14], v31 offset:16
	s_waitcnt vmcnt(15) lgkmcnt(1)
	v_fmac_f32_e32 v1, v15, v7
	s_waitcnt vmcnt(14)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v1, v16, v8
	s_waitcnt vmcnt(13)
	v_fmac_f32_e32 v1, v17, v9
	s_waitcnt vmcnt(12)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v1, v18, v10
	ds_load_b128 v[7:10], v31 offset:32
	s_waitcnt vmcnt(11) lgkmcnt(1)
	v_fmac_f32_e32 v1, v19, v11
	s_waitcnt vmcnt(10)
	v_fmac_f32_e32 v1, v20, v12
	s_waitcnt vmcnt(9)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v1, v21, v13
	s_waitcnt vmcnt(8)
	v_fmac_f32_e32 v1, v22, v14
	ds_load_b128 v[11:14], v31 offset:48
	s_waitcnt vmcnt(7) lgkmcnt(1)
	v_fmac_f32_e32 v1, v23, v7
	s_waitcnt vmcnt(6)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v1, v24, v8
	s_waitcnt vmcnt(5)
	v_fmac_f32_e32 v1, v25, v9
	s_waitcnt vmcnt(4)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v1, v26, v10
	s_waitcnt vmcnt(3) lgkmcnt(0)
	v_fmac_f32_e32 v1, v27, v11
	s_waitcnt vmcnt(2)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v1, v28, v12
	s_waitcnt vmcnt(1)
	v_fmac_f32_e32 v1, v29, v13
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v1, v30, v14
	s_cbranch_scc0 .LBB2_20
; %bb.21:
	v_lshlrev_b64 v[2:3], 2, v[2:3]
	s_mul_f32 s2, s4, 0x3fb8aa3b
	s_add_u32 s0, s14, s0
	s_addc_u32 s1, s15, s1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	s_xor_b32 s3, s2, 0x80000000
	v_add_co_u32 v5, vcc_lo, s8, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v6, null, s9, v3, vcc_lo
	s_rndne_f32 s5, s2
	s_fmamk_f32 s3, s4, 0x3fb8aa3b, s3
	s_cmp_nlt_f32 s4, 0xc2ce8ed0
	global_load_b32 v5, v[5:6], off
	s_sub_f32 s2, s2, s5
	s_fmamk_f32 s3, s4, 0x32a5705f, s3
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_ngt_f32 s4, 0x42b17218
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_3)
	s_add_f32 s2, s2, s3
	v_exp_f32_e32 v6, s2
	s_cvt_i32_f32 s2, s5
	s_delay_alu instid0(TRANS32_DEP_1) | instid1(SALU_CYCLE_3)
	v_ldexp_f32 v6, v6, s2
	s_mov_b32 s2, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v6, 0, v6, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	v_cndmask_b32_e32 v6, 0x7f800000, v6, vcc_lo
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fma_f32 v1, -v6, v1, v5
	v_mov_b32_e32 v5, 0
	v_mul_f32_e32 v7, v4, v1
	v_mov_b32_e32 v1, 0
.LBB2_22:                               ; =>This Inner Loop Header: Depth=1
	v_lshl_or_b32 v4, s2, 7, v0
	s_lshl_b32 s3, s2, 2
	s_add_i32 s2, s2, 8
	v_mov_b32_e32 v20, s3
	s_cmpk_eq_i32 s2, 0x80
	v_lshlrev_b64 v[8:9], 2, v[4:5]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v24, vcc_lo, s0, v8
	v_add_co_ci_u32_e64 v25, null, s1, v9, vcc_lo
	ds_load_b128 v[8:11], v20
	ds_load_b128 v[12:15], v20 offset:512
	ds_load_b128 v[16:19], v20 offset:16
	s_clause 0x7
	global_load_b32 v4, v[24:25], off
	global_load_b32 v26, v[24:25], off offset:512
	global_load_b32 v27, v[24:25], off offset:1024
	global_load_b32 v28, v[24:25], off offset:1536
	global_load_b32 v29, v[24:25], off offset:2048
	global_load_b32 v30, v[24:25], off offset:2560
	global_load_b32 v31, v[24:25], off offset:3072
	global_load_b32 v32, v[24:25], off offset:3584
	s_waitcnt lgkmcnt(2)
	v_mul_f32_e32 v9, v7, v9
	v_mul_f32_e32 v8, v7, v8
	ds_load_b128 v[20:23], v20 offset:528
	s_waitcnt vmcnt(6)
	v_fmac_f32_e32 v9, v6, v26
	v_fmac_f32_e32 v8, v6, v4
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_dual_mul_f32 v4, v7, v10 :: v_dual_fmac_f32 v1, v12, v8
	s_waitcnt vmcnt(5)
	v_dual_fmac_f32 v4, v6, v27 :: v_dual_fmac_f32 v1, v13, v9
	v_mul_f32_e32 v10, v7, v11
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v11, v7, v16
	v_mul_f32_e32 v13, v7, v18
	v_fmac_f32_e32 v1, v14, v4
	s_waitcnt vmcnt(4)
	v_fmac_f32_e32 v10, v6, v28
	s_waitcnt vmcnt(3)
	v_dual_fmac_f32 v11, v6, v29 :: v_dual_mul_f32 v14, v7, v19
	s_waitcnt vmcnt(1)
	v_fmac_f32_e32 v13, v6, v31
	s_clause 0x3
	global_store_b32 v[24:25], v8, off
	global_store_b32 v[24:25], v9, off offset:512
	global_store_b32 v[24:25], v4, off offset:1024
	global_store_b32 v[24:25], v10, off offset:1536
	s_waitcnt vmcnt(0)
	v_dual_fmac_f32 v1, v15, v10 :: v_dual_fmac_f32 v14, v6, v32
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_mul_f32 v12, v7, v17 :: v_dual_fmac_f32 v1, v20, v11
	v_fmac_f32_e32 v12, v6, v30
	s_clause 0x3
	global_store_b32 v[24:25], v11, off offset:2048
	global_store_b32 v[24:25], v12, off offset:2560
	global_store_b32 v[24:25], v13, off offset:3072
	global_store_b32 v[24:25], v14, off offset:3584
	v_fmac_f32_e32 v1, v21, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v1, v22, v13
	v_fmac_f32_e32 v1, v23, v14
	s_cbranch_scc0 .LBB2_22
; %bb.23:
	v_add_co_u32 v2, vcc_lo, s16, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v3, null, s17, v3, vcc_lo
	global_store_b32 v[2:3], v1, off
	s_endpgm
.LBB2_24:
	s_mov_b32 s4, 0
	s_mov_b32 s21, -1
	s_trap 2
	s_branch .LBB2_1
.LBB2_25:
	s_or_b32 exec_lo, exec_lo, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_eq_u32 exec_lo, -1
	s_mov_b32 s0, -1
	s_cbranch_scc0 .LBB2_14
.LBB2_26:                               ; %Flow186
	s_and_not1_b32 vcc_lo, exec_lo, s0
	s_cbranch_vccnz .LBB2_65
.LBB2_27:                               ; %_Z16__shfl_down_syncIyfET0_T_S0_ji.exit36.i
	s_waitcnt lgkmcnt(0)
	ds_swizzle_b32 v11, v10 offset:swizzle(BITMASK_PERM,"1pppp")
	s_cmp_lg_u32 exec_lo, -1
	s_mov_b32 s0, -1
	s_cbranch_scc1 .LBB2_61
; %bb.28:                               ; %Flow188
	s_and_not1_b32 vcc_lo, exec_lo, s0
	s_cbranch_vccnz .LBB2_65
.LBB2_29:                               ; %_Z16__shfl_down_syncIyfET0_T_S0_ji.exit38.i
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v10, v10, v11
	s_cmp_lg_u32 exec_lo, -1
	s_mov_b32 s0, -1
	ds_bpermute_b32 v11, v7, v10
	s_cbranch_scc1 .LBB2_62
; %bb.30:                               ; %Flow190
	s_and_not1_b32 vcc_lo, exec_lo, s0
	s_cbranch_vccnz .LBB2_65
.LBB2_31:                               ; %_Z16__shfl_down_syncIyfET0_T_S0_ji.exit40.i
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v10, v10, v11
	s_cmp_lg_u32 exec_lo, -1
	s_mov_b32 s0, -1
	ds_bpermute_b32 v11, v8, v10
	s_cbranch_scc1 .LBB2_63
; %bb.32:                               ; %Flow192
	s_and_not1_b32 vcc_lo, exec_lo, s0
	s_cbranch_vccnz .LBB2_65
.LBB2_33:                               ; %_Z16__shfl_down_syncIyfET0_T_S0_ji.exit42.i
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v10, v10, v11
	s_cmp_lg_u32 exec_lo, -1
	s_mov_b32 s0, -1
	ds_bpermute_b32 v11, v9, v10
	s_cbranch_scc1 .LBB2_64
; %bb.34:                               ; %Flow194
	s_and_not1_b32 vcc_lo, exec_lo, s0
	s_cbranch_vccnz .LBB2_65
.LBB2_35:                               ; %_Z16__shfl_down_syncIyfET0_T_S0_ji.exit44.i
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v10, v10, v11
	v_cmp_eq_u32_e64 s22, 0, v0
	s_barrier
	buffer_gl0_inv
	ds_bpermute_b32 v11, v5, v10
	s_and_saveexec_b32 s0, s22
	s_cbranch_execz .LBB2_37
; %bb.36:
	s_waitcnt lgkmcnt(0)
	v_dual_add_f32 v10, v10, v11 :: v_dual_mov_b32 v11, 0
	ds_store_b32 v11, v10 offset:1024
.LBB2_37:                               ; %_Z9block_sumfPf.exit
	s_or_b32 exec_lo, exec_lo, s0
	v_mov_b32_e32 v10, 0
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_gl0_inv
	s_mov_b32 s4, -1
	ds_load_b32 v10, v10 offset:1024
	s_cmp_eq_u32 exec_lo, -1
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v10, 0x358637bd, v10
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_f32_e64 s0, 0x800000, v10
	v_mul_f32_e32 v11, 0x4b800000, v10
	s_barrier
	buffer_gl0_inv
	s_barrier
	buffer_gl0_inv
	s_cbranch_scc0 .LBB2_66
; %bb.38:                               ; %Flow196
	v_cndmask_b32_e64 v10, v10, v11, s0
	s_and_not1_b32 vcc_lo, exec_lo, s4
	s_cbranch_vccnz .LBB2_84
.LBB2_39:                               ; %_Z16__shfl_down_syncIyfET0_T_S0_ji.exit.i64
	v_mul_f32_e32 v11, v4, v4
	s_cmp_lg_u32 exec_lo, -1
	s_mov_b32 s4, -1
	ds_swizzle_b32 v12, v11 offset:swizzle(BITMASK_PERM,"1pppp")
	s_cbranch_scc1 .LBB2_77
; %bb.40:                               ; %Flow198
	s_and_not1_b32 vcc_lo, exec_lo, s4
	s_cbranch_vccnz .LBB2_84
.LBB2_41:                               ; %_Z16__shfl_down_syncIyfET0_T_S0_ji.exit28.i65
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v11, v11, v12
	s_cmp_lg_u32 exec_lo, -1
	s_mov_b32 s4, -1
	ds_bpermute_b32 v12, v7, v11
	s_cbranch_scc1 .LBB2_78
; %bb.42:                               ; %Flow200
	s_and_not1_b32 vcc_lo, exec_lo, s4
	s_cbranch_vccnz .LBB2_84
.LBB2_43:                               ; %_Z16__shfl_down_syncIyfET0_T_S0_ji.exit30.i67
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v11, v11, v12
	s_cmp_lg_u32 exec_lo, -1
	s_mov_b32 s4, -1
	ds_bpermute_b32 v12, v8, v11
	s_cbranch_scc1 .LBB2_79
; %bb.44:                               ; %Flow202
	s_and_not1_b32 vcc_lo, exec_lo, s4
	s_cbranch_vccnz .LBB2_84
.LBB2_45:                               ; %_Z16__shfl_down_syncIyfET0_T_S0_ji.exit32.i69
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v11, v11, v12
	s_cmp_lg_u32 exec_lo, -1
	s_mov_b32 s4, -1
	ds_bpermute_b32 v12, v9, v11
	s_cbranch_scc1 .LBB2_80
; %bb.46:                               ; %Flow204
	s_and_not1_b32 vcc_lo, exec_lo, s4
	s_cbranch_vccnz .LBB2_84
.LBB2_47:                               ; %_Z16__shfl_down_syncIyfET0_T_S0_ji.exit34.i71
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v11, v11, v12
	ds_bpermute_b32 v5, v5, v11
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB2_49
; %bb.48:
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v5, v11, v5
	v_lshrrev_b32_e32 v11, 3, v0
	ds_store_b32 v11, v5 offset:1024
.LBB2_49:
	s_or_b32 exec_lo, exec_lo, s4
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v5, 0
	s_barrier
	buffer_gl0_inv
	s_and_saveexec_b32 s3, s1
	s_cbranch_execz .LBB2_67
; %bb.50:
	v_lshlrev_b32_e32 v5, 2, v0
	ds_load_b32 v5, v5 offset:1024
	s_or_b32 exec_lo, exec_lo, s3
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_eq_u32 exec_lo, -1
	s_mov_b32 s1, -1
	s_cbranch_scc1 .LBB2_68
.LBB2_51:
	s_mov_b32 s21, -1
	s_trap 2
	s_cbranch_execz .LBB2_69
	s_branch .LBB2_84
.LBB2_52:
	s_mov_b32 s21, -1
	s_trap 2
	s_cbranch_execz .LBB2_4
.LBB2_53:
	s_mov_b32 s3, 0
.LBB2_54:                               ; %Flow179
                                        ; implicit-def: $sgpr22
                                        ; implicit-def: $vgpr5
                                        ; implicit-def: $vgpr7
                                        ; implicit-def: $vgpr8
                                        ; implicit-def: $sgpr0
                                        ; implicit-def: $vgpr1
                                        ; implicit-def: $vgpr4
.LBB2_55:                               ; %Flow177
	s_mov_b32 s20, s2
	s_and_b32 vcc_lo, exec_lo, s21
	s_cbranch_vccz .LBB2_16
.LBB2_56:                               ; %UnifiedUnreachableBlock
	; divergent unreachable
	s_and_not1_b32 vcc_lo, exec_lo, s3
	s_cbranch_vccz .LBB2_17
.LBB2_57:                               ; %UnifiedReturnBlock
	s_endpgm
.LBB2_58:
	s_mov_b32 s6, 0
	s_mov_b32 s21, -1
	s_trap 2
	s_cbranch_execz .LBB2_6
	s_branch .LBB2_54
.LBB2_59:
	s_mov_b32 s6, 0
	s_mov_b32 s21, -1
	s_trap 2
	s_cbranch_execz .LBB2_8
	s_branch .LBB2_54
.LBB2_60:
	s_mov_b32 s6, 0
	s_mov_b32 s21, -1
	s_trap 2
	s_cbranch_execz .LBB2_10
	s_branch .LBB2_54
.LBB2_61:
	s_mov_b32 s21, -1
	s_trap 2
	s_cbranch_execz .LBB2_29
	s_branch .LBB2_65
.LBB2_62:
	s_mov_b32 s21, -1
	s_trap 2
	s_cbranch_execz .LBB2_31
	s_branch .LBB2_65
.LBB2_63:
	s_mov_b32 s21, -1
	s_trap 2
	s_cbranch_execz .LBB2_33
	s_branch .LBB2_65
.LBB2_64:
	s_mov_b32 s21, -1
	s_trap 2
	s_cbranch_execz .LBB2_35
.LBB2_65:
	s_mov_b32 s3, 0
                                        ; implicit-def: $sgpr22
                                        ; implicit-def: $vgpr5
                                        ; implicit-def: $vgpr7
                                        ; implicit-def: $vgpr8
                                        ; implicit-def: $sgpr0
	s_branch .LBB2_55
.LBB2_66:
	s_mov_b32 s4, 0
	s_mov_b32 s21, -1
	s_trap 2
	v_cndmask_b32_e64 v10, v10, v11, s0
	s_cbranch_execz .LBB2_39
	s_branch .LBB2_84
.LBB2_67:
	s_or_b32 exec_lo, exec_lo, s3
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_eq_u32 exec_lo, -1
	s_mov_b32 s1, -1
	s_cbranch_scc0 .LBB2_51
.LBB2_68:                               ; %Flow206
	s_and_not1_b32 vcc_lo, exec_lo, s1
	s_cbranch_vccnz .LBB2_84
.LBB2_69:                               ; %_Z16__shfl_down_syncIyfET0_T_S0_ji.exit36.i73
	s_waitcnt lgkmcnt(0)
	ds_swizzle_b32 v11, v5 offset:swizzle(BITMASK_PERM,"1pppp")
	s_cmp_lg_u32 exec_lo, -1
	s_mov_b32 s1, -1
	s_cbranch_scc1 .LBB2_81
; %bb.70:                               ; %Flow208
	s_and_not1_b32 vcc_lo, exec_lo, s1
	s_cbranch_vccnz .LBB2_84
.LBB2_71:                               ; %_Z16__shfl_down_syncIyfET0_T_S0_ji.exit38.i74
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v5, v5, v11
	s_cmp_lg_u32 exec_lo, -1
	s_mov_b32 s1, -1
	ds_bpermute_b32 v7, v7, v5
	s_cbranch_scc1 .LBB2_82
; %bb.72:                               ; %Flow210
	s_and_not1_b32 vcc_lo, exec_lo, s1
	s_cbranch_vccnz .LBB2_84
.LBB2_73:                               ; %_Z16__shfl_down_syncIyfET0_T_S0_ji.exit40.i75
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v5, v5, v7
	s_cmp_lg_u32 exec_lo, -1
	s_mov_b32 s1, -1
	ds_bpermute_b32 v7, v8, v5
	s_cbranch_scc1 .LBB2_83
; %bb.74:                               ; %Flow212
	s_and_not1_b32 vcc_lo, exec_lo, s1
	s_cbranch_vccnz .LBB2_84
.LBB2_75:                               ; %_Z16__shfl_down_syncIyfET0_T_S0_ji.exit42.i76
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v7, v5, v7
	s_cmp_lg_u32 exec_lo, -1
	s_mov_b32 s3, -1
	ds_bpermute_b32 v5, v9, v7
	s_cbranch_scc0 .LBB2_85
; %bb.76:
	s_mov_b32 s3, 0
	s_mov_b32 s21, -1
	s_trap 2
	s_branch .LBB2_85
.LBB2_77:
	s_mov_b32 s4, 0
	s_mov_b32 s21, -1
	s_trap 2
	s_cbranch_execz .LBB2_41
	s_branch .LBB2_84
.LBB2_78:
	s_mov_b32 s4, 0
	s_mov_b32 s21, -1
	s_trap 2
	s_cbranch_execz .LBB2_43
	s_branch .LBB2_84
.LBB2_79:
	s_mov_b32 s4, 0
	s_mov_b32 s21, -1
	s_trap 2
	s_cbranch_execz .LBB2_45
	s_branch .LBB2_84
.LBB2_80:
	s_mov_b32 s4, 0
	s_mov_b32 s21, -1
	s_trap 2
	s_cbranch_execz .LBB2_47
	s_branch .LBB2_84
.LBB2_81:
	s_mov_b32 s21, -1
	s_trap 2
	s_cbranch_execz .LBB2_71
	s_branch .LBB2_84
.LBB2_82:
	s_mov_b32 s21, -1
	s_trap 2
	s_cbranch_execz .LBB2_73
	s_branch .LBB2_84
.LBB2_83:
	s_mov_b32 s21, -1
	s_trap 2
	s_cbranch_execz .LBB2_75
.LBB2_84:
	s_mov_b32 s3, 0
                                        ; implicit-def: $vgpr5
                                        ; implicit-def: $vgpr7
.LBB2_85:                               ; %Flow197
	s_delay_alu instid0(VALU_DEP_1)
	v_rsq_f32_e32 v8, v10
	s_branch .LBB2_55
.Lfunc_end2:
	.size	_Z10rec_kernelPKfS0_S0_S0_S0_PfS1_i, .Lfunc_end2-_Z10rec_kernelPKfS0_S0_S0_S0_PfS1_i
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z10rec_kernelPKfS0_S0_S0_S0_PfS1_i
		.amdhsa_group_segment_fixed_size 1152
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 320
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 33
		.amdhsa_next_free_sgpr 23
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_shared_vgpr_count 0
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end2-_Z10rec_kernelPKfS0_S0_S0_S0_PfS1_i)<<4)&1008)>>4
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
                                        ; -- End function
	.set .L_Z10rec_kernelPKfS0_S0_S0_S0_PfS1_i.num_vgpr, 33
	.set .L_Z10rec_kernelPKfS0_S0_S0_S0_PfS1_i.num_agpr, 0
	.set .L_Z10rec_kernelPKfS0_S0_S0_S0_PfS1_i.numbered_sgpr, 23
	.set .L_Z10rec_kernelPKfS0_S0_S0_S0_PfS1_i.num_named_barrier, 0
	.set .L_Z10rec_kernelPKfS0_S0_S0_S0_PfS1_i.private_seg_size, 0
	.set .L_Z10rec_kernelPKfS0_S0_S0_S0_PfS1_i.uses_vcc, 1
	.set .L_Z10rec_kernelPKfS0_S0_S0_S0_PfS1_i.uses_flat_scratch, 0
	.set .L_Z10rec_kernelPKfS0_S0_S0_S0_PfS1_i.has_dyn_sized_stack, 0
	.set .L_Z10rec_kernelPKfS0_S0_S0_S0_PfS1_i.has_recursion, 0
	.set .L_Z10rec_kernelPKfS0_S0_S0_S0_PfS1_i.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 3036
; TotalNumSgprs: 25
; NumVgprs: 33
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 1152 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 4
; NumSGPRsForWavesPerEU: 25
; NumVGPRsForWavesPerEU: 33
; Occupancy: 16
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	_Z11rec0_kernelPKfS0_S0_S0_S0_PfS1_i ; -- Begin function _Z11rec0_kernelPKfS0_S0_S0_S0_PfS1_i
	.globl	_Z11rec0_kernelPKfS0_S0_S0_S0_PfS1_i
	.p2align	8
	.type	_Z11rec0_kernelPKfS0_S0_S0_S0_PfS1_i,@function
_Z11rec0_kernelPKfS0_S0_S0_S0_PfS1_i:   ; @_Z11rec0_kernelPKfS0_S0_S0_S0_PfS1_i
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_mov_b32 s4, -1
	s_mov_b32 s3, 0
	s_cmp_eq_u32 exec_lo, -1
	s_mov_b32 s21, 0
	s_cbranch_scc0 .LBB3_24
.LBB3_1:                                ; %Flow
	v_readfirstlane_b32 s20, v0
	s_and_not1_b32 vcc_lo, exec_lo, s4
	s_cbranch_vccnz .LBB3_15
; %bb.2:                                ; %_Z16__shfl_down_syncIyfET0_T_S0_ji.exit.i
	s_load_b256 s[4:11], s[0:1], 0x0
	v_lshl_add_u32 v2, s2, 7, v0
	s_load_b256 s[12:19], s[0:1], 0x20
	s_cmp_lg_u32 exec_lo, -1
	s_mov_b32 s3, -1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshlrev_b64 v[4:5], 2, v[2:3]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v6, vcc_lo, s6, v4
	v_add_co_ci_u32_e64 v7, null, s7, v5, vcc_lo
	global_load_b32 v6, v[6:7], off
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v7, v6, v6
	ds_swizzle_b32 v8, v7 offset:swizzle(BITMASK_PERM,"1pppp")
	s_cbranch_scc1 .LBB3_52
; %bb.3:                                ; %Flow178
	s_and_not1_b32 vcc_lo, exec_lo, s3
	s_cbranch_vccnz .LBB3_53
.LBB3_4:                                ; %_Z16__shfl_down_syncIyfET0_T_S0_ji.exit28.i
	v_mbcnt_lo_u32_b32 v1, -1, 0
	s_mov_b32 s3, 0
	s_cmp_lg_u32 exec_lo, -1
	s_mov_b32 s6, -1
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v8, v7, v8
	v_cmp_gt_u32_e32 vcc_lo, 24, v1
	v_cndmask_b32_e64 v9, 0, 8, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v9, v9, v1
	v_lshlrev_b32_e32 v7, 2, v9
	ds_bpermute_b32 v9, v7, v8
	s_cbranch_scc1 .LBB3_58
; %bb.5:                                ; %Flow180
	s_and_not1_b32 vcc_lo, exec_lo, s6
	s_cbranch_vccnz .LBB3_54
.LBB3_6:                                ; %_Z16__shfl_down_syncIyfET0_T_S0_ji.exit30.i
	v_cmp_gt_u32_e32 vcc_lo, 28, v1
	s_cmp_lg_u32 exec_lo, -1
	s_mov_b32 s6, -1
	v_cndmask_b32_e64 v10, 0, 4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v10, v10, v1
	s_waitcnt lgkmcnt(0)
	v_dual_add_f32 v9, v8, v9 :: v_dual_lshlrev_b32 v8, 2, v10
	ds_bpermute_b32 v10, v8, v9
	s_cbranch_scc1 .LBB3_59
; %bb.7:                                ; %Flow182
	s_and_not1_b32 vcc_lo, exec_lo, s6
	s_cbranch_vccnz .LBB3_54
.LBB3_8:                                ; %_Z16__shfl_down_syncIyfET0_T_S0_ji.exit32.i
	v_cmp_gt_u32_e32 vcc_lo, 30, v1
	s_cmp_lg_u32 exec_lo, -1
	s_mov_b32 s6, -1
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v10, v9, v10
	v_cndmask_b32_e64 v11, 0, 2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v11, v11, v1
	v_lshlrev_b32_e32 v9, 2, v11
	ds_bpermute_b32 v11, v9, v10
	s_cbranch_scc1 .LBB3_60
; %bb.9:                                ; %Flow184
	s_and_not1_b32 vcc_lo, exec_lo, s6
	s_cbranch_vccnz .LBB3_54
.LBB3_10:                               ; %_Z16__shfl_down_syncIyfET0_T_S0_ji.exit34.i
	v_add_co_u32 v4, vcc_lo, s4, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v5, null, s5, v5, vcc_lo
	v_cmp_ne_u32_e32 vcc_lo, 31, v1
	v_and_b32_e32 v12, 31, v0
	global_load_b32 v4, v[4:5], off
	v_add_co_ci_u32_e64 v1, null, 0, v1, vcc_lo
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v10, v10, v11
	v_cmp_eq_u32_e64 s3, 0, v12
	s_delay_alu instid0(VALU_DEP_3)
	v_lshlrev_b32_e32 v5, 2, v1
	ds_bpermute_b32 v11, v5, v10
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB3_12
; %bb.11:
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v10, v10, v11
	v_lshrrev_b32_e32 v11, 3, v0
	ds_store_b32 v11, v10 offset:1024
.LBB3_12:
	s_or_b32 exec_lo, exec_lo, s4
	s_load_b32 s0, s[0:1], 0x4c
	v_mov_b32_e32 v10, 0
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	buffer_gl0_inv
	s_bfe_u32 s0, s0, 0xb0005
	s_delay_alu instid0(SALU_CYCLE_1)
	v_cmp_gt_u32_e64 s1, s0, v0
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB3_25
; %bb.13:
	v_lshlrev_b32_e32 v10, 2, v0
	ds_load_b32 v10, v10 offset:1024
	s_or_b32 exec_lo, exec_lo, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_eq_u32 exec_lo, -1
	s_mov_b32 s0, -1
	s_cbranch_scc1 .LBB3_26
.LBB3_14:
	s_mov_b32 s21, -1
	s_trap 2
	s_cbranch_execz .LBB3_27
	s_branch .LBB3_65
.LBB3_15:
                                        ; implicit-def: $sgpr22
                                        ; implicit-def: $vgpr5
                                        ; implicit-def: $vgpr7
                                        ; implicit-def: $vgpr8
                                        ; implicit-def: $sgpr0
                                        ; implicit-def: $vgpr1
                                        ; implicit-def: $vgpr4
                                        ; implicit-def: $vgpr6
                                        ; implicit-def: $vgpr2_vgpr3
                                        ; implicit-def: $vgpr0
                                        ; implicit-def: $sgpr4_sgpr5_sgpr6_sgpr7_sgpr8_sgpr9_sgpr10_sgpr11_sgpr12_sgpr13_sgpr14_sgpr15_sgpr16_sgpr17_sgpr18_sgpr19
	s_and_b32 vcc_lo, exec_lo, s21
	s_cbranch_vccnz .LBB3_56
.LBB3_16:                               ; %Flow215
	s_and_not1_b32 vcc_lo, exec_lo, s3
	s_cbranch_vccnz .LBB3_57
.LBB3_17:                               ; %_Z16__shfl_down_syncIyfET0_T_S0_ji.exit44.i77
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v5, v7, v5
	v_lshlrev_b32_e32 v1, 2, v1
	s_barrier
	buffer_gl0_inv
	ds_bpermute_b32 v1, v1, v5
	s_and_saveexec_b32 s1, s22
	s_cbranch_execz .LBB3_19
; %bb.18:
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v1, v5, v1
	v_mov_b32_e32 v5, 0
	ds_store_b32 v5, v1 offset:1024
.LBB3_19:                               ; %_Z9block_sumfPf.exit78
	s_or_b32 exec_lo, exec_lo, s1
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v1, 0
	s_barrier
	buffer_gl0_inv
	v_mul_f32_e32 v9, 0x45800000, v8
	ds_load_b32 v5, v1 offset:1024
	s_ashr_i32 s21, s20, 31
	s_mov_b32 s5, 0
	s_lshl_b64 s[2:3], s[20:21], 2
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v5, 0x358637bd, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v7, 0x4b800000, v5
	v_cmp_gt_f32_e32 vcc_lo, 0x800000, v5
	v_cndmask_b32_e32 v5, v5, v7, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rsq_f32_e32 v5, v5
	v_mul_f32_e32 v7, 0x45800000, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v5, v5, v7, vcc_lo
	v_cndmask_b32_e64 v7, v8, v9, s0
	s_add_u32 s0, s10, s2
	s_addc_u32 s1, s11, s3
	v_dual_mul_f32 v4, v4, v5 :: v_dual_lshlrev_b32 v5, 2, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v6, v6, v7
	v_mul_f32_e32 v4, 0x3db504f3, v4
	s_barrier
	buffer_gl0_inv
	s_barrier
	buffer_gl0_inv
	ds_store_2addr_stride64_b32 v5, v6, v4 offset1:2
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_gl0_inv
	global_load_b32 v7, v1, s[0:1]
	s_add_u32 s0, s12, s2
	s_addc_u32 s1, s13, s3
	v_lshlrev_b64 v[5:6], 2, v[0:1]
	global_load_b32 v4, v1, s[0:1]
	s_lshl_b64 s[0:1], s[20:21], 16
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	s_add_u32 s2, s14, s0
	s_addc_u32 s3, s15, s1
	v_add_co_u32 v5, vcc_lo, s2, v5
	v_add_co_ci_u32_e64 v6, null, s3, v6, vcc_lo
	s_mov_b64 s[2:3], 0
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s4, v7
.LBB3_20:                               ; =>This Inner Loop Header: Depth=1
	v_add_co_u32 v7, vcc_lo, v5, s2
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v8, null, s3, v6, vcc_lo
	v_mov_b32_e32 v31, s5
	s_add_u32 s2, s2, 0x2000
	s_addc_u32 s3, s3, 0
	s_clause 0x7
	global_load_b32 v15, v[7:8], off
	global_load_b32 v16, v[7:8], off offset:512
	global_load_b32 v17, v[7:8], off offset:1024
	global_load_b32 v18, v[7:8], off offset:1536
	global_load_b32 v19, v[7:8], off offset:2048
	global_load_b32 v20, v[7:8], off offset:2560
	global_load_b32 v21, v[7:8], off offset:3072
	global_load_b32 v22, v[7:8], off offset:3584
	v_add_co_u32 v7, vcc_lo, 0x1000, v7
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v8, null, 0, v8, vcc_lo
	s_add_i32 s5, s5, 64
	s_cmp_eq_u32 s2, 0x10000
	s_clause 0x7
	global_load_b32 v23, v[7:8], off
	global_load_b32 v24, v[7:8], off offset:512
	global_load_b32 v25, v[7:8], off offset:1024
	global_load_b32 v26, v[7:8], off offset:1536
	global_load_b32 v27, v[7:8], off offset:2048
	global_load_b32 v28, v[7:8], off offset:2560
	global_load_b32 v29, v[7:8], off offset:3072
	global_load_b32 v30, v[7:8], off offset:3584
	ds_load_b128 v[7:10], v31
	ds_load_b128 v[11:14], v31 offset:16
	s_waitcnt vmcnt(15) lgkmcnt(1)
	v_fmac_f32_e32 v1, v15, v7
	s_waitcnt vmcnt(14)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v1, v16, v8
	s_waitcnt vmcnt(13)
	v_fmac_f32_e32 v1, v17, v9
	s_waitcnt vmcnt(12)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v1, v18, v10
	ds_load_b128 v[7:10], v31 offset:32
	s_waitcnt vmcnt(11) lgkmcnt(1)
	v_fmac_f32_e32 v1, v19, v11
	s_waitcnt vmcnt(10)
	v_fmac_f32_e32 v1, v20, v12
	s_waitcnt vmcnt(9)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v1, v21, v13
	s_waitcnt vmcnt(8)
	v_fmac_f32_e32 v1, v22, v14
	ds_load_b128 v[11:14], v31 offset:48
	s_waitcnt vmcnt(7) lgkmcnt(1)
	v_fmac_f32_e32 v1, v23, v7
	s_waitcnt vmcnt(6)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v1, v24, v8
	s_waitcnt vmcnt(5)
	v_fmac_f32_e32 v1, v25, v9
	s_waitcnt vmcnt(4)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v1, v26, v10
	s_waitcnt vmcnt(3) lgkmcnt(0)
	v_fmac_f32_e32 v1, v27, v11
	s_waitcnt vmcnt(2)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v1, v28, v12
	s_waitcnt vmcnt(1)
	v_fmac_f32_e32 v1, v29, v13
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v1, v30, v14
	s_cbranch_scc0 .LBB3_20
; %bb.21:
	v_lshlrev_b64 v[2:3], 2, v[2:3]
	s_mul_f32 s2, s4, 0x3fb8aa3b
	s_add_u32 s0, s14, s0
	s_addc_u32 s1, s15, s1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	s_xor_b32 s3, s2, 0x80000000
	v_add_co_u32 v5, vcc_lo, s8, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v6, null, s9, v3, vcc_lo
	s_rndne_f32 s5, s2
	s_fmamk_f32 s3, s4, 0x3fb8aa3b, s3
	s_cmp_nlt_f32 s4, 0xc2ce8ed0
	global_load_b32 v5, v[5:6], off
	s_sub_f32 s2, s2, s5
	s_fmamk_f32 s3, s4, 0x32a5705f, s3
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_ngt_f32 s4, 0x42b17218
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_3)
	s_add_f32 s2, s2, s3
	v_exp_f32_e32 v6, s2
	s_cvt_i32_f32 s2, s5
	s_delay_alu instid0(TRANS32_DEP_1) | instid1(SALU_CYCLE_3)
	v_ldexp_f32 v6, v6, s2
	s_mov_b32 s2, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v6, 0, v6, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	v_cndmask_b32_e32 v6, 0x7f800000, v6, vcc_lo
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fma_f32 v1, -v6, v1, v5
	v_mov_b32_e32 v5, 0
	v_mul_f32_e32 v7, v4, v1
	v_mov_b32_e32 v1, 0
.LBB3_22:                               ; =>This Inner Loop Header: Depth=1
	v_lshl_or_b32 v4, s2, 7, v0
	s_lshl_b32 s3, s2, 2
	s_add_i32 s2, s2, 8
	v_mov_b32_e32 v20, s3
	s_cmpk_eq_i32 s2, 0x80
	v_lshlrev_b64 v[8:9], 2, v[4:5]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v24, vcc_lo, s0, v8
	v_add_co_ci_u32_e64 v25, null, s1, v9, vcc_lo
	ds_load_b128 v[8:11], v20
	ds_load_b128 v[12:15], v20 offset:512
	ds_load_b128 v[16:19], v20 offset:16
	s_clause 0x7
	global_load_b32 v4, v[24:25], off
	global_load_b32 v26, v[24:25], off offset:512
	global_load_b32 v27, v[24:25], off offset:1024
	global_load_b32 v28, v[24:25], off offset:1536
	global_load_b32 v29, v[24:25], off offset:2048
	global_load_b32 v30, v[24:25], off offset:2560
	global_load_b32 v31, v[24:25], off offset:3072
	global_load_b32 v32, v[24:25], off offset:3584
	s_waitcnt lgkmcnt(2)
	v_mul_f32_e32 v9, v7, v9
	v_mul_f32_e32 v8, v7, v8
	ds_load_b128 v[20:23], v20 offset:528
	s_waitcnt vmcnt(6)
	v_fmac_f32_e32 v9, v6, v26
	v_fmac_f32_e32 v8, v6, v4
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_dual_mul_f32 v4, v7, v10 :: v_dual_fmac_f32 v1, v12, v8
	s_waitcnt vmcnt(5)
	v_dual_fmac_f32 v4, v6, v27 :: v_dual_fmac_f32 v1, v13, v9
	v_mul_f32_e32 v10, v7, v11
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v11, v7, v16
	v_mul_f32_e32 v13, v7, v18
	v_fmac_f32_e32 v1, v14, v4
	s_waitcnt vmcnt(4)
	v_fmac_f32_e32 v10, v6, v28
	s_waitcnt vmcnt(3)
	v_dual_fmac_f32 v11, v6, v29 :: v_dual_mul_f32 v14, v7, v19
	s_waitcnt vmcnt(1)
	v_fmac_f32_e32 v13, v6, v31
	s_clause 0x3
	global_store_b32 v[24:25], v8, off
	global_store_b32 v[24:25], v9, off offset:512
	global_store_b32 v[24:25], v4, off offset:1024
	global_store_b32 v[24:25], v10, off offset:1536
	s_waitcnt vmcnt(0)
	v_dual_fmac_f32 v1, v15, v10 :: v_dual_fmac_f32 v14, v6, v32
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_mul_f32 v12, v7, v17 :: v_dual_fmac_f32 v1, v20, v11
	v_fmac_f32_e32 v12, v6, v30
	s_clause 0x3
	global_store_b32 v[24:25], v11, off offset:2048
	global_store_b32 v[24:25], v12, off offset:2560
	global_store_b32 v[24:25], v13, off offset:3072
	global_store_b32 v[24:25], v14, off offset:3584
	v_fmac_f32_e32 v1, v21, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v1, v22, v13
	v_fmac_f32_e32 v1, v23, v14
	s_cbranch_scc0 .LBB3_22
; %bb.23:
	v_add_co_u32 v2, vcc_lo, s16, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v3, null, s17, v3, vcc_lo
	global_store_b32 v[2:3], v1, off
	s_endpgm
.LBB3_24:
	s_mov_b32 s4, 0
	s_mov_b32 s21, -1
	s_trap 2
	s_branch .LBB3_1
.LBB3_25:
	s_or_b32 exec_lo, exec_lo, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_eq_u32 exec_lo, -1
	s_mov_b32 s0, -1
	s_cbranch_scc0 .LBB3_14
.LBB3_26:                               ; %Flow186
	s_and_not1_b32 vcc_lo, exec_lo, s0
	s_cbranch_vccnz .LBB3_65
.LBB3_27:                               ; %_Z16__shfl_down_syncIyfET0_T_S0_ji.exit36.i
	s_waitcnt lgkmcnt(0)
	ds_swizzle_b32 v11, v10 offset:swizzle(BITMASK_PERM,"1pppp")
	s_cmp_lg_u32 exec_lo, -1
	s_mov_b32 s0, -1
	s_cbranch_scc1 .LBB3_61
; %bb.28:                               ; %Flow188
	s_and_not1_b32 vcc_lo, exec_lo, s0
	s_cbranch_vccnz .LBB3_65
.LBB3_29:                               ; %_Z16__shfl_down_syncIyfET0_T_S0_ji.exit38.i
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v10, v10, v11
	s_cmp_lg_u32 exec_lo, -1
	s_mov_b32 s0, -1
	ds_bpermute_b32 v11, v7, v10
	s_cbranch_scc1 .LBB3_62
; %bb.30:                               ; %Flow190
	s_and_not1_b32 vcc_lo, exec_lo, s0
	s_cbranch_vccnz .LBB3_65
.LBB3_31:                               ; %_Z16__shfl_down_syncIyfET0_T_S0_ji.exit40.i
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v10, v10, v11
	s_cmp_lg_u32 exec_lo, -1
	s_mov_b32 s0, -1
	ds_bpermute_b32 v11, v8, v10
	s_cbranch_scc1 .LBB3_63
; %bb.32:                               ; %Flow192
	s_and_not1_b32 vcc_lo, exec_lo, s0
	s_cbranch_vccnz .LBB3_65
.LBB3_33:                               ; %_Z16__shfl_down_syncIyfET0_T_S0_ji.exit42.i
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v10, v10, v11
	s_cmp_lg_u32 exec_lo, -1
	s_mov_b32 s0, -1
	ds_bpermute_b32 v11, v9, v10
	s_cbranch_scc1 .LBB3_64
; %bb.34:                               ; %Flow194
	s_and_not1_b32 vcc_lo, exec_lo, s0
	s_cbranch_vccnz .LBB3_65
.LBB3_35:                               ; %_Z16__shfl_down_syncIyfET0_T_S0_ji.exit44.i
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v10, v10, v11
	v_cmp_eq_u32_e64 s22, 0, v0
	s_barrier
	buffer_gl0_inv
	ds_bpermute_b32 v11, v5, v10
	s_and_saveexec_b32 s0, s22
	s_cbranch_execz .LBB3_37
; %bb.36:
	s_waitcnt lgkmcnt(0)
	v_dual_add_f32 v10, v10, v11 :: v_dual_mov_b32 v11, 0
	ds_store_b32 v11, v10 offset:1024
.LBB3_37:                               ; %_Z9block_sumfPf.exit
	s_or_b32 exec_lo, exec_lo, s0
	v_mov_b32_e32 v10, 0
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_gl0_inv
	s_mov_b32 s4, -1
	ds_load_b32 v10, v10 offset:1024
	s_cmp_eq_u32 exec_lo, -1
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v10, 0x358637bd, v10
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_f32_e64 s0, 0x800000, v10
	v_mul_f32_e32 v11, 0x4b800000, v10
	s_barrier
	buffer_gl0_inv
	s_barrier
	buffer_gl0_inv
	s_cbranch_scc0 .LBB3_66
; %bb.38:                               ; %Flow196
	v_cndmask_b32_e64 v10, v10, v11, s0
	s_and_not1_b32 vcc_lo, exec_lo, s4
	s_cbranch_vccnz .LBB3_84
.LBB3_39:                               ; %_Z16__shfl_down_syncIyfET0_T_S0_ji.exit.i64
	v_mul_f32_e32 v11, v4, v4
	s_cmp_lg_u32 exec_lo, -1
	s_mov_b32 s4, -1
	ds_swizzle_b32 v12, v11 offset:swizzle(BITMASK_PERM,"1pppp")
	s_cbranch_scc1 .LBB3_77
; %bb.40:                               ; %Flow198
	s_and_not1_b32 vcc_lo, exec_lo, s4
	s_cbranch_vccnz .LBB3_84
.LBB3_41:                               ; %_Z16__shfl_down_syncIyfET0_T_S0_ji.exit28.i65
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v11, v11, v12
	s_cmp_lg_u32 exec_lo, -1
	s_mov_b32 s4, -1
	ds_bpermute_b32 v12, v7, v11
	s_cbranch_scc1 .LBB3_78
; %bb.42:                               ; %Flow200
	s_and_not1_b32 vcc_lo, exec_lo, s4
	s_cbranch_vccnz .LBB3_84
.LBB3_43:                               ; %_Z16__shfl_down_syncIyfET0_T_S0_ji.exit30.i67
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v11, v11, v12
	s_cmp_lg_u32 exec_lo, -1
	s_mov_b32 s4, -1
	ds_bpermute_b32 v12, v8, v11
	s_cbranch_scc1 .LBB3_79
; %bb.44:                               ; %Flow202
	s_and_not1_b32 vcc_lo, exec_lo, s4
	s_cbranch_vccnz .LBB3_84
.LBB3_45:                               ; %_Z16__shfl_down_syncIyfET0_T_S0_ji.exit32.i69
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v11, v11, v12
	s_cmp_lg_u32 exec_lo, -1
	s_mov_b32 s4, -1
	ds_bpermute_b32 v12, v9, v11
	s_cbranch_scc1 .LBB3_80
; %bb.46:                               ; %Flow204
	s_and_not1_b32 vcc_lo, exec_lo, s4
	s_cbranch_vccnz .LBB3_84
.LBB3_47:                               ; %_Z16__shfl_down_syncIyfET0_T_S0_ji.exit34.i71
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v11, v11, v12
	ds_bpermute_b32 v5, v5, v11
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB3_49
; %bb.48:
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v5, v11, v5
	v_lshrrev_b32_e32 v11, 3, v0
	ds_store_b32 v11, v5 offset:1024
.LBB3_49:
	s_or_b32 exec_lo, exec_lo, s4
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v5, 0
	s_barrier
	buffer_gl0_inv
	s_and_saveexec_b32 s3, s1
	s_cbranch_execz .LBB3_67
; %bb.50:
	v_lshlrev_b32_e32 v5, 2, v0
	ds_load_b32 v5, v5 offset:1024
	s_or_b32 exec_lo, exec_lo, s3
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_eq_u32 exec_lo, -1
	s_mov_b32 s1, -1
	s_cbranch_scc1 .LBB3_68
.LBB3_51:
	s_mov_b32 s21, -1
	s_trap 2
	s_cbranch_execz .LBB3_69
	s_branch .LBB3_84
.LBB3_52:
	s_mov_b32 s21, -1
	s_trap 2
	s_cbranch_execz .LBB3_4
.LBB3_53:
	s_mov_b32 s3, 0
.LBB3_54:                               ; %Flow179
                                        ; implicit-def: $sgpr22
                                        ; implicit-def: $vgpr5
                                        ; implicit-def: $vgpr7
                                        ; implicit-def: $vgpr8
                                        ; implicit-def: $sgpr0
                                        ; implicit-def: $vgpr1
                                        ; implicit-def: $vgpr4
.LBB3_55:                               ; %Flow177
	s_mov_b32 s20, s2
	s_and_b32 vcc_lo, exec_lo, s21
	s_cbranch_vccz .LBB3_16
.LBB3_56:                               ; %UnifiedUnreachableBlock
	; divergent unreachable
	s_and_not1_b32 vcc_lo, exec_lo, s3
	s_cbranch_vccz .LBB3_17
.LBB3_57:                               ; %UnifiedReturnBlock
	s_endpgm
.LBB3_58:
	s_mov_b32 s6, 0
	s_mov_b32 s21, -1
	s_trap 2
	s_cbranch_execz .LBB3_6
	s_branch .LBB3_54
.LBB3_59:
	s_mov_b32 s6, 0
	s_mov_b32 s21, -1
	s_trap 2
	s_cbranch_execz .LBB3_8
	s_branch .LBB3_54
.LBB3_60:
	s_mov_b32 s6, 0
	s_mov_b32 s21, -1
	s_trap 2
	s_cbranch_execz .LBB3_10
	s_branch .LBB3_54
.LBB3_61:
	s_mov_b32 s21, -1
	s_trap 2
	s_cbranch_execz .LBB3_29
	s_branch .LBB3_65
.LBB3_62:
	s_mov_b32 s21, -1
	s_trap 2
	s_cbranch_execz .LBB3_31
	s_branch .LBB3_65
.LBB3_63:
	s_mov_b32 s21, -1
	s_trap 2
	s_cbranch_execz .LBB3_33
	s_branch .LBB3_65
.LBB3_64:
	s_mov_b32 s21, -1
	s_trap 2
	s_cbranch_execz .LBB3_35
.LBB3_65:
	s_mov_b32 s3, 0
                                        ; implicit-def: $sgpr22
                                        ; implicit-def: $vgpr5
                                        ; implicit-def: $vgpr7
                                        ; implicit-def: $vgpr8
                                        ; implicit-def: $sgpr0
	s_branch .LBB3_55
.LBB3_66:
	s_mov_b32 s4, 0
	s_mov_b32 s21, -1
	s_trap 2
	v_cndmask_b32_e64 v10, v10, v11, s0
	s_cbranch_execz .LBB3_39
	s_branch .LBB3_84
.LBB3_67:
	s_or_b32 exec_lo, exec_lo, s3
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_eq_u32 exec_lo, -1
	s_mov_b32 s1, -1
	s_cbranch_scc0 .LBB3_51
.LBB3_68:                               ; %Flow206
	s_and_not1_b32 vcc_lo, exec_lo, s1
	s_cbranch_vccnz .LBB3_84
.LBB3_69:                               ; %_Z16__shfl_down_syncIyfET0_T_S0_ji.exit36.i73
	s_waitcnt lgkmcnt(0)
	ds_swizzle_b32 v11, v5 offset:swizzle(BITMASK_PERM,"1pppp")
	s_cmp_lg_u32 exec_lo, -1
	s_mov_b32 s1, -1
	s_cbranch_scc1 .LBB3_81
; %bb.70:                               ; %Flow208
	s_and_not1_b32 vcc_lo, exec_lo, s1
	s_cbranch_vccnz .LBB3_84
.LBB3_71:                               ; %_Z16__shfl_down_syncIyfET0_T_S0_ji.exit38.i74
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v5, v5, v11
	s_cmp_lg_u32 exec_lo, -1
	s_mov_b32 s1, -1
	ds_bpermute_b32 v7, v7, v5
	s_cbranch_scc1 .LBB3_82
; %bb.72:                               ; %Flow210
	s_and_not1_b32 vcc_lo, exec_lo, s1
	s_cbranch_vccnz .LBB3_84
.LBB3_73:                               ; %_Z16__shfl_down_syncIyfET0_T_S0_ji.exit40.i75
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v5, v5, v7
	s_cmp_lg_u32 exec_lo, -1
	s_mov_b32 s1, -1
	ds_bpermute_b32 v7, v8, v5
	s_cbranch_scc1 .LBB3_83
; %bb.74:                               ; %Flow212
	s_and_not1_b32 vcc_lo, exec_lo, s1
	s_cbranch_vccnz .LBB3_84
.LBB3_75:                               ; %_Z16__shfl_down_syncIyfET0_T_S0_ji.exit42.i76
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v7, v5, v7
	s_cmp_lg_u32 exec_lo, -1
	s_mov_b32 s3, -1
	ds_bpermute_b32 v5, v9, v7
	s_cbranch_scc0 .LBB3_85
; %bb.76:
	s_mov_b32 s3, 0
	s_mov_b32 s21, -1
	s_trap 2
	s_branch .LBB3_85
.LBB3_77:
	s_mov_b32 s4, 0
	s_mov_b32 s21, -1
	s_trap 2
	s_cbranch_execz .LBB3_41
	s_branch .LBB3_84
.LBB3_78:
	s_mov_b32 s4, 0
	s_mov_b32 s21, -1
	s_trap 2
	s_cbranch_execz .LBB3_43
	s_branch .LBB3_84
.LBB3_79:
	s_mov_b32 s4, 0
	s_mov_b32 s21, -1
	s_trap 2
	s_cbranch_execz .LBB3_45
	s_branch .LBB3_84
.LBB3_80:
	s_mov_b32 s4, 0
	s_mov_b32 s21, -1
	s_trap 2
	s_cbranch_execz .LBB3_47
	s_branch .LBB3_84
.LBB3_81:
	s_mov_b32 s21, -1
	s_trap 2
	s_cbranch_execz .LBB3_71
	s_branch .LBB3_84
.LBB3_82:
	s_mov_b32 s21, -1
	s_trap 2
	s_cbranch_execz .LBB3_73
	s_branch .LBB3_84
.LBB3_83:
	s_mov_b32 s21, -1
	s_trap 2
	s_cbranch_execz .LBB3_75
.LBB3_84:
	s_mov_b32 s3, 0
                                        ; implicit-def: $vgpr5
                                        ; implicit-def: $vgpr7
.LBB3_85:                               ; %Flow197
	s_delay_alu instid0(VALU_DEP_1)
	v_rsq_f32_e32 v8, v10
	s_branch .LBB3_55
.Lfunc_end3:
	.size	_Z11rec0_kernelPKfS0_S0_S0_S0_PfS1_i, .Lfunc_end3-_Z11rec0_kernelPKfS0_S0_S0_S0_PfS1_i
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z11rec0_kernelPKfS0_S0_S0_S0_PfS1_i
		.amdhsa_group_segment_fixed_size 1152
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 320
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 33
		.amdhsa_next_free_sgpr 23
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_shared_vgpr_count 0
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end3-_Z11rec0_kernelPKfS0_S0_S0_S0_PfS1_i)<<4)&1008)>>4
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
                                        ; -- End function
	.set .L_Z11rec0_kernelPKfS0_S0_S0_S0_PfS1_i.num_vgpr, 33
	.set .L_Z11rec0_kernelPKfS0_S0_S0_S0_PfS1_i.num_agpr, 0
	.set .L_Z11rec0_kernelPKfS0_S0_S0_S0_PfS1_i.numbered_sgpr, 23
	.set .L_Z11rec0_kernelPKfS0_S0_S0_S0_PfS1_i.num_named_barrier, 0
	.set .L_Z11rec0_kernelPKfS0_S0_S0_S0_PfS1_i.private_seg_size, 0
	.set .L_Z11rec0_kernelPKfS0_S0_S0_S0_PfS1_i.uses_vcc, 1
	.set .L_Z11rec0_kernelPKfS0_S0_S0_S0_PfS1_i.uses_flat_scratch, 0
	.set .L_Z11rec0_kernelPKfS0_S0_S0_S0_PfS1_i.has_dyn_sized_stack, 0
	.set .L_Z11rec0_kernelPKfS0_S0_S0_S0_PfS1_i.has_recursion, 0
	.set .L_Z11rec0_kernelPKfS0_S0_S0_S0_PfS1_i.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 3036
; TotalNumSgprs: 25
; NumVgprs: 33
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 1152 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 4
; NumSGPRsForWavesPerEU: 25
; NumVGPRsForWavesPerEU: 33
; Occupancy: 16
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	_Z11rec1_kernelPKfS0_S0_S0_S0_PfS1_i ; -- Begin function _Z11rec1_kernelPKfS0_S0_S0_S0_PfS1_i
	.globl	_Z11rec1_kernelPKfS0_S0_S0_S0_PfS1_i
	.p2align	8
	.type	_Z11rec1_kernelPKfS0_S0_S0_S0_PfS1_i,@function
_Z11rec1_kernelPKfS0_S0_S0_S0_PfS1_i:   ; @_Z11rec1_kernelPKfS0_S0_S0_S0_PfS1_i
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_b256 s[4:11], s[0:1], 0x0
	v_lshl_add_u32 v1, s2, 7, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v2, 31, v1
	v_lshlrev_b64 v[1:2], 2, v[1:2]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v3, vcc_lo, s6, v1
	v_add_co_ci_u32_e64 v4, null, s7, v2, vcc_lo
	global_load_b32 v3, v[3:4], off
	v_add_co_u32 v4, vcc_lo, s4, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v5, null, s5, v2, vcc_lo
	v_and_b32_e32 v6, 31, v0
	global_load_b32 v4, v[4:5], off
	v_cmp_gt_u32_e32 vcc_lo, 24, v6
	v_cndmask_b32_e64 v5, 0, 8, vcc_lo
	v_add_lshl_u32 v7, v5, v6, 2
	s_waitcnt vmcnt(1)
	v_mul_f32_e32 v8, v3, v3
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	ds_swizzle_b32 v5, v8 offset:swizzle(BITMASK_PERM,"1pppp")
	;;#ASMEND
	v_fmac_f32_e32 v5, v3, v3
	v_cmp_gt_u32_e32 vcc_lo, 28, v6
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	ds_bpermute_b32 v10, v7, v5
	;;#ASMEND
	v_cmp_eq_u32_e64 s3, 0, v6
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_add_f32_e32 v5, v5, v10
	v_cndmask_b32_e64 v9, 0, 4, vcc_lo
	v_cmp_gt_u32_e32 vcc_lo, 30, v6
	v_add_lshl_u32 v9, v9, v6, 2
	v_cndmask_b32_e64 v8, 0, 2, vcc_lo
	v_cmp_ne_u32_e32 vcc_lo, 31, v6
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	ds_bpermute_b32 v10, v9, v5
	;;#ASMEND
	v_add_f32_e32 v5, v5, v10
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_add_lshl_u32 v8, v8, v6, 2
	v_add_co_ci_u32_e64 v10, null, 0, v6, vcc_lo
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	ds_bpermute_b32 v11, v8, v5
	;;#ASMEND
	v_dual_add_f32 v5, v5, v11 :: v_dual_lshlrev_b32 v10, 2, v10
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	ds_bpermute_b32 v6, v10, v5
	;;#ASMEND
	s_and_saveexec_b32 s4, s3
; %bb.1:
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v5, v5, v6
	v_lshrrev_b32_e32 v6, 3, v0
	ds_store_b32 v6, v5 offset:1024
; %bb.2:
	s_or_b32 exec_lo, exec_lo, s4
	s_load_b32 s4, s[0:1], 0x4c
	v_mov_b32_e32 v11, 0
	v_lshlrev_b32_e32 v5, 2, v0
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	buffer_gl0_inv
	s_bfe_u32 s4, s4, 0xb0005
	s_delay_alu instid0(SALU_CYCLE_1)
	v_cmp_gt_u32_e64 s5, s4, v0
	s_and_saveexec_b32 s4, s5
; %bb.3:
	ds_load_b32 v11, v5 offset:1024
; %bb.4:
	s_or_b32 exec_lo, exec_lo, s4
	s_mov_b32 s4, -1
	s_mov_b32 s12, 0
	s_cmp_eq_u32 exec_lo, -1
	s_mov_b32 s6, 0
	s_cbranch_scc0 .LBB4_30
; %bb.5:                                ; %Flow
	s_and_not1_b32 vcc_lo, exec_lo, s4
	s_cbranch_vccnz .LBB4_35
.LBB4_6:                                ; %_Z16__shfl_down_syncIyfET0_T_S0_ji.exit.i
	s_waitcnt lgkmcnt(0)
	ds_swizzle_b32 v12, v11 offset:swizzle(BITMASK_PERM,"1pppp")
	s_cmp_lg_u32 exec_lo, -1
	s_mov_b32 s4, -1
	s_cbranch_scc1 .LBB4_31
; %bb.7:                                ; %Flow169
	s_and_not1_b32 vcc_lo, exec_lo, s4
	s_cbranch_vccnz .LBB4_35
.LBB4_8:                                ; %_Z16__shfl_down_syncIyfET0_T_S0_ji.exit32.i
	v_mbcnt_lo_u32_b32 v6, -1, 0
	s_cmp_lg_u32 exec_lo, -1
	s_mov_b32 s4, -1
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v12, v11, v12
	v_cmp_gt_u32_e32 vcc_lo, 24, v6
	v_cndmask_b32_e64 v13, 0, 8, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v13, v13, v6
	v_lshlrev_b32_e32 v11, 2, v13
	ds_bpermute_b32 v13, v11, v12
	s_cbranch_scc1 .LBB4_32
; %bb.9:                                ; %Flow171
	s_and_not1_b32 vcc_lo, exec_lo, s4
	s_cbranch_vccnz .LBB4_35
.LBB4_10:                               ; %_Z16__shfl_down_syncIyfET0_T_S0_ji.exit34.i
	v_cmp_gt_u32_e32 vcc_lo, 28, v6
	s_cmp_lg_u32 exec_lo, -1
	s_mov_b32 s4, -1
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v13, v12, v13
	v_cndmask_b32_e64 v14, 0, 4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v14, v14, v6
	v_lshlrev_b32_e32 v12, 2, v14
	ds_bpermute_b32 v14, v12, v13
	s_cbranch_scc1 .LBB4_33
; %bb.11:                               ; %Flow173
	s_and_not1_b32 vcc_lo, exec_lo, s4
	s_cbranch_vccnz .LBB4_35
.LBB4_12:                               ; %_Z16__shfl_down_syncIyfET0_T_S0_ji.exit36.i
	v_cmp_gt_u32_e32 vcc_lo, 30, v6
	s_cmp_lg_u32 exec_lo, -1
	s_mov_b32 s4, -1
	v_cndmask_b32_e64 v15, 0, 2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v15, v15, v6
	s_waitcnt lgkmcnt(0)
	v_dual_add_f32 v14, v13, v14 :: v_dual_lshlrev_b32 v13, 2, v15
	ds_bpermute_b32 v15, v13, v14
	s_cbranch_scc1 .LBB4_34
; %bb.13:                               ; %Flow175
	s_and_not1_b32 vcc_lo, exec_lo, s4
	s_cbranch_vccnz .LBB4_35
.LBB4_14:                               ; %_Z16__shfl_down_syncIyfET0_T_S0_ji.exit38.i
	v_cmp_ne_u32_e32 vcc_lo, 31, v6
	v_cmp_eq_u32_e64 s7, 0, v0
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_gl0_inv
	v_add_co_ci_u32_e64 v6, null, 0, v6, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v14, v14, v15 :: v_dual_lshlrev_b32 v15, 2, v6
	ds_bpermute_b32 v15, v15, v14
	s_and_saveexec_b32 s4, s7
	s_cbranch_execz .LBB4_16
; %bb.15:
	s_waitcnt lgkmcnt(0)
	v_dual_add_f32 v14, v14, v15 :: v_dual_mov_b32 v15, 0
	ds_store_b32 v15, v14 offset:1024
.LBB4_16:                               ; %_Z14block_sum_handfPf.exit
	s_or_b32 exec_lo, exec_lo, s4
	v_mov_b32_e32 v14, 0
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_gl0_inv
	v_mul_f32_e32 v16, v4, v4
	ds_load_b32 v15, v14 offset:1024
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_gl0_inv
	s_barrier
	buffer_gl0_inv
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	ds_swizzle_b32 v16, v16 offset:swizzle(BITMASK_PERM,"1pppp")
	;;#ASMEND
	v_fmac_f32_e32 v16, v4, v4
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	ds_bpermute_b32 v7, v7, v16
	;;#ASMEND
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_f32_e32 v7, v16, v7
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	ds_bpermute_b32 v9, v9, v7
	;;#ASMEND
	v_add_f32_e32 v9, v7, v9
	v_add_f32_e32 v7, 0x358637bd, v15
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	ds_bpermute_b32 v8, v8, v9
	;;#ASMEND
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_f32_e32 v8, v9, v8
	v_cmp_gt_f32_e64 s4, 0x800000, v7
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	ds_bpermute_b32 v9, v10, v8
	;;#ASMEND
	s_and_saveexec_b32 s12, s3
; %bb.17:
	v_add_f32_e32 v8, v8, v9
	v_lshrrev_b32_e32 v9, 3, v0
	ds_store_b32 v9, v8 offset:1024
; %bb.18:
	s_or_b32 exec_lo, exec_lo, s12
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_gl0_inv
	s_and_saveexec_b32 s3, s5
; %bb.19:
	ds_load_b32 v14, v5 offset:1024
; %bb.20:
	s_or_b32 exec_lo, exec_lo, s3
	v_mul_f32_e32 v8, 0x4b800000, v7
	s_cmp_eq_u32 exec_lo, -1
	s_mov_b32 s3, -1
	s_cbranch_scc0 .LBB4_46
; %bb.21:                               ; %Flow177
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v9, v7, v8, s4
	s_and_not1_b32 vcc_lo, exec_lo, s3
	s_cbranch_vccnz .LBB4_50
.LBB4_22:                               ; %_Z16__shfl_down_syncIyfET0_T_S0_ji.exit.i69
	s_waitcnt lgkmcnt(0)
	ds_swizzle_b32 v7, v14 offset:swizzle(BITMASK_PERM,"1pppp")
	s_cmp_lg_u32 exec_lo, -1
	s_mov_b32 s3, -1
	s_cbranch_scc1 .LBB4_47
; %bb.23:                               ; %Flow179
	s_and_not1_b32 vcc_lo, exec_lo, s3
	s_cbranch_vccnz .LBB4_50
.LBB4_24:                               ; %_Z16__shfl_down_syncIyfET0_T_S0_ji.exit32.i70
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v7, v14, v7
	s_cmp_lg_u32 exec_lo, -1
	s_mov_b32 s3, -1
	ds_bpermute_b32 v8, v11, v7
	s_cbranch_scc1 .LBB4_48
; %bb.25:                               ; %Flow181
	s_and_not1_b32 vcc_lo, exec_lo, s3
	s_cbranch_vccnz .LBB4_50
.LBB4_26:                               ; %_Z16__shfl_down_syncIyfET0_T_S0_ji.exit34.i72
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v7, v7, v8
	s_cmp_lg_u32 exec_lo, -1
	s_mov_b32 s3, -1
	ds_bpermute_b32 v8, v12, v7
	s_cbranch_scc1 .LBB4_49
; %bb.27:                               ; %Flow183
	s_and_not1_b32 vcc_lo, exec_lo, s3
	s_cbranch_vccnz .LBB4_50
.LBB4_28:                               ; %_Z16__shfl_down_syncIyfET0_T_S0_ji.exit36.i74
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v8, v7, v8
	s_cmp_lg_u32 exec_lo, -1
	s_mov_b32 s12, -1
	ds_bpermute_b32 v7, v13, v8
	s_cbranch_scc0 .LBB4_51
; %bb.29:
	s_mov_b32 s12, 0
	s_trap 2
	v_rsq_f32_e32 v9, v9
	s_cbranch_execnz .LBB4_36
	s_branch .LBB4_37
.LBB4_30:
	s_mov_b32 s6, -1
	s_trap 2
	s_cbranch_execz .LBB4_6
	s_branch .LBB4_35
.LBB4_31:
	s_mov_b32 s6, -1
	s_trap 2
	s_cbranch_execz .LBB4_8
	s_branch .LBB4_35
.LBB4_32:
	s_mov_b32 s6, -1
	s_trap 2
	s_cbranch_execz .LBB4_10
	s_branch .LBB4_35
.LBB4_33:
	s_mov_b32 s6, -1
	s_trap 2
	s_cbranch_execz .LBB4_12
	s_branch .LBB4_35
.LBB4_34:
	s_mov_b32 s6, -1
	s_trap 2
	s_cbranch_execz .LBB4_14
.LBB4_35:
                                        ; implicit-def: $sgpr7
                                        ; implicit-def: $vgpr7
                                        ; implicit-def: $vgpr8
                                        ; implicit-def: $vgpr9
                                        ; implicit-def: $sgpr4
                                        ; implicit-def: $vgpr6
	s_and_b32 vcc_lo, exec_lo, s6
	s_cbranch_vccz .LBB4_37
.LBB4_36:                               ; %UnifiedUnreachableBlock
	; divergent unreachable
.LBB4_37:                               ; %Flow186
	s_and_not1_b32 vcc_lo, exec_lo, s12
	s_cbranch_vccz .LBB4_39
; %bb.38:                               ; %UnifiedReturnBlock
	s_endpgm
.LBB4_39:                               ; %_Z16__shfl_down_syncIyfET0_T_S0_ji.exit38.i76
	s_waitcnt lgkmcnt(0)
	v_dual_add_f32 v7, v8, v7 :: v_dual_lshlrev_b32 v6, 2, v6
	s_load_b256 s[12:19], s[0:1], 0x20
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_gl0_inv
	ds_bpermute_b32 v6, v6, v7
	s_and_saveexec_b32 s0, s7
	s_cbranch_execz .LBB4_41
; %bb.40:
	s_waitcnt lgkmcnt(0)
	v_dual_add_f32 v6, v7, v6 :: v_dual_mov_b32 v7, 0
	ds_store_b32 v7, v6 offset:1024
.LBB4_41:                               ; %_Z14block_sum_handfPf.exit79
	s_or_b32 exec_lo, exec_lo, s0
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v6, 0
	s_barrier
	buffer_gl0_inv
	s_ashr_i32 s3, s2, 31
	ds_load_b32 v7, v6 offset:1024
	s_lshl_b64 s[0:1], s[2:3], 2
	v_mul_f32_e32 v10, 0x45800000, v9
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v7, 0x358637bd, v7
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v8, 0x4b800000, v7
	v_cmp_gt_f32_e32 vcc_lo, 0x800000, v7
	v_cndmask_b32_e32 v7, v7, v8, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rsq_f32_e32 v7, v7
	v_mul_f32_e32 v8, 0x45800000, v7
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v7, v7, v8, vcc_lo
	v_cndmask_b32_e64 v8, v9, v10, s4
	s_add_u32 s4, s10, s0
	s_addc_u32 s5, s11, s1
	s_add_u32 s0, s12, s0
	v_dual_mul_f32 v4, v4, v7 :: v_dual_mul_f32 v3, v3, v8
	s_addc_u32 s1, s13, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v4, 0x3db504f3, v4
	s_barrier
	buffer_gl0_inv
	s_barrier
	buffer_gl0_inv
	ds_store_2addr_stride64_b32 v5, v3, v4 offset1:2
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_gl0_inv
	s_clause 0x1
	global_load_b32 v4, v6, s[4:5]
	global_load_b32 v3, v6, s[0:1]
	s_lshl_b64 s[0:1], s[2:3], 16
	s_mov_b32 s5, 0
	s_add_u32 s2, s14, s0
	s_addc_u32 s3, s15, s1
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s4, v4
	v_add_co_u32 v4, s2, s2, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v5, null, s3, 0, s2
	s_mov_b64 s[2:3], 0
.LBB4_42:                               ; =>This Inner Loop Header: Depth=1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v7, vcc_lo, v4, s2
	v_add_co_ci_u32_e64 v8, null, s3, v5, vcc_lo
	s_add_u32 s2, s2, 0x2000
	s_addc_u32 s3, s3, 0
	s_clause 0x7
	global_load_b32 v15, v[7:8], off
	global_load_b32 v16, v[7:8], off offset:512
	global_load_b32 v17, v[7:8], off offset:1024
	global_load_b32 v18, v[7:8], off offset:1536
	global_load_b32 v19, v[7:8], off offset:2048
	global_load_b32 v20, v[7:8], off offset:2560
	global_load_b32 v21, v[7:8], off offset:3072
	global_load_b32 v22, v[7:8], off offset:3584
	v_add_co_u32 v7, vcc_lo, 0x1000, v7
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v8, null, 0, v8, vcc_lo
	v_mov_b32_e32 v31, s5
	s_add_i32 s5, s5, 64
	s_clause 0x7
	global_load_b32 v23, v[7:8], off
	global_load_b32 v24, v[7:8], off offset:512
	global_load_b32 v25, v[7:8], off offset:1024
	global_load_b32 v26, v[7:8], off offset:1536
	global_load_b32 v27, v[7:8], off offset:2048
	global_load_b32 v28, v[7:8], off offset:2560
	global_load_b32 v29, v[7:8], off offset:3072
	global_load_b32 v30, v[7:8], off offset:3584
	ds_load_b128 v[7:10], v31
	ds_load_b128 v[11:14], v31 offset:16
	s_cmp_eq_u32 s2, 0x10000
	s_waitcnt vmcnt(15) lgkmcnt(1)
	v_fmac_f32_e32 v6, v15, v7
	s_waitcnt vmcnt(14)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v6, v16, v8
	s_waitcnt vmcnt(13)
	v_fmac_f32_e32 v6, v17, v9
	s_waitcnt vmcnt(12)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v6, v18, v10
	ds_load_b128 v[7:10], v31 offset:32
	s_waitcnt vmcnt(11) lgkmcnt(1)
	v_fmac_f32_e32 v6, v19, v11
	s_waitcnt vmcnt(10)
	v_fmac_f32_e32 v6, v20, v12
	s_waitcnt vmcnt(9)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v6, v21, v13
	s_waitcnt vmcnt(8)
	v_fmac_f32_e32 v6, v22, v14
	ds_load_b128 v[11:14], v31 offset:48
	s_waitcnt vmcnt(7) lgkmcnt(1)
	v_fmac_f32_e32 v6, v23, v7
	s_waitcnt vmcnt(6)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v6, v24, v8
	s_waitcnt vmcnt(5)
	v_fmac_f32_e32 v6, v25, v9
	s_waitcnt vmcnt(4)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v6, v26, v10
	s_waitcnt vmcnt(3) lgkmcnt(0)
	v_fmac_f32_e32 v6, v27, v11
	s_waitcnt vmcnt(2)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v6, v28, v12
	s_waitcnt vmcnt(1)
	v_fmac_f32_e32 v6, v29, v13
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v6, v30, v14
	s_cbranch_scc0 .LBB4_42
; %bb.43:
	v_add_co_u32 v4, vcc_lo, s8, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v5, null, s9, v2, vcc_lo
	s_mul_f32 s2, s4, 0x3fb8aa3b
	s_add_u32 s0, s14, s0
	s_addc_u32 s1, s15, s1
	global_load_b32 v4, v[4:5], off
	s_xor_b32 s3, s2, 0x80000000
	s_rndne_f32 s5, s2
	s_fmamk_f32 s3, s4, 0x3fb8aa3b, s3
	s_cmp_nlt_f32 s4, 0xc2ce8ed0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_sub_f32 s2, s2, s5
	s_fmamk_f32 s3, s4, 0x32a5705f, s3
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_ngt_f32 s4, 0x42b17218
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_3)
	s_add_f32 s2, s2, s3
	v_exp_f32_e32 v5, s2
	s_cvt_i32_f32 s2, s5
	s_delay_alu instid0(TRANS32_DEP_1) | instid1(SALU_CYCLE_3)
	v_ldexp_f32 v5, v5, s2
	s_mov_b32 s2, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v5, 0, v5, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	v_cndmask_b32_e32 v5, 0x7f800000, v5, vcc_lo
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v6, -v5, v6, v4
	v_dual_mov_b32 v4, 0 :: v_dual_mul_f32 v7, v3, v6
	v_mov_b32_e32 v6, 0
.LBB4_44:                               ; =>This Inner Loop Header: Depth=1
	v_lshl_or_b32 v3, s2, 7, v0
	s_lshl_b32 s3, s2, 2
	s_add_i32 s2, s2, 8
	v_mov_b32_e32 v20, s3
	s_cmpk_eq_i32 s2, 0x80
	v_lshlrev_b64 v[8:9], 2, v[3:4]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v24, vcc_lo, s0, v8
	v_add_co_ci_u32_e64 v25, null, s1, v9, vcc_lo
	ds_load_b128 v[8:11], v20
	ds_load_b128 v[12:15], v20 offset:512
	ds_load_b128 v[16:19], v20 offset:16
	s_clause 0x7
	global_load_b32 v3, v[24:25], off
	global_load_b32 v26, v[24:25], off offset:512
	global_load_b32 v27, v[24:25], off offset:1024
	global_load_b32 v28, v[24:25], off offset:1536
	global_load_b32 v29, v[24:25], off offset:2048
	global_load_b32 v30, v[24:25], off offset:2560
	global_load_b32 v31, v[24:25], off offset:3072
	global_load_b32 v32, v[24:25], off offset:3584
	s_waitcnt lgkmcnt(2)
	v_mul_f32_e32 v9, v7, v9
	v_mul_f32_e32 v8, v7, v8
	ds_load_b128 v[20:23], v20 offset:528
	s_waitcnt vmcnt(6)
	v_fmac_f32_e32 v9, v5, v26
	v_dual_fmac_f32 v8, v5, v3 :: v_dual_mul_f32 v3, v7, v10
	v_mul_f32_e32 v10, v7, v11
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v11, v7, v16
	s_waitcnt vmcnt(5)
	v_dual_fmac_f32 v6, v12, v8 :: v_dual_fmac_f32 v3, v5, v27
	s_waitcnt vmcnt(3)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v11, v5, v29
	v_fmac_f32_e32 v10, v5, v28
	v_dual_fmac_f32 v6, v13, v9 :: v_dual_mul_f32 v13, v7, v18
	v_mul_f32_e32 v12, v7, v17
	s_clause 0x3
	global_store_b32 v[24:25], v8, off
	global_store_b32 v[24:25], v9, off offset:512
	global_store_b32 v[24:25], v3, off offset:1024
	global_store_b32 v[24:25], v10, off offset:1536
	v_fmac_f32_e32 v6, v14, v3
	v_mul_f32_e32 v14, v7, v19
	s_waitcnt vmcnt(1)
	v_fmac_f32_e32 v13, v5, v31
	v_fmac_f32_e32 v12, v5, v30
	v_fmac_f32_e32 v6, v15, v10
	s_waitcnt vmcnt(0)
	v_fmac_f32_e32 v14, v5, v32
	s_clause 0x3
	global_store_b32 v[24:25], v11, off offset:2048
	global_store_b32 v[24:25], v12, off offset:2560
	global_store_b32 v[24:25], v13, off offset:3072
	global_store_b32 v[24:25], v14, off offset:3584
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v6, v20, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v6, v21, v12
	v_fmac_f32_e32 v6, v22, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v6, v23, v14
	s_cbranch_scc0 .LBB4_44
; %bb.45:
	v_add_co_u32 v0, vcc_lo, s16, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v1, null, s17, v2, vcc_lo
	global_store_b32 v[0:1], v6, off
	s_endpgm
.LBB4_46:
	s_mov_b32 s6, -1
	s_trap 2
	v_cndmask_b32_e64 v9, v7, v8, s4
	s_cbranch_execz .LBB4_22
	s_branch .LBB4_50
.LBB4_47:
	s_mov_b32 s6, -1
	s_trap 2
	s_cbranch_execz .LBB4_24
	s_branch .LBB4_50
.LBB4_48:
	s_mov_b32 s6, -1
	s_trap 2
	s_cbranch_execz .LBB4_26
	s_branch .LBB4_50
.LBB4_49:
	s_mov_b32 s6, -1
	s_trap 2
	s_cbranch_execz .LBB4_28
.LBB4_50:
	s_mov_b32 s12, 0
                                        ; implicit-def: $vgpr7
                                        ; implicit-def: $vgpr8
.LBB4_51:                               ; %Flow178
	s_delay_alu instid0(VALU_DEP_1)
	v_rsq_f32_e32 v9, v9
	s_and_b32 vcc_lo, exec_lo, s6
	s_cbranch_vccnz .LBB4_36
	s_branch .LBB4_37
.Lfunc_end4:
	.size	_Z11rec1_kernelPKfS0_S0_S0_S0_PfS1_i, .Lfunc_end4-_Z11rec1_kernelPKfS0_S0_S0_S0_PfS1_i
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z11rec1_kernelPKfS0_S0_S0_S0_PfS1_i
		.amdhsa_group_segment_fixed_size 1152
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 320
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 33
		.amdhsa_next_free_sgpr 20
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_shared_vgpr_count 0
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end4-_Z11rec1_kernelPKfS0_S0_S0_S0_PfS1_i)<<4)&1008)>>4
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
                                        ; -- End function
	.set .L_Z11rec1_kernelPKfS0_S0_S0_S0_PfS1_i.num_vgpr, 33
	.set .L_Z11rec1_kernelPKfS0_S0_S0_S0_PfS1_i.num_agpr, 0
	.set .L_Z11rec1_kernelPKfS0_S0_S0_S0_PfS1_i.numbered_sgpr, 20
	.set .L_Z11rec1_kernelPKfS0_S0_S0_S0_PfS1_i.num_named_barrier, 0
	.set .L_Z11rec1_kernelPKfS0_S0_S0_S0_PfS1_i.private_seg_size, 0
	.set .L_Z11rec1_kernelPKfS0_S0_S0_S0_PfS1_i.uses_vcc, 1
	.set .L_Z11rec1_kernelPKfS0_S0_S0_S0_PfS1_i.uses_flat_scratch, 0
	.set .L_Z11rec1_kernelPKfS0_S0_S0_S0_PfS1_i.has_dyn_sized_stack, 0
	.set .L_Z11rec1_kernelPKfS0_S0_S0_S0_PfS1_i.has_recursion, 0
	.set .L_Z11rec1_kernelPKfS0_S0_S0_S0_PfS1_i.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 2904
; TotalNumSgprs: 22
; NumVgprs: 33
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 1152 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 4
; NumSGPRsForWavesPerEU: 22
; NumVGPRsForWavesPerEU: 33
; Occupancy: 16
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	_Z11rec2_kernelPKfS0_S0_S0_S0_PfS1_i ; -- Begin function _Z11rec2_kernelPKfS0_S0_S0_S0_PfS1_i
	.globl	_Z11rec2_kernelPKfS0_S0_S0_S0_PfS1_i
	.p2align	8
	.type	_Z11rec2_kernelPKfS0_S0_S0_S0_PfS1_i,@function
_Z11rec2_kernelPKfS0_S0_S0_S0_PfS1_i:   ; @_Z11rec2_kernelPKfS0_S0_S0_S0_PfS1_i
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_b256 s[8:15], s[0:1], 0x0
	v_lshl_add_u32 v1, s2, 7, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v2, 31, v1
	v_lshlrev_b64 v[1:2], 2, v[1:2]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v3, vcc_lo, s10, v1
	v_add_co_ci_u32_e64 v4, null, s11, v2, vcc_lo
	global_load_b32 v3, v[3:4], off
	v_add_co_u32 v4, vcc_lo, s8, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v5, null, s9, v2, vcc_lo
	v_and_b32_e32 v12, 31, v0
	global_load_b32 v4, v[4:5], off
	v_cmp_gt_u32_e32 vcc_lo, 24, v12
	v_cndmask_b32_e64 v5, 0, 8, vcc_lo
	v_add_lshl_u32 v7, v5, v12, 2
	s_waitcnt vmcnt(1)
	v_mul_f32_e32 v6, v3, v3
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	ds_swizzle_b32 v5, v6 offset:swizzle(BITMASK_PERM,"1pppp")
	;;#ASMEND
	v_fmac_f32_e32 v5, v3, v3
	v_cmp_gt_u32_e32 vcc_lo, 28, v12
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	ds_bpermute_b32 v9, v7, v5
	;;#ASMEND
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_add_f32_e32 v5, v5, v9
	v_cndmask_b32_e64 v8, 0, 4, vcc_lo
	v_cmp_gt_u32_e32 vcc_lo, 30, v12
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v11, v4, v4
	v_add_lshl_u32 v8, v8, v12, 2
	v_cndmask_b32_e64 v6, 0, 2, vcc_lo
	v_cmp_ne_u32_e32 vcc_lo, 31, v12
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_add_lshl_u32 v9, v6, v12, 2
	v_add_co_ci_u32_e64 v10, null, 0, v12, vcc_lo
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	ds_bpermute_b32 v6, v8, v5
	;;#ASMEND
	v_add_f32_e32 v5, v5, v6
	v_cmp_eq_u32_e32 vcc_lo, 0, v12
	v_lshlrev_b32_e32 v6, 2, v10
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	ds_bpermute_b32 v10, v9, v5
	;;#ASMEND
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_add_f32_e32 v5, v5, v10
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	ds_bpermute_b32 v10, v6, v5
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	ds_swizzle_b32 v11, v11 offset:swizzle(BITMASK_PERM,"1pppp")
	;;#ASMEND
	v_fmac_f32_e32 v11, v4, v4
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	ds_bpermute_b32 v13, v7, v11
	;;#ASMEND
	v_add_f32_e32 v11, v11, v13
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	ds_bpermute_b32 v13, v8, v11
	;;#ASMEND
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_f32_e32 v11, v11, v13
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	ds_bpermute_b32 v13, v9, v11
	;;#ASMEND
	v_add_f32_e32 v11, v11, v13
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	ds_bpermute_b32 v12, v6, v11
	;;#ASMEND
	s_and_saveexec_b32 s3, vcc_lo
	s_cbranch_execz .LBB5_2
; %bb.1:
	v_lshrrev_b32_e32 v13, 3, v0
	v_add_f32_e32 v11, v11, v12
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_add_f32 v5, v5, v10 :: v_dual_add_nc_u32 v10, 0x400, v13
	ds_store_2addr_b32 v10, v5, v11 offset1:32
.LBB5_2:
	s_or_b32 exec_lo, exec_lo, s3
	s_clause 0x1
	s_load_b32 s3, s[0:1], 0x4c
	s_load_b256 s[4:11], s[0:1], 0x20
	v_dual_mov_b32 v10, 0 :: v_dual_lshlrev_b32 v5, 2, v0
	v_mov_b32_e32 v11, 0
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_gl0_inv
	s_bfe_u32 s0, s3, 0xb0005
	s_delay_alu instid0(SALU_CYCLE_1)
	v_cmp_gt_u32_e32 vcc_lo, s0, v0
	s_and_saveexec_b32 s0, vcc_lo
; %bb.3:
	ds_load_b32 v11, v5 offset:1024
; %bb.4:
	s_or_b32 exec_lo, exec_lo, s0
	s_and_saveexec_b32 s0, vcc_lo
; %bb.5:
	ds_load_b32 v10, v5 offset:1152
; %bb.6:
	s_or_b32 exec_lo, exec_lo, s0
	s_waitcnt lgkmcnt(0)
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	ds_swizzle_b32 v12, v11 offset:swizzle(BITMASK_PERM,"1pppp")
	;;#ASMEND
	v_add_f32_e32 v11, v11, v12
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	ds_swizzle_b32 v12, v10 offset:swizzle(BITMASK_PERM,"1pppp")
	;;#ASMEND
	v_add_f32_e32 v10, v10, v12
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	ds_bpermute_b32 v12, v7, v11
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	ds_bpermute_b32 v7, v7, v10
	;;#ASMEND
	v_add_f32_e32 v11, v11, v12
	s_mov_b32 s10, 0
	v_add_f32_e32 v7, v10, v7
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	ds_bpermute_b32 v10, v8, v11
	;;#ASMEND
	s_mov_b32 s0, exec_lo
	v_add_f32_e32 v10, v11, v10
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	ds_bpermute_b32 v8, v8, v7
	;;#ASMEND
	v_add_f32_e32 v8, v7, v8
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	ds_bpermute_b32 v7, v9, v10
	;;#ASMEND
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v7, v10, v7
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	ds_bpermute_b32 v9, v9, v8
	;;#ASMEND
	v_add_f32_e32 v8, v8, v9
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	ds_bpermute_b32 v9, v6, v7
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	ds_bpermute_b32 v6, v6, v8
	;;#ASMEND
	s_barrier
	buffer_gl0_inv
	v_cmpx_eq_u32_e32 0, v0
; %bb.7:
	v_dual_add_f32 v6, v8, v6 :: v_dual_add_f32 v7, v7, v9
	v_add_nc_u32_e64 v8, 0x400, 0
	ds_store_2addr_b32 v8, v7, v6 offset1:32
; %bb.8:                                ; %_Z19block_sum_pair_handffPfS_S_S_.exit
	s_or_b32 exec_lo, exec_lo, s0
	v_add_nc_u32_e64 v6, 0x400, 0
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_gl0_inv
	s_ashr_i32 s3, s2, 31
	ds_load_2addr_b32 v[6:7], v6 offset1:32
	s_lshl_b64 s[16:17], s[2:3], 2
	s_waitcnt lgkmcnt(0)
	v_dual_add_f32 v7, 0x358637bd, v7 :: v_dual_add_f32 v6, 0x358637bd, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v8, 0x4b800000, v7 :: v_dual_mul_f32 v9, 0x4b800000, v6
	v_cmp_gt_f32_e32 vcc_lo, 0x800000, v7
	v_cmp_gt_f32_e64 s0, 0x800000, v6
	v_cndmask_b32_e32 v7, v7, v8, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v6, v6, v9, s0
	v_rsq_f32_e32 v7, v7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rsq_f32_e32 v6, v6
	v_dual_mul_f32 v8, 0x45800000, v7 :: v_dual_mul_f32 v9, 0x45800000, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v7, v7, v8, vcc_lo
	v_cndmask_b32_e64 v6, v6, v9, s0
	s_add_u32 s0, s14, s16
	s_addc_u32 s1, s15, s17
	s_add_u32 s14, s4, s16
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_dual_mul_f32 v4, v4, v7 :: v_dual_mul_f32 v3, v3, v6
	s_addc_u32 s15, s5, s17
	v_mul_f32_e32 v4, 0x3db504f3, v4
	s_barrier
	buffer_gl0_inv
	ds_store_2addr_stride64_b32 v5, v3, v4 offset1:2
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_gl0_inv
	s_load_b32 s5, s[0:1], 0x0
	s_load_b32 s4, s[14:15], 0x0
	s_lshl_b64 s[0:1], s[2:3], 16
	v_mov_b32_e32 v3, 0
	s_add_u32 s2, s6, s0
	s_addc_u32 s3, s7, s1
	v_add_co_u32 v4, s2, s2, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v5, null, s3, 0, s2
	s_mov_b64 s[2:3], 0
.LBB5_9:                                ; =>This Inner Loop Header: Depth=1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v6, vcc_lo, v4, s2
	v_add_co_ci_u32_e64 v7, null, s3, v5, vcc_lo
	s_add_u32 s2, s2, 0x2000
	s_addc_u32 s3, s3, 0
	s_clause 0x7
	global_load_b32 v14, v[6:7], off
	global_load_b32 v15, v[6:7], off offset:512
	global_load_b32 v16, v[6:7], off offset:1024
	global_load_b32 v17, v[6:7], off offset:1536
	global_load_b32 v18, v[6:7], off offset:2048
	global_load_b32 v19, v[6:7], off offset:2560
	global_load_b32 v20, v[6:7], off offset:3072
	global_load_b32 v21, v[6:7], off offset:3584
	v_add_co_u32 v6, vcc_lo, 0x1000, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v7, null, 0, v7, vcc_lo
	v_mov_b32_e32 v30, s10
	s_add_i32 s10, s10, 64
	s_clause 0x7
	global_load_b32 v22, v[6:7], off
	global_load_b32 v23, v[6:7], off offset:512
	global_load_b32 v24, v[6:7], off offset:1024
	global_load_b32 v25, v[6:7], off offset:1536
	global_load_b32 v26, v[6:7], off offset:2048
	global_load_b32 v27, v[6:7], off offset:2560
	global_load_b32 v28, v[6:7], off offset:3072
	global_load_b32 v29, v[6:7], off offset:3584
	ds_load_b128 v[6:9], v30
	ds_load_b128 v[10:13], v30 offset:16
	s_cmp_eq_u32 s2, 0x10000
	s_waitcnt vmcnt(15) lgkmcnt(0)
	v_fmac_f32_e32 v3, v14, v6
	s_waitcnt vmcnt(14)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v3, v15, v7
	s_waitcnt vmcnt(13)
	v_fmac_f32_e32 v3, v16, v8
	s_waitcnt vmcnt(12)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v3, v17, v9
	ds_load_b128 v[6:9], v30 offset:32
	s_waitcnt vmcnt(11)
	v_fmac_f32_e32 v3, v18, v10
	s_waitcnt vmcnt(10)
	v_fmac_f32_e32 v3, v19, v11
	s_waitcnt vmcnt(9)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v3, v20, v12
	s_waitcnt vmcnt(8)
	v_fmac_f32_e32 v3, v21, v13
	ds_load_b128 v[10:13], v30 offset:48
	s_waitcnt vmcnt(7) lgkmcnt(1)
	v_fmac_f32_e32 v3, v22, v6
	s_waitcnt vmcnt(6)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v3, v23, v7
	s_waitcnt vmcnt(5)
	v_fmac_f32_e32 v3, v24, v8
	s_waitcnt vmcnt(4)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v3, v25, v9
	s_waitcnt vmcnt(3) lgkmcnt(0)
	v_fmac_f32_e32 v3, v26, v10
	s_waitcnt vmcnt(2)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v3, v27, v11
	s_waitcnt vmcnt(1)
	v_fmac_f32_e32 v3, v28, v12
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v3, v29, v13
	s_cbranch_scc0 .LBB5_9
; %bb.10:
	v_add_co_u32 v4, vcc_lo, s12, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v5, null, s13, v2, vcc_lo
	s_mul_f32 s2, s5, 0x3fb8aa3b
	s_add_u32 s0, s6, s0
	s_addc_u32 s1, s7, s1
	global_load_b32 v4, v[4:5], off
	s_xor_b32 s3, s2, 0x80000000
	s_rndne_f32 s6, s2
	s_fmamk_f32 s3, s5, 0x3fb8aa3b, s3
	s_cmp_nlt_f32 s5, 0xc2ce8ed0
	v_mov_b32_e32 v6, 0
	s_sub_f32 s2, s2, s6
	s_fmamk_f32 s3, s5, 0x32a5705f, s3
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_ngt_f32 s5, 0x42b17218
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_3)
	s_add_f32 s2, s2, s3
	v_exp_f32_e32 v5, s2
	s_cvt_i32_f32 s2, s6
	s_delay_alu instid0(TRANS32_DEP_1) | instid1(SALU_CYCLE_3)
	v_ldexp_f32 v5, v5, s2
	s_mov_b32 s2, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v5, 0, v5, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	v_cndmask_b32_e32 v5, 0x7f800000, v5, vcc_lo
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v3, -v5, v3, v4
	v_dual_mov_b32 v4, 0 :: v_dual_mul_f32 v7, s4, v3
.LBB5_11:                               ; =>This Inner Loop Header: Depth=1
	v_lshl_or_b32 v3, s2, 7, v0
	s_lshl_b32 s3, s2, 2
	s_add_i32 s2, s2, 8
	v_mov_b32_e32 v20, s3
	s_cmpk_eq_i32 s2, 0x80
	v_lshlrev_b64 v[8:9], 2, v[3:4]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v24, vcc_lo, s0, v8
	v_add_co_ci_u32_e64 v25, null, s1, v9, vcc_lo
	ds_load_b128 v[8:11], v20
	ds_load_b128 v[12:15], v20 offset:512
	ds_load_b128 v[16:19], v20 offset:16
	s_clause 0x7
	global_load_b32 v3, v[24:25], off
	global_load_b32 v26, v[24:25], off offset:512
	global_load_b32 v27, v[24:25], off offset:1024
	global_load_b32 v28, v[24:25], off offset:1536
	global_load_b32 v29, v[24:25], off offset:2048
	global_load_b32 v30, v[24:25], off offset:2560
	global_load_b32 v31, v[24:25], off offset:3072
	global_load_b32 v32, v[24:25], off offset:3584
	s_waitcnt lgkmcnt(2)
	v_mul_f32_e32 v9, v7, v9
	v_mul_f32_e32 v8, v7, v8
	ds_load_b128 v[20:23], v20 offset:528
	s_waitcnt vmcnt(6)
	v_fmac_f32_e32 v9, v5, v26
	v_dual_fmac_f32 v8, v5, v3 :: v_dual_mul_f32 v3, v7, v10
	v_mul_f32_e32 v10, v7, v11
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v11, v7, v16
	s_waitcnt vmcnt(5)
	v_dual_fmac_f32 v6, v12, v8 :: v_dual_fmac_f32 v3, v5, v27
	s_waitcnt vmcnt(3)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v11, v5, v29
	v_fmac_f32_e32 v10, v5, v28
	v_dual_fmac_f32 v6, v13, v9 :: v_dual_mul_f32 v13, v7, v18
	v_mul_f32_e32 v12, v7, v17
	s_clause 0x3
	global_store_b32 v[24:25], v8, off
	global_store_b32 v[24:25], v9, off offset:512
	global_store_b32 v[24:25], v3, off offset:1024
	global_store_b32 v[24:25], v10, off offset:1536
	v_fmac_f32_e32 v6, v14, v3
	v_mul_f32_e32 v14, v7, v19
	s_waitcnt vmcnt(1)
	v_fmac_f32_e32 v13, v5, v31
	v_fmac_f32_e32 v12, v5, v30
	v_fmac_f32_e32 v6, v15, v10
	s_waitcnt vmcnt(0)
	v_fmac_f32_e32 v14, v5, v32
	s_clause 0x3
	global_store_b32 v[24:25], v11, off offset:2048
	global_store_b32 v[24:25], v12, off offset:2560
	global_store_b32 v[24:25], v13, off offset:3072
	global_store_b32 v[24:25], v14, off offset:3584
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v6, v20, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v6, v21, v12
	v_fmac_f32_e32 v6, v22, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v6, v23, v14
	s_cbranch_scc0 .LBB5_11
; %bb.12:
	v_add_co_u32 v0, vcc_lo, s8, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v1, null, s9, v2, vcc_lo
	global_store_b32 v[0:1], v6, off
	s_endpgm
.Lfunc_end5:
	.size	_Z11rec2_kernelPKfS0_S0_S0_S0_PfS1_i, .Lfunc_end5-_Z11rec2_kernelPKfS0_S0_S0_S0_PfS1_i
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z11rec2_kernelPKfS0_S0_S0_S0_PfS1_i
		.amdhsa_group_segment_fixed_size 1280
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 320
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 33
		.amdhsa_next_free_sgpr 18
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_shared_vgpr_count 0
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end5-_Z11rec2_kernelPKfS0_S0_S0_S0_PfS1_i)<<4)&1008)>>4
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
                                        ; -- End function
	.set .L_Z11rec2_kernelPKfS0_S0_S0_S0_PfS1_i.num_vgpr, 33
	.set .L_Z11rec2_kernelPKfS0_S0_S0_S0_PfS1_i.num_agpr, 0
	.set .L_Z11rec2_kernelPKfS0_S0_S0_S0_PfS1_i.numbered_sgpr, 18
	.set .L_Z11rec2_kernelPKfS0_S0_S0_S0_PfS1_i.num_named_barrier, 0
	.set .L_Z11rec2_kernelPKfS0_S0_S0_S0_PfS1_i.private_seg_size, 0
	.set .L_Z11rec2_kernelPKfS0_S0_S0_S0_PfS1_i.uses_vcc, 1
	.set .L_Z11rec2_kernelPKfS0_S0_S0_S0_PfS1_i.uses_flat_scratch, 0
	.set .L_Z11rec2_kernelPKfS0_S0_S0_S0_PfS1_i.has_dyn_sized_stack, 0
	.set .L_Z11rec2_kernelPKfS0_S0_S0_S0_PfS1_i.has_recursion, 0
	.set .L_Z11rec2_kernelPKfS0_S0_S0_S0_PfS1_i.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 2532
; TotalNumSgprs: 20
; NumVgprs: 33
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 1280 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 4
; NumSGPRsForWavesPerEU: 20
; NumVGPRsForWavesPerEU: 33
; Occupancy: 16
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	_Z12rec2c_kernelPKfS0_S0_S0_S0_PfS1_i ; -- Begin function _Z12rec2c_kernelPKfS0_S0_S0_S0_PfS1_i
	.globl	_Z12rec2c_kernelPKfS0_S0_S0_S0_PfS1_i
	.p2align	8
	.type	_Z12rec2c_kernelPKfS0_S0_S0_S0_PfS1_i,@function
_Z12rec2c_kernelPKfS0_S0_S0_S0_PfS1_i:  ; @_Z12rec2c_kernelPKfS0_S0_S0_S0_PfS1_i
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_mov_b32 s4, -1
	s_mov_b32 s21, 0
	s_cmp_eq_u32 exec_lo, -1
	s_mov_b32 s3, 0
	s_cbranch_scc0 .LBB6_53
.LBB6_1:                                ; %Flow
	v_readfirstlane_b32 s20, v0
	s_and_not1_b32 vcc_lo, exec_lo, s4
	s_cbranch_vccnz .LBB6_44
; %bb.2:                                ; %_Z16__shfl_down_syncIyfET0_T_S0_ji.exit.i
	s_load_b256 s[4:11], s[0:1], 0x0
	v_lshl_add_u32 v2, s2, 7, v0
	s_load_b256 s[12:19], s[0:1], 0x20
	s_cmp_lg_u32 exec_lo, -1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshlrev_b64 v[4:5], 2, v[2:3]
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v6, vcc_lo, s6, v4
	v_add_co_ci_u32_e64 v7, null, s7, v5, vcc_lo
	s_mov_b32 s6, -1
	global_load_b32 v6, v[6:7], off
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v7, v6, v6
	ds_swizzle_b32 v8, v7 offset:swizzle(BITMASK_PERM,"1pppp")
	s_cbranch_scc1 .LBB6_54
; %bb.3:                                ; %Flow164
	s_and_not1_b32 vcc_lo, exec_lo, s6
	s_cbranch_vccnz .LBB6_58
.LBB6_4:                                ; %_Z16__shfl_down_syncIyfET0_T_S0_ji.exit56.i
	v_mbcnt_lo_u32_b32 v1, -1, 0
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v7, v7, v8
	s_cmp_lg_u32 exec_lo, -1
	s_mov_b32 s6, -1
	v_cmp_gt_u32_e32 vcc_lo, 24, v1
	v_cndmask_b32_e64 v9, 0, 8, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v9, v9, v1
	v_lshlrev_b32_e32 v9, 2, v9
	ds_bpermute_b32 v8, v9, v7
	s_cbranch_scc1 .LBB6_55
; %bb.5:                                ; %Flow166
	s_and_not1_b32 vcc_lo, exec_lo, s6
	s_cbranch_vccnz .LBB6_58
.LBB6_6:                                ; %_Z16__shfl_down_syncIyfET0_T_S0_ji.exit58.i
	v_cmp_gt_u32_e32 vcc_lo, 28, v1
	s_cmp_lg_u32 exec_lo, -1
	s_mov_b32 s6, -1
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v7, v7, v8
	v_cndmask_b32_e64 v10, 0, 4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v10, v10, v1
	v_lshlrev_b32_e32 v8, 2, v10
	ds_bpermute_b32 v10, v8, v7
	s_cbranch_scc1 .LBB6_56
; %bb.7:                                ; %Flow168
	s_and_not1_b32 vcc_lo, exec_lo, s6
	s_cbranch_vccnz .LBB6_58
.LBB6_8:                                ; %_Z16__shfl_down_syncIyfET0_T_S0_ji.exit60.i
	v_cmp_gt_u32_e32 vcc_lo, 30, v1
	s_cmp_lg_u32 exec_lo, -1
	s_mov_b32 s6, -1
	v_cndmask_b32_e64 v11, 0, 2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v11, v11, v1
	s_waitcnt lgkmcnt(0)
	v_dual_add_f32 v10, v7, v10 :: v_dual_lshlrev_b32 v7, 2, v11
	ds_bpermute_b32 v11, v7, v10
	s_cbranch_scc1 .LBB6_57
; %bb.9:                                ; %Flow170
	s_and_not1_b32 vcc_lo, exec_lo, s6
	s_cbranch_vccnz .LBB6_58
.LBB6_10:                               ; %_Z16__shfl_down_syncIyfET0_T_S0_ji.exit62.i
	v_cmp_ne_u32_e32 vcc_lo, 31, v1
	s_cmp_lg_u32 exec_lo, -1
	s_mov_b32 s6, -1
	v_add_co_ci_u32_e64 v1, null, 0, v1, vcc_lo
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_add_f32 v11, v10, v11 :: v_dual_lshlrev_b32 v10, 2, v1
	ds_bpermute_b32 v12, v10, v11
	s_cbranch_scc1 .LBB6_62
; %bb.11:                               ; %Flow172
	s_and_not1_b32 vcc_lo, exec_lo, s6
	s_cbranch_vccnz .LBB6_63
.LBB6_12:                               ; %_Z16__shfl_down_syncIyfET0_T_S0_ji.exit64.i
	v_add_co_u32 v4, vcc_lo, s4, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v5, null, s5, v5, vcc_lo
	s_cmp_lg_u32 exec_lo, -1
	s_mov_b32 s4, -1
	global_load_b32 v4, v[4:5], off
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v5, v4, v4
	ds_swizzle_b32 v13, v5 offset:swizzle(BITMASK_PERM,"1pppp")
	s_cbranch_scc1 .LBB6_67
; %bb.13:                               ; %Flow174
	s_and_not1_b32 vcc_lo, exec_lo, s4
	s_cbranch_vccnz .LBB6_71
.LBB6_14:                               ; %_Z16__shfl_down_syncIyfET0_T_S0_ji.exit66.i
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v5, v5, v13
	s_cmp_lg_u32 exec_lo, -1
	s_mov_b32 s4, -1
	ds_bpermute_b32 v13, v9, v5
	s_cbranch_scc1 .LBB6_68
; %bb.15:                               ; %Flow176
	s_and_not1_b32 vcc_lo, exec_lo, s4
	s_cbranch_vccnz .LBB6_71
.LBB6_16:                               ; %_Z16__shfl_down_syncIyfET0_T_S0_ji.exit68.i
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v5, v5, v13
	s_cmp_lg_u32 exec_lo, -1
	s_mov_b32 s4, -1
	ds_bpermute_b32 v13, v8, v5
	s_cbranch_scc1 .LBB6_69
; %bb.17:                               ; %Flow178
	s_and_not1_b32 vcc_lo, exec_lo, s4
	s_cbranch_vccnz .LBB6_71
.LBB6_18:                               ; %_Z16__shfl_down_syncIyfET0_T_S0_ji.exit70.i
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v5, v5, v13
	s_cmp_lg_u32 exec_lo, -1
	s_mov_b32 s4, -1
	ds_bpermute_b32 v13, v7, v5
	s_cbranch_scc1 .LBB6_70
; %bb.19:                               ; %Flow180
	s_and_not1_b32 vcc_lo, exec_lo, s4
	s_cbranch_vccnz .LBB6_71
.LBB6_20:                               ; %_Z16__shfl_down_syncIyfET0_T_S0_ji.exit72.i
	s_waitcnt lgkmcnt(0)
	v_dual_add_f32 v5, v5, v13 :: v_dual_and_b32 v14, 31, v0
	s_mov_b32 s4, exec_lo
	ds_bpermute_b32 v13, v10, v5
	v_cmpx_eq_u32_e32 0, v14
	s_cbranch_execz .LBB6_22
; %bb.21:
	v_lshrrev_b32_e32 v14, 3, v0
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v5, v5, v13
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_add_f32 v11, v11, v12 :: v_dual_add_nc_u32 v12, 0x400, v14
	ds_store_2addr_b32 v12, v11, v5 offset1:32
.LBB6_22:
	s_or_b32 exec_lo, exec_lo, s4
	s_load_b32 s0, s[0:1], 0x4c
	v_dual_mov_b32 v5, 0 :: v_dual_lshlrev_b32 v12, 2, v0
	v_mov_b32_e32 v11, 0
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_gl0_inv
	s_bfe_u32 s0, s0, 0xb0005
	s_delay_alu instid0(SALU_CYCLE_1)
	v_cmp_gt_u32_e32 vcc_lo, s0, v0
	s_and_saveexec_b32 s0, vcc_lo
	s_cbranch_execz .LBB6_64
; %bb.23:
	ds_load_b32 v11, v12 offset:1024
	s_or_b32 exec_lo, exec_lo, s0
	s_and_saveexec_b32 s0, vcc_lo
	s_cbranch_execnz .LBB6_65
.LBB6_24:
	s_or_b32 exec_lo, exec_lo, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_eq_u32 exec_lo, -1
	s_mov_b32 s0, -1
	s_cbranch_scc0 .LBB6_66
.LBB6_25:                               ; %Flow182
	s_and_not1_b32 vcc_lo, exec_lo, s0
	s_cbranch_vccnz .LBB6_71
.LBB6_26:                               ; %_Z16__shfl_down_syncIyfET0_T_S0_ji.exit74.i
	s_waitcnt lgkmcnt(0)
	ds_swizzle_b32 v13, v11 offset:swizzle(BITMASK_PERM,"1pppp")
	s_cmp_lg_u32 exec_lo, -1
	s_mov_b32 s0, -1
	s_cbranch_scc1 .LBB6_72
; %bb.27:                               ; %Flow184
	s_and_not1_b32 vcc_lo, exec_lo, s0
	s_cbranch_vccnz .LBB6_71
.LBB6_28:                               ; %_Z16__shfl_down_syncIyfET0_T_S0_ji.exit76.i
	ds_swizzle_b32 v12, v5 offset:swizzle(BITMASK_PERM,"1pppp")
	s_cmp_lg_u32 exec_lo, -1
	s_mov_b32 s0, -1
	s_cbranch_scc1 .LBB6_73
; %bb.29:                               ; %Flow186
	s_and_not1_b32 vcc_lo, exec_lo, s0
	s_cbranch_vccnz .LBB6_71
.LBB6_30:                               ; %_Z16__shfl_down_syncIyfET0_T_S0_ji.exit78.i
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v11, v11, v13
	s_cmp_lg_u32 exec_lo, -1
	s_mov_b32 s0, -1
	ds_bpermute_b32 v13, v9, v11
	s_cbranch_scc1 .LBB6_74
; %bb.31:                               ; %Flow188
	s_and_not1_b32 vcc_lo, exec_lo, s0
	s_cbranch_vccnz .LBB6_71
.LBB6_32:                               ; %_Z16__shfl_down_syncIyfET0_T_S0_ji.exit80.i
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v5, v5, v12
	s_cmp_lg_u32 exec_lo, -1
	s_mov_b32 s0, -1
	ds_bpermute_b32 v12, v9, v5
	s_cbranch_scc1 .LBB6_75
; %bb.33:                               ; %Flow190
	s_and_not1_b32 vcc_lo, exec_lo, s0
	s_cbranch_vccnz .LBB6_71
.LBB6_34:                               ; %_Z16__shfl_down_syncIyfET0_T_S0_ji.exit82.i
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v9, v11, v13
	s_cmp_lg_u32 exec_lo, -1
	s_mov_b32 s0, -1
	ds_bpermute_b32 v11, v8, v9
	s_cbranch_scc1 .LBB6_76
; %bb.35:                               ; %Flow192
	s_and_not1_b32 vcc_lo, exec_lo, s0
	s_cbranch_vccnz .LBB6_71
.LBB6_36:                               ; %_Z16__shfl_down_syncIyfET0_T_S0_ji.exit84.i
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v5, v5, v12
	s_cmp_lg_u32 exec_lo, -1
	s_mov_b32 s0, -1
	ds_bpermute_b32 v8, v8, v5
	s_cbranch_scc1 .LBB6_77
; %bb.37:                               ; %Flow194
	s_and_not1_b32 vcc_lo, exec_lo, s0
	s_cbranch_vccnz .LBB6_71
.LBB6_38:                               ; %_Z16__shfl_down_syncIyfET0_T_S0_ji.exit86.i
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v11, v9, v11
	s_cmp_lg_u32 exec_lo, -1
	s_mov_b32 s0, -1
	ds_bpermute_b32 v12, v7, v11
	s_cbranch_scc1 .LBB6_78
; %bb.39:                               ; %Flow196
	s_and_not1_b32 vcc_lo, exec_lo, s0
	s_cbranch_vccnz .LBB6_71
.LBB6_40:                               ; %_Z16__shfl_down_syncIyfET0_T_S0_ji.exit88.i
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v9, v5, v8
	s_cmp_lg_u32 exec_lo, -1
	s_mov_b32 s0, -1
	ds_bpermute_b32 v8, v7, v9
	s_cbranch_scc1 .LBB6_79
; %bb.41:                               ; %Flow198
	s_and_not1_b32 vcc_lo, exec_lo, s0
	s_cbranch_vccnz .LBB6_80
.LBB6_42:                               ; %_Z16__shfl_down_syncIyfET0_T_S0_ji.exit90.i
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v7, v11, v12
	s_cmp_lg_u32 exec_lo, -1
	s_mov_b32 s21, -1
	ds_bpermute_b32 v5, v10, v7
	s_cbranch_scc0 .LBB6_59
; %bb.43:
	s_mov_b32 s21, 0
	s_mov_b32 s3, -1
	s_trap 2
	s_branch .LBB6_59
.LBB6_44:
                                        ; implicit-def: $vgpr5
                                        ; implicit-def: $vgpr7
                                        ; implicit-def: $vgpr8
                                        ; implicit-def: $vgpr9
                                        ; implicit-def: $vgpr4
                                        ; implicit-def: $vgpr1
                                        ; implicit-def: $vgpr6
                                        ; implicit-def: $vgpr2_vgpr3
                                        ; implicit-def: $vgpr0
                                        ; implicit-def: $sgpr4_sgpr5_sgpr6_sgpr7_sgpr8_sgpr9_sgpr10_sgpr11_sgpr12_sgpr13_sgpr14_sgpr15_sgpr16_sgpr17_sgpr18_sgpr19
	s_and_b32 vcc_lo, exec_lo, s3
	s_cbranch_vccnz .LBB6_60
.LBB6_45:                               ; %Flow201
	s_and_not1_b32 vcc_lo, exec_lo, s21
	s_cbranch_vccnz .LBB6_61
.LBB6_46:                               ; %_Z16__shfl_down_syncIyfET0_T_S0_ji.exit92.i
	s_waitcnt lgkmcnt(0)
	v_dual_add_f32 v8, v9, v8 :: v_dual_lshlrev_b32 v1, 2, v1
	s_mov_b32 s4, 0
	s_mov_b32 s0, exec_lo
	s_barrier
	ds_bpermute_b32 v1, v1, v8
	s_waitcnt lgkmcnt(0)
	buffer_gl0_inv
	v_cmpx_eq_u32_e32 0, v0
	s_cbranch_execz .LBB6_48
; %bb.47:
	v_add_f32_e32 v1, v8, v1
	v_add_f32_e32 v5, v7, v5
	v_add_nc_u32_e64 v7, 0x400, 0
	ds_store_2addr_b32 v7, v5, v1 offset1:32
.LBB6_48:                               ; %_Z19block_sum_pair_compffPfS_S_S_.exit
	s_or_b32 exec_lo, exec_lo, s0
	v_add_nc_u32_e64 v1, 0x400, 0
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_gl0_inv
	s_ashr_i32 s21, s20, 31
	ds_load_2addr_b32 v[7:8], v1 offset1:32
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v1, 0x358637bd, v8
	v_add_f32_e32 v5, 0x358637bd, v7
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_mul_f32_e32 v7, 0x4b800000, v1
	v_cmp_gt_f32_e32 vcc_lo, 0x800000, v1
	v_mul_f32_e32 v8, 0x4b800000, v5
	v_cmp_gt_f32_e64 s0, 0x800000, v5
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v1, v1, v7, vcc_lo
	v_cndmask_b32_e64 v5, v5, v8, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_rsq_f32_e32 v1, v1
	v_rsq_f32_e32 v5, v5
	s_delay_alu instid0(TRANS32_DEP_2)
	v_mul_f32_e32 v7, 0x45800000, v1
	s_delay_alu instid0(TRANS32_DEP_1) | instid1(VALU_DEP_1)
	v_dual_mul_f32 v8, 0x45800000, v5 :: v_dual_cndmask_b32 v7, v1, v7
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v5, v5, v8, s0
	s_lshl_b64 s[0:1], s[20:21], 2
	v_dual_mul_f32 v4, v4, v7 :: v_dual_mov_b32 v1, 0
	v_lshlrev_b32_e32 v7, 2, v0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_mul_f32_e32 v5, v6, v5
	s_add_u32 s2, s10, s0
	v_mul_f32_e32 v4, 0x3db504f3, v4
	s_addc_u32 s3, s11, s1
	s_add_u32 s0, s12, s0
	s_addc_u32 s1, s13, s1
	s_barrier
	buffer_gl0_inv
	ds_store_2addr_stride64_b32 v7, v5, v4 offset1:2
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_gl0_inv
	s_clause 0x1
	global_load_b32 v7, v1, s[2:3]
	global_load_b32 v4, v1, s[0:1]
	v_lshlrev_b64 v[5:6], 2, v[0:1]
	s_lshl_b64 s[0:1], s[20:21], 16
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	s_add_u32 s2, s14, s0
	s_addc_u32 s3, s15, s1
	v_add_co_u32 v5, vcc_lo, s2, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v6, null, s3, v6, vcc_lo
	s_mov_b64 s[2:3], 0
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s5, v7
.LBB6_49:                               ; =>This Inner Loop Header: Depth=1
	v_add_co_u32 v7, vcc_lo, v5, s2
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v8, null, s3, v6, vcc_lo
	v_mov_b32_e32 v31, s4
	s_add_u32 s2, s2, 0x2000
	s_addc_u32 s3, s3, 0
	s_clause 0x7
	global_load_b32 v15, v[7:8], off
	global_load_b32 v16, v[7:8], off offset:512
	global_load_b32 v17, v[7:8], off offset:1024
	global_load_b32 v18, v[7:8], off offset:1536
	global_load_b32 v19, v[7:8], off offset:2048
	global_load_b32 v20, v[7:8], off offset:2560
	global_load_b32 v21, v[7:8], off offset:3072
	global_load_b32 v22, v[7:8], off offset:3584
	v_add_co_u32 v7, vcc_lo, 0x1000, v7
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v8, null, 0, v8, vcc_lo
	s_add_i32 s4, s4, 64
	s_cmp_eq_u32 s2, 0x10000
	s_clause 0x7
	global_load_b32 v23, v[7:8], off
	global_load_b32 v24, v[7:8], off offset:512
	global_load_b32 v25, v[7:8], off offset:1024
	global_load_b32 v26, v[7:8], off offset:1536
	global_load_b32 v27, v[7:8], off offset:2048
	global_load_b32 v28, v[7:8], off offset:2560
	global_load_b32 v29, v[7:8], off offset:3072
	global_load_b32 v30, v[7:8], off offset:3584
	ds_load_b128 v[7:10], v31
	ds_load_b128 v[11:14], v31 offset:16
	s_waitcnt vmcnt(15) lgkmcnt(1)
	v_fmac_f32_e32 v1, v15, v7
	s_waitcnt vmcnt(14)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v1, v16, v8
	s_waitcnt vmcnt(13)
	v_fmac_f32_e32 v1, v17, v9
	s_waitcnt vmcnt(12)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v1, v18, v10
	ds_load_b128 v[7:10], v31 offset:32
	s_waitcnt vmcnt(11) lgkmcnt(1)
	v_fmac_f32_e32 v1, v19, v11
	s_waitcnt vmcnt(10)
	v_fmac_f32_e32 v1, v20, v12
	s_waitcnt vmcnt(9)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v1, v21, v13
	s_waitcnt vmcnt(8)
	v_fmac_f32_e32 v1, v22, v14
	ds_load_b128 v[11:14], v31 offset:48
	s_waitcnt vmcnt(7) lgkmcnt(1)
	v_fmac_f32_e32 v1, v23, v7
	s_waitcnt vmcnt(6)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v1, v24, v8
	s_waitcnt vmcnt(5)
	v_fmac_f32_e32 v1, v25, v9
	s_waitcnt vmcnt(4)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v1, v26, v10
	s_waitcnt vmcnt(3) lgkmcnt(0)
	v_fmac_f32_e32 v1, v27, v11
	s_waitcnt vmcnt(2)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v1, v28, v12
	s_waitcnt vmcnt(1)
	v_fmac_f32_e32 v1, v29, v13
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v1, v30, v14
	s_cbranch_scc0 .LBB6_49
; %bb.50:
	v_lshlrev_b64 v[2:3], 2, v[2:3]
	s_mul_f32 s2, s5, 0x3fb8aa3b
	s_add_u32 s0, s14, s0
	s_addc_u32 s1, s15, s1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	s_xor_b32 s3, s2, 0x80000000
	v_add_co_u32 v5, vcc_lo, s8, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v6, null, s9, v3, vcc_lo
	s_rndne_f32 s4, s2
	s_fmamk_f32 s3, s5, 0x3fb8aa3b, s3
	s_cmp_nlt_f32 s5, 0xc2ce8ed0
	global_load_b32 v5, v[5:6], off
	s_sub_f32 s2, s2, s4
	s_fmamk_f32 s3, s5, 0x32a5705f, s3
	s_cselect_b32 vcc_lo, -1, 0
	s_cmp_ngt_f32 s5, 0x42b17218
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_3)
	s_add_f32 s2, s2, s3
	v_exp_f32_e32 v6, s2
	s_cvt_i32_f32 s2, s4
	s_delay_alu instid0(TRANS32_DEP_1) | instid1(SALU_CYCLE_3)
	v_ldexp_f32 v6, v6, s2
	s_mov_b32 s2, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v6, 0, v6, vcc_lo
	s_cselect_b32 vcc_lo, -1, 0
	v_cndmask_b32_e32 v6, 0x7f800000, v6, vcc_lo
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fma_f32 v1, -v6, v1, v5
	v_mov_b32_e32 v5, 0
	v_mul_f32_e32 v7, v4, v1
	v_mov_b32_e32 v1, 0
.LBB6_51:                               ; =>This Inner Loop Header: Depth=1
	v_lshl_or_b32 v4, s2, 7, v0
	s_lshl_b32 s3, s2, 2
	s_add_i32 s2, s2, 8
	v_mov_b32_e32 v20, s3
	s_cmpk_eq_i32 s2, 0x80
	v_lshlrev_b64 v[8:9], 2, v[4:5]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v24, vcc_lo, s0, v8
	v_add_co_ci_u32_e64 v25, null, s1, v9, vcc_lo
	ds_load_b128 v[8:11], v20
	ds_load_b128 v[12:15], v20 offset:512
	ds_load_b128 v[16:19], v20 offset:16
	s_clause 0x7
	global_load_b32 v4, v[24:25], off
	global_load_b32 v26, v[24:25], off offset:512
	global_load_b32 v27, v[24:25], off offset:1024
	global_load_b32 v28, v[24:25], off offset:1536
	global_load_b32 v29, v[24:25], off offset:2048
	global_load_b32 v30, v[24:25], off offset:2560
	global_load_b32 v31, v[24:25], off offset:3072
	global_load_b32 v32, v[24:25], off offset:3584
	s_waitcnt lgkmcnt(2)
	v_mul_f32_e32 v9, v7, v9
	v_mul_f32_e32 v8, v7, v8
	ds_load_b128 v[20:23], v20 offset:528
	s_waitcnt vmcnt(6)
	v_fmac_f32_e32 v9, v6, v26
	v_fmac_f32_e32 v8, v6, v4
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_dual_mul_f32 v4, v7, v10 :: v_dual_fmac_f32 v1, v12, v8
	s_waitcnt vmcnt(5)
	v_dual_fmac_f32 v4, v6, v27 :: v_dual_fmac_f32 v1, v13, v9
	v_mul_f32_e32 v10, v7, v11
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v11, v7, v16
	v_mul_f32_e32 v13, v7, v18
	v_fmac_f32_e32 v1, v14, v4
	s_waitcnt vmcnt(4)
	v_fmac_f32_e32 v10, v6, v28
	s_waitcnt vmcnt(3)
	v_dual_fmac_f32 v11, v6, v29 :: v_dual_mul_f32 v14, v7, v19
	s_waitcnt vmcnt(1)
	v_fmac_f32_e32 v13, v6, v31
	s_clause 0x3
	global_store_b32 v[24:25], v8, off
	global_store_b32 v[24:25], v9, off offset:512
	global_store_b32 v[24:25], v4, off offset:1024
	global_store_b32 v[24:25], v10, off offset:1536
	s_waitcnt vmcnt(0)
	v_dual_fmac_f32 v1, v15, v10 :: v_dual_fmac_f32 v14, v6, v32
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_mul_f32 v12, v7, v17 :: v_dual_fmac_f32 v1, v20, v11
	v_fmac_f32_e32 v12, v6, v30
	s_clause 0x3
	global_store_b32 v[24:25], v11, off offset:2048
	global_store_b32 v[24:25], v12, off offset:2560
	global_store_b32 v[24:25], v13, off offset:3072
	global_store_b32 v[24:25], v14, off offset:3584
	v_fmac_f32_e32 v1, v21, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v1, v22, v13
	v_fmac_f32_e32 v1, v23, v14
	s_cbranch_scc0 .LBB6_51
; %bb.52:
	v_add_co_u32 v2, vcc_lo, s16, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v3, null, s17, v3, vcc_lo
	global_store_b32 v[2:3], v1, off
	s_endpgm
.LBB6_53:
	s_mov_b32 s4, 0
	s_mov_b32 s3, -1
	s_trap 2
	s_branch .LBB6_1
.LBB6_54:
	s_mov_b32 s6, 0
	s_mov_b32 s3, -1
	s_trap 2
	s_cbranch_execz .LBB6_4
	s_branch .LBB6_58
.LBB6_55:
	s_mov_b32 s6, 0
	s_mov_b32 s3, -1
	s_trap 2
	s_cbranch_execz .LBB6_6
	s_branch .LBB6_58
.LBB6_56:
	s_mov_b32 s6, 0
	s_mov_b32 s3, -1
	s_trap 2
	s_cbranch_execz .LBB6_8
	s_branch .LBB6_58
.LBB6_57:
	s_mov_b32 s6, 0
	s_mov_b32 s3, -1
	s_trap 2
	s_cbranch_execz .LBB6_10
.LBB6_58:
                                        ; implicit-def: $vgpr5
                                        ; implicit-def: $vgpr7
                                        ; implicit-def: $vgpr8
                                        ; implicit-def: $vgpr9
                                        ; implicit-def: $vgpr4
                                        ; implicit-def: $vgpr1
.LBB6_59:                               ; %Flow165
	s_mov_b32 s20, s2
	s_and_b32 vcc_lo, exec_lo, s3
	s_cbranch_vccz .LBB6_45
.LBB6_60:                               ; %UnifiedUnreachableBlock
	; divergent unreachable
	s_and_not1_b32 vcc_lo, exec_lo, s21
	s_cbranch_vccz .LBB6_46
.LBB6_61:                               ; %UnifiedReturnBlock
	s_endpgm
.LBB6_62:
	s_mov_b32 s3, -1
	s_trap 2
	s_cbranch_execz .LBB6_12
.LBB6_63:
                                        ; implicit-def: $vgpr5
                                        ; implicit-def: $vgpr7
                                        ; implicit-def: $vgpr8
                                        ; implicit-def: $vgpr9
                                        ; implicit-def: $vgpr4
	s_branch .LBB6_59
.LBB6_64:
	s_or_b32 exec_lo, exec_lo, s0
	s_and_saveexec_b32 s0, vcc_lo
	s_cbranch_execz .LBB6_24
.LBB6_65:
	ds_load_b32 v5, v12 offset:1152
	s_or_b32 exec_lo, exec_lo, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_eq_u32 exec_lo, -1
	s_mov_b32 s0, -1
	s_cbranch_scc1 .LBB6_25
.LBB6_66:
	s_mov_b32 s3, -1
	s_trap 2
	s_cbranch_execz .LBB6_26
	s_branch .LBB6_71
.LBB6_67:
	s_mov_b32 s4, 0
	s_mov_b32 s3, -1
	s_trap 2
	s_cbranch_execz .LBB6_14
	s_branch .LBB6_71
.LBB6_68:
	s_mov_b32 s4, 0
	s_mov_b32 s3, -1
	s_trap 2
	s_cbranch_execz .LBB6_16
	s_branch .LBB6_71
.LBB6_69:
	s_mov_b32 s4, 0
	s_mov_b32 s3, -1
	s_trap 2
	s_cbranch_execz .LBB6_18
	s_branch .LBB6_71
.LBB6_70:
	s_mov_b32 s4, 0
	s_mov_b32 s3, -1
	s_trap 2
	s_cbranch_execz .LBB6_20
.LBB6_71:
                                        ; implicit-def: $vgpr5
                                        ; implicit-def: $vgpr7
                                        ; implicit-def: $vgpr8
                                        ; implicit-def: $vgpr9
	s_branch .LBB6_59
.LBB6_72:
	s_mov_b32 s3, -1
	s_trap 2
	s_cbranch_execz .LBB6_28
	s_branch .LBB6_71
.LBB6_73:
	s_mov_b32 s3, -1
	s_trap 2
	s_cbranch_execz .LBB6_30
	s_branch .LBB6_71
.LBB6_74:
	s_mov_b32 s3, -1
	s_trap 2
	s_cbranch_execz .LBB6_32
	s_branch .LBB6_71
.LBB6_75:
	s_mov_b32 s3, -1
	s_trap 2
	s_cbranch_execz .LBB6_34
	s_branch .LBB6_71
.LBB6_76:
	s_mov_b32 s3, -1
	s_trap 2
	s_cbranch_execz .LBB6_36
	s_branch .LBB6_71
.LBB6_77:
	s_mov_b32 s3, -1
	s_trap 2
	s_cbranch_execz .LBB6_38
	s_branch .LBB6_71
.LBB6_78:
	s_mov_b32 s3, -1
	s_trap 2
	s_cbranch_execz .LBB6_40
	s_branch .LBB6_71
.LBB6_79:
	s_mov_b32 s3, -1
	s_trap 2
	s_cbranch_execz .LBB6_42
.LBB6_80:
                                        ; implicit-def: $vgpr5
                                        ; implicit-def: $vgpr7
	s_branch .LBB6_59
.Lfunc_end6:
	.size	_Z12rec2c_kernelPKfS0_S0_S0_S0_PfS1_i, .Lfunc_end6-_Z12rec2c_kernelPKfS0_S0_S0_S0_PfS1_i
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z12rec2c_kernelPKfS0_S0_S0_S0_PfS1_i
		.amdhsa_group_segment_fixed_size 1280
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 320
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 33
		.amdhsa_next_free_sgpr 22
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_shared_vgpr_count 0
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end6-_Z12rec2c_kernelPKfS0_S0_S0_S0_PfS1_i)<<4)&1008)>>4
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
                                        ; -- End function
	.set .L_Z12rec2c_kernelPKfS0_S0_S0_S0_PfS1_i.num_vgpr, 33
	.set .L_Z12rec2c_kernelPKfS0_S0_S0_S0_PfS1_i.num_agpr, 0
	.set .L_Z12rec2c_kernelPKfS0_S0_S0_S0_PfS1_i.numbered_sgpr, 22
	.set .L_Z12rec2c_kernelPKfS0_S0_S0_S0_PfS1_i.num_named_barrier, 0
	.set .L_Z12rec2c_kernelPKfS0_S0_S0_S0_PfS1_i.private_seg_size, 0
	.set .L_Z12rec2c_kernelPKfS0_S0_S0_S0_PfS1_i.uses_vcc, 1
	.set .L_Z12rec2c_kernelPKfS0_S0_S0_S0_PfS1_i.uses_flat_scratch, 0
	.set .L_Z12rec2c_kernelPKfS0_S0_S0_S0_PfS1_i.has_dyn_sized_stack, 0
	.set .L_Z12rec2c_kernelPKfS0_S0_S0_S0_PfS1_i.has_recursion, 0
	.set .L_Z12rec2c_kernelPKfS0_S0_S0_S0_PfS1_i.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 2876
; TotalNumSgprs: 24
; NumVgprs: 33
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 1280 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 4
; NumSGPRsForWavesPerEU: 24
; NumVGPRsForWavesPerEU: 33
; Occupancy: 16
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.set amdgpu.max_num_named_barrier, 0
	.text
	.type	__hip_cuid_e7fe99a3618b5683,@object ; @__hip_cuid_e7fe99a3618b5683
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_e7fe99a3618b5683
__hip_cuid_e7fe99a3618b5683:
	.byte	0                               ; 0x0
	.size	__hip_cuid_e7fe99a3618b5683, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_e7fe99a3618b5683
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         40
        .size:           8
        .value_kind:     global_buffer
      - .offset:         48
        .size:           4
        .value_kind:     by_value
      - .offset:         52
        .size:           4
        .value_kind:     by_value
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 56
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z12gates_kernelPKfS0_S0_S0_PfS1_ii
    .private_segment_fixed_size: 0
    .sgpr_count:     15
    .sgpr_spill_count: 0
    .symbol:         _Z12gates_kernelPKfS0_S0_S0_PfS1_ii.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     16
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .actual_access:  read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
      - .offset:         40
        .size:           4
        .value_kind:     by_value
      - .offset:         44
        .size:           4
        .value_kind:     by_value
      - .offset:         48
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         52
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         56
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         60
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         62
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         64
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         66
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         68
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         70
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         88
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         96
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         104
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         112
        .size:           2
        .value_kind:     hidden_grid_dims
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 304
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z11conv_kernelPKfS0_S0_PfS1_ii
    .private_segment_fixed_size: 0
    .sgpr_count:     20
    .sgpr_spill_count: 0
    .symbol:         _Z11conv_kernelPKfS0_S0_PfS1_ii.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     17
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .actual_access:  read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         40
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         48
        .size:           8
        .value_kind:     global_buffer
      - .offset:         56
        .size:           4
        .value_kind:     by_value
      - .offset:         64
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         68
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         72
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         76
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         78
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         80
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         82
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         84
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         86
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         104
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         112
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         120
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         128
        .size:           2
        .value_kind:     hidden_grid_dims
    .gfx1250_revision: B0
    .group_segment_fixed_size: 1152
    .kernarg_segment_align: 8
    .kernarg_segment_size: 320
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z10rec_kernelPKfS0_S0_S0_S0_PfS1_i
    .private_segment_fixed_size: 0
    .sgpr_count:     25
    .sgpr_spill_count: 0
    .symbol:         _Z10rec_kernelPKfS0_S0_S0_S0_PfS1_i.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     33
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .actual_access:  read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         40
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         48
        .size:           8
        .value_kind:     global_buffer
      - .offset:         56
        .size:           4
        .value_kind:     by_value
      - .offset:         64
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         68
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         72
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         76
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         78
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         80
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         82
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         84
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         86
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         104
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         112
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         120
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         128
        .size:           2
        .value_kind:     hidden_grid_dims
    .gfx1250_revision: B0
    .group_segment_fixed_size: 1152
    .kernarg_segment_align: 8
    .kernarg_segment_size: 320
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z11rec0_kernelPKfS0_S0_S0_S0_PfS1_i
    .private_segment_fixed_size: 0
    .sgpr_count:     25
    .sgpr_spill_count: 0
    .symbol:         _Z11rec0_kernelPKfS0_S0_S0_S0_PfS1_i.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     33
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .actual_access:  read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         40
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         48
        .size:           8
        .value_kind:     global_buffer
      - .offset:         56
        .size:           4
        .value_kind:     by_value
      - .offset:         64
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         68
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         72
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         76
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         78
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         80
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         82
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         84
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         86
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         104
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         112
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         120
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         128
        .size:           2
        .value_kind:     hidden_grid_dims
    .gfx1250_revision: B0
    .group_segment_fixed_size: 1152
    .kernarg_segment_align: 8
    .kernarg_segment_size: 320
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z11rec1_kernelPKfS0_S0_S0_S0_PfS1_i
    .private_segment_fixed_size: 0
    .sgpr_count:     22
    .sgpr_spill_count: 0
    .symbol:         _Z11rec1_kernelPKfS0_S0_S0_S0_PfS1_i.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     33
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .actual_access:  read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         40
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         48
        .size:           8
        .value_kind:     global_buffer
      - .offset:         56
        .size:           4
        .value_kind:     by_value
      - .offset:         64
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         68
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         72
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         76
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         78
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         80
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         82
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         84
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         86
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         104
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         112
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         120
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         128
        .size:           2
        .value_kind:     hidden_grid_dims
    .gfx1250_revision: B0
    .group_segment_fixed_size: 1280
    .kernarg_segment_align: 8
    .kernarg_segment_size: 320
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z11rec2_kernelPKfS0_S0_S0_S0_PfS1_i
    .private_segment_fixed_size: 0
    .sgpr_count:     20
    .sgpr_spill_count: 0
    .symbol:         _Z11rec2_kernelPKfS0_S0_S0_S0_PfS1_i.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     33
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .actual_access:  read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         40
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         48
        .size:           8
        .value_kind:     global_buffer
      - .offset:         56
        .size:           4
        .value_kind:     by_value
      - .offset:         64
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         68
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         72
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         76
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         78
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         80
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         82
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         84
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         86
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         104
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         112
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         120
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         128
        .size:           2
        .value_kind:     hidden_grid_dims
    .gfx1250_revision: B0
    .group_segment_fixed_size: 1280
    .kernarg_segment_align: 8
    .kernarg_segment_size: 320
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z12rec2c_kernelPKfS0_S0_S0_S0_PfS1_i
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .sgpr_spill_count: 0
    .symbol:         _Z12rec2c_kernelPKfS0_S0_S0_S0_PfS1_i.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     33
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1151
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
