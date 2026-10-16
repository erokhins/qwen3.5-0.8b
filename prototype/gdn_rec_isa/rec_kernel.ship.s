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
