# ECE273_Project
ECE273 project (Spring 2020)
Written by Guoren Zhong, June 8th, 2020

This is the codes for topic 2 of ECE273 project: Blind deconvolution.

There are 7 files in total written in MATLAB, and below is the instruction:

"Blind_Deconvolution_NNM.m" and "Nonconvex_blind_deconvolution_regGrad.m" are the basic algorithm implementations of NNM algorithm and regGrad algorithm, which do not have any plotting outputs. They can be used to test the algorithm with single input.

"Figure1_result_convex_NNM.m" outputs two phase transition diagram for "sparse" and "short" input respectively. To run this, start with small inputs, or it may take a century to generate the outputs.

"Figure2_comparing_NNM_and_nonblind.m" outputs the comparison for blind deconvolution and nonblind deconvolution. **

"Figure3_comparing_NNM_and_regGrad.m" outputs the comparison for NNM and regGrad algorithms.

"Figure4_robustness_sparsity.m" outputs the comparison for sparse inputs and dense inputs. **

"Figure5_robustness_low_rank.m" consists of 2 experiments for testing the robustness against violating low-rank condition. **

\*\*Note: to plot Fig.2, Fig.4, and Fig.5, you need to run the "Convex Approach --- NNM" block in "Figure3_comparing_NNM_and_regGrad.m" to get the data for NNM first (should be named as "P_success_convex").
