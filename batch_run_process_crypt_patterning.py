#!/usr/bin/python3

from multiprocessing.pool import Pool
import subprocess
from subprocess import Popen
import sys
import time


def work(input):
    job = subprocess.call(input)


if __name__ == '__main__':

    # Initial sweep to set up time-scales by fitting MSD curve 03-06-21 *************************************************************************************
    #
    # param_Ft = ['0.000', '1.000', '2.000', '3.000']
    # param_tdiff = ['0.500', '1.000', '2.000', '4.000']
    # param_rseed = ['0.600', '1.000', '1.400', '1.800']
    # replicates = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    #
    # is_pattern = '0'
    #
    # # k_diff=0.05
    # source = "crypt_patterning.cu"
    #
    # compile_call = ["nvcc", "-std=c++11", "-arch=sm_61", "-Wno-deprecated-gpu-targets", source, "-o", "exec"]
    # call = subprocess.run(compile_call, stdout=subprocess.PIPE)
    #
    # for r in replicates:
    #     for Ft in param_Ft:
    #         for t_diff in param_tdiff:
    #             for r_seed in param_rseed:
    #                 model_call = ["./exec", is_pattern, Ft, t_diff, r_seed, r]
    #                 call = subprocess.run(model_call, encoding='utf-8', stdout=subprocess.PIPE)
    #
    # all_inputs=[]
    #
    # output_path = '/g/sharpeba/members/Miquel_Marin/crypt_patterning_output/crypt_patterning_sweep_time_scales_03-06-21/'
    #
    # script_1 = "compute_MSD_stem_cell_ratios.py"
    #
    # for r in replicates:
    #     for Ft in param_Ft:
    #         for t_diff in param_tdiff:
    #             for r_seed in param_rseed:
    #                 file_pattern = output_path + 'crypt_simulation_no_pattern_Ft_'+Ft+'_t-diff_'+t_diff+'_r-seed_'+r_seed+'_rep_'+rep+'_*'
    #
    #                 command = ["python3", script_1, file_pattern, output_path + "time_series/"]
    #                 all_inputs.append(command)
    #
    # n_proc = int(sys.argv[1])
    # print("n processes =", n_proc)
    #
    # tp = Pool(processes=n_proc)
    #
    # print(tp.map(work, all_inputs))
    #
    # tp.close()
    # tp.join()


    # Initial sweep to set up time-scales by fitting MSD curve (refining parameters) 15-06-21 *************************************************************************************

    # param_Ft = ['1.00', '1.17', '1.33', '1.50', '1.67', '1.83', '2.00', '2.17', '2.33', '2.50']
    # param_rseed = ['1.00', '1.11', '1.22', '1.33', '1.44', '1.56', '1.67', '1.78', '1.89', '2.00']
    # replicates = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    #
    # t_diff = '10.0'
    # is_pattern = '0'
    # # k_diff = 0.1
    # source = "crypt_patterning.cu"
    #
    # compile_call = ["nvcc", "-std=c++11", "-arch=sm_61", "-Wno-deprecated-gpu-targets", source, "-o", "exec"]
    # call = subprocess.run(compile_call, stdout=subprocess.PIPE)
    #
    # for r in replicates:
    #     for Ft in param_Ft:
    #         for t_diff in param_tdiff:
    #             for r_seed in param_rseed:
    #                 model_call = ["./exec", is_pattern, Ft, t_diff, r_seed, r]
    #                 call = subprocess.run(model_call, encoding='utf-8', stdout=subprocess.PIPE)
    #
    # all_inputs=[]
    #
    # output_path = '/g/sharpeba/members/Miquel_Marin/crypt_patterning_output/crypt_patterning_sweep_time_scales_15-06-21/'
    #
    # script_1 = "compute_MSD_stem_cell_ratios.py"
    #
    # for r in replicates:
    #     for Ft in param_Ft:
    #         for r_seed in param_rseed:
    #             file_pattern = output_path + 'crypt_simulation_no_pattern_Ft_'+Ft+'_t-diff_'+t_diff+'_r-seed_'+r_seed+'_rep_'+r+'_*'
    #
    #             command = ["python3", script_1, file_pattern, output_path + "time_series/"]
    #             all_inputs.append(command)
    #
    # n_proc = int(sys.argv[1])
    # print("n processes =", n_proc)
    #
    # tp = Pool(processes=n_proc)
    #
    # print(tp.map(work, all_inputs))
    #
    # tp.close()
    # tp.join()

    # Second sweep to explore spatial scales (crypt radius and crypt separation) 28-10-21 *************************************************************************************

    # param_kdeg = ['0.01', '0.02', '0.03', '0.04', '0.05', '0.06', '0.07', '0.08', '0.09', '0.10']
    # param_pratio = ['0.01', '0.02', '0.03', '0.04', '0.05', '0.06', '0.07', '0.08', '0.09', '0.10']
    # replicates = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    #
    # t_diff = '10.0'
    # Ft = '1.0'
    # r_seed = '1.6'
    # is_pattern = '0'
    # # k_diff = 0.1
    # source = "crypt_patterning.cu"
    #
    # # compile_call = ["nvcc", "-std=c++14", "-arch=sm_61", "-Wno-deprecated-gpu-targets", source, "-o", "exec"]
    # # call = subprocess.run(compile_call, stdout=subprocess.PIPE)
    # #
    # # for r in replicates:
    # #     for k_deg in param_kdeg:
    # #         for p_ratio in param_pratio:
    # #             model_call = ["./exec", is_pattern, Ft, t_diff, r_seed, k_deg, p_ratio, r]
    # #             call = subprocess.run(model_call, encoding='utf-8', stdout=subprocess.PIPE)
    # #             # print(model_call)
    #
    # all_inputs=[]
    #
    # output_path = "/home/miquel/data/crypt_patterning_output/crypt_patterning_spatial_scales_28-10-21/"
    #
    # script_1 = "compute_crypt_histogram.py"
    # for r in replicates:
    #     for k_deg in param_kdeg:
    #         for p_ratio in param_pratio:
    #             file_pattern = output_path + 'crypt_simulation_no_pattern_Ft_'+Ft+'_t-diff_'+t_diff+'_r-seed_'+r_seed+'_k-w-deg_'+k_deg+'_paneth-ratio_'+p_ratio+'_rep_'+r+'_1.vtk'
    #             # print(file_pattern)
    #             command = [sys.executable, script_1, file_pattern, output_path + "hists/"]
    #             # print(command)
    #             # call = subprocess.run(command, encoding='utf-8', stdout=subprocess.PIPE)
    #             all_inputs.append(command)
    #
    # n_proc = int(sys.argv[1])
    # print("n processes =", n_proc)
    #
    # tp = Pool(processes=n_proc)
    #
    # print(tp.map(work, all_inputs))
    #
    # tp.close()
    # tp.join()


    # param_kdeg = ['0.070', '0.074', '0.078', '0.082', '0.086', '0.090', '0.094', '0.098', '0.102', '0.106', '0.110']
    # param_pratio = ['0.070', '0.074', '0.078', '0.082', '0.086', '0.090', '0.094', '0.098', '0.102', '0.106', '0.110']
    # replicates = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    #
    # t_diff = '10.0'
    # Ft = '1.0'
    # r_seed = '1.6'
    # is_pattern = '0'
    # # k_diff = 0.1
    # source = "crypt_patterning.cu"
    #
    # compile_call = ["nvcc", "-std=c++14", "-arch=sm_61", "-Wno-deprecated-gpu-targets", source, "-o", "exec"]
    # call = subprocess.run(compile_call, stdout=subprocess.PIPE)
    #
    # for r in replicates:
    #     for k_deg in param_kdeg:
    #         for p_ratio in param_pratio:
    #             model_call = ["./exec", is_pattern, Ft, t_diff, r_seed, k_deg, p_ratio, r]
    #             call = subprocess.run(model_call, encoding='utf-8', stdout=subprocess.PIPE)
    #             # print(model_call)
    #
    # all_inputs=[]
    #
    # output_path = "/g/sharpe/members/Miquel_Marin/crypt_patterning_output/crypt_patterning_spatial_scales_12-01-21/"
    #
    # script_1 = "compute_crypt_histogram.py"
    # for r in replicates:
    #     for k_deg in param_kdeg:
    #         for p_ratio in param_pratio:
    #             file_pattern = output_path + 'crypt_simulation_no_pattern_Ft_'+Ft+'_t-diff_'+t_diff+'_r-seed_'+r_seed+'_k-w-deg_'+k_deg+'_paneth-ratio_'+p_ratio+'_rep_'+r+'_1.vtk'
    #             # print(file_pattern)
    #             command = [sys.executable, script_1, file_pattern, output_path + "hists/"]
    #             # print(command)
    #             # call = subprocess.run(command, encoding='utf-8', stdout=subprocess.PIPE)
    #             all_inputs.append(command)
    #
    # n_proc = int(sys.argv[1])
    # print("n processes =", n_proc)
    #
    # tp = Pool(processes=n_proc)
    #
    # print(tp.map(work, all_inputs))
    #
    # tp.close()
    # tp.join()


    # # Second sweep to set up time-scales by fitting MSD curve (refining parameters) 28-01-22 *************************************************************************************
    #
    # param_Ft = ['1.00', '1.17', '1.33', '1.50', '1.67', '1.83', '2.00', '2.17', '2.33', '2.50']
    # param_rseed = ['1.00', '1.11', '1.22', '1.33', '1.44', '1.56', '1.67', '1.78', '1.89', '2.00']
    # replicates = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    #
    # t_diff = '10.0'
    # is_pattern = '0'
    # # k_diff = 0.1
    # source = "crypt_patterning.cu"
    #
    # # compile_call = ["nvcc", "-std=c++11", "-arch=sm_61", "-Wno-deprecated-gpu-targets", source, "-o", "exec"]
    # # call = subprocess.run(compile_call, stdout=subprocess.PIPE)
    # #
    # # for r in replicates:
    # #     for Ft in param_Ft:
    # #         for r_seed in param_rseed:
    # #             model_call = ["./exec", is_pattern, Ft, t_diff, r_seed, "0.05", "0.05", r]
    # #             call = subprocess.run(model_call, encoding='utf-8', stdout=subprocess.PIPE)
    # #
    # #             model_call = ["./exec", is_pattern, Ft, t_diff, r_seed, "0.098", "0.09", r]
    # #             call = subprocess.run(model_call, encoding='utf-8', stdout=subprocess.PIPE)
    #
    #
    # all_inputs=[]
    #
    # output_path = '/g/sharpe/members/Miquel_Marin/crypt_patterning_output/crypt_patterning_sweep_time_scales_28-01-22/'
    #
    # script_1 = "compute_MSD_stem_cell_ratios.py"
    #
    # for r in replicates:
    #     for Ft in param_Ft:
    #         for r_seed in param_rseed:
    #             k_deg = '0.05'
    #             p_ratio = '0.05'
    #             file_pattern = output_path + 'crypt_simulation_no_pattern_Ft_'+Ft+'_t-diff_'+t_diff+'_r-seed_'+r_seed+'_k-w-deg_'+k_deg+'_paneth-ratio_'+p_ratio+'_rep_'+r+'_*'
    #
    #             command = ["python3", script_1, file_pattern, output_path + "time_series/"]
    #             all_inputs.append(command)
    #
    #             k_deg = '0.098'
    #             p_ratio = '0.09'
    #             file_pattern = output_path + 'crypt_simulation_no_pattern_Ft_'+Ft+'_t-diff_'+t_diff+'_r-seed_'+r_seed+'_k-w-deg_'+k_deg+'_paneth-ratio_'+p_ratio+'_rep_'+r+'_*'
    #
    #             command = ["python3", script_1, file_pattern, output_path + "time_series/"]
    #             all_inputs.append(command)
    #
    #
    #
    # n_proc = int(sys.argv[1])
    # print("n processes =", n_proc)
    #
    # tp = Pool(processes=n_proc)
    #
    # print(tp.map(work, all_inputs))
    #
    # tp.close()
    # tp.join()

    # Running patterns with variations in pattern distance *************************************************************************************
    # param_lpattern = ['6.00', '9.24', '18.0']
    # replicates = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    #
    # Ft = '1.67'
    # t_diff = '10.0'
    # r_seed = '1.56'
    # k_w_deg = '0.098'
    # p_ratio = '0.09'
    # r_pattern = '2.55'
    # is_pattern = '1'
    #
    # # k_diff = 0.1
    # source = "crypt_patterning.cu"
    #
    # compile_call = ["nvcc", "-std=c++11", "-arch=sm_61", "-Wno-deprecated-gpu-targets", source, "-o", "exec"]
    # call = subprocess.run(compile_call, stdout=subprocess.PIPE)
    #
    # for r in replicates:
    #     for l_pattern in param_lpattern:
    #         model_call = ["./exec", is_pattern, Ft, t_diff, r_seed, k_w_deg, p_ratio, r_pattern, l_pattern, r]
    #         call = subprocess.run(model_call, encoding='utf-8', stdout=subprocess.PIPE)


    # all_inputs=[]

    # output_path = '/g/sharpe/members/Miquel_Marin/crypt_patterning_output/crypt_patterning_small sweep_w_patterns_07-02-22/'
    #
    # script_1 = "compute_MSD_stem_cell_ratios.py"
    #
    # for r in replicates:
    #     for Ft in param_Ft:
    #         for r_seed in param_rseed:
    #             k_deg = '0.05'
    #             p_ratio = '0.05'
    #             file_pattern = output_path + 'crypt_simulation_no_pattern_Ft_'+Ft+'_t-diff_'+t_diff+'_r-seed_'+r_seed+'_k-w-deg_'+k_deg+'_paneth-ratio_'+p_ratio+'_rep_'+r+'_*'
    #
    #             command = ["python3", script_1, file_pattern, output_path + "time_series/"]
    #             all_inputs.append(command)
    #
    #             k_deg = '0.098'
    #             p_ratio = '0.09'
    #             file_pattern = output_path + 'crypt_simulation_no_pattern_Ft_'+Ft+'_t-diff_'+t_diff+'_r-seed_'+r_seed+'_k-w-deg_'+k_deg+'_paneth-ratio_'+p_ratio+'_rep_'+r+'_*'
    #
    #             command = ["python3", script_1, file_pattern, output_path + "time_series/"]
    #             all_inputs.append(command)



    # n_proc = int(sys.argv[1])
    # print("n processes =", n_proc)
    #
    # tp = Pool(processes=n_proc)
    #
    # print(tp.map(work, all_inputs))
    #
    # tp.close()
    # tp.join()


    # Running patterns with variations in pattern distance with corrected pattern lengthscales 30-08-22 *************************************************************************************
    replicates = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    Ft = '1.67'
    t_diff = '10.0'
    r_seed = '1.56'
    k_w_deg = '0.098'
    p_ratio = '0.09'
    r_pattern = '1.65'
    is_pattern = '1'


    r_pattern = 1.75
    scale_factor = r_pattern / 4.5
    param_lpattern = [str(15*scale_factor)[:4], str(20*scale_factor)[:4], str(40*scale_factor)[:5]]
    r_pattern = str(r_pattern)

    # k_diff = 0.1
    source = "crypt_patterning.cu"

    compile_call = ["nvcc", "-std=c++14", "-arch=sm_86", "-Wno-deprecated-gpu-targets", source, "-o", "exec"]
    call = subprocess.run(compile_call, stdout=subprocess.PIPE)

    for r in replicates:
        for l_pattern in param_lpattern:
            print("l",l_pattern,"rep",r)
            model_call = ["./exec", is_pattern, Ft, t_diff, r_seed, k_w_deg, p_ratio, r_pattern, l_pattern, r]
            call = subprocess.run(model_call, encoding='utf-8', stdout=subprocess.PIPE)

    output_path = '/home/miquel/data/crypt_simulation_output/crypt_patterning_small_sweep_w_patterns_30-08-22/'

    script_1 = "compute_crypt_histogram.py"

    all_inputs = []
    for r in replicates:
        for l_pattern in param_lpattern:
                file_pattern = output_path + 'crypt_simulation_w_pattern_Ft_'+Ft+'_t-diff_'+t_diff+'_r-seed_'+r_seed+'_k-w-deg_'+k_w_deg+'_paneth-ratio_'+p_ratio+'_r-pattern_'+r_pattern+'_l-pattern_'+l_pattern+'_rep_'+r+'_1.vtk'

                command = ["python3", script_1, file_pattern, output_path + "hists/"]
                all_inputs.append(command)


    n_proc = 8
    print("n processes =", n_proc)

    tp = Pool(processes=n_proc)

    print(tp.map(work, all_inputs))

    tp.close()
    tp.join()
