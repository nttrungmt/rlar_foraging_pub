import os, sys
from band_plot import *
import argparse
#======================================CHANGE======================================================
#dir_string = "./log/plot_data/"

fname_ID_range = range(35)
winsize = 100
alpha = 0.01
num_plot_points = 200
max_x = -1 #10000 # -1 if not used
plot_fname_bases = []

###plot_fname_bases.append('Hoff_20200225_164932_1500_150_num_food_return_')
###plot_fname_bases.append('Hoff_20200225_164932_3000_150_num_food_return_')
###plot_fname_bases.append('Hoff_20200413_121737_3000_150_num_food_return_')
###plot_fname_bases.append('Hoff_20200419_172857_3000_150_num_food_return_')
#plot_fname_bases.append('Hoff_20200419_172857_3000_150_num_food_return_tb_')

#plot_fname_bases.append('dqnPen_0.1_5e-05_1024_512_per_20200314_num_food_return_')
#plot_fname_bases.append('dqnPen_0.1_0.0001_1024_512_per_20200324_num_food_return_')
#plot_fname_bases.append('dqnPen_0.1z_0.0001_1024_512_per_20200324_num_food_return_')
#plot_fname_bases.append('dqnPen_0.1_lrdecay_0.0001_1024_512_per_20200331_num_food_return_')
#plot_fname_bases.append('dqnPen_0.1GpuCrh_0.0001_2048_512_per_20200331_num_food_return_')
#plot_fname_bases.append('dqnPen_0.1GpuCrh_0.0001_1000_400_per_20200402_num_food_return_')
#plot_fname_bases.append('dqnPen_0.1GpuObsv2Norm_0.0001_64_256_per_20200410_num_food_return_')
#plot_fname_bases.append('dqnPen0.1_Obsv2NormLrDec_0.0001_32_256_per_20200413_num_food_return_')
#plot_fname_bases.append('dqnPen0.1zGpuObsv2Norm_0.0001_32_256_per_20200415_num_food_return_')
#plot_fname_bases.append('dqnPen0.1zObsv2Norm_0.0001_32_256_per_20200416_')
#plot_fname_bases.append('dqnPen0.1zObsv2Norm_0.0001_32_400_per_20200419_')
#plot_fname_bases.append('dqnPen0.1zObsv2Norm_0.0001_32_400_per_20200419_num_food_return_')
#plot_fname_bases.append('dqnPen0.1zGpuObsv2Rwd2nbA_0.0001_32_400_per_20200420_num_food_return_')
#plot_fname_bases.append('dqnPen0.1zGpuObsv2Rwd2nb_0.0001_64_400_per_20200422_num_food_return_')
#plot_fname_bases.append('dqnPen0.1zObsv2Norm_1e-05_64_400_per_20200423_num_food_return_')

#plot_fname_bases.append('CpuPen0.1Obsv2_0.0001_64_256_per_20200428_num_food_return_')
#plot_fname_bases.append('CpuPen0.1Obsv2Rwdnb0.5_0.0001_64_256_per_20200430_num_food_return_')
#plot_fname_bases.append('CpuPen0.1z0Obsv2_0.0001_64_256_per_20200428_num_food_return_')
#plot_fname_bases.append('CpuPen0.1z0Obsv2Rwdnb0.5_0.0001_64_256_per_20200430_num_food_return_')
#plot_fname_bases.append('CpuPen0.1z1Obsv2_0.0001_64_256_per_20200502_num_food_return_')
#plot_fname_bases.append('CpuPen0.1z1Obsv2Rwdnb0.5_0.0001_64_256_per_20200502_num_food_return_')

#plot_fname_bases.append('Pen0.1GpuObsv2_0.0001_64_256_per_20200429_num_food_return_')
#plot_fname_bases.append('Pen0.1z0GpuObsv2_0.0001_64_256_per_20200429_num_food_return_')
#plot_fname_bases.append('Pen0.1GpuObsv2Rwdnb0.5_0.0001_64_256_per_20200501_num_food_return_')
#plot_fname_bases.append('Pen0.1z0GpuObsv2Rwdnb0.5_0.0001_64_256_per_20200501_num_food_return_')
#plot_fname_bases.append('Pen0.1z1GpuObsv2_0.0001_64_256_per_20200503_num_food_return_')
#plot_fname_bases.append('Pen0.1z1GpuObsv2Rwdnb0.5_0.0001_64_256_per_20200503_num_food_return_')

#plot_fname_bases.append('dqnPen0.1Obsv2_0.0001_64_256_per_20200505_num_food_return_')
#plot_fname_bases.append('dqnPen0.1Obsv2Rwdnb0.5_0.0001_64_256_per_20200507_num_food_return_')
#plot_fname_bases.append('dqnPen0.1z0Obsv2_0.0001_64_256_per_20200505_num_food_return_')
#plot_fname_bases.append('dqnPen0.1z0Obsv2Rwdnb0.5_0.0001_64_256_per_20200509_num_food_return_')
#plot_fname_bases.append('dqnPen0.1z1Obsv2_0.0001_64_256_per_20200507_num_food_return_')
#plot_fname_bases.append('dqnPen0.1z1Obsv2Rwdnb0.5_0.0001_64_256_per_20200509_num_food_return_')

#plot_fname_bases.append('dqnGpuPen0.1Obsv2_0.0001_64_256_per_20200505_num_food_return_')
#plot_fname_bases.append('dqnGpuPen0.1Obsv2z0_0.0001_64_256_per_20200505_num_food_return_')
#plot_fname_bases.append('dqnGpuPen0.1Obsv2_0.0001_64_256_per_20200507_num_food_return_')
#plot_fname_bases.append('dqnGpuPen0.1Obsv2Rwdnb0.5_0.0001_64_256_per_20200510_num_food_return_')
#plot_fname_bases.append('dqnGpuPen0.1Obsv2z0_0.0001_64_256_per_20200507_num_food_return_')
#plot_fname_bases.append('dqnGpuPen0.1Obsv2z1_0.0001_64_256_per_20200510_num_food_return_')

###################################################################
parser = argparse.ArgumentParser(description='ma-foraging-mdn')
parser.add_argument('--mode', type=str, default='cpu', help='Data location to load data')
parser.add_argument('--method', type=str, default='winavg', help='The moving average to used: winavg or cumavg')
parser.add_argument('--type', type=int, default=5, help='First plot or second')
###################################################################
args = parser.parse_args()
###################################################################
#mode=sys.argv[2] if len(sys.argv)>2 else 'cpu'
mode = args.mode
fig = args.type     #5 
#fig = 6
if mode.lower()=="gpu":
    dir_string = "./log/plot_data_gpu/"
else:
    if(fig == 5):
        dir_string = "./log/plot_data_merge_2/"
        plot_fname_bases.append('Handcoded_')
        plot_fname_bases.append('DQN_No_HF_')
        plot_fname_bases.append('DQN_Full_HF_')
        plot_fname_bases.append('DQN_RLaR_2_')
    else:
        dir_string = "./log/plot_data_merge_2/"
        plot_fname_bases.append('Handcoded_')
        plot_fname_bases.append('DQN_RLaR_1_')
        plot_fname_bases.append('DQN_RLaR_2_')
        plot_fname_bases.append('DQN_RLaR_5_')

# dir_string = "./log/plot_data_#robots/"
# plot_fname_bases.append('test_20_')
# plot_fname_bases.append('test_25_20200612_')
# plot_fname_bases.append('test_30_20200611_')
# plot_fname_bases.append('test_40_20200612_')

#plot_fname_bases.append('FL_')
#plot_fname_bases.append('FL_withHAT_')

#plot_fname_bases.append('BDN_IL_HAT_CCorC_Noise0.1_0.9999992_alpha0.001_cThresh0.7_')
#plot_fname_bases.append('BDN_IL_CHAT_Noise0.1_0.9999992_alpha0.001_cThresh0.7_')
#plot_fname_bases.append('BDN_CCHAT_Noise0.1_0.001_')

#plot_fname_bases.append('FM_IL_HAT_Noise0.1_0.999977_0.005_cThresh0.0_')
#plot_fname_bases.append('FM_IL_HAT_CCorC_Noise0.1_0.999977_alpha0.005_cThresh0.7_')
#plot_fname_bases.append('FM_IL_CHAT_Noise0.1_0.999977_alpha0.005_cThresh0.7_')
#plot_fname_bases.append('FM_CCHAT_Noise0.1_0.005_')
#plot_fname_bases.append('FM_IL_Noise0.1_alpha0.005_')

#plot_fname_bases.append('SwrmRewards_alpha_0.01_')
#plot_fname_bases.append('SwrmRNDRewards_alpha_0.01_')
#plot_fname_bases.append('SwrmRLRewards_alpha_0.01_')

#plot_fname_bases.append('')
line_types=[1,3,4,7,8,9,10,12,13]
#==================================================================================================

out_fnames = []
max_dsize = 0
#method = sys.argv[1] if len(sys.argv)> 1 else "winavg"
method = args.method
for f in plot_fname_bases:
    d, outf = create_plot_data(fname_ID_range, dir_string, f, winsize, alpha, ext=".csv", method=method)
    out_fnames.append( outf )
    #print len(d)
    if len(d) > max_dsize:
        max_dsize = len(d)

data_key_skip = max(d.keys()) / max_dsize
max_num_keys = max_dsize if max_x < 0 else max_x / data_key_skip
e = max(1, int( max_num_keys / num_plot_points ))
#print(e)
#os.system("rm x")
f = open('script','w')
s = "set style fill transparent solid 0.3 noborder\n"
#s += "set terminal png size 1600,900 font \",20\"\n"
s += "set key right bottom noenhanced\n"
s += "set xlabel \"Episode\"\n"
s += "set ylabel \"Number of returned food\"\n"
#s += "set title \""+method+'-'+mode.upper()+"\"\n"
s += "set title font \",28\"\n"
s += "set key font \",24\"\n"
s += "set xlabel font \",22\"\n"
s += "set ylabel font \", 22\"\n"
s += "set xtics font \",14\"\n"
s += "set ytics font \",14\"\n"
s += "set grid ytics mytics\n"
s += "set xtics 200\n"
s += "set mytics 5\n"
s += "set grid\n"
if max_x > 0:
    s+= "set xrange[0:" +str(max_x)+"]\n"
s += "plot "
f.write(s)
s=""
for i in range(len(out_fnames)):
    s += "'"+out_fnames[i]+"' every "+str(e)+" using 1:($2-$3):($2+$3) with filledcurves notitle, '' every "
    s += str(e)+" using 1:2 with lp lt "+str(line_types[i])+" pt "+str(i+1)+" ps 1.5 lw 2 title '"+plot_fname_bases[i]+"',"
f.write(s)
f.close()
os.system('gnuplot -persist script')
