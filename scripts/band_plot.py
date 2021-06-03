#Input: A series of file names ending in '..integer.raw'.
#Do: window each time series, find mean and sd, plot mean w confidence bands (for visualization only)
#Output: create a plot data file (x y ystderror) w a proper file name in the input directory

import os
import csv

#-----------------------CHANGE-----------------------------------------------------
dir_string = "./data/"

fname_base_string = "GN_IL_HAT_CCorC_Noise0.1_0.999977_alpha0.005_cThresh0.7_"

n_cols = 1 #x y1 .. yN, N=n_cols
fname_ID_range = range(8)
winsize = 20
alpha = 1.0
num_plot_points = 50
#winsum_string = "~/code/keepaway/tools/genwinsum "
winsum_string_BB = "python ./scripts/winsum.py "
cumsum_string_BB = "python ./scripts/cumavg.py "
#----------------------------------------------------------------------------------

def mean(data):
    """Return the sample arithmetic mean of data."""
    n = len(data)
    if n < 1:
        raise ValueError('mean requires at least one data point')
    return sum(data)/float(n) # in Python 2 use sum(data)/float(n)

def _ss(data):
    """Return sum of square deviations of sequence data."""
    c = mean(data)
    ss = sum((x-c)**2 for x in data)
    return ss

def stddev(data, ddof=1):
    """Calculates the population standard deviation when ddof=0
    otherwise computes the sample standard deviation."""
    n = len(data)
    if n < 2:
        print("HERE", data)
        raise ValueError('variance requires at least two data points')
    ss = _ss(data)
    pvar = ss/(n-ddof)
    return pvar**0.5

def create_plot_data(fname_ID_range, dir_string, fname_base_string, winsize, alpha, ext=".raw", n_cols=1, method="winavg"):
    data_map = {}
    if ext==".raw":
        outfname = dir_string + fname_base_string + "plot.data"
    else:
        outfname = dir_string + fname_base_string + "plot"+ext+".data"
    outf = open(outfname, 'w')
                     
    for fid in fname_ID_range:
        #windowing_string = winsum_string + str(winsize) + " " + str(alpha) + " < " + dir_string + fname_base_string + str(fid) +".raw" + " > x"
        if(method == "winavg"):
            windowing_string = winsum_string_BB+str(n_cols)+' '+dir_string + fname_base_string + str(fid) +ext+' '+str(winsize) + " " + str(alpha) + " > x"
        else:
            windowing_string = cumsum_string_BB+str(n_cols)+' '+dir_string + fname_base_string + str(fid) +ext+' '+str(winsize) + " " + str(alpha) + " > x"
        #print windowing_string
        os.system(windowing_string)
        #t = input("Wait")
        inp = csv.reader(open('x','r'), delimiter=' ')
        for row in inp:
            #print(row)
            x = int(row[0])
            if x not in data_map:
                data_map[x] = []
                for i in range(n_cols):
                    data_map[x].append([])
            for i in range(n_cols):
                y = float(row[i+1])
                data_map[x][i].append( y )
                data_map[x][i].append( y )      #20200313 trung add to have more than 1 data points
                #print(x,y)
    
    #n = len(fname_ID_range) #number of series 20210504
    for x in sorted(data_map.keys()):
        s = str(x) + ' ' 
        for i in range(n_cols):
            if (len(data_map[x][i]) >= 2):
                n = len(data_map[x][i]) #number of series 20210504
                avg = mean(data_map[x][i])
                sd = (2.032/(n**0.5))*stddev(data_map[x][i])    #update sd cor confident level 95% of student's t statistics
                s += str(avg) + ' ' + str(sd) + ' '
            else:
                print("Too few points for Key=",x, "Column=",i, len(data_map[x][i]))
                s += '0 0 '
        outf.write(s+'\n')

    outf.close()
    #os.system("rm x")
    return data_map, outfname

if __name__ == '__main__':
    data_map, outfname = create_plot_data(fname_ID_range, dir_string, fname_base_string, winsize, alpha)
    e = int( len(data_map) / num_plot_points )
    f = open('script','w')
    s = "set style fill transparent solid 0.2 noborder\n"
    f.write(s)
    s = "plot '"+outfname+"' every "+str(e)+" using 1:($2-$3):($2+$3) with filledcurves notitle, '' every "
    s+=str(e)+" using 1:2 with lp lt 1 pt 7 ps 1.5 lw 2 title '"+fname_base_string+"'"
    f.write(s)
    f.close()
    os.system('gnuplot -persist script')
#os.system("rm script")
