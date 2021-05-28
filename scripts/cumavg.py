#Assumes an N column file: i x1 x2 ... xN
#Outputs a stream of N+1 columns: t x1 .. xN --cumulative-averaged
#Currently setup to convert episode length to reward for a certain reward function

import sys,csv

nargs = len(sys.argv)
if nargs < 4:
    print("Usage: ",sys.argv[0]," <n-cols> <data file> <min. window size>")
else:
    N, fName, min_window, alpha = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
    inf = open(fName,'r')
    inp = csv.reader(inf, delimiter=',')
    next(inp)
    iBuffer = [] # window for i
    xiBuffer = [] # windows for xi
    for i in range(N):
        xiBuffer.append([])
    runningAvg = [0.0]*N
    for row in inp:
        if len(row) != N+1:
            #print "wrong #cols:",len(row), row
            continue
        #if row[0]=='0':
        #    print int(row[0]), float(row[1])
        #    runningAvg = float(row[1])
        else:
            cur_x = int(row[0])
            iBuffer.append( cur_x )
            for i in range(N):
                num = float(row[i+1])
                #====MODIFY==================================================
                gamma = 0.999
                max_steps = 200
                #if (num < max_steps):
                #    val = ((-1.0)*(1-gamma**(num-1))/(1-gamma)) + 100.0*(gamma**(num-1))
                    #val = 200 #-1.0* (num-1) + 100.0
                #else:
                #    val = ((-1.0)*(1-gamma**(num))/(1-gamma))
                    #val = -1.0*(num)
                #xiBuffer[i].append( val )
                xiBuffer[i].append( num )
                #============================================================
            if (len(xiBuffer[i]) < min_window):
                continue
            s = str(iBuffer[0])+' '
            for i in range(N):
                length = len(xiBuffer[i])
                if (length <= min_window): 
                    runningAvg[i] = (sum(xiBuffer[i]) / (1.0 * len(xiBuffer[i])))
                else:
                    #runningAvg[i] = (((length-1)*runningAvg[i]) + xiBuffer[i][-1] ) / length
                    runningAvg[i] = ( ((1-alpha)*runningAvg[i]) + alpha*xiBuffer[i][-1] )       #20200612
                s += str(runningAvg[i])+' '
            print(s) # sum(xiBuffer) / (1.0 * windowSize)
            del iBuffer[0]

