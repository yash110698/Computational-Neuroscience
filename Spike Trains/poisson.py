import random as rnd
import math
import numpy
import matplotlib.pyplot as plt

def get_spike_train(rate,big_t,tau_ref):

    if 1<=rate*tau_ref:
        print("firing rate not possible given refractory period f/p")
        return []


    exp_rate=rate/(1-tau_ref*rate)

    spike_train=[]

    t=rnd.expovariate(exp_rate)

    while t< big_t:
        spike_train.append(t)
        t+=tau_ref+rnd.expovariate(exp_rate)

    return spike_train

##################################################################################################################

def Fano(spike_train, width):
    spike_count = []
    count = 0
    bin = width

    for i in range(0,len(spike_train)):
        while(spike_train[i] > bin):
            spike_count.append(count)
            count = 0
            bin += width
        count+=1
        if(i == len(spike_train)-1):
            spike_count.append(count)
            while(bin < big_t-0.00001):
                spike_count.append(0)
                bin += width

    Mu=len(spike_train)/len(spike_count)
    variance=0
    for i in range(0,len(spike_count)):
        variance += pow((spike_count[i]-Mu),2)
    variance /= len(spike_count)
    ff = variance/Mu

    print("Fano Factor  (", (int)(width*1000),"ms)  \t\t= ",ff,sep='')


##################################################################################################################
def COV(spike_train):
    spike_interval=[]
    interval=0.0
    sum=0.0

    for i in range(1,len(spike_train)):
        interval = spike_train[i]- spike_train[i-1]
        spike_interval.append(interval)
        sum+=interval

    Mu = sum/len(spike_interval)
    variance=0

    for i in range(0,len(spike_interval)):
        variance += pow((spike_interval[i]-Mu),2)

    variance /= len(spike_interval)
    sd = math.sqrt(variance)
    cov = sd/Mu

    print("Coefficient of Variation \t=",cov)


##################################################################################################################

def load_data(filename,T):

    data_array = [T(line.strip()) for line in open(filename, 'r')]

    return data_array

##################################################################################################################

def FanoBinary(spike_train, width):
    spike_count=[]
    wd=width/2
    bin=wd
    count=0
    Mu=0

    for i in range(0,len(spikes)):
        if( i < (bin) ):
            count+=spikes[i]
        elif( i == (bin) ):
            spike_count.append(count)
            Mu+=count
            bin+=wd
            count=spikes[i]
        if( i==len(spikes)-1):
            spike_count.append(count)
            Mu+=count

    Mu/=len(spike_count)
    variance=0

    for i in range(0,len(spike_count)):
        variance += pow((spike_count[i]-Mu),2)

    variance /= len(spike_count)
    ff = variance/Mu

    print("Fano Factor  (", (int)(width),"ms)  \t\t= ",ff,sep='')

##################################################################################################################
def COVBinary(spike_train):

    spike_interval=[]
    sum=0.0
    pos=0

    for i in range(0,len(spike_train)):
        if(spike_train[i] == 1):
            pos=i
            break

    for i in range(pos+1,len(spike_train)):
        if(spike_train[i] == 1):
            interval = (i-pos)*2
            spike_interval.append(interval)
            sum+=interval
            pos=i

    Mu = sum/len(spike_interval)
    variance=0.0

    for i in range(0,len(spike_interval)):
        variance += pow((spike_interval[i]-Mu),2)

    variance /= len(spike_interval)
    sd = math.sqrt(variance)
    cov = sd/Mu

    print("Coefficient of Variation \t=",cov)


##################################################################################################################


Hz=1.0
sec=1.0
ms=0.001

rate = 35.0 *Hz         #firing rate
tau_ref = 0*ms          #refractory period
big_t = 1000*sec        #total time

print("\n--------------------QUESTION 1--------------------\n")
print("Total Time Period = ", big_t, " sec")
print("Firing Rate = ", rate, " Hz")

spike_train = get_spike_train(rate,big_t,tau_ref)
print("\n\nREFRACTORY TIME = ", tau_ref, " sec")
print("No. of spikes = ", len(spike_train),"\n")
Fano(spike_train, 0.010)
Fano(spike_train, 0.050)
Fano(spike_train, 0.100)
COV(spike_train)


tau_ref = 5*ms        #refractory period
spike_train = get_spike_train(rate,big_t,tau_ref)
print("\n\nREFRACTORY TIME = ", tau_ref, " sec")
print("No. of spikes = ", len(spike_train),"\n")
Fano(spike_train, 0.010)
Fano(spike_train, 0.050)
Fano(spike_train, 0.100)
COV(spike_train)


print("\n\n--------------------QUESTION 2--------------------\n")

rate=500.0*Hz         #firing rate
big_t=1200*sec        #total time
spikes=load_data("rho.dat",int)
print("Total Time Period = ", big_t, " sec")
print("Firing Rate = ", rate, " Hz\n")
FanoBinary(spikes, 10)
FanoBinary(spikes, 50)
FanoBinary(spikes, 100)
COVBinary(spikes)

##################################################################################################################
print("\n\n--------------------QUESTION 3--------------------\n")
autoc = numpy.zeros(101)

for i in range(0,len(spikes)):
    if(spikes[i] == 1):
        index=0
        for flag in range(i-50,i+51):
            if(flag>=0 and flag <len(spikes) and spikes[flag]==1):
                autoc[index]+=1
            index+=1

for i in range(0,len(autoc)):
    autoc[i] /= (len(spikes)/100)

x = numpy.arange(-100,101,2,dtype=int)
plt.plot(x, autoc)
plt.ylabel('autocorrelation')
plt.xlabel('Interval (s)')
plt.title('Q3: Autocorrelogram')
plt.show()

##################################################################################################################
print("\n\n--------------------QUESTION 4--------------------\n")
stimulus=load_data("stim.dat",float)
STA = numpy.zeros(51)

for i in range(50,len(spikes)):
    if(spikes[i] == 1):
        index=0
        for flag in range(i-50,i+1):
            STA[index] += stimulus[flag]*(flag)*2*ms
            index+=1

for i in range(0,len(STA)):
    STA[i] /= (len(stimulus)*50)

x = numpy.arange(-100,1,2,dtype=int)
plt.plot(x, STA)
plt.ylabel('spike average')
plt.xlabel('Interval (s)')
plt.title('Q4: Spike-Triggered Average')
plt.show()


##################################################################################################################
