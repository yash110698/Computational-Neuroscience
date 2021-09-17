import random as rnd
import math
import numpy
import matplotlib.pyplot as plt

sec=1.0
ms=0.001
mV=0.001
nA=0.001
MOhm=pow(10, 0)
nS=pow(10, -3)

##################################################################################################################
def IFNeuronSynapse( V1, V2, Es):

    V1*=mV
    V2*=mV
    Es*=mV

    Vrest = -80*mV
    Vth   = -54*mV
    El    = -70*mV
    RmIe  =  18*mV
    Tm = 20*ms
    Ts = 10*ms
    deltaS = 0.5
    Rmgs = 0.15
    dt = 0.25*ms
    timeLimit = 1*sec
    timeLoop = int(timeLimit/dt)

    s1=0.0
    s2=0.0
    RmIs_1=0.0
    RmIs_2=0.0
    volt_1=[]
    volt_2=[]

    for x in range(0,timeLoop):
        s1 -= (s1*dt/Ts)
        s2 -= (s2*dt/Ts)
        RmIs_1 = s1*Rmgs*(Es-V1)
        RmIs_2 = s2*Rmgs*(Es-V2)
        V1 += dt* ( (El - V1 + RmIe + RmIs_1)/Tm )
        V2 += dt* ( (El - V2 + RmIe + RmIs_2)/Tm )
        if( V1 > Vth ):
            V1 = Vrest
            s2+=deltaS
        if( V2 > Vth ):
            V2 = Vrest
            s1+=deltaS
        volt_1.append(V1)
        volt_2.append(V2)

    for x in range(0,timeLoop):
        volt_1[x]*=1000
        volt_2[x]*=1000

    x = numpy.arange(1, (timeLimit/ms)+1, 0.25, dtype=float)
    plt.plot(x, volt_1, label="neuron 1")
    plt.plot(x, volt_2, label="neuron 2")
    plt.ylabel('Membrane Potential (mV)')
    plt.xlabel('Time (ms)')
    plt.legend(bbox_to_anchor=(1.13,1),loc=4)
    if(Es == 0.0):
        plt.title('Q2) Synapses are excitatory,  (Es = 0mV)')
        plt.savefig('A}Q2_E0.png')
    else:
        plt.title('Q2) Synapses are inhibitory,  (Es = -80mV)')
        plt.savefig('A}Q2_E80.png')
    plt.show()

def IFNeuron():

    Vrest = -70*mV
    Vth   = -40*mV
    V     = Vrest

    El = -70*mV
    Ie = 3.1*nA
    Rm = 10*MOhm
    Tm = 10*ms
    dt = 0.25*ms
    timeLimit = 1*sec
    timeLoop = int(timeLimit/dt)

    voltage=[]
    voltage.append(V)

    for x in range(1,timeLoop):
        dv = dt* ( (El - V + Rm*Ie)/Tm )
        V = V + dv
        if( V > Vth ):
            V = Vrest
        voltage.append(V)

    for x in range(0,timeLoop):
        voltage[x]*=1000

    x = numpy.arange(1, (timeLimit/ms)+1, 0.25, dtype=float)
    plt.plot(x, voltage, label="neuron")
    plt.ylabel('Membrane Potential (mV)')
    plt.xlabel('Time (ms)')
    plt.legend(bbox_to_anchor=(1.13,1),loc=4)
    plt.title('Q1) Integrate & Fire Neuron Model')
    plt.savefig('A}Q1).png')
    plt.show()
##################################################################################################################
def question1():

    Vreset= -65*mV
    Vrest = -65*mV
    Vth   = -50*mV
    V     = -65*mV
    El    = -65*mV
    Rm = 100*MOhm
    Ie = 0*nS
    Tm = 10*ms
    Ts = 2*ms
    Es = 0.0*mV
    deltaS = 0.5
    r = 15                          #firing rate of Pre-Synaptic neuron(s)

    dt = 0.25*ms                    #timestep
    timeLimit = 1*sec               #total simulation time
    timeLoop = int(timeLimit/dt)    #no. of loop iterations for simulation

    N = 40                          #no of Synapses
    g_bar = [4*nS]*N                #array storing synaptic weights/strength
    s = [0.0]*N                     #array storing synaptic values/inputs
    voltage=[]                      #array storing output voltages of Post-Synaptic neuron

    PostFiringRate = 0              #firing rate of Post-Synaptic neuron

    for x in range(0,timeLoop):
        sum_s = 0.0
        for i in range(0,N):
            s[i] -= (s[i]*dt/Ts)
            if(rnd.random() < (r*dt)):
                s[i] += deltaS - (s[i]*dt/Ts)
            sum_s += s[i]*g_bar[i]
        Is = (Es-V) * sum_s
        dv = (El - V + (Rm*Is))*(dt/Tm)
        V += dv
        if( V > Vth ):
            V = Vrest
            PostFiringRate +=1
        voltage.append(V)

    for x in range(0,timeLoop):
      voltage[x]*=1000

    print("\n--------------------------------QUESTION 1--------------------------------")
    print("No. of Synapses ",N)
    print("   >>> Post-Synaptic Neuron Firing Rate (~20Hz) =",PostFiringRate,"Hz\n")

    xaxis = numpy.arange(1, (timeLimit/ms)+1, 0.25, dtype=float)
    plt.plot(xaxis, voltage, label="Post-synaptic neuron")
    plt.ylabel('Membrane Potential (mV)')
    plt.xlabel('Time (ms)')
    plt.legend(bbox_to_anchor=(1.13,1.06),loc=4)
    plt.title('Q1) Voltage of Neuron with 40 synapses  (Firing Rate = %.1f Hz)' %PostFiringRate)
    plt.savefig('Q1)NeuronVoltage.png')
    plt.show()
##################################################################################################################
def simulation2(STDP, Gbar, isPlot):

    if(STDP):
        print("STDP on")
    else:
        print("STDP off")
    print("gbar %.2f nS"%Gbar)

    Vreset= -65*mV
    Vrest = -65*mV
    Vth   = -50*mV
    V     = -65*mV
    El    = -65*mV
    Rm = 100*MOhm
    Ie = 0*nS
    Tm = 10*ms
    Ts = 2*ms
    Es = 0.0*mV
    deltaS = 0.5
    r = 15

    dt = 0.25*ms
    timeLimit = 300*sec
    timeLoop = int(timeLimit/dt)

    N = 40
    g_bar = [Gbar*nS]*N
    s = [0.0]*N

    Tpre = [0.0*sec]*N
    Tpost = -1000.0*sec
    delta_T = 0.0
    voltage=[]

    PostFiringRate = 0
    PostSpikes = 0
    isPotent = False
    final30 = 0
    timeBin = 10*sec
    spikesBin = 0
    rateBin = []

    for x in range(0,timeLoop):
        sum_s = 0.0
        for i in range(0,N):
            s[i] -= (s[i]*dt/Ts)
            if(isPotent==True and  Tpost!=Tpre[i]):
                g_bar[i] = plasticity( STDP, g_bar[i], Tpost - Tpre[i])         #POTENTIATION
            if(rnd.random() < (r*dt)):
                s[i] += deltaS - (s[i]*dt/Ts)
                Tpre[i] = (x+1)*dt                                              #Tpre update spike time
                if(Tpost>0):
                    g_bar[i] = plasticity( STDP, g_bar[i], Tpost - Tpre[i])     #DEPRESSION
            sum_s += s[i]*g_bar[i]

        Is = (Es-V) * sum_s
        dv = (El - V + (Rm*Is))*(dt/Tm)
        V += dv
        isPotent = False
        if( V > Vth ):
            V = Vrest
            PostSpikes +=1
            spikesBin +=1
            isPotent = True
            Tpost = (x+1)*dt                                                    #Tpost update spike time
            if((x+1)*dt >= (timeLimit-30.0)):
                final30 += 1
        if((x+1)*dt % timeBin == 0.0):
            rateBin.append(spikesBin/timeBin)
            spikesBin = 0
        voltage.append(V)

    PostFiringRate = PostSpikes/timeLimit
    xaxis=[]
    for x in range(0,timeLoop):
        voltage[x]*=1000
        xaxis.append((x+1)*dt)

    if(isPlot):
        gbarMean = 0
        for i in range(0,N):
            g_bar[i] = g_bar[i] / nS
            gbarMean += g_bar[i]
        gbarMean /= N
        print("   >>> Mean of steady-state Synaptic Strength after one run (~2.03 nS) = %.2f nS"%gbarMean)

        plt.hist( g_bar, 8, range=(0,4), label= "Mean = %.2f nS" %gbarMean)
        plt.ylabel('frequency')
        plt.xlabel('Steady-state Synaptic Weights (nS)')
        plt.legend()
        plt.title('Q2) Steady-state Synaptic Weights after one run' )
        plt.savefig('Q2)Histogram.png')
        plt.show()

        plt.plot( numpy.arange(1, timeLimit, 10), rateBin, label="post-synaptic neuron")
        plt.ylabel('Average Firing Rate (Hz)')
        plt.xlabel('Time (sec)')
        plt.legend(loc=1)
        plt.title('Q2) Average Firing Rate vs time  (10sec time bins)')
        plt.savefig('Q2)FiringRate')
        plt.show()


    if(not isPlot):
        print("   >>> Average Firing Rate (last 30sec) =",final30/30.0,"Hz")
    print()
##################################################################################################################
def simulation3(STDP, r):

    if(False):#(STDP and (r==10 or r==20)):
        print("STDP on")
        print("input firing rate %dHz"%r)
        print("   >>> Histogram of Steady-state Synaptic Weights")

    Vreset= -65*mV
    Vrest = -65*mV
    Vth   = -50*mV
    V     = -65*mV
    El    = -65*mV
    Rm = 100*MOhm
    Ie = 0*nS
    Tm = 10*ms
    Ts = 2*ms
    Es = 0.0*mV
    deltaS = 0.5
    #r = 15

    dt = 0.25*ms
    timeLimit = 300*sec
    timeLoop = int(timeLimit/dt)
    voltage=[]

    N = 40
    g_bar = [4*nS]*N
    s = [0.0]*N

    Tpre = [0.0*sec]*N
    Tpost = -1000.0*sec
    delta_T = 0.0
    isPotent = False

    PostFiringRate = 0
    PostSpikes = 0
    PreSpikes = 0
    final30 = 0

    for x in range(0,timeLoop):
        sum_s = 0.0
        for i in range(0,N):
            s[i] -= (s[i]*dt/Ts)
            if(isPotent==True and  Tpost!=Tpre[i]):
                g_bar[i] = plasticity( STDP, g_bar[i], Tpost - Tpre[i])         #POTENTIATION
            if(rnd.random() < (r*dt)):
                PreSpikes +=1
                s[i] += deltaS - (s[i]*dt/Ts)
                Tpre[i] = (x+1)*dt                                              #Tpre update spike time
                if(Tpost>0):
                    g_bar[i] = plasticity( STDP, g_bar[i], Tpost - Tpre[i])     #DEPRESSION
            sum_s += s[i]*g_bar[i]
        Is = (Es-V) * sum_s
        dv = (El - V + (Rm*Is))*(dt/Tm)
        V += dv
        isPotent = False
        if( V > Vth ):
            V = Vrest
            PostSpikes +=1
            if((x+1)*dt >= (timeLimit-30.0)):
                final30 += 1
            isPotent = True
            Tpost = (x+1)*dt                                                    #Tpost update spike time
        voltage.append(V)

    PostFiringRate = PostSpikes/timeLimit
    final30 /= 30.0

    for x in range(0,timeLoop):
        voltage[x]*=1000

    if(STDP and (r==10 or r==20)):
        gbarMean = 0
        for i in range(0,N):
            g_bar[i] = g_bar[i] / nS
            gbarMean += g_bar[i]
        gbarMean /= N

        plt.hist( g_bar, 8, range=(0,4), label= "Mean = %.2f nS" %gbarMean)
        plt.ylabel('frequency')
        plt.xlabel('Steady-state Synaptic Weights (nS)')
        plt.legend()
        plt.title('Q3) Steady-state Synaptic Weights when (r = %dHz)' %r)
        plt.savefig('Q3)Hist_R%d.png'%r)
        plt.show()

    #print("   >>> Average Firing Rate (last 30sec) =",final30,"Hz\n")
    return final30
##################################################################################################################
def simulation4(B):
    #print("B is %dHz"%B)

    Vreset= -65*mV
    Vrest = -65*mV
    Vth   = -50*mV
    V     = -65*mV
    El    = -65*mV
    Rm = 100*MOhm
    Ie = 0*nS
    Tm = 10*ms
    Ts = 2*ms
    Es = 0.0*mV
    deltaS = 0.5
    r = 0.0

    dt = 0.25*ms
    timeLimit = 300*sec
    timeLoop = int(timeLimit/dt)
    voltage=[]

    N = 40
    g_bar = [4*nS]*N
    s = [0.0]*N
    STDP=True

    Tpre = [0.0*sec]*N
    Tpost = -1000.0*sec
    delta_T = 0.0
    isPotent = False
    Time = 0.0

    r0=20
    f=10
    for x in range(0,timeLoop):
        Time = (x+1)*dt
        r = r0 + ( B * math.sin(2*math.pi*f*Time) )

        sum_s = 0.0
        for i in range(0,N):
            s[i] -= (s[i]*dt/Ts)
            if(isPotent==True and  Tpost!=Tpre[i]):
                g_bar[i] = plasticity( STDP, g_bar[i], Tpost - Tpre[i])         #POTENTIATION
            if(rnd.random() < (r*dt)):
                s[i] += deltaS - (s[i]*dt/Ts)
                Tpre[i] = Time                                                      #Tpre update spike time
                if(Tpost>0):
                    g_bar[i] = plasticity( STDP, g_bar[i], Tpost - Tpre[i])     #DEPRESSION
            sum_s += s[i]*g_bar[i]
        Is = (Es-V) * sum_s
        dv = (El - V + (Rm*Is))*(dt/Tm)
        V += dv
        isPotent = False
        if( V > Vth ):
            V = Vrest
            isPotent = True
            Tpost = Time                                                            #Tpost update spike time
        voltage.append(V)

    gbarMean = 0.0
    SD = 0.0
    for i in range(0,N):
        g_bar[i] /= nS
        gbarMean += g_bar[i]
    gbarMean /= N
    for i in range(0,N):
        SD += pow(g_bar[i]-gbarMean,2)
    SD = math.sqrt(SD/N)

    if(B==0 or B==20):
        plt.hist( g_bar, 8, range=(0,4), label= 'Mean = %.2f nS' %gbarMean)
        plt.ylabel('frequency')
        plt.xlabel('Steady-state Synaptic Weights (nS)')
        plt.legend()
        plt.title('Q4) Steady-state Synaptic Weights when (B = %dHz)'%B )
        plt.savefig('Q4)Hist_B%d.png'%B)
        plt.show()

    #print("   >>> Mean synaptic strength = %.2f"%gbarMean)
    #print("   >>> SD   synaptic strength = %.2f"%SD)
    #print("\n---------------------------------------------------------------------------")
    return gbarMean,SD
##################################################################################################################
def plasticity( STDP, g_bar, delta_T):
    if(STDP==False):
        return g_bar
    A_n = 0.25*nS
    A_p = 0.2*nS
    T_n = 20*ms
    T_p = 20*ms
    if(delta_T > 0.0):
        deltaf =  A_p * numpy.exp(-abs(delta_T)/T_p)
    if(delta_T <= 0.0):
        deltaf = -A_n * numpy.exp(-abs(delta_T)/T_n)
    return cap(g_bar + deltaf)

def cap(g_bar):
    lower = 0.0
    upper = 4.0*nS
    if(g_bar > upper):
        g_bar = upper
    if(g_bar < lower):
        g_bar = lower
    return g_bar
##################################################################################################################
def question2():
    print("\n--------------------------------QUESTION 2--------------------------------")
    simulation2( True,  4.0, True)      #Mean gbar = 2.03 nS
    simulation2( True,  2.06, False)    #rate(last 30sec) = 0.033Hz
    simulation2( False, 2.06, False)    #rate(last 30sec) = 0.0Hz

def question3():
    print("\n--------------------------------QUESTION 3--------------------------------")
    onRate=[]
    offRate=[]
    inpRate=[]
    for i in range(10,21):
        inpRate.append( i )
        onRate.append ( simulation3( True , i) )
        offRate.append( simulation3( False, i) )
    plt.plot(inpRate,  onRate, label="STDP on")
    plt.plot(inpRate, offRate, label="STDP off")
    plt.ylabel('Steady-State Output Firing Rate (Hz)')
    plt.xlabel('Input Firing Rate (Hz)')
    plt.legend()
    plt.title('Q3) Steady-State Output Firing Rate vs Input Firing Rate')
    plt.savefig('Q3)FiringRate.png')
    plt.show()

def question4():
    print("\n--------------------------------QUESTION 4--------------------------------")

    Mean,SD,B   =   [0.0]*5, [0.0]*5, [0,5,10,15,20]
    for i in range(0,5):
        Mean[i],SD[i]= simulation4(B[i])

    print("\nB:")
    print(['%d Hz' % i for i in B])
    print("\nMean:")
    print(['%0.2f Hz' % i for i in Mean])#range: 1.5Hz - 4.0Hz
    print("\nSD:")
    print(['%0.2f Hz' % i for i in SD])#range: 1.5Hz - 0.0Hz

    plt.plot(B, Mean, label="Mean")
    plt.ylabel('Mean of steady-state Synaptic Strengths (nS)')
    plt.xlabel('B (Hz)')
    plt.legend()
    plt.title('Q4) Mean of steady-state Synaptic Strengths vs B values')
    plt.savefig('Q4)Mean.png')
    plt.show()

    plt.plot(B, SD,color='red', label="Standard Deviation")
    plt.ylabel('SD of steady-state Synaptic Strengths (nS)')
    plt.xlabel('B (Hz)')
    plt.legend()
    plt.title('Q4) SD of steady-state Synaptic Strengths vs B values')
    plt.savefig('Q4)SD.png')
    plt.show()

    if(False):
        plt.plot(B, Mean, label="Mean")
        plt.plot(B, SD,color='red', label="Standard Deviation")
        plt.ylabel('Steady-state Synaptic Strengths (nS)')
        plt.xlabel('B (Hz)')
        plt.legend()
        plt.title('Q4) Mean & SD of steady-state Synaptic Strengths vs B values')
        plt.savefig('Q4)Mean_SD.png')
        plt.show()

##################################################################################################################

def runA():
    print("\n--------------------------------- PART A ---------------------------------")
    #QUESTION 1
    IFNeuron()
    #QUESTION 2
    V1,V2 = rnd.randrange(-80, -54, 1), rnd.randrange(-80, -54, 1)
    #V1,V2 = -66, -76
    print("V1 = %d mV \nV2 = %d mV" %(V1,V2))
    IFNeuronSynapse( V1, V2, 0)
    IFNeuronSynapse( V1, V2, -80)

def runB():
    print("\n--------------------------------- PART B ---------------------------------")
    question1()
    question2()
    question3()
    question4()


runA()
runB()




##################################################################################################################
