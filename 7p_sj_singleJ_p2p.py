import numpy as np
import math
import matplotlib.pyplot as plt

d0 = 0
t0 = 0
a0 = 0

df = 85
v_max = 100
v_tg = 0
v_st = 0
alpha = (1, 0.75, 0.5, 0.25)
acceleration = np.empty([4,25000])
velocity = np.empty([4,25000])
position = np.empty([4,25000])
jerk = np.empty([4,25000])
t_i = np.zeros((4,7))
T = np.zeros((7,))
t_a = np.linspace(0,2.5,num=25000)
a = np.zeros((25000,))
v = np.zeros((25000,))
d = np.zeros((25000,))
j = np.zeros((25000,))

for f in range(len(alpha)):

    a_max = (1 - 0.5*alpha[f])*150


    D_m = df - d0
    c1 = ((v_tg**2)-(v_st**2))/(2*a_max)
    c2 = (v_max**2)/a_max

    if D_m <= c1:
        a_m1 = a_max
        a_m3 = 0
        v_m = math.sqrt((v_st**2)+2*a_max*D_m)
        v_en = v_m
    elif D_m >= c2-c1:
        a_m1 = a_max
        a_m3 = -a_max
        v_m = v_max
        v_en = v_tg
    else:
        a_m1 = a_max
        a_m3 = -a_max
        v_m = math.sqrt(a_max*D_m + (v_st**2 + v_tg**2)/2)
        v_en = v_tg


    T_m1 = (v_m - v_st)/a_m1
    T_m3 = (v_en - v_m)/a_m3
    D_m1 = ((v_m**2) - (v_st**2))/(2*a_m1)
    D_m3 = ((v_en**2) - (v_m**2))/(2*a_m3)
    T_m2 = (D_m - D_m1 - D_m3)/v_m

    T[0] = 0.5*alpha[f]*T_m1
    T[1] = (1-alpha[f])*T_m1
    T[2] = T[0]
    T[3] = T_m2
    T[4] = 0.5*alpha[f]*T_m3
    T[5] = (1-alpha[f])*T_m3
    T[6] = T[4]

    T_m = T[0]+T[1]+T[2]+T[3]+T[4]+T[5]+T[6]

    J_1p = (np.pi*a_m1)/(alpha[f]*(1-(0.5*alpha[f]))*T_m1)
    J_5p = -(np.pi*a_m3)/(alpha[f]*(1-(0.5*alpha[f]))*T_m3)
    
    temp = 0
    for l in range(7):
        temp = temp + T[l]
        t_i[f,l] = temp

    J1_c = (J_1p*T[0]/np.pi)
    J5_c = (J_5p*T[4]/np.pi)

    for i in range(len(t_a)):
        if t_a[i]>=0 and t_a[i]<t_i[f,0]:
            tau = t_a[i]-0
            a[i]=J1_c*(1-np.cos(np.pi*(tau/T[0])))
            v[i]=v_st+J1_c*(tau-(T[0]*np.sin(np.pi*tau/T[0])/np.pi))
            d[i]=v_st*tau+0.5*J1_c*(tau**2)+(J1_c*(T[0]**2)/(np.pi**2))*(np.cos(np.pi*tau/T[0])-1)
            j[i]=J_1p*np.sin(np.pi*tau/T[0])
        elif t_a[i]>=t_i[f,0] and t_a[i]<t_i[f,1]:
            tau = t_a[i]-t_i[f,0]
            a[i]=2*J1_c
            v[i]=v_st+J1_c*(2*tau+T[0])
            d[i]=v_st*T[0]+(J1_c*T[0]**2)*(0.5-(2/(np.pi**2)))+(v_st+J1_c*T[0])*tau+J1_c*tau**2
            j[i]=0
        elif t_a[i]>=t_i[f,1] and t_a[i]<t_i[f,2]:
            tau = t_a[i]-t_i[f,1]
            a[i]=J1_c*(np.cos(np.pi*tau/T[0])+1)
            v[i]=v_st+J1_c*(T[0]+2*T[1]+tau+T[0]*np.sin(np.pi*tau/T[0])/np.pi)
            d[i]=v_st*(T[0]+T[1])+J1_c*((0.5-(1/(np.pi**2)))*(T[0]**2)+T[0]*T[1]+T[1]**2)+(v_st+J1_c*(T[0]+2*T[1]))*tau+0.5*J1_c*(tau**2)-((J1_c*T[0]**2)/(np.pi**2))*np.cos(np.pi*tau/T[0])
            j[i]=-J_1p*np.sin(np.pi*tau/T[0])
        elif t_a[i]>=t_i[f,2] and t_a[i]<t_i[f,3]:
            tau = t_a[i]-t_i[f,2]
            a[i]=0
            v[i]=v_st+2*J1_c*(T[0]+T[1])
            d[i]=v_st*(2*T[0]+T[1])+J1_c*(2*T[0]**2+3*T[0]*T[1]+T[1]**2)+(v_st+2*J1_c*(T[0]+T[1]))*tau
            j[i]=0
        elif t_a[i]>=t_i[f,3] and t_a[i]<t_i[f,4]:
            tau = t_a[i]-t_i[f,3]
            a[i]=J5_c*(np.cos(np.pi*tau/T[4])-1)
            v[i]=v_st+2*J5_c*(T[0]+T[1])+J5_c*(T[4]*(np.sin(np.pi*tau/T[4])/np.pi)-tau)
            d[i]=v_st*(2*T[0]+T[1]+T[3])+tau*(v_st+2*J1_c*(T[0]+T[1]))+J1_c*((2*T[0]**2)+3*T[0]*T[1]+(T[1]**2)+2*T[0]*T[3]+2*T[1]*T[3])-0.5*J5_c*(tau**2)+(J5_c*(T[4]**2)/(np.pi**2))*(1-np.cos(np.pi*tau/T[4]))
            j[i]=-J_5p*np.sin(np.pi*tau/T[4])
        elif t_a[i]>=t_i[f,4] and t_a[i]<t_i[f,5]:
            tau = t_a[i]-t_i[f,4]
            a[i]=-2*J5_c
            v[i]=v_st+2*J5_c*(T[0]+T[1])-J5_c*(2*tau+T[4])
            d[i]=v_st*(2*T[0]+T[1]+T[3]+T[4])+(J5_c*(T[4]**2))*(-0.5+(2/np.pi**2))+J1_c*(T[0]+T[1])*(2*T[0]+T[1]+2*T[3]+2*T[4])+tau*(v_st+2*J1_c*(T[0]+T[1])-(J5_c*T[4]))-J5_c*tau**2
            j[i]=0
        elif t_a[i]>=t_i[f,5] and t_a[i]<=t_i[f,6]:
            tau = t_a[i]-t_i[f,5]
            a[i]=-J5_c*(1+np.cos(np.pi*tau/T[4]))
            v[i]=v_st+2*J1_c*(T[0]+T[1])-J5_c*(2*T[5]+T[4])-J5_c*(tau+T[4]*np.sin(np.pi*tau/T[4])/np.pi)
            d[i]=v_st*(2*T[0]+T[1]+T[3]+T[4]+T[5])-0.5*(J5_c*tau**2)+J5_c*((-0.5+(1/np.pi**2))*(T[4]**2)-T[4]*T[5]-(T[5]**2))+(J5_c*(T[4]**2)/(np.pi**2))*np.cos(np.pi*tau/T[4])+J1_c*(T[0]+T[1])*(2*T[0]+T[1]+2*T[3]+2*T[4]+2*T[5])+tau*(v_st+2*J1_c*(T[0]+T[1])-J5_c*(2*T[5]+T[4]))
            j[i]=J_5p*np.sin(np.pi*tau/T[4])
        else:
            a[i] = math.nan 
            v[i] = math.nan
            d[i] = math.nan
            j[i] = math.nan
    
    acceleration[f,:] = a
    velocity[f,:] = v
    position[f,:] = d
    jerk[f,:] = j

lg = [r'$\alpha$ = 1',r'$\alpha$ = 0.75',r'$\alpha$ = 0.5',r'$\alpha$ = 0']

fig, single_j = plt.subplots(2,2)
single_j[1,0].plot(t_a, acceleration[0], color='b', linestyle = 'solid')
single_j[1,0].plot(t_a, acceleration[1], color='g', linestyle = 'dashed')
single_j[1,0].plot(t_a, acceleration[2], color='r', linestyle = 'dashdot')
single_j[1,0].plot(t_a, acceleration[3], color='y', linestyle = 'dotted')
single_j[1,0].axvline(x = t_i[0,6], color = 'b', linestyle = 'dotted')
single_j[1,0].axvline(x = t_i[1,6], color = 'g', linestyle = 'dotted')
single_j[1,0].axvline(x = t_i[2,6], color = 'r', linestyle = 'dotted')
single_j[1,0].axvline(x = t_i[3,6], color = 'y', linestyle = 'dotted')
single_j[1,0].set_title("Acceleration vs Time")
single_j[1,0].set_xlabel("Time (s)")
single_j[1,0].set_ylabel("Acceleration" r'$(deg/s^2)$')
single_j[1,0].legend(labels=lg)

single_j[0,1].plot(t_a, velocity[0], color='b', linestyle = 'solid')
single_j[0,1].plot(t_a, velocity[1], color='g', linestyle = 'dashed')
single_j[0,1].plot(t_a, velocity[2], color='r', linestyle = 'dashdot')
single_j[0,1].plot(t_a, velocity[3], color='y', linestyle = 'dotted')
single_j[0,1].axvline(x = t_i[0,6], color = 'b', linestyle = 'dotted')
single_j[0,1].axvline(x = t_i[1,6], color = 'g', linestyle = 'dotted')
single_j[0,1].axvline(x = t_i[2,6], color = 'r', linestyle = 'dotted')
single_j[0,1].axvline(x = t_i[3,6], color = 'y', linestyle = 'dotted')
single_j[0,1].set_title("Velocity vs Time")
single_j[0,1].set_xlabel("Time (s)")
single_j[0,1].set_ylabel("Velocity" r'$(deg/s)$')
single_j[0,1].legend(labels=lg)

single_j[0,0].plot(t_a, position[0], color='b', linestyle = 'solid')
single_j[0,0].plot(t_a, position[1], color='g', linestyle = 'dashed')
single_j[0,0].plot(t_a, position[2], color='r', linestyle = 'dashdot')
single_j[0,0].plot(t_a, position[3], color='y', linestyle = 'dotted')
single_j[0,0].axvline(x = t_i[0,6], color = 'b', linestyle = 'dotted')
single_j[0,0].axvline(x = t_i[1,6], color = 'g', linestyle = 'dotted')
single_j[0,0].axvline(x = t_i[2,6], color = 'r', linestyle = 'dotted')
single_j[0,0].axvline(x = t_i[3,6], color = 'y', linestyle = 'dotted')
single_j[0,0].set_title("Position vs Time")
single_j[0,0].set_xlabel("Time (s)")
single_j[0,0].set_ylabel("Position" r'$(deg)$')
single_j[0,0].legend(labels=lg)

single_j[1,1].plot(t_a, jerk[0], color='b', linestyle = 'solid')
single_j[1,1].plot(t_a, jerk[1], color='g', linestyle = 'dashed')
single_j[1,1].plot(t_a, jerk[2], color='r', linestyle = 'dashdot')
single_j[1,1].plot(t_a, jerk[3], color='y', linestyle = 'dotted')
single_j[1,1].axvline(x = t_i[0,6], color = 'b', linestyle = 'dotted')
single_j[1,1].axvline(x = t_i[1,6], color = 'g', linestyle = 'dotted')
single_j[1,1].axvline(x = t_i[2,6], color = 'r', linestyle = 'dotted')
single_j[1,1].axvline(x = t_i[3,6], color = 'y', linestyle = 'dotted')
single_j[1,1].set_title("Jerk vs Time")
single_j[1,1].set_xlabel("Time (s)")
single_j[1,1].set_ylabel("Jerk" r'$(deg/s^3)$')
single_j[1,1].legend(labels=lg)

plt.show()