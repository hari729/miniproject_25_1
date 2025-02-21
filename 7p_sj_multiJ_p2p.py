import numpy as np
import math
import matplotlib.pyplot as plt
#import pinocchio as pin

plt.close('all')

initial_pos = np.array([-10, 20, 15, 150, 30, 120])
final_pos = np.array([55, 35, 30, 10, 70, 25])
v_max = np.array([100, 95, 100, 150, 130, 110])
a_max = np.array([60, 60, 75, 70, 90, 80])

v_st = 0
v_tg = 0
alpha = 0.5

no_joints = len(initial_pos)
path_length = abs(final_pos - initial_pos)
theta = final_pos - initial_pos
A_max = np.empty([no_joints],dtype=np.float16)
v_m = np.empty([no_joints],dtype=np.float16)
v_en = np.empty([no_joints],dtype=np.float16)
a_m1 = np.empty([no_joints],dtype=np.float16)
a_m3 = np.empty([no_joints],dtype=np.float16)
T_m1 = np.empty([no_joints],dtype=np.float16)
T_m2 = np.empty([no_joints],dtype=np.float16)
T_m3 = np.empty([no_joints],dtype=np.float16)
T_j = np.empty([no_joints],dtype=np.float16)
T = np.empty([no_joints,7],dtype=np.float16)
D_m1 = np.empty([no_joints],dtype=np.float16)
D_m3 = np.empty([no_joints],dtype=np.float16)

acceleration = np.empty([6,4500],dtype=np.float16)
velocity = np.empty([6,4500],dtype=np.float16)
position = np.empty([6,4500],dtype=np.float16)
jerk = np.empty([6,4500],dtype=np.float16)
t_i = np.zeros((6,7),dtype=np.float16)
t_a = np.linspace(0,4.5,num=4500,dtype=np.float16)
a = np.zeros((4500,),dtype=np.float16)
v = np.zeros((4500,),dtype=np.float16)
d = np.zeros((4500,),dtype=np.float16)
j = np.zeros((4500,),dtype=np.float16)
J1_c = np.empty([6],dtype=np.float16)
J5_c = np.empty([6],dtype=np.float16)
J_1p = np.empty([6],dtype=np.float16)
J_5p = np.empty([6],dtype=np.float16)
direction = np.empty([no_joints],dtype=np.float16)

torque = np.empty([6,4500],dtype=np.float16)

for k in range(no_joints):
    direction[k] = path_length[k]/theta[k]

for i in range(no_joints):

    A_max[i] = (1 - 0.5*alpha)*a_max[i]
    c1 = ((v_tg**2)-(v_st**2))/(2*A_max[i])
    c2 = (v_max[i]**2)/A_max[i]

    if path_length[i] <= c1:
        a_m1[i] = A_max[i]
        a_m3[i] = 0
        v_m[i] = math.sqrt((v_st**2)+2*A_max[i]*path_length[i])
        v_en[i] = v_m[i]
    elif path_length[i] >= c2-c1:
        a_m1[i] = A_max[i]
        a_m3[i] = -A_max[i]
        v_m[i] = v_max[i]
        v_en[i] = v_tg
    else:
        a_m1[i] = A_max[i]
        a_m3[i] = -A_max[i]
        v_m[i] = math.sqrt(A_max[i]*path_length[i] + (v_st**2 + v_tg**2)/2)
        v_en[i] = v_tg

    T_m1[i] = (v_m[i] - v_st)/a_m1[i]
    T_m3[i] = (v_en[i] - v_m[i])/a_m3[i]
    D_m1[i] = ((v_m[i]**2) - (v_st**2))/(2*a_m1[i])
    D_m3[i] = ((v_en[i]**2) - (v_m[i]**2))/(2*a_m3[i])
    T_m2[i] = (path_length[i] - D_m1[i] - D_m3[i])/v_m[i]

    T[i,0] = 0.5*alpha*T_m1[i]
    T[i,1] = (1-alpha)*T_m1[i]
    T[i,2] = T[i,0]
    T[i,3] = T_m2[i]
    T[i,4] = 0.5*alpha*T_m3[i]
    T[i,5] = (1-alpha)*T_m3[i]
    T[i,6] = T[i,4]

    T_j[i] = T[i,0]+T[i,1]+T[i,2]+T[i,3]+T[i,4]+T[i,5]+T[i,6]

T_m = np.max(T_j)

slowest_joint = np.argmax(T_j)

print("Execution Time = ",T_m)
print("Slowest Joint = ",slowest_joint+1)



for l in range(no_joints):

    a_m3[l] = -path_length[l]/(T_m3[slowest_joint]*(T_m - T_m3[slowest_joint]))
    a_m1[l] = -a_m3[l]
    A_max[l] = a_m1[l]

    T_m1[l] = T_m1[slowest_joint]
    T_m3[l] = T_m3[slowest_joint]
    T_m2[l] = T_m2[slowest_joint]

    T[l,0] = 0.5*alpha*T_m1[l]
    T[l,1] = (1-alpha)*T_m1[l]
    T[l,2] = T[l,0]
    T[l,3] = T_m2[l]
    T[l,4] = 0.5*alpha*T_m3[l]
    T[l,5] = (1-alpha)*T_m3[l]
    T[l,6] = T[l,4]
  
    T_j[l] = T[l,0]+T[l,1]+T[l,2]+T[l,3]+T[l,4]+T[l,5]+T[l,6]

    temp = 0
    for o in range(7):
        temp = temp + T[l,o]
        t_i[l,o] = temp

    J_1p[l] = (np.pi*a_m1[l])/(alpha*(1-(0.5*alpha))*T_m1[l])
    J_5p[l] = -(np.pi*a_m3[l])/(alpha*(1-(0.5*alpha))*T_m3[l])

    J1_c[l] = (J_1p[l]*T[l,0]/np.pi)
    J5_c[l] = (J_5p[l]*T[l,4]/np.pi)


    for k in range(len(t_a)):
            if t_a[k]>=0 and t_a[k]<t_i[l,0]:
                tau = t_a[k]-0
                a[k]=J1_c[l]*(1-np.cos(np.pi*(tau/T[l,0])))
                v[k]=v_st+J1_c[l]*(tau-(T[l,0]*np.sin(np.pi*tau/T[l,0])/np.pi))
                d[k]=initial_pos[l] + direction[l]*(v_st*tau+0.5*J1_c[l]*(tau**2)+(J1_c[l]*(T[l,0]**2)/(np.pi**2))*(np.cos(np.pi*tau/T[l,0])-1))
                j[k]=direction[l]*J_1p[l]*np.sin(np.pi*tau/T[l,0])
            elif t_a[k]>=t_i[l,0] and t_a[k]<t_i[l,1]:
                tau = t_a[k]-t_i[l,0]
                a[k]=2*J1_c[l]
                v[k]=v_st+J1_c[l]*(2*tau+T[l,0])
                d[k]=initial_pos[l] + direction[l]*(v_st*T[l,0]+(J1_c[l]*T[l,0]**2)*(0.5-(2/(np.pi**2)))+(v_st+J1_c[l]*T[l,0])*tau+J1_c[l]*tau**2)
                j[k]=0
            elif t_a[k]>=t_i[l,1] and t_a[k]<t_i[l,2]:
                tau = t_a[k]-t_i[l,1]
                a[k]=J1_c[l]*(np.cos(np.pi*tau/T[l,0])+1)
                v[k]=v_st+J1_c[l]*(T[l,0]+2*T[l,1]+tau+T[l,0]*np.sin(np.pi*tau/T[l,0])/np.pi)
                d[k]=initial_pos[l] + direction[l]*(v_st*(T[l,0]+T[l,1])+J1_c[l]*((0.5-(1/(np.pi**2)))*(T[l,0]**2)+T[l,0]*T[l,1]+T[l,1]**2)+(v_st+J1_c[l]*(T[l,0]+2*T[l,1]))*tau+0.5*J1_c[l]*(tau**2)-((J1_c[l]*T[l,0]**2)/(np.pi**2))*np.cos(np.pi*tau/T[l,0]))
                j[k]=direction[l]*-J_1p[l]*np.sin(np.pi*tau/T[l,0])
            elif t_a[k]>=t_i[l,2] and t_a[k]<t_i[l,3]:
                tau = t_a[k]-t_i[l,2]
                a[k]=0
                v[k]=v_st+2*J1_c[l]*(T[l,0]+T[l,1])
                d[k]=initial_pos[l] + direction[l]*(v_st*(2*T[l,0]+T[l,1])+J1_c[l]*(2*T[l,0]**2+3*T[l,0]*T[l,1]+T[l,1]**2)+(v_st+2*J1_c[l]*(T[l,0]+T[l,1]))*tau)
                j[k]=0
            elif t_a[k]>=t_i[l,3] and t_a[k]<t_i[l,4]:
                tau = t_a[k]-t_i[l,3]
                a[k]=J5_c[l]*(np.cos(np.pi*tau/T[l,4])-1)
                v[k]=v_st+2*J5_c[l]*(T[l,0]+T[l,1])+J5_c[l]*(T[l,4]*(np.sin(np.pi*tau/T[l,4])/np.pi)-tau)
                d[k]=initial_pos[l] + direction[l]*(v_st*(2*T[l,0]+T[l,1]+T[l,3])+tau*(v_st+2*J1_c[l]*(T[l,0]+T[l,1]))+J1_c[l]*((2*T[l,0]**2)+3*T[l,0]*T[l,1]+(T[l,1]**2)+2*T[l,0]*T[l,3]+2*T[l,1]*T[l,3])-0.5*J5_c[l]*(tau**2)+(J5_c[l]*(T[l,4]**2)/(np.pi**2))*(1-np.cos(np.pi*tau/T[l,4])))
                j[k]=direction[l]*-J_5p[l]*np.sin(np.pi*tau/T[l,4])
            elif t_a[k]>=t_i[l,4] and t_a[k]<t_i[l,5]:
                tau = t_a[k]-t_i[l,4]
                a[k]=-2*J5_c[l]
                v[k]=v_st+2*J5_c[l]*(T[l,0]+T[l,1])-J5_c[l]*(2*tau+T[l,4])
                d[k]=initial_pos[l] + direction[l]*(v_st*(2*T[l,0]+T[l,1]+T[l,3]+T[l,4])+(J5_c[l]*(T[l,4]**2))*(-0.5+(2/np.pi**2))+J1_c[l]*(T[l,0]+T[l,1])*(2*T[l,0]+T[l,1]+2*T[l,3]+2*T[l,4])+tau*(v_st+2*J1_c[l]*(T[l,0]+T[l,1])-(J5_c[l]*T[l,4]))-J5_c[l]*tau**2)
                j[k]=0
            elif t_a[k]>=t_i[l,5] and t_a[k]<=t_i[l,6]:
                tau = t_a[k]-t_i[l,5]
                a[k]=-J5_c[l]*(1+np.cos(np.pi*tau/T[l,4]))
                v[k]=v_st+2*J1_c[l]*(T[l,0]+T[l,1])-J5_c[l]*(2*T[l,5]+T[l,4])-J5_c[l]*(tau+T[l,4]*np.sin(np.pi*tau/T[l,4])/np.pi)
                d[k]=initial_pos[l] + direction[l]*(v_st*(2*T[l,0]+T[l,1]+T[l,3]+T[l,4]+T[l,5])-0.5*(J5_c[l]*tau**2)+J5_c[l]*((-0.5+(1/np.pi**2))*(T[l,4]**2)-T[l,4]*T[l,5]-(T[l,5]**2))+(J5_c[l]*(T[l,4]**2)/(np.pi**2))*np.cos(np.pi*tau/T[l,4])+J1_c[l]*(T[l,0]+T[l,1])*(2*T[l,0]+T[l,1]+2*T[l,3]+2*T[l,4]+2*T[l,5])+tau*(v_st+2*J1_c[l]*(T[l,0]+T[l,1])-J5_c[l]*(2*T[l,5]+T[l,4])))
                j[k]=direction[l]*J_5p[l]*np.sin(np.pi*tau/T[l,4])
            else:
                a[k] = math.nan 
                v[k] = math.nan
                d[k] = math.nan
                j[k] = math.nan
        
    acceleration[l,:] = a * (theta[l]/path_length[l])
    velocity[l,:] = v * (theta[l]/path_length[l])
    position[l,:] = d
    jerk[l,:] = j


lg = ["J 1","j 2","j 3","j 4","j 5","j 6"]

# robot = 
# model = pinocchio.buildSampleModelManipulator()
# data = model.createData()

# for e in range (6):
#     for f in range (4500):
#         torque[e,f] = pin.rnea(model, data, position[e,f], velocity[e,f], acceleration[e,f])

fig, multi_j = plt.subplots(2,2)

# multi_j[0,0].set_xlim([0,4])
multi_j[0,0].yaxis.set_ticks(np.arange(-10,160,20))
multi_j[0,0].plot(t_a, position[0], color='b', linestyle = 'solid')
multi_j[0,0].plot(t_a, position[1], color='g', linestyle = 'solid')
multi_j[0,0].plot(t_a, position[2], color='r', linestyle = 'dashed')
multi_j[0,0].plot(t_a, position[3], color='orange', linestyle = 'dashed')
multi_j[0,0].plot(t_a, position[4], color='lime', linestyle = 'dotted')
multi_j[0,0].plot(t_a, position[5], color='c', linestyle = 'dotted')
multi_j[0,0].axvline(x = 0, color = 'grey', linestyle = 'dashdot')

multi_j[0,0].axvline(x = t_i[slowest_joint,6], color = 'orange', linestyle = 'dotted')

multi_j[0,0].set_title("Position vs Time")
multi_j[0,0].set_xlabel("Time (s)")
multi_j[0,0].set_ylabel("Position" r'$(deg)$')
multi_j[0,0].legend(labels=lg)

multi_j[1,0].plot(t_a, acceleration[0], color='b', linestyle = 'solid')
multi_j[1,0].plot(t_a, acceleration[1], color='g', linestyle = 'solid')
multi_j[1,0].plot(t_a, acceleration[2], color='r', linestyle = 'dashed')
multi_j[1,0].plot(t_a, acceleration[3], color='orange', linestyle = 'dashed')
multi_j[1,0].plot(t_a, acceleration[4], color='lime', linestyle = 'dotted')
multi_j[1,0].plot(t_a, acceleration[5], color='c', linestyle = 'dotted')
multi_j[1,0].axvline(x = 0, color = 'grey', linestyle = 'dashdot')
multi_j[1,0].axvline(x = t_i[slowest_joint,6], color = 'orange', linestyle = 'dotted')
multi_j[1,0].set_title("Acceleration vs Time")
multi_j[1,0].set_xlabel("Time (s)")
multi_j[1,0].set_ylabel("Acceleration" r'$(deg/s^2)$')

multi_j[0,1].plot(t_a, velocity[0], color='b', linestyle = 'solid')
multi_j[0,1].plot(t_a, velocity[1], color='g', linestyle = 'solid')
multi_j[0,1].plot(t_a, velocity[2], color='r', linestyle = 'dashed')
multi_j[0,1].plot(t_a, velocity[3], color='orange', linestyle = 'dashed')
multi_j[0,1].plot(t_a, velocity[4], color='lime', linestyle = 'dotted')
multi_j[0,1].plot(t_a, velocity[5], color='c', linestyle = 'dotted')
multi_j[0,1].axvline(x = 0, color = 'grey', linestyle = 'dashdot')
multi_j[0,1].axvline(x = t_i[slowest_joint,6], color = 'orange', linestyle = 'dotted')
multi_j[0,1].set_title("Velocity vs Time")
multi_j[0,1].set_xlabel("Time (s)")
multi_j[0,1].set_ylabel("Velocity" r'$(deg/s)$')

multi_j[1,1].plot(t_a, jerk[0], color='b', linestyle = 'solid')
multi_j[1,1].plot(t_a, jerk[1], color='g', linestyle = 'solid')
multi_j[1,1].plot(t_a, jerk[2], color='r', linestyle = 'dashed')
multi_j[1,1].plot(t_a, jerk[3], color='orange', linestyle = 'dashed')
multi_j[1,1].plot(t_a, jerk[4], color='lime', linestyle = 'dotted')
multi_j[1,1].plot(t_a, jerk[5], color='c', linestyle = 'dotted')
multi_j[1,1].axvline(x = 0, color = 'grey', linestyle = 'dashdot')
multi_j[1,1].axvline(x = t_i[slowest_joint,6], color = 'orange', linestyle = 'dotted')
multi_j[1,1].set_title("Jerk vs Time")
multi_j[1,1].set_xlabel("Time (s)")
multi_j[1,1].set_ylabel("Jerk" r'$(deg/s^3)$')

plt.show()
