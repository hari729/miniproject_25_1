# A_max[l] = a_m1[l]
    # c1 = ((v_tg**2)-(v_st**2))/(2*A_max[l])
    # c2 = (v_max[l]**2)/A_max[l]

    # if path_length[l] <= c1:
    #     a_m1[l] = A_max[l]
    #     a_m3[l] = 0
    #     v_m[l] = math.sqrt((v_st**2)+2*A_max[l]*path_length[l])
    #     v_en[l] = v_m[l]
    # elif path_length[l] >= c2-c1:
    #     a_m1[l] = A_max[l]
    #     a_m3[l] = -A_max[l]
    #     v_m[l] = v_max[l]
    #     v_en[l] = v_tg
    # else:
    #     a_m1[l] = A_max[l]
    #     a_m3[l] = -A_max[l]
    #     v_m[l] = math.sqrt(A_max[l]*path_length[l] + (v_st**2 + v_tg**2)/2)
    #     v_en[l] = v_tg

    # tp1 = 2*a_m3[l]
    # tp2 = 2*T_m*a_m3[l]**2
    # tp3 = -2*path_length[l]*a_m3[l]**2
    # coeff = [tp1, tp2, tp3]
    # v_roots = np.roots(coeff) 
    # print(v_roots)
    #v_m[l] = min(v_roots)


    # if T_m2[l] < 0:
    #     v_m[l] = math.sqrt(path_length[l]*2*a_m1[l]*a_m3[l]/(a_m3[l]-a_m1[l]))
    #     if v_m[l] > A_max[l]*T_m/2:
    #             v_m[l] = A_max[l]*T_m/2
    #     elif v_m[l] < v_st:
    #             v_m[l] = v_st
    
    #     T_m1[l] = (v_m[l] - v_st)/a_m1[l]
    #     T_m3[l] = (v_en[l] - v_m[l])/a_m3[l]
    #     D_m1[l] = ((v_m[l]**2) - (v_st**2))/(2*a_m1[l])
    #     D_m3[l] = ((v_en[l]**2) - (v_m[l]**2))/(2*a_m3[l])
    #     T_m2[l] = 0

    # v_roots = np.roots([1,-T_m*a_m3[l],path_length[l]*a_m3[l]])
    # v_m[l] = abs(min(v_roots))
    # print(v_roots)

    if T_m >= (v_st+v_tg)/A_max[l] and a_m1[l] == -A_max[l]:
        if a_m3[l] == A_max[l]:
            v_m[l] = 0
            v_en[l] = A_max[l]*T_m
            print("c1")
        elif v_en[l] == v_tg:
            if v_m[l] == 0:
                a_m3[l] = v_en[l]/T_m3[l]
                print("c2")
            elif T_m2[l] == 0:
                a_m3[l] = (v_en[l]-v_m[l])*A_max[l]/(A_max[l]*T_m - v_st + v_m[l])
                print("c3") 
    elif (v_st+v_tg)/A_max[l] >= T_m >= v_st/A_max[l] and a_m1[l] == -A_max[l]:
        if a_m3[l] == A_max[l] and v_m[l] == 0:
            v_en[l] = A_max[l]*T_m
            print("c4")
    elif (v_st+v_tg)/A_max[l] >= T_m >= (v_tg-v_st)/A_max[l] and T_m >= v_st/A_max[l] and a_m1[l] == -A_max[l]:
        if a_m3[l] == A_max[l] and T_m2[l] == 0:
            v_en[l] = A_max[l]*T_m + 2*v_m[l] - v_st
            print("c5")
        elif a_m3[l] == A_max[l] and v_en[l] == v_tg:
            a_m3[l] = (v_en[l]-v_m[l])*A_max[l]/(A_max[l]*T_m - v_st + v_m[l])
            print("c6")
    elif (v_tg-v_st)/A_max[l] >= T_m >= v_st/A_max[l] and a_m1[l] == -A_max[l]:
        if a_m3[l] == A_max[l] and T_m2[l] == 0:
            v_en[l] = A_max[l]*T_m + 2*v_m[l] - v_st
            print("c7")
        elif a_me[l] == A_max[l]:
            v_m[l] = (v_st+v_en[l]-A_max[l]*T_m)/2
            print("c8")
        elif v_en[l] == v_tg:
            a_m3[l] = (v_tg-v_m[l])*A_max[l]/(A_max[l]*T_m - v_st + v_m[l])
            print("c9")
    elif (2*v_max[l]-v_st-v_tg)/A_max[l] <= T_m and a_m1[l] == A_max[l]:
        print(l)
        if v_en[l] == v_tg and T[l,1] == 0:
            a_m3[l] = (v_tg-v_m[l])*A_max[l]/(A_max[l]*T_m - v_m[l] + v_st)
            print("c10")
        elif v_m[l] == v_max[l] and v_en[l] == v_tg:
            a_m3[l] = (v_max[l]-v_tg)/T_m3[l]
            print("c11")
    elif (2*v_max[l]-v_st-v_tg)/A_max[l] >= T_m >= (v_tg-v_st)/A_max[l] and a_m1[l] == A_max[l] and v_en[l] == v_tg and v_st <= v_m[l] <= A_max[l]*T_m/2 :
            a_m3[l] = (v_tg-v_m[l])*A_max[l]/(A_max[l]*T_m - v_m[l] + v_st)
            print("c12")
    else:
        print(l,"None")
    
    