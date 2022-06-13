import csv
from gurobipy import GRB
from gurobipy import Model
from gurobipy import quicksum
from gurobipy import min_
import SimulationConfiguration as SimConfig
import pandas as pd

p = {}
sumcharging = {}


class Optimierung():
    def ladeplan(self, lenID):
        # Lesen der Daten
        # Tabelle anpassen für mehrere Fzg, CO2 Emission erneut rechnen, die Daten aus Prognose einfügen
        global p
        global sumcharging
        global E_EV
        global P_max
        global OTM_Target
        global C_Cost
        global C_Emission

        P_max = SimConfig.P_max
        E_EV = SimConfig.E_EV
        OTM_Target = SimConfig.OTM_Target
        C_Cost = SimConfig.C_Cost
        C_Emission = SimConfig.C_Emission

        i = 0
        z = 0
        Z = 0
        # datei hat prognose aus level 2 und 3 als alle
        with open("./prognose/Prognose_final.csv") as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                Z = Z + 1
                if z != 0:
                    # lenID = number of evs? -> 8 für timeindex,
                    for j in range(0, 8 + 2 * (lenID - 1)):
                        # grid, pvpower, gl, sp, co2, chargingev1, soctocharge ev1
                        if row[j] == "":  # todo evtl hier ändern
                            # p liste deren werte um 1 reihe noch unten verschoben sit
                            p[i, j + 1] = 0
                        else:
                            p[i, j + 1] = float(row[j])
                            # var_test = float(row[j])
                            # print(0)
                    i = i + 1
                z = z + 1

        Z = Z - 1
        # print(z)
        # print(Z)
        # print(p)

        model = Model("flow_shop")
        for i in range(1, lenID + 1):
            globals()['x' + str(i)] = {}
            globals()['a' + str(i)] = {}
            globals()['S' + str(i)] = {}
            globals()['E' + str(i)] = {}
            # globals()['pv'+str(i)] = {}
            # globals( )['pv_a' + str(i)] = {}

        # x(i) wie viel geladen wird(kW), a(i) ob geladen wird
        # S: Standzeit, E: zu ladene SOC
        for i in range(Z):
            for j in range(1, lenID + 1):
                exec(
                    'x{}[i] = model.addVar(vtype="C", name="x{}(%s)" % (i))'.format(j, j))
                exec(
                    'a{}[i] = model.addVar(vtype="b", name="a{}(%s)" % (i))'.format(j, j))
                # exec('pv{}[i] = model.addVar(vtype="C", name="pv{}(%s)" % (i))'.format(j, j))
                # exec('pv_a{}[i] = model.addVar(vtype="C", name="pv_a{}(%s)" % (i))'.format(j, j))
            # x1[i] = model.addVar(vtype="C", name="x(%s)" % (i))
            # x2[i] = model.addVar(vtype="C", name="x2(%s)" % (i))
        # for i in range(Z):
        # a1[i]= model.addVar(vtype="b", name="a(%s)" % (i))
        # a2[i]= model.addVar(vtype="b", name="a2(%s)" % (i))
        model.update()

        # CONSTRAINTS
        # mehrere Nebenbedingungen gelten fuer alle EV
        # fuer Mehrdimensinal (mehrere Fzge) erneut anpassen

        for i in range(Z):
            sumcharging[i] = 0
            for j in range(1, lenID + 1):
                # min charge
                exec('model.addConstr((x{}[i]) >= 4.14)'.format(j))
                # max charge
                exec('model.addConstr((x{}[i]) <= P_max)'.format(j))
                # für netzbegrenzung
                exec(
                    'sumcharging[i] = x{}[i]*a{}[i]+sumcharging[i]'.format(j, j))
                # exec('model.addConstr((pv_a{}[i]) == (x{}[i]*a{}[i]))'.format(j, j, j))
                # exec('model.addConstr((pv{}[i]) == min_(pv_a{}[i], p[i,3]))'.format(j,j,j))

            # model.addConstr((x1[i]) >= 1.3)
            # max je nach EV erneut anpassen
            # model.addConstr((x1[i]) <= 11)
            # model.addConstr((x2[i]) >= 1.3)
            # model.addConstr((x2[i]) <= 11)
            # model.addConstr(x1[i]*a1[i]+x2[i]*a2[i]+x3[i]*a3[i]+x4[i]*a4[i] <= p[i,2]+p[i,3]-p[i,4])
            model.addConstr(sumcharging[i] <= p[i, 2] + p[i, 3] - p[i, 4])

        model.update()

        for i1 in range(1, lenID + 1):
            k = 0
            for i in range(Z):
                if p[i, (7 + 2 * (i1 - 1))] == 1:
                    if i == 0:
                        exec('S{}[k]=[i]'.format(i1))
                    else:
                        if p[i - 1, (7 + 2 * (i1 - 1))] == 0:
                            exec('S{}[k]=[i]'.format(i1))
                        else:
                            exec('S{}[k].append(i)'.format(i1))
                if p[i, (8 + 2 * (i1 - 1))] != 0:
                    exec('E{}[k]=p[i,(8+2*(i1-1))]'.format(i1))
                    k = k + 1

                # i1 ev id
                # k anzahl der trips
            for j in range(k):
                exec('model.addConstr(quicksum(quicksum((1/4)*x{}[i]*a{}[i] for i in S{}[h]) for h in range(j+1)) \
                              >= quicksum(E{}[h] * E_EV for h in range(j+1)), name = "NB_EV_{}")'.format(i1, i1, i1, i1,
                                                                                                         i1))

        '''k = 0

        for i in range(Z):
            if p[i,7]==1:
                if i==0:
                    S1[k]=[i]
                else:
                    if p[i-1,7]==0:
                        S1[k]=[i]
                    else:
                        S1[k].append(i)
            if p[i,8] != 0:
                E1[k]=p[i,8]
                k=k+1

        for j in range(k):
            print("j",j)
            model.addConstr(quicksum(quicksum((1/4)*x1[i]*a1[i] for i in S1[h]) for h in range(j+1)) >= quicksum(E1[h]*50 for h in range(j+1)))

        k = 0

        for i in range(Z):       
            if p[i,9]==1:
                if i==0:
                    S2[k]=[i]
                else:
                    if p[i-1,9]==0:
                        S2[k]=[i]
                    else:
                        S2[k].append(i)
            if p[i,10] != 0:
                E2[k]=p[i,10]
                k=k+1
        for j in range(k):
            print("j",j)
            model.addConstr(quicksum(quicksum((1/4)*x2[i]*a2[i] for i in S2[h]) for h in range(j+1)) >= quicksum(E2[h]*50 for h in range(j+1)))
        print("\n S1:",S1)
        print("\n E1:",E1)
        print("\n k1:",k1)

        print("\n S2:",S2)
        print("\n E2:",E2)
        print("\n k2:",k2)'''

        # for k in S:
        # model.addConstr(quicksum((1/4)*x[i]*a[i] for i in S[k]) >= E[k]*50)

        # For-Schleife fuer jedes Fzg erstellen
        # x & a Entscheidungsvariabeln
        # 50 Akkukapazitaet, als Variabel fuer andere EV anpassen
        # nicht einfacher formulieren

        # Nebenbedingung Netzanschluss nicht einfach gleich 20
        # 20: grid - building + pv
        # andere Nebenbedingungen einfuegen

        model.update()

        model.write('test.lp')

        # OBJECTIVE
        # Hier eine Kostenzfk
        # Aendern mit Zfk fuer Kosten(5) und CO2(6) mit Gewichten und CO2-Zfk
        # wenn fuer mehrere EV dann nochmal aussen eine Quicksum setzen - weil kein Vektor mehr sondern Matrix
        '''sumprice=[]
        for j in range(1,3):
            exec('sumprice[i] = x{}[i]*a{}[i]*p[i,5]+sumprice[i]'.format(j,j))
        model.setObjective(quicksum(sumprice[i] for i in range(Z)), GRB.MINIMIZE)'''
        # model.setObjective(quicksum(x1[i]*a1[i]*p[i,5]+x2[i]*a2[i]*p[i,5] for i in range(Z)), GRB.MINIMIZE)
        # if OTM_Target == 'Cost':
        #     for j in range(1,lenID+1):
        #         globals()['sumprice'+str(j)] = quicksum(globals()['x'+str(j)][i]*globals()['a'+str(j)][i]*p[i,5] for i in range(Z))
        # else:
        #     for j in range(1,lenID+1):
        #         globals()['sumprice'+str(j)] = quicksum(globals()['x'+str(j)][i]*globals()['a'+str(j)][i]*p[i,6] for i in range(Z))
        # new ZF
        for j in range(1, lenID + 1):
            globals()['sumprice' + str(j)] = quicksum(
                globals()['x' + str(j)][i] * globals()['a' + str(j)
                                                       ][i] * (p[i, 6] * C_Emission + C_Cost * p[i, 5])
                for i in range(Z))
            # globals()['sumprice'+str(j)] = quicksum(globals()['x'+str(j)][i]*globals()['a'+str(j)][i]*(p[i,6]*C_Emission+
            #         C_Cost*p[i,5])-globals()['pv'+str(j)][i]*(p[i,6]*C_Emission+
            #         C_Cost*p[i,5]) for i in range(Z))

        # sumcharging1 = quicksum(x1[i]*a1[i]*p[i,5] for i in range(Z))
        # sumcharging2 = quicksum(x2[i]*a2[i]*p[i,5] for i in range(Z))

        # https://www.gurobi.com/documentation/9.0/refman/mip_models.html
        # https://www.gurobi.com/documentation/9.0/examples/change_parameters.html
        model.Params.OptimalityTol = 1e-4

        sumprice = 0
        for j in range(1, lenID + 1):
            sumprice = sumprice + globals()['sumprice' + str(j)]

        model.setObjective(sumprice, GRB.MINIMIZE)
        model.setParam(GRB.Param.DualReductions, 0)

        model.__data = sumcharging

        for j in range(1, lenID + 1):
            exec('model.__data= x{},a{}'.format(j, j))
        model.update()

        model.optimize()

        if model.status == GRB.INFEASIBLE:
            # Loop until we reduce to a model that can be solved
            try:
                removed = []
                while True:
                    model.computeIIS()
                    print('\nThe following constraint cannot be satisfied:')
                    i = 0
                    all_q_constrs = model.getQConstrs()
                    constraints = []
                    for qc in all_q_constrs:
                        if "NB_EV" in qc.QCName:
                            constraints.append(qc)
                    for c in constraints:
                        i += 1
                        if c.IISQConstr:
                            print('%s' % c.QCName)
                            # Remove a single constraint from the model
                            removed.append(str(c.QCName))
                            model.remove(c)
                            break
                    print('')
            except:
                model.optimize()
                print('Result - these are the removed Constraints: ')
                print(removed)

        # if model.status == GRB.INFEASIBLE:
        #     print('INFEASIBLE')
        #     vars = model.getVars( )
        #     all_q_constrs = model.getQConstrs()
        #     constraints = []
        #     for qc in all_q_constrs:
        #           if "test" == qc.QCName:
        #               constraints.append(qc)
        #     print(constraints)
        #     rhspen = [0.0] * len(constraints)
        #     model.feasRelax(2,False, None, None, None, constraints, rhspen)
        #     #model.feasRelax(1, False, vars, None, ubpen, None, None)
        #     model.optimize( )
        #
        # #for j in x:
        #  #   print(x[j].X)
        #   #  print(a[j].X)

        # OUTPUT
        for i in range(1, lenID + 1):
            globals()['Ladeplan' + str(i)] = {}
            # Ladeplan1={}
        # Ladeplan2={}
        Time = {}
# ----------------------------------------------------------------------
# Hier Anzahl EV Anpassen
        for i in range(Z):
            Ladeplan1[i] = x1[i].X * a1[i].X
            Ladeplan2[i] = x2[i].X * a2[i].X
            Ladeplan3[i] = x3[i].X * a3[i].X
            Ladeplan4[i] = x4[i].X * a4[i].X
            Ladeplan5[i] = x5[i].X * a5[i].X
            Ladeplan6[i] = x6[i].X * a6[i].X
            Ladeplan1[i] = x7[i].X * a7[i].X
            Ladeplan2[i] = x8[i].X * a8[i].X
            Ladeplan3[i] = x9[i].X * a9[i].X
            Ladeplan4[i] = x10[i].X * a10[i].X
            Ladeplan5[i] = x11[i].X * a11[i].X
            Ladeplan6[i] = x12[i].X * a12[i].X
            # Ladeplan13[i] = x13[i].X * a13[i].X
            # Ladeplan14[i] = x14[i].X * a14[i].X
            # Ladeplan15[i] = x15[i].X * a15[i].X
            # Ladeplan16[i] = x16[i].X * a16[i].X
            # Ladeplan17[i] = x17[i].X * a17[i].X
            # Ladeplan18[i] = x18[i].X * a18[i].X
            # Ladeplan19[i] = x19[i].X * a19[i].X
            # Ladeplan20[i] = x20[i].X * a20[i].X
            # Ladeplan21[i] = x21[i].X * a21[i].X
            # Ladeplan22[i] = x22[i].X * a22[i].X
            # Ladeplan23[i] = x23[i].X * a23[i].X
            # Ladeplan24[i] = x24[i].X * a24[i].X
            # Time[i]=p[i,1]
            # Ladeplan2[i]=x2[i].X*a2[i].X
            # Ladeplan3[i]=x3[i].X*a3[i].X
        Ladeleistung1 = x1[0].X * a1[0].X
        Ladeleistung2 = x2[0].X * a2[0].X
        Ladeleistung3 = x3[0].X * a3[0].X
        Ladeleistung4 = x4[0].X * a4[0].X
        Ladeleistung5 = x5[0].X * a5[0].X
        Ladeleistung6 = x6[0].X * a6[0].X
        Ladeleistung7 = x7[0].X*a7[0].X
        Ladeleistung8 = x8[0].X*a8[0].X
        Ladeleistung9 = x9[0].X*a9[0].X
        Ladeleistung10 = x10[0].X*a10[0].X
        Ladeleistung11 = x11[0].X*a11[0].X
        Ladeleistung12 = x12[0].X*a12[0].X
        # Ladeleistung13=x13[0].X*a13[0].X
        # Ladeleistung14=x14[0].X*a14[0].X
        # Ladeleistung15=x15[0].X*a15[0].X
        # Ladeleistung16=x16[0].X*a16[0].X
        # Ladeleistung17=x17[0].X*a17[0].X
        # Ladeleistung18=x18[0].X*a18[0].X
        # Ladeleistung19=x19[0].X*a19[0].X
        # Ladeleistung20=x20[0].X*a20[0].X
        # Ladeleistung21 =x21[0].X*a21[0].X
        # Ladeleistung22 =x22[0].X*a22[0].X
        # Ladeleistung23 =x23[0].X*a23[0].X
        # Ladeleistung24 =x24[0].X*a24[0].X

        # Kosten=float(model.ObjVal)
        Kosten = 0

        Ladeplan_gesamt = pd.DataFrame(data=[Ladeplan1, Ladeplan2, Ladeplan3, Ladeplan4, Ladeplan5, Ladeplan6, Ladeplan7, Ladeplan8, Ladeplan9, Ladeplan10, Ladeplan11, Ladeplan12
                                             # ,Ladeplan13, Ladeplan14, Ladeplan15, Ladeplan16, Ladeplan17, Ladeplan18,
                                             # Ladeplan19, Ladeplan20, Ladeplan21, Ladeplan22, Ladeplan23, Ladeplan24
                                             ]).T

        for i in range(Z):
            Kosten = Kosten + x1[i].X * a1[i].X * p[i, 5]

        # PowerAll = 0
        # for j in range(1,lenID+2):
        # PowerAll = PowerAll + globals()['x'+str(j)][0].X*globals()['a'+str(j)][0].X

        Zeit = model.Runtime

        # print("\n\n Kosten:",Kosten)
        # print("\n\n Zeit:",Zeit)
        # print("\n\n Ladeplan:")
        '''for i in Ladeplan1:
            print(i,Ladeplan1[i])'''
        objval = model.objVal
        file_logger = 'Optimierung_Ladepläne_results.csv'
        with open(file_logger, 'a+', newline='') as write_obj:
            # Create a writer object from csv module
            csv_writer = csv.writer(write_obj)
            # Add contents of list as last row in the csv file
            list_of_elem = [p[0, 1], objval, p[0, 5], p[0, 6], Ladeleistung1, Ladeleistung2, Ladeleistung3, Ladeleistung4, Ladeleistung5, Ladeleistung6, Ladeleistung7, Ladeleistung8, Ladeleistung9, Ladeleistung10, Ladeleistung11, Ladeleistung12
                            # ,Ladeleistung13,Ladeleistung14,Ladeleistung15,Ladeleistung16,Ladeleistung17,Ladeleistung18,
                            # Ladeleistung19,Ladeleistung20,Ladeleistung21,Ladeleistung22,Ladeleistung23,Ladeleistung24
                            ]
            csv_writer.writerow(list_of_elem)
        return (Ladeleistung1, Ladeleistung2, Ladeleistung3, Ladeleistung4, Ladeleistung5, Ladeleistung6,
                Ladeleistung7, Ladeleistung8, Ladeleistung9, Ladeleistung10, Ladeleistung11, Ladeleistung12,
                # Ladeleistung13,Ladeleistung14,Ladeleistung15,Ladeleistung16,Ladeleistung17,Ladeleistung18,
                # Ladeleistung19,Ladeleistung20,Ladeleistung21,Ladeleistung22,Ladeleistung23,Ladeleistung24,
                Ladeplan_gesamt)

    # 1. in main zählen wie viele wirklich laden (soc <100 ? wie viele sind da)
    # daraus ladeleistung
    # dann später in get fkt die (id) lädt er oder nicht? wie soc etc
    # und wenn 1 dann bekommt er ladeleistung sonst nicht
    def equal(self, number_of_evs, load_limit):
        number_of_charging_evs = number_of_evs
        if number_of_charging_evs != 0:
            ladeleistung = load_limit / number_of_charging_evs
            if ladeleistung > SimConfig.P_max:
                ladeleistung = SimConfig.P_max
            while ladeleistung < 4.14:
                if number_of_charging_evs < 1:
                    print(
                        'FEHLER: zu wenig verfügbare Leistung um ein Fahrzeug bei minimaler Ladeleistung zu laden')
                    break
                print(
                    'WARNUNG: ein Fahrzeug kann nicht mit der min. Ladeleistung geladen werden')
                number_of_charging_evs -= 1
                ladeleistung = load_limit / number_of_charging_evs
        else:
            ladeleistung = 0
        return ladeleistung

    def fcfs(self, load_limit, list_of_evs_with_prio):
        priority_list_sorted = sorted(
            list_of_evs_with_prio, key=lambda ev: ev[1], reverse=True)
        load_limit = load_limit
        total_nmb_of_evs = SimConfig.Number_ESD
        ladeleistungen_list = [0] * total_nmb_of_evs
        nmbr_ev_min_charge = len(list_of_evs_with_prio) - 1
        for ev in priority_list_sorted:
            EV_id = ev[0]
            if load_limit - SimConfig.P_max - 4.14 * nmbr_ev_min_charge >= 0:
                ladeleistung = SimConfig.P_max
                ladeleistungen_list[EV_id - 1] = ladeleistung
                load_limit -= ladeleistung
                nmbr_ev_min_charge -= 1
            elif load_limit - 4.14 * nmbr_ev_min_charge >= 0:
                ladeleistung = load_limit - 4.14 * (nmbr_ev_min_charge)
                ladeleistungen_list[EV_id - 1] = ladeleistung
                load_limit -= ladeleistung
                nmbr_ev_min_charge -= 1
            else:
                print('FEHLER: ein ladenes Fahrzeug bekommt nicht die min_charge')

        return ladeleistungen_list


if __name__ == '__main__':
    test = Optimierung()
    test.ladeplan(6)
#  i1, i2, i3 , i4, i5, i6,i7,i8,i9,i10 = test.fcfs(10,[1,2,3,4,5])
# i1, i2, i3, i4, i5 = test.fcfs(30, [1, 2, 3, 4, 5], [5,9,4,0,9]) #waswenn2x gleicheprio
# i1, i2, i3, i4, i5, i6 = test.fcfs(30, [(1,4),(2,5),(3,9),(4,8), (5,9)]) #waswenn2x gleicheprio

# print(i1, i2, i3 , i4, i5, i6)
# print(8-(8-1.3))#????
