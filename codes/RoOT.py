######################################################################
# Copyright (C) 2020 University of Washington and VillageReach 
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>
#
# SPDX-License-Identifier: GPL-3.0-only
######################################################################

## Code for defining delivery routes using indexing method
## Developed by Larissa P.G. Petroianu and Yi Chu - University of Washington
## Advisor: Prof. Zelda B. Zabinsky
## March 2020

##### work and finalize each vehicle
import math
import csv
import pandas as pd
from tkinter import *
from tkinter import filedialog
import pygtrie as trie
import time
import numpy as np
import random
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import xlsxwriter  ####NEW
from openpyxl.styles.builtins import total
from pandas.io.formats.printing import justify
from datetime import date

# print('START ********')

root = Tk()

root.geometry('500x150+400+200')

root.title("Routing Tool")


# Get file name based on file dialog box

def choose_file(label):
    fileName = filedialog.askopenfilename()

    label.config(text=fileName)


def retrieve_input():
    global inputValue

    # inputValue = textBox.get("1.0","end-1c")

    inputValue = lblFileName.cget("text")

    root.destroy()


root.msg = Label(root, text="Route Optimization Tool (RoOT)")

root.msg.pack()

# textBox = Text(root, height=2, width=40)

# textBox.pack()

lblFileName = Label(root, text="")

lblFileName.pack()

buttonChooseFile = Button(root, text='Choose an input file for route optimization',
                           command=lambda: choose_file(lblFileName))

buttonChooseFile.pack(pady=10)

buttonCommit = Button(root, height=1, text="Run route optimization",
                      command=lambda: retrieve_input())

buttonCommit.pack()

mainloop()

file_name = inputValue

output_name = inputValue.split(".")[0] + '_result_'+ str(date.today()) +'.xlsx'
# print(output_name)

# print('RoOT was developed at the University of Washington and VillageReach')
print('Loading data, please wait, it will take a few seconds...')

data = pd.ExcelFile(file_name)

data_capacities = pd.read_excel(data, 'center_capacities', 0, index_col=None,
                                na_values=['NA'], skiprows=1)

data_demand = pd.read_excel(data, 'demand', 0, index_col=None, skiprows=2)

data_products = pd.read_excel(data, 'products', 0, index_col=None,
                              na_values=['NA'], skiprows=1)
data_transit = pd.read_excel(data, 'distance_data', 0, index_col=None,
                             na_values=['NA'], skiprows=2)
data_vehicle = pd.read_excel(data, 'vehicle', 0, index_col=None,
                             na_values=['NA'], skiprows=1)
data_input = pd.read_excel(data, 'parameters', 0, index_col=None, na_values=['NA'])

# data_arcs = pd.read_excel(data, 'available_roads', 0, index_col=None, na_values=['NA'],
#                           skiprows=2)
data_arcs = pd.read_excel(data, 'road_condition', 0, index_col=None, na_values=['NA'],
                          skiprows=2)
# data_warning = pd.read_excel(data, 'warning_capacity', 0, index_col=None, na_values=['NA'])

data_drop = pd.read_excel(data, 'dropdowns', 0, index_col=None, na_values=['NA'])

###################NEW!!!!!!!!!!
origin = data_input.iloc[1, 1]
start_time = float(data_input.iloc[2, 2]) * 60 + float(data_input.iloc[2, 3])
lft = float(data_input.iloc[3, 2]) * 60 + float(data_input.iloc[3, 3]) - start_time
W = float(data_input.iloc[4, 1]) * 60
W_transit = 0.1 * float(data_input.iloc[5, 1])
W_penalties = 0.1 * float(data_input.iloc[6, 1])
Description = data_input.iloc[0, 1]

time_to_run = int(data_drop.iloc[0,10])
# time_to_run = 1800  ####NEW
# print("time = "+str(time_to_run))

HC1 = []
HC = []
for index, row in data_capacities.iterrows():
    HC1.append(row[0])

# check demand
Check = {}
for i in range(0, len(HC1)):
    if str(HC1[i]) != 'nan':
        #         print(HC1[i])
        #         print(data_demand['sum'][i])
        Check[HC1[i]] = float(data_demand['sum'][i])
        if (Check[HC1[i]] > 0 or str(HC1[i]) == origin):
            HC.append(HC1[i])
    else:
        break
# print(HC)

# create list of products and volumes
P_original = []
P = ['cold', 'dry']
COLD = []
DRY = []
vol_p = {}
doses = {}  ########NEW

# print(data_products['Requires Cold Storage?'][11])

# product volumes
for index, row in data_products.iterrows():
    if str(row[1]) != 'nan':
        product = row[1]
        P_original.append(product)
        vol_p[product] = float(row[6])  ########NEW
        a = row[2]
        if a == "Yes":
            COLD.append(product)
            doses[product] = float(row[3])  ########NEW
        else:
            DRY.append(product)
    else:
        break

    # create lists and dictionaries
# list of centers, dictionaries of distances,
## available arcs, penalties for arcs

dist = {}
avail_arc = {}
Gamma = {}
list_pen_roads = []  ######NEW
list_pen_cars = []  ######NEW

pen_dict_road = {}  ######NEW
for index, linha in data_drop.iterrows():
    pen_dict_road[linha[4]] = linha[5]
    list_pen_roads.append(linha[4])

pen_dict_car = {}  ######NEW
for index, linha in data_drop.iterrows():
    pen_dict_car[linha[7]] = linha[8]
    list_pen_cars.append(linha[7])
#
#
# print(pen_dict.keys())
# print(pen_dict.values())

# print(data_arcs.iloc[HC1.index(HC[1]),HC1.index(HC[1])+2])
# print(data_transit.iloc[0,1])
for i in range(0, len(HC)):
    for j in range(0, len(HC)):
        if i != j:
            dist[HC[i], HC[j]] = float(data_transit.iloc[HC1.index(HC[i]), HC1.index(HC[j]) + 2])
            temp = data_arcs.iloc[HC1.index(HC[i]), HC1.index(HC[j]) + 2]
            #         print (temp)
            #         print (pen_dict[temp])
            if pen_dict_road[temp] == 0:
                avail_arc[HC[i], HC[j]] = 0
            else:
                avail_arc[HC[i], HC[j]] = 1
    
            Gamma[HC[i], HC[j]] = float(pen_dict_road[temp])
    #         print(Gamma[HC[i],HC[j]])
        else:
            dist[HC[i], HC[j]] = 0
            avail_arc[HC[i], HC[j]] = 0
            Gamma[HC[i], HC[j]] = 0

# print(dist["HOSPITAL CENTRAL","HOSPITAL CENTRAL"])

#
# Create list of vehicles and dictionary of
## velocity, penalty, capacities and maximum time
V = []  # vehicles
velocity = {}
Beta = {}  # vehicle penalties
c_cold = {}  # vehicle cold capacity
c_dry = {}  # vehicle dry capacity
lft_v = {}  # route maximum time
cost_km = {}  # cost per kilometer #######NEW
dose_v = {}  # quantity of doses carried by the vehicle #########NEW
cost_personnel = {}

# populate sets and parameters
for index, row in data_vehicle.iterrows():
    if row[1] == 'Available':
        temp = data_vehicle.iloc[index, 8]  ######NEW
        valor1 = row[0]
        V.append(valor1)
        velocity[valor1] = float(row[2])
        Beta[valor1] = float(pen_dict_car[temp])  ######NEW
        c_cold[valor1] = float(row[6])
        c_dry[valor1] = float(row[7])
        lft_v[valor1] = lft
        cost_km[valor1] = float(row[4]) / float(row[3])  ########NEW
        cost_personnel[valor1] = round(float(row[10]) * float(row[11]), 2)  ########NEW
#         dose_v[valor1] = 0  ###########NEW

# transit time
h = {}
for i in HC:
    for j in HC:
        for v in V:
            if (i == j):
                h[i, j, v] = 0
            else:
                h[i, j, v] = float(dist[i, j] * 60 / velocity[v])
#         print(h[i,j,v])

total_dose = 0  ########NEW

# demand in HC i for product p for each scenario
d = {}
for i in HC:
    d[i, 'cold'] = 0
    d[i, 'dry'] = 0

d_original = {}
for i in HC:
    d_original[i] = []

cap_cold = {}
cap_dry = {}

for v in V:
    cap_cold[v] = c_cold[v]
    cap_dry[v] = c_dry[v]

##################################################################
#####################################################################
demand_doses = {}  #######NEW
for i in range(0, len(HC)):
    demand_doses[HC[i]] = 0
    p_list = []
    cap_cold[HC[i]] = round(data_demand.iloc[HC1.index(HC[i]), 1] * 100, 2)
    #     print(cap_cold[HC[i]])
    cap_dry[HC[i]] = round(data_demand.iloc[HC1.index(HC[i]), 2] * 100, 2)
    #     print(cap_dry[HC[i]])
    for j in range(0, len(P_original)):
        p = P_original[j]
        #         total_dose = total_dose + float(data_demand.iloc[HC1.index(HC[i]),
        #                                                          P_original.index(p) + 4])

        if p in COLD:
#             if float(data_demand.iloc[HC1.index(HC[i]), P_original.index(p) + 4]) != 0:
            d[HC[i], 'cold'] = ((math.ceil(float(data_demand.iloc[HC1.index(HC[i]),
                                                        P_original.index(p) + 4]) / doses[p]) * doses[p]) *
                                float(vol_p[p]) + float(d[HC[i], 'cold']))
            p_list.append([p, (math.ceil(float(data_demand.iloc[HC1.index(HC[i]),P_original.index(p) + 4]) / 
                                         doses[p]) * doses[p])])  ####NEW
 
            demand_doses[HC[i]] = (demand_doses[HC[i]] +
                                   (math.ceil(float(data_demand.iloc[HC1.index(HC[i]),P_original.index(p) + 4]) / 
                                         doses[p]) * doses[p]))
            

#     total_dose = total_dose + demand_doses[HC[i]]  
#         
        elif p in DRY:
            #             print(p)
            d[HC[i], 'dry'] = (float(data_demand.iloc[HC1.index(HC[i]),
                                                      P_original.index(p) + 4]) *
                               float(vol_p[p]) + float(d[HC[i], 'dry']))
            p_list.append([p, data_demand.iloc[HC1.index(HC[i]), P_original.index(p) + 4]])  ####NEW
#             print(d[HC[i],'dry'])
    d_original[HC[i]].append(p_list)
    total_dose = total_dose + demand_doses[HC[i]]  
#     print(d_original[HC[i]])####NEW
#########################################################################
###########################################################################

# #refrigerator capacity in HC i for product p
r = {}
dry = {}
for i in range(0, len(HC)):
    r[HC[i]] = float(data_capacities.iloc[HC1.index(HC[i]), 3])
    #     print(r[HC[i]])
    dry[HC[i]] = float(data_capacities.iloc[HC1.index(HC[i]), 4])
#     print(dry[HC[i]])

# big number
B = 100000

# total calculations for future use
total_index = 0
total_h = 0
total_gamma = 0
total_beta = 0
total_cold = 0
total_dry = 0
total_initial_avail = 0
total_velocity = 0

for v in V:
    total_beta = total_beta + Beta[v]
    total_cold = total_cold + c_cold[v]
    total_dry = total_dry + c_dry[v]
    total_velocity = total_velocity + velocity[v]

for i in HC:
    total_initial_avail = total_initial_avail + avail_arc[origin, i]
    for j in HC:
        total_gamma = total_gamma + Gamma[i, j]

for i in HC:
    for j in HC:
        for v in V:
            total_h = total_h + h[i, j, v]

# calculate means for indexing
mean_h = total_h / (len(HC) * len(HC) * len(V) / 2)  #### changed len(v) to len(V)
mean_gamma = total_gamma / len(Gamma)
mean_beta = total_beta / len(V)
mean_cold = total_cold / len(V)
mean_dry = total_dry / len(V)
mean_velocity = total_velocity / len(V)
mean_avail = total_initial_avail / len(HC)
times = {}
initial_hour = math.floor(start_time / 60)
initial_minutes = math.floor(start_time % 60)
time_hours = str(initial_hour) + ' h ' + str(initial_minutes) + ' min '
# lower_bound_data = []
lb_flag = False
for v in V:
    times[v] = [initial_hour * 60]
eliteSet = np.empty(
    3)  #########HERE, 1st entry corresponds to best, 2nd entry corresponds to uncertain, 3rd corresponds to random###########
eliteSet[0] = 0.5
eliteSet[1] = 0.5
eliteSet[2] = 0
print('Processing...')


def lower_bound(total_centers, current_center, next_center, current_vehicle, total_cost):
    """
    This method calculate a lower bound given a partially determined route
    The lower bound consists of four parts:
    Part1: time already spent
    Part2: time from the current center to the next one
    Part3: time to traverse the minimum spanning tree
    Part4: if the current_center is the origin,
           the mininum time from any HC going back to the origin is added
    """
    # print("######LOWER BOUND ###########")
    part1 = total_cost

    if len(total_centers) == 0 and current_center == origin:
        return part1

    # get the max possible speed
    part2 = W_transit * math.exp(-(h[current_center, next_center, current_vehicle] / mean_h)) + \
            W_penalties * (math.exp(-(Gamma[current_center, next_center] / mean_gamma)) +
                           math.exp(-(Beta[current_vehicle] / mean_beta)))

    if len(total_centers) == 0 and current_center != origin:
        return part1 + part2

    # construct the distance matrix
    matrix_size = len(total_centers) + 1

    cost_list = []
    for v_ in V:
        cost_matrix = csr_matrix((matrix_size, matrix_size), dtype=np.float64).toarray()
        row_ = 0
        col_ = 1
        # distance from origin to all the unvisited HC
        for j_ in total_centers:
            cost_matrix[row_, col_] = W_transit * math.exp(-(h[origin, j_, v_] / mean_h)) + \
                                      W_penalties * (math.exp(-(Gamma[origin, j_] / mean_gamma)) +
                                                     math.exp(-(Beta[v_] / mean_beta)))
            if current_center == origin and j_ == next_center:
                cost_matrix[row_, col_] = 0
            col_ += 1
        row_ += 1

        for i_ in total_centers:
            col_ = 1
            for j_ in total_centers:
                if row_ >= col_:
                    col_ += 1
                    continue
                cost_matrix[row_, col_] = W_transit * math.exp(-(h[i_, j_, v_] / mean_h)) + \
                                          W_penalties * (math.exp(-(Gamma[i_, j_] / mean_gamma)) +
                                                         math.exp(-(Beta[v_] / mean_beta)))
                col_ += 1
            row_ += 1
        cost_list.append(cost_matrix)
    min_cost = cost_list[0]
    for i_ in range(1, len(cost_list)):
        min_cost = np.minimum(cost_list[i_], min_cost)
    # calculate the mininum spanning tree
    min_span_tree = minimum_spanning_tree(min_cost).toarray().astype(float)

    part3 = sum(sum(min_span_tree))

    # we need to add the go back time
    if next_center == origin:
        temp_ = min_cost[0][np.nonzero(min_cost[0])]
        part4 = min(temp_)
    else:
        part4 = 0
    # print("The current lower bound is " + str(LB) + " and the incumbent is " + str(incumbent))
    return part1 + part2 + part3 + part4


def global_lower_bound(total_centers, total_cost):
    """
    This method calculate the lower bound at the very beginning
    when no decision has been made.
    """
    # get the max possible speed
    total_centers_t = total_centers.copy()
    if origin in total_centers_t:
        total_centers_t.remove(origin)
    matrix_size = len(total_centers_t) + 1
    cost_list = []
    for v_ in V:
        cost_matrix = csr_matrix((matrix_size, matrix_size), dtype=np.float64).toarray()
        row_ = 0
        col_ = 1
        for j_ in total_centers_t:
            cost_matrix[row_, col_] = W_transit * math.exp(-(h[origin, j_, v_] / mean_h)) + \
                                      W_penalties * (math.exp(-(Gamma[origin, j_] / mean_gamma)) +
                                                     math.exp(-(Beta[v_] / mean_beta)))
            col_ += 1
        row_ += 1
        # distance between unvisited HCs
        for i_ in total_centers_t:
            col_ = 1
            for j_ in total_centers_t:
                if row_ >= col_:
                    col_ += 1
                    continue
                cost_matrix[row_, col_] = W_transit * math.exp(-(h[i_, j_, v_] / mean_h)) + \
                                          W_penalties * (math.exp(-(Gamma[i_, j_] / mean_gamma)) +
                                                         math.exp(-(Beta[v_] / mean_beta)))
                col_ += 1
            row_ += 1
        cost_list.append(cost_matrix)
    min_cost = cost_list[0]
    for i_ in range(1, len(cost_list)):
        min_cost = np.minimum(cost_list[i_], min_cost)
    # calculate the mininum spanning tree
    min_span_tree = minimum_spanning_tree(min_cost).toarray().astype(float)
    part3 = sum(sum(min_span_tree))
    # we need to add the go back time
    temp_ = min_cost[0]
    ind = np.nonzero(temp_)
    tem = temp_[ind]
    part4 = min(tem)

    return total_cost + part3 + part4


# define next vehicle to be used, according to index 1
# vehicle will be chosen if it is available and has capacity
def next_vehicle(moves, available_vehicles, c_cold_, c_dry_, lft_v_):
    ind = 0
    vehicle = available_vehicles[0]
    for v_ in available_vehicles:
        if c_cold_[v_] > 0 and c_dry_[v_] > 0 and lft_v_[v_] > 0:
            index_temp = (math.exp(-(Beta[v_] / mean_beta)) +
                          math.exp(+(c_cold_[v_] / mean_cold)) +
                          math.exp(+(c_dry_[v_] / mean_dry)) +
                          math.exp(+(velocity[v_] / mean_velocity)))
            if index_temp > ind:
                ind = index_temp
                vehicle = v_
    # global total_index
    # total_index = total_index + ind
    moves.append(vehicle)
    return vehicle, moves


# define next node, according to index 2
# node will be chosen the vehicle has time to reach it.
# time and routes calculations here
def next_node(current_center, total_centers, vehicle_, c_cold_, c_dry_, lft_v_, total_cost):
    available_centers = total_centers.copy()
    check = total_centers.copy()
    next_center_lower_bound = np.inf
    flag = True
    if len(available_centers) == 0:
        return origin, True, np.inf
    if current_center != origin:
        next_center = origin
        ind = 0
    else:
        next_center = total_centers[0]
        ind = W_transit * math.exp(-(h[current_center, next_center, vehicle_] / mean_h) +
                                   W_penalties * math.exp(-(Gamma[current_center, next_center] / mean_gamma)))
    for j_ in available_centers:
        if j_ != next_center:
            # test if going to HC j is feasible
            if feasibility(current_center, j_, vehicle_, c_cold_, c_dry_, lft_v_):
                # test if going to HC j vialates the optimality
                temp_total = available_centers.copy()
                temp_total.remove(j_)
                lowerBound = lower_bound(temp_total, current_center, j_, vehicle_, total_cost)
                if lowerBound < incumbent + 1e-10:
                    index_temp = W_transit * math.exp(-(h[current_center, j_, vehicle_] / mean_h) +
                                                      W_penalties * math.exp(-(Gamma[current_center, j_] / mean_gamma)))
                    # +math.exp((1-avail_arc[origin,j])/mean_avail))
                    if index_temp > ind:
                        ind = index_temp
                        next_center = j_
                        next_center_lower_bound = lowerBound
                else:
                    check.remove(j_)
            else:
                check.remove(j_)
    #     print(current_center, next_center, c_cold_, c_dry_, lft_v_, lft_v)
    if len(check) == 0 and next_center != origin:
        # print("EMPTY!!!!")
        flag = False
    # if flag and next_center_lower_bound != np.inf:
    #     lower_bound_data.append(next_center_lower_bound)

    return next_center, flag, next_center_lower_bound


def goto(moves, current_center, next_center, vehicle_, available_vehicles, c_cold_, c_dry_, lft_v_):
    if next_center == origin:
        available_vehicles.remove(vehicle_)
        if len(available_vehicles) == 0:
            available_vehicles = V.copy()
            c_cold_ = c_cold.copy()
            c_dry_ = c_dry.copy()
            lft_v_ = lft_v.copy()
    # total_index = total_index + index
    lft_v_[vehicle_] = lft_v_[vehicle_] - (h[current_center, next_center, vehicle_] + W)
    c_cold_[vehicle_] = c_cold_[vehicle_] - d[next_center, 'cold']
    c_dry_[vehicle_] = c_dry_[vehicle_] - d[next_center, 'dry']
    moves.append([current_center, next_center, vehicle_])
    return moves, available_vehicles, c_cold_, c_dry_, lft_v_


# vehicle and route feasibility, according to capacities, time,
# and availability of the route
def feasibility(current_center, next_center, vehicle_, c_cold_, c_dry_, lft_v_):
    _cold = c_cold_[vehicle_] - d[next_center, 'cold']
    _dry = c_dry_[vehicle_] - d[next_center, 'dry']
    _lft = lft_v_[vehicle_] - (h[current_center, next_center, vehicle_] + h[next_center, origin, vehicle_] + W)
    if (_cold < 0) or (_dry < 0) or (_lft < 0) or (avail_arc[current_center, next_center] == 0):
        return False
    return True


# main to run the approximation, calling the methods above
def decision_tree(path_, t_, incumbent_, solutions_, global_lb_, branch_lb_):
    # if haven't go down the tree
    if not path_:
        # initialize all variables
        c_cold_t = c_cold.copy()
        c_dry_t = c_dry.copy()
        lft_v_t = lft_v.copy()
        total_centers_t = HC.copy()
        total_centers_t.remove(origin)
        available_vehicles_t = V.copy()
        # times_t = times.copy()
        moves_t = []  # moves_t, moves are used to store the decisions made within one solution
        current_center_t = origin
        total_cost_t = 0
        pos = []  # posible next moving
        for v_ in available_vehicles_t:
            pos.append(v_)
    # if have go down the tree and have a solution
    else:
        # initialize all variables
        c_cold_t = c_cold.copy()
        c_dry_t = c_dry.copy()
        lft_v_t = lft_v.copy()
        moves_t = []
        total_centers_t = HC.copy()
        total_centers_t.remove(origin)
        available_vehicles_t = V.copy()
        # times_t = times.copy()
        total_cost_t = 0
        pos = []
        steps = path_.split("/")
        # read decisions of the branch with lowest cost, we will go down this branch to explore next
        for step in steps:
            divide = step.split(",")
            if len(divide) == 3:
                current_ = divide[0]
                center_ = divide[1]
                current_v_ = divide[2]
                moves_t.append([current_, center_, current_v_])
            else:
                moves_t.append(divide[0])
        # restore all variables up to len(moves_t)
        for i_ in range(len(moves_t)):
            # if length == 3, the decision is choosing next center for a vehicle
            if isinstance(moves_t[i_], list):
                lft_v_t[moves_t[i_][2]] = lft_v_t[moves_t[i_][2]] - (
                            h[moves_t[i_][0], moves_t[i_][1], moves_t[i_][2]] + W)
                c_cold_t[moves_t[i_][2]] = c_cold_t[moves_t[i_][2]] - d[moves_t[i_][1], 'cold']
                c_dry_t[moves_t[i_][2]] = c_dry_t[moves_t[i_][2]] - d[moves_t[i_][1], 'dry']
                if moves_t[i_][1] != origin:
                    total_cost_t = (total_cost_t +
                                    W_transit * math.exp(
                                -(h[moves_t[i_][0], moves_t[i_][1], moves_t[i_][2]] / mean_h)) +
                                    W_penalties * (math.exp(-(Gamma[moves_t[i_][0], moves_t[i_][1]] / mean_gamma)) +
                                                   math.exp(-(Beta[moves_t[i_][2]] / mean_beta))))
                    # list_t = times_t[moves_t[i_][2]]
                    # time_t = float(list_t[len(list_t) - 1]) + float(
                    #     (h[moves_t[i_][0], moves_t[i_][1], moves_t[i_][2]] + W))
                    # list_t.append(time_t)
                    # times_t[moves_t[i_][2]] = list_t
                    total_centers_t.remove(moves_t[i_][1])
                # else, the decision is choosing next vehicle
                else:
                    total_cost_t = (total_cost_t +
                                    W_transit * math.exp(-(h[moves_t[i_][0], origin, moves_t[i_][2]] / mean_h)) +
                                    W_penalties * (math.exp(-(Gamma[moves_t[i_][0], origin] / mean_gamma)) +
                                                   math.exp(-(Beta[moves_t[i_][2]] / mean_beta))))
                    available_vehicles_t.remove(moves_t[i_][2])
                    if len(available_vehicles_t) == 0:
                        available_vehicles_t = V.copy()
                        c_cold_t = c_cold.copy()
                        c_dry_t = c_dry.copy()
                        lft_v_t = lft_v.copy()
        # depending on length of the last decision, restore corresponding variables
        if isinstance(moves_t[-1], list):
            current_center_t = moves_t[-1][1]
            current_v_t = moves_t[-1][2]
        else:
            if len(moves_t) == 1:
                current_center_t = origin
                current_v_t = available_vehicles_t[0]
            else:
                current_center_t = moves_t[-2][1]
                current_v_t = moves_t[-1]
        # for v_ in available_vehicles_t:
        #     if c_cold_t[v_] < 0 or c_dry_t[v_] < 0 or lft_v_t[v_] < 0:
        #         available_vehicles_t.remove(v_)
        if isinstance(moves_t[-1], list):
            if current_center_t == origin:
                for v_ in available_vehicles_t:
                    pos.append(v_)
            else:
                for node in total_centers_t:
                    temp_total = total_centers_t.copy()
                    temp_total.remove(node)
                    lowerBound = lower_bound(temp_total, current_center_t, node, current_v_t, total_cost_t)
                    if feasibility(current_center_t, node, moves_t[-1][2], c_cold_t, c_dry_t, lft_v_t) \
                            and lowerBound < incumbent_ + 1e-10:
                        pos.append([node, moves_t[-1][2], lowerBound])
                temp_total = total_centers_t.copy()
                depot_lb = lower_bound(temp_total, current_center_t, origin, current_v_t, total_cost_t)
                if depot_lb < incumbent_ + 1e-10:
                    pos.append([origin, moves_t[-1][2], depot_lb])
                # if total_c_ == 5.901847060949091:
                #     print('moves', moves_t)
                #     print('pos', total_centers_t, pos)
        else:
            for node in total_centers_t:
                temp_total = total_centers_t.copy()
                temp_total.remove(node)
                lowerBound = lower_bound(temp_total, current_center_t, node, current_v_t, total_cost_t)
                if feasibility(current_center_t, node, moves_t[-1], c_cold_t, c_dry_t, lft_v_t) \
                        and lowerBound < incumbent_ + 1e-10:
                    pos.append([node, moves_t[-1], lowerBound])
        if len(total_centers_t) == 0:
            global_lb_.append(branch_lb_)
            return t_, incumbent_, solutions_, global_lb_

    # enumerate through all 'breadth(possibilities)' for current 'depth' on decision tree
    for possible in pos:
        # again, restore variables
        c_cold_ = c_cold_t.copy()
        c_dry_ = c_dry_t.copy()
        lft_v_ = lft_v_t.copy()
        # times_ = times_t.copy()
        available_vehicles = available_vehicles_t.copy()
        total_centers = total_centers_t.copy()
        total_cost = total_cost_t
        moves = moves_t.copy()
        current_center = current_center_t
        # if len(total_centers) == 0:
        #     next_center_lower_bound = np.inf
        # len(possible) == 2 means we are choosing center
        if isinstance(possible, list):
            select_c = possible[0]
            select_v = possible[1]
            current_lb = possible[2]
            moves, available_vehicles, c_cold_, c_dry_, lft_v_ = goto(moves,
                                                                      current_center,
                                                                      select_c,
                                                                      select_v,
                                                                      available_vehicles,
                                                                      c_cold_,
                                                                      c_dry_,
                                                                      lft_v_,
                                                                      )
            total_cost = (total_cost + W_transit * math.exp(-(h[current_center, select_c, select_v] / mean_h)) +
                          W_penalties * (math.exp(-(Gamma[current_center, select_c] / mean_gamma)) +
                                         math.exp(-(Beta[select_v] / mean_beta))))
            current_center = select_c
            if select_c != origin:
                total_centers.remove(select_c)
            else:
                select_v = 'None'
        # else, we are choosing vehicle
        else:
            select_v = possible
            current_lb = global_lower_bound(total_centers, total_cost)
            moves.append(select_v)
        # this step is updating the Trie t, which is used to store all solutions we've visited
        string = ''
        for i_ in range(len(moves) - 1):
            if isinstance(moves[i_], list):
                string += (str(moves[i_][0]) + ',' + str(moves[i_][1]) + ',' + str(moves[i_][2]) + '/')
            else:
                string += (str(moves[i_]) + '/')
        if isinstance(moves[-1], list):
            string += (str(moves[-1][0]) + ',' + str(moves[-1][1]) + ',' + str(moves[-1][2]))
        else:
            string += (str(moves[-1]))
        move = string
        if len(total_centers) == 0:
            if current_center != origin:
                total_cost = (total_cost + W_transit * math.exp(-(h[current_center, origin, select_v] / mean_h)) +
                              W_penalties * (math.exp(-(Gamma[current_center, select_c] / mean_gamma)) +
                                             math.exp(-(Beta[select_v] / mean_beta))))
                current_center = origin
            current_lb = total_cost
            global_lb_.append(current_lb)
        # while still have center to goto or still haven't returned back to origin
        while len(total_centers) != 0 or current_center != origin:
            if select_v in available_vehicles:
                select_c, flag, next_center_lower_bound = next_node(current_center, total_centers,
                                                                    select_v, c_cold_, c_dry_, lft_v_, total_cost)
                # not flag means all possible moves are pruned for current depth in the tree
                if not flag:
                    # if that's the case, mark current branch cost inf and break out of decision_tree function
                    total_cost = np.inf
                    string = ''
                    for i_ in range(len(moves) - 1):
                        if isinstance(moves[i_], list):
                            string += (str(moves[i_][0]) + ',' + str(moves[i_][1]) + ',' + str(moves[i_][2]) + '/')
                        else:
                            string += (str(moves[i_]) + '/')
                    if isinstance(moves[-1], list):
                        string += (str(moves[-1][0]) + ',' + str(moves[-1][1]) + ',' + str(moves[-1][2]))
                    else:
                        string += (str(moves[-1]))
                    t_[string] = (total_cost, move, current_lb)
                    break

                moves, available_vehicles, c_cold_, c_dry_, lft_v_ = goto(moves,
                                                                          current_center,
                                                                          select_c,
                                                                          select_v,
                                                                          available_vehicles,
                                                                          c_cold_,
                                                                          c_dry_,
                                                                          lft_v_,
                                                                          )
            else:
                select_v, moves = next_vehicle(moves, available_vehicles, c_cold_, c_dry_, lft_v_)
                select_c, flag, next_center_lower_bound = next_node(current_center, total_centers, select_v,
                                                                    c_cold_, c_dry_, lft_v_, total_cost)
                if not flag:
                    total_cost = np.inf
                    string = ''
                    for i_ in range(len(moves) - 1):
                        if isinstance(moves[i_], list):
                            string += (str(moves[i_][0]) + ',' + str(moves[i_][1]) + ',' + str(moves[i_][2]) + '/')
                        else:
                            string += (str(moves[i_]) + '/')
                    if isinstance(moves[-1], list):
                        string += (str(moves[-1][0]) + ',' + str(moves[-1][1]) + ',' + str(moves[-1][2]))
                    else:
                        string += (str(moves[-1]))
                    t_[string] = (total_cost, move, current_lb)
                    break
                moves, available_vehicles, c_cold_, c_dry_, lft_v_ = goto(moves,
                                                                          current_center,
                                                                          select_c,
                                                                          select_v,
                                                                          available_vehicles,
                                                                          c_cold_,
                                                                          c_dry_,
                                                                          lft_v_,
                                                                          )
            if select_c != origin:
                total_centers.remove(select_c)
            total_cost = (total_cost + W_transit * math.exp(-(h[current_center, select_c, select_v] / mean_h)) +
                          W_penalties * (math.exp(-(Gamma[current_center, select_c] / mean_gamma)) +
                                         math.exp(-(Beta[select_v] / mean_beta))))
            current_center = select_c
            # print('after', select_v, lft_v_)
        # if no center left, means we found a feasible solution, update Trie t and return
        if len(total_centers) == 0:
            string = ''
            for i_ in range(len(moves) - 1):
                if isinstance(moves[i_], list):
                    string += (str(moves[i_][0]) + ',' + str(moves[i_][1]) + ',' + str(moves[i_][2]) + '/')
                else:
                    string += (str(moves[i_]) + '/')
            if isinstance(moves[-1], list):
                string += (str(moves[-1][0]) + ',' + str(moves[-1][1]) + ',' + str(moves[-1][2]))
            else:
                string += (str(moves[-1]))
            t_[string] = (total_cost, move, current_lb)
            if total_cost < incumbent_:
#                 print('time', time.time() - start, 'cost', total_cost)  ####### changed
                solutions_ = moves
                incumbent_ = total_cost
                # update the lowerbound
                # if lb_flag:
                #     new_lower_bound = min(lower_bound_data)
                # else:
                #     new_lower_bound = GLOBAL
    return t_, incumbent_, solutions_, global_lb_


def traverse(t_, start_, incumbent_, solutions_, global_lb_):
    """
    traverse the decision tree, evaluate all possible solutions
    """
    # while Trie t is not empty (still possible solution not visited)
    ll = 0
    gap = 100
    while t_ and time.time() - start_ <= time_to_run:
        print(".", end = "", flush=True)
        for branches in t_.items():
            if branches[1][2] > incumbent_ + 1e-10:
                del t_[branches[0]]
        # for kk in t.items():
        #     print('branch', kk[1][1], kk[1][2])
        if len(t.items()) == 0:
            print('optimality', incumbent_)
            break
        else:
            glb = min(t_.items(), key=lambda a_: a_[1][2])[1][2]
        if min(global_lb_) < glb:
            glb = min(global_lb_)
        
        if gap != incumbent_ - glb:
            gap = incumbent_ - glb
#             print('gap', incumbent_ - glb)
            
        if glb == incumbent_:
            print('optimality found', incumbent_)
            break
        
        gap = incumbent_ - glb
        
        # print('gap', incumbent_ - global_lb)
        # find the elite branches with minimum cost and go down that branch
        branches_best = sorted(t_.items(), key=lambda a_: a_[1][0])
        n = np.empty(3, dtype=np.int)
        for i_ in range(len(eliteSet)):
            if eliteSet[i_] != 0:
                n[i_] = int(np.ceil(len(t_.items()) * 0.12 * eliteSet[i_]))
        if eliteSet[0] != 0:
            branches_search = branches_best[:n[0]].copy()
            branches_remain = branches_best[n[0]:].copy()
        else:
            branches_search = t_.items().copy()
            branches_remain = t_.items().copy()
        if eliteSet[1] != 0 and len(branches_remain) != 0:
            branches_uncertain = sorted(branches_remain, key=lambda a_: abs(a_[1][0] - a_[1][2]))
            branches_search = branches_search + branches_uncertain[:n[1]]
            branches_remain = branches_uncertain[n[1]:]
        if eliteSet[2] != 0 and len(branches_remain) != 0:
            pick = random.sample(range(0, len(branches_remain)), n[2])
            for i_ in range(len(pick)):
                branches_search = branches_search + [branches_remain[pick[i_]]]

        # print('!!!!!!!!!!')
        # for l in branches:
        #     print(l[1][1])
        # print('!!!!!!!!!!')
        # print(branches)
        for branch in branches_search:
            #print(".", end = "", flush=True)
            # delete the min branch since we've already visited it by now
            del t_[branch[0]]
            # path_ stores moves up to the depth we've explored
            path_ = branch[1][1]
            branch_lb = branch[1][2]
            # enter decision_tree again
            t_, incumbent_, solutions_, global_lb_ = \
                decision_tree(path_, t_, incumbent_, solutions_, global_lb_, branch_lb)
            # lower_bound_data.remove(temp_lb)
        ll += 1

# 
#     print("solution = ")
#     print("solutions type = ", type(solutions_))
#     print("solutions vehicle = ", solutions_[0], "centers = ", solutions_[1][0],solutions_[1][1])
#     print(solutions_)
    create_excel(solutions_)


def create_excel(solution):
#     # create file with solution
#     #     for i in range(len(solution)):
#     #         print(solution[i])
# 
    vehicle_dict = {}
    for i in range(len(solution)):
        #         print("tamanho =",len(solution[i]))
        #         print("type = ",type(solution[i]))
        #         print(isinstance(solution[i], str))
        #         print("solution = ",solution[i])
        if isinstance(solution[i], str):
            #             print("vehicle =",solution[i])
            if not solution[i] in vehicle_dict:
                vehicle_dict[solution[i]] = []
        else:
            #             print("vehicle_list =",solution[i][2])
            #             print("center =",solution[i][0])
            temp_centers = vehicle_dict[solution[i][2]]
            temp_centers.append(solution[i][0])
            vehicle_dict[solution[i][2]] = temp_centers
        #             print("car =",solution[i][2])
    #             print("list =",vehicle_dict[solution[i][2]])
    #
    routes = {}
    pen_route = {}
    count = 1
    name_0 = "Route "
    name = ""
    for k in vehicle_dict.keys():
        previous = origin
        for c in vehicle_dict[k]:
            #             print("car =",k)
            temp = [k, origin]
            #             print("list =",temp)
            if c == origin:
                name = name_0 + str(count)
                #                 print(name)
                count = count + 1
                routes[name] = temp
            #                 print(routes[name])
            else:
                #                 print("center =",c)
                #                 print("name =",name)
                #                 print(routes[name])
                temp_centers = routes[name]
                temp_centers.append(c)
                routes[name] = temp_centers
 
    #                 print("penalties =",list_pen_roads[int(Gamma[previous,c])-1])
 
#         print(routes)
    print("")
    print("writing output file")
  
    workbook = xlsxwriter.Workbook(output_name)
    worksheet = workbook.add_worksheet('Routes')
    gray = workbook.add_format({'bold': True})
    gray.set_bg_color('silver')
    gray.set_border()
    gray.set_text_wrap()
    gray.set_align('vcenter')
 
    bold = workbook.add_format({'bold': 1})
    bold.set_border()
    bold.set_text_wrap()
    bold.set_align('vcenter')
 
    simple = workbook.add_format()
    simple.set_border()
    simple.set_text_wrap()
    simple.set_align('vcenter')
 
    merge_format = workbook.add_format({'bold': 1})
    merge_format.set_border()
    merge_format.set_align('vcenter')
 
    worksheet.set_column(0, 30, 19)#(0, 17, 17)
 
    worksheet.write('A1', 'ROUTE DESCRIPTION:', gray)
    worksheet.merge_range('B1:D1', Description, merge_format)
    worksheet.write('F1', "OUTPUT SHEET 1 OF 2", gray)
    
    worksheet.merge_range('A3:C3', 'SUMMARY ROUTE AND PRODUCT DISTRIBUTION', gray)
    worksheet.write('A5', 'TOTAL DISTANCE (Km):', gray)
 
    worksheet.write('B5', 'TOTAL FUEL COST:', gray)
 
    worksheet.write('C5', 'TOTAL PER DIEM COST:', gray)
 
    worksheet.write('D5', 'TOTAL COST (FUEL + PER DIEM):', gray)
 
    worksheet.write('E5', 'TOTAL DOSES DELIVERED:', gray)
 
    worksheet.write('F5', 'COST PER DOSE:', gray)
    
    worksheet.merge_range('A8:C8', 'DETAILED ROUTE INFORMATION FOR '+str(len(routes)) +" ROUTES",
                           gray)
 
    row = 9
    col = 0
 
    total_distance = 0
    total_cost = 0
    total_doses_delivered = 0
    total_personnel = 0
    total_fuel = 0
 
    for key in routes.keys():
        route_distance = 0
        count = 1
 
        worksheet.write_string(row, col, 'ROUTE:', gray)
        worksheet.write_string(row + 1, col, key, bold)
 
        worksheet.write_string(row, col + 1, 'VEHICLE:', gray)
        worksheet.write_string(row + 1, col + 1, routes[key][0], bold)
 
        worksheet.write_string(row, col + 2, 'VEHICLE CONDITION:', gray)
        worksheet.write_string(row + 1, col + 2, list_pen_cars[int(Beta[routes[key][0]]) - 1], bold)
 
        worksheet.write_string(row, col + 3, 'DISTANCE FOR ROUTE (Km):', gray)
 
        worksheet.write_string(row, col + 4, 'FUEL COST FOR ROUTE:', gray)
 
        worksheet.write_string(row, col + 5, 'PER DIEM COST FOR ROUTE:', gray)
        worksheet.write_number(row + 1, col + 5, cost_personnel[routes[key][0]], simple)
 
        worksheet.write_string(row, col + 6, 'TOTAL DOSES DELIVERED:', gray)
        worksheet.write_string(row, col + 7, 'TOTAL COST PER DOSE:', gray)
 
        worksheet.write_string(row, col + 8, 'REFRIGERATED UTILIZATION OF VEHICLE (%):', gray)
 
        worksheet.write_string(row, col + 9, 'NON-REFRIGERATED UTILIZATION OF VEHICLE (%):', gray)
 
        temp_time = 8 * 60
        temp_hour = math.floor(temp_time / 60)
        temp_minutes = math.floor(temp_time % 60)
        temp_time_hours = str(temp_hour) + ' h ' + str(temp_minutes) + ' min'
        worksheet.write_string(row + 1, col + 11, temp_time_hours, simple)
 
        worksheet.write_string(row, col + 10, 'CENTERS:', gray)
 
        refri_util = 0
        dry_util = 0
        doses_vehicle = 0
 
        for i in range(1, len(routes[key])):
            # centers
            worksheet.write_string(row + i, col + 10, routes[key][i], simple)
 
            if i < len(routes[key]) - 1:
                # road penalties
                worksheet.write_string(row + i, col + 12,
                                       list_pen_roads[int(Gamma[routes[key][i], routes[key][i + 1]]) - 1],
                                       simple)
                # time
                temp_time = temp_time + h[routes[key][i], routes[key][i + 1], routes[key][0]] + W
                temp_hour = math.floor(temp_time / 60)
                temp_minutes = math.floor(temp_time % 60)
                temp_time_hours = str(temp_hour) + ' h ' + str(temp_minutes) + ' min'
                worksheet.write_string(row + i + 1, col + 11, temp_time_hours, simple)
 
                # distance
                route_distance = route_distance + dist[routes[key][i], routes[key][i + 1]]
                # quantity carried in volume
#                 refri_util = refri_util + r[routes[key][i + 1]]
                refri_util = refri_util + d[routes[key][i + 1],'cold']
#                 dry_util = dry_util + dry[routes[key][i + 1]]
                dry_util = dry_util + d[routes[key][i + 1],'dry']
 
                # total doses
                doses_vehicle = doses_vehicle + demand_doses[routes[key][i + 1]]
                total_doses_delivered = total_doses_delivered + demand_doses[routes[key][i + 1]]
 
        route_distance = route_distance + dist[routes[key][i], origin]
        total_distance = total_distance + route_distance
        worksheet.write_number(row + 1, col + 3, round(route_distance, 2), simple)
 
        ##vehicle utilization
        worksheet.write_number(row + 1, col + 8, round(refri_util * 100 /
                                                       c_cold[routes[key][0]], 2), simple)
#         worksheet.write_number(row + 2, col + 8, refri_util, simple)
#         worksheet.write_number(row + 3, col + 8, c_cold[routes[key][0]], simple)
        
        worksheet.write_number(row + 1, col + 9, round(dry_util * 100 /
                                                       c_dry[routes[key][0]], 2), simple)
        
#         worksheet.write_number(row + 2, col + 9, dry_util, simple)
#         worksheet.write_number(row + 3, col + 9, c_dry[routes[key][0]], simple)
 
        # route cost
        route_cost = route_distance * cost_km[routes[key][0]]
        worksheet.write_number(row + 1, col + 4, round(route_cost, 2), simple)  # fuel cost
 
        total_personnel = total_personnel + cost_personnel[routes[key][0]]
        total_fuel = total_fuel + route_distance * cost_km[routes[key][0]]
 
        total_cost = total_cost + route_distance * cost_km[routes[key][0]] + cost_personnel[routes[key][0]]
 
        # total doses per vehicle
        worksheet.write_number(row + 1, col + 6, round(doses_vehicle, 2), simple)
 
        # cost per dose per vehicle
        worksheet.write_number(row + 1, col + 7, round((route_cost + cost_personnel[routes[key][0]]) /
                                                       doses_vehicle, 2), simple)
 
        # penalties
        worksheet.write_string(row + len(routes[key]), col + 10, origin, simple)
        worksheet.write_string(row + len(routes[key]) - 1, col + 12,
                               list_pen_roads[int(Gamma[routes[key][i], origin]) - 1],
                               simple)
 
        worksheet.write_string(row, col + 11, 'TIME TO LEAVE THE CENTER:', gray)
 
        worksheet.write_string(row, col + 12, 'ROAD CONDITION:', gray)
 
        row = row + len(routes[key]) + 3
 
    # total distance
    worksheet.write('A6', round(total_distance, 2), bold)
    # total fuel cost
    worksheet.write('B6', round(total_fuel, 2), bold)  ####fuel
 
    #####personnel cost
    worksheet.write('C6', round(total_personnel, 2), bold)
 
    ### total cost
    worksheet.write('D6', round(total_cost, 2), bold)
 
    # total_doses
    worksheet.write('E6', round(total_doses_delivered, 2), bold)
    # cost/dose
    worksheet.write('F6', round(total_cost / total_doses_delivered, 2), bold)
 
    worksheet2 = workbook.add_worksheet('Products')
    worksheet2.set_column(0, 30, 19)
    worksheet2.write('A1', 'ROUTE DESCRIPTION:', gray)
    worksheet2.merge_range('B1:D1', Description, merge_format)
    worksheet2.write('F1', "OUTPUT SHEET 2 OF 2", gray)
    worksheet2.merge_range('A3:B3', 'PRODUCTS DELIVERED', gray)
 
    row = 4
    col = 0
 
    for key in routes.keys():
        worksheet2.write_string(row, col, 'ROUTE:', gray)
        worksheet2.write_string(row, col + 1, key, bold)
        worksheet2.write_string(row + 1, col, 'VEHICLE:', gray)
        worksheet2.write_string(row + 1, col + 1, routes[key][0], bold)
        worksheet2.write_string(row + 2, col, 'STARTING LOCATION:', gray)
        worksheet2.write_string(row + 2, col + 1, origin, bold)
        row = row + 4
        for i in range(2, len(routes[key])):
            worksheet2.write_string(row, col, 'CENTER:', gray)
            worksheet2.write_string(row, col + 1, routes[key][i], bold)
            worksheet2.write_string(row + 1, col, 'REFRIGERATED UTILIZATION AT HEALTH CENTER (%):', gray)
            worksheet2.write_number(row + 1, col + 1, round((d[routes[key][i], 'cold'] /
                                                             r[routes[key][i]]) * 100, 2), bold)
            worksheet2.write_string(row + 2, col, 'NON-REFRIGERATED UTILIZATION AT HEALTH CENTER (%):', gray)
            worksheet2.write_number(row + 2, col + 1, round((d[routes[key][i], 'dry'] /
                                                             dry[routes[key][i]]) * 100, 2), bold)
            worksheet2.write_string(row + 3, col, 'PRODUCT:', gray)
            worksheet2.write_string(row + 3, col + 1, 'QUANTITY (DOSE OR UNITS)', gray)
            row = row + 3
            for j in range(0, len(d_original[routes[key][i]][0])):
                row = row + 1
                worksheet2.write_string(row, col, d_original[routes[key][i]][0][j][0], simple)  ##product
                worksheet2.write_number(row, col + 1, d_original[routes[key][i]][0][j][1], simple)  ##demand
            row = row + 2
        row = 4
        col = col + 3
 
    workbook.close()
 
    root2 = Tk()
    root2.geometry('600x150+400+200')
    root2.title("Routing Tool")
 
    final_text = "Route Optimization Tool (RoOT)\n \nYour results are in:\n \n" + output_name
 
    msg = Message(root2, text=final_text, width=500, justify=CENTER)
 
    closing_button = Button(root2, height=1, width=10,
                            text="Close window", command=root2.destroy)
 
    msg.pack(pady=10)
 
    msg.pack()
    closing_button.pack()
 
    root2.mainloop()


if __name__ == '__main__':
    start = time.time()
    path = ''
    t = trie.StringTrie()
    incumbent = math.inf
    solutions = []
    global_lb = []
    global_lb.append(np.inf)
    t, incumbent, solutions, global_lb = decision_tree(path, t, incumbent, solutions, global_lb, np.inf)
    traverse(t, start, incumbent, solutions, global_lb)
    end = time.time()
    print('time', end - start)

############changes on #329 and
