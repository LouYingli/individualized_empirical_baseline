# -*- coding: utf-8 -*-
"""
Created on Thu May 12 14:41:35 2022

@author: YingliLou
"""

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

import numpy as np
import pandas as pd


data = np.array(pd.read_csv('./database/energy_data.csv',skiprows=0))
KeyVariable = np.array(pd.read_csv('key_neutral_building_characteristics.csv',skiprows=0))
sample = np.array(pd.read_csv('input_file.csv',skiprows=0)).T
sample = pd.DataFrame(list(sample[2]),
                      index =['Climate','Total_floor_area', 'Aspect_ratio','Floor_to_floor_height',
                              'Window_to_wall_ratio','Building_orientation','People_density',
                                'Heating_setpoint_temperature','Cooling_setpoint_temperature',
                                'Service_water_usage','Heating_humidity_ratio','Cooling_humidity_ratio',
                                'Weekly_operation_hour','Electric_equipment_power_density','Ventilation'])
climate = sample.loc['Climate'][0]
sample = sample.transpose()

climateList = ['pre1A','pre2A','pre2B','pre3A','pre3B','pre3C','pre4A',
           'pre4B','pre4C','pre5A','pre5B','pre6A','pre6B','pre7','pre8',
           'post1A','post2A','post2B','post3A','post3B','post3C','post4A',
           'post4B','post4C','post5A','post5B','post6A','post6B','post7','post8']
index = climateList.index(climate)
KeyVariable = list(KeyVariable[index,1:])
KeyVariable = [item for item in KeyVariable if not(pd.isnull(item)) == True]

clf = ExtraTreesRegressor()

Input1=[] #FloorArea m2
Input2=[] #AspectRatio 
Input3=[] #Floor-to-Floor-height  m
Input4=[] #People-density person/m2
Input5=[] #Htg C
Input6=[] #Clg C
Input7=[] #ServiceWaterUsage # gpm default 0.4945*3.79 lpm
Input8=[] #kg-H2O/kg-air
Input9=[] #kg-H2O/kg-air
Input10=[] #OperationHour hours/week
Input11=[] #ReduceElectricEquipmentLoadsByPercentage   W/m2  Pre-1980: 13.08 W/m2   Post-1980: 8.19 W/m2
Input12=[] #wwr
Input13=[] #ventilation m3/s-person 
Input14=[] #Rotationr Degree
Output=[] # site energy  MJ/m2
for i in range(len(data)):
    if data[i][1] == climate:
        Input1.append(data[i][2])
        Input2.append(data[i][3])
        Input3.append(data[i][4])
        Input4.append(data[i][5])
        Input5.append(data[i][6])
        Input6.append(data[i][7])
        Input7.append(data[i][8])
        Input8.append(data[i][9])
        Input9.append(data[i][10])
        Input10.append(data[i][11])
        Input11.append(data[i][12])
        Input12.append(data[i][13])
        Input13.append(data[i][14])
        Input14.append(data[i][15])
        Output.append(data[i][17])
temp=pd.DataFrame(list(zip(Input1,Input2,Input3,Input4,Input5,Input6,Input7,Input8,
                           Input9,Input10,Input11,Input12,Input13,Input14,Output)),
    columns =['Total_floor_area', 'Aspect_ratio','Floor_to_floor_height',
              'Window_to_wall_ratio','Building_orientation','People_density',
              'Heating_setpoint_temperature','Cooling_setpoint_temperature',
              'Service_water_usage','Heating_humidity_ratio','Cooling_humidity_ratio',
              'Weekly_operation_hour','Electric_equipment_power_density','Ventilation','SiteEnergy'])
            
model=clf.fit(temp[KeyVariable], temp['SiteEnergy'])
predict = clf.predict(sample[KeyVariable])

text_file = open("individualized_empirical_baseline.txt", "w")
n = text_file.write(str(np.round(predict[0],2))+' MJ/m²-yr')
text_file.close()

 
