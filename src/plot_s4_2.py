#!/home/luis/anaconda3/bin/python3
#__________________________________
#           PLOT S4
#             v4.0
#----------------------------------
# This program makes some graphs of 
# 's4' variable, for each frequency
# and constellation. SBAS data is 
# included in GPS & GALILEO graphs. 
# Author: Luis D.
# :)

from matplotlib.dates import DateFormatter, HourLocator
from matplotlib.ticker import AutoMinorLocator
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import glob
import os 

# Declare variables
root_path = "/home/luis/Desktop/Proyects_Files/LISN/GPSs/Tareas/Graficas_cintilaciones/"
input_files_path = root_path + "Input_data/Data_set/"
input_files_path_op = root_path + "Input_data/Data_procesada/"
output_files_path = root_path + "Output_data/"
file_s4 = "ljic_200806.s4"

class ScintillationPlot():
    def __init__(self):
        self.pi = 3.14
    
    def read_s4_file(self, input_file):
        self.df = pd.read_csv(input_file, header=None, sep="\t")
        print("Done!")
        return self.df
    
    # Extract s4 info, for each frequency and constellation
    def process_dataframe(self):
        # Delete columns without s4, elevation and azimuth info
        for i in self.df.columns:
            if i>3:
                m = (i-4)%24
                if m >= 9: del self.df[i]
        
        # Reindex the columns 
        self.df.columns = np.arange(len(self.df.columns)) 

        # A new df is created and filled 
        df2 = pd.DataFrame(columns=range(12))

        T = 9 # Period of dataset
        for i in range(len(self.df)):
            for h in range(self.df.iloc[i,3]): # that cell contains the number of measurements 
                m = self.df.iloc[i, [0, 1, 2, 4+T*h, 5+T*h, 6+T*h, 7+T*h, 8+T*h, 9+T*h, 10+T*h, 11+T*h, 12+T*h]]
                m.index = range(len(m))
                df2 = df2.append(m)

        df2.index = range(len(df2))

        # Change datatype: str -> int 
        for i in range(3):
            df2[i] = df2[i].astype("int")

        # Change datatype: str -> float
        for i in range(6):
            df2[6+i] = df2[6+i].astype("str").str.strip().astype("float")

        # Rename columns 
        df2.columns = ["YY", "DOY", "SOD", "PRN", "Azimuth", "Elevation", "S4_sig1", "S4_sig1_corr", "S4_sig2", 
                        "S4_sig2_corr", "S4_sig3", "S4_sig3_corr"]

        # Calculate the corrected S4
        # These values are calculated according Septentrio manual, section ISMR
        def get_correctedS4(row):
            s4 = row[0]
            correction= row[1]
            
            # Treat nan numbers 
            if pd.isnull(s4) or pd.isnull(correction):
                return np.nan
            else:
                # Calculate the corrected S4
                x = s4**2-correction**2
                if x>0:
                    return x**0.5
                else:
                    return 0    

        for i in range(3):        
            df2[f"S4_sig{i+1}"] = df2[[f"S4_sig{i+1}",f"S4_sig{i+1}_corr"]].apply(
                get_correctedS4, axis=1)

        # Delete some columns
        del df2["S4_sig1_corr"]
        del df2["S4_sig2_corr"]
        del df2["S4_sig3_corr"]

        # Sort values by "PRN" and "SOD" 
        df2 = df2.sort_values(["PRN","SOD"])

        # Reindex 
        df2.index = range(len(df2))

        # Convert to datetime data type 
        def change2datetime(row):
            yy = int(row[0])
            doy = int(row[1])
            sod = int(row[2])
            
            if sod < 0: 
                doy = doy -1
                sod = 60*60*24+sod      
            
            cdate = str(yy)+"-"+str(doy)+"-"+str(datetime.timedelta(seconds=sod))
            fecha = datetime.datetime.strptime(cdate, "%y-%j-%X")
            
            return fecha  

        m = df2.apply(change2datetime, axis=1)

        # Create a new column
        df2.insert(0,column="DateTime",value=0)
        df2["DateTime"] = m

        # Delete some columns 
        del df2["YY"]
        del df2["DOY"]
        del df2["SOD"]

        # Datetime as index
        df2.set_index("DateTime", inplace=True)
        self.df2 = df2.copy()

        print("Done!")

        return self.df2
    
    # Filter S4 data based on the angle and the S4 value
    def filter_dataframe(self):
        # Aux function
        def filter_elev_s4(row):
            elev = row[0]
            s4 = row[1]
            threshold_s4 = 0.3
            threshold_elev = 35 # Unit: º
            
            if elev < threshold_elev:
                return [s4, np.nan, np.nan]
            elif s4 < threshold_s4:
                return [np.nan, s4, np.nan]
            else:
                return [np.nan, np.nan, s4]

        # Create 3 additional columns per each S4_sigx column
        for j in range(3):
            j += 1
            df_aux = self.df2[["Elevation", f"S4_sig{j}"]].apply(filter_elev_s4, axis=1, result_type="expand")
            df_aux.rename(columns = {0:f"S4_sig{j}_1", 1:f"S4_sig{j}_2", 2:f"S4_sig{j}_3"}, inplace=True)
            self.df2 = pd.concat([self.df2, df_aux], join='inner', axis=1)

        print("Done!")
        return self.df2    

    ####-Plot methods     
    # Identify the available constellations 
    def extract_const(self):
        const = self.df2["PRN"].str[0].unique() # extract the first character of each cell 
        return const

    # Extract PRNs of a constellation and freq, in which there is no null data    
    def extract_prns(self, const='G', freq='S4_sig1'): # const: char (e.g. 'G')
        prns = self.df2["PRN"].unique().tolist()
        PRNs = [value for value in prns if const in value]
        PRNs.sort(key=lambda x: int(x[1:])) # sort in ascendent order 
        
        # Check no null columns in the prns
        prn_values = []
        for value in PRNs:
            mask = self.df2["PRN"] == value
            df_test = self.df2[mask]
            if df_test[freq].isna().sum() < len(df_test): # when the column is not null 
                prn_values.append(value)
        
        return prn_values

    # Extract S4 data
    def get_s4(self, prn='G10', freq='S4_sig1'): # prn: str (e.g. 'C14') ; freq: str (e.g. 'S4_sig1')  
        mask = self.df2["PRN"] == prn
        df_aux = self.df2[mask]
        df_final_1 = df_aux[freq + "_1"]
        df_final_2 = df_aux[freq + "_2"]
        df_final_3 = df_aux[freq + "_3"]
        
        return {1: df_final_1, 2: df_final_2, 3: df_final_3} 
        
    # Extract elevation info
    def get_elevation(self, prn='G10', freq='S4_sig1'):
        mask = self.df2["PRN"] == prn
        df_aux = self.df2[mask]
        df_final = df_aux["Elevation"]
        
        return df_final

    # Convert SBAS code to SVID (number only)
    def convert2SVID(self, prn='G10'):
        if prn[0] == "S":
            nn = int(prn[1:])
            if 20 <= nn <= 40:
                return str(nn + 100)
            elif 41 <= nn <= 58:
                return str(nn + 157)
        else:
            return prn
        
    # Get the frequency name and value for a given PRN code and Freq code
    def get_freq_name(self, const='G', freq_code=1):
        if freq_code == 1:
            if const == 'G':
                return {"name":'L1CA', "value":"1575.42"}
            elif const == 'R':
                return {"name":'L1CA', "value":"1602"} # change 
            elif const == 'S':
                return {"name":'L1CA', "value":"1575.42"}
            elif const == 'J':
                return {"name":'L1CA', "value":"1575.42"}
            elif const == 'E':
                return {"name":'L1BC', "value":"1575.42"}
            elif const == 'C':
                return {"name":'B1', "value":"1575.42"}
            elif const == 'I':
                return {"name":'B1', "value":"1176.45"}
            else: 
                return "Insert a right code!"
        elif freq_code == 2:
            if const == 'G':
                return {"name":'L2C', "value":"1227.60"}
            elif const == 'R':
                return {"name":'L2C', "value":"1246"} # change 
            elif const == 'J':
                return {"name":'L2C', "value":"1227.60"}
            elif const == 'E':
                return {"name":'E5a', "value":'1176.45'}
            elif const == 'C':
                return {"name":'B2', "value":'1176.45'}
            elif const == 'S':
                return {"name":'L5', "value":'1176.45'}
            else: 
                return "Insert a right code!"
        elif freq_code == 3:
            if const == 'G':
                return {"name":'L5', "value":'1176.45'}
            elif const == 'J':
                return {"name":'L5', "value":'1176.45'}
            elif const == 'E':
                return {"name":'E5b', "value":'1207.14'}
            elif const == 'C':
                return {"name":'B3', "value":'1268.52'}
            else: 
                return "Insert a right code!"
        else:
            return "Insert a right code!"
        
    # Get the name for a given constelation code
    def get_const_name(self, const='G'):
        if const == 'G': return 'GPS'
        elif const == 'R': return 'GLONASS'
        elif const == 'E': return 'GALILEO'
        elif const == 'S': return 'SBAS'
        elif const == 'C': return 'BEIDOU'
        elif const == 'J': return 'QZSS'
        elif const == 'I': return 'IRNSS'
        else:
            return 'Incorrect PRN code!'

    # Convert GPS into SBAS frequencies    
    def convert_GPS2SBAS_frequency(self, freq='S4_sig1'):
        if freq == 'S4_sig1': return freq
        elif freq == 'S4_sig3': return 'S4_sig2'
        
    # Append SBAS prns into another prns list 
    def append_sbas_prns(self, const, freq, PRNs):
        const_sbas = 'S'
        if const == 'G':
            while freq != 'S4_sig2':
                freq_sbas = self.convert_GPS2SBAS_frequency(freq)
                PRNs_SBAS = self.extract_prns(const_sbas, freq_sbas)
                PRNs += PRNs_SBAS
                break
            return PRNs
        elif const == 'E':
            while freq != 'S4_sig3':
                freq_sbas = freq
                PRNs_SBAS = self.extract_prns(const_sbas, freq_sbas)
                PRNs += PRNs_SBAS
                break
            return PRNs
        else:
            return PRNs
        
    


G1 = ScintillationPlot()
G1.read_s4_file(input_files_path+file_s4)
G1.process_dataframe()
df1 = G1.filter_dataframe()
#print(df1.head())
#print(df1.columns)

n_const = G1.extract_const()
print(f"Available constellations: ")
print(n_const)

n_prns = G1.extract_prns(const='G')
#print(f"Available PRNs: ")
#print(n_prns)

s4_data = G1.get_s4(freq='S4_sig2')
s4_1 = s4_data[1].head()
#print("The values of S4_sig1_1 are:")
#print(s4_1)

elevation = G1.get_elevation()
#print("The elevation values are: ")
#print(elevation.head())

svid = G1.convert2SVID(prn="S45")
print("The SVID is:")
print(svid)

freq_name = G1.get_freq_name()
print("The freq name is:")
print(freq_name["name"])

const_name = G1.get_const_name('S')
print("The const name is:")
print(const_name)

new_freq = G1.convert_GPS2SBAS_frequency('S4_sig1')
print("The new frequency is: ")
print(new_freq)

prns_2 = G1.append_sbas_prns(const="G", freq='S4_sig1', PRNs=n_prns)
print("The new prns list is:")
print(prns_2)