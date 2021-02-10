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
import matplotlib.dates as mdates 
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
file_s4 = "ljic_200806.s4" # Test file 

class ScintillationPlot():
    def __init__(self):
        self.pi = 3.14
    
    def read_s4_file(self, input_file):
        self.input_file = input_file
        self.df = pd.read_csv(self.input_file, header=None, sep="\t")
        
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

        print("df ready to plot!")
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
    def _convert2SVID(self, prn='G10'):
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
    def _convert_GPS2SBAS_frequency(self, freq='S4_sig1'):
        if freq == 'S4_sig1': return freq
        elif freq == 'S4_sig3': return 'S4_sig2'
        
    # Append SBAS prns into another prns list 
    def _append_sbas_prns(self, const, freq, PRNs):
        const_sbas = 'S'
        if const == 'G':
            while freq != 'S4_sig2':
                freq_sbas = self._convert_GPS2SBAS_frequency(freq)
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
        
    # Change frequency for SBAS const
    def _change_frequency(self, const='G', freq='S4_sig1'):
        if const == 'G':
            return self._convert_GPS2SBAS_frequency(freq)
        elif const == 'E':
            return freq
        else:
            return freq
        
    # Get figure name
    def _figure_name(self):
        file_name = self.input_file[len(input_files_path):]
        figure_name = file_name[:-3]  
        return figure_name

    # Check no null column in the frequency column
    def _check_noNull_values(self, const, freq):
        mask = self.df2["PRN"].str.contains(const)
        df_aux = self.df2[mask]
        if df_aux[freq].isna().sum() < len(df_aux):
            return True 
        else:
            return False

    # Plot s4 and elevation info: there is a subplot for each PRN 
    def plot1_s4(self, const='G', freq='S4_sig1', sbas=False):
        if self._check_noNull_values(const, freq): 
            # Get file UTC date
            figure_name = self._figure_name()
            fecha = figure_name[5:] # e.g. 200926
            fecha2 = datetime.datetime.strptime(fecha, "%y%m%d")
            fecha3 = datetime.datetime.strftime(fecha2,"%Y/%m/%d")

            fecha2_tomorrow = fecha2 + pd.DateOffset(days=1)
            fecha2_tomorrow = fecha2_tomorrow.to_pydatetime()

            # Get UTC day range, to add a vertical strip
            fecha_morning_first = fecha2 + pd.DateOffset(hours=11) 
            fecha_morning_first = fecha_morning_first.to_pydatetime()
            
            fecha_morning_last = fecha2 + pd.DateOffset(hours=23)
            fecha_morning_last = fecha_morning_last.to_pydatetime()

            # Get the PRNs
            PRNs = self.extract_prns(const, freq)
            
            # Include SBAS data if corresponds
            if sbas: PRNs = self._append_sbas_prns(const, freq, PRNs)
            
            # Create the figure with the subplots 
            n_rows = (len(PRNs)+1)//2
            n_cols = 2
            
            fig, axs = plt.subplots(n_rows, n_cols, figsize=(7*n_cols,1*n_rows), sharex="col", sharey="row",
                            gridspec_kw={'hspace': 0, 'wspace': 0})   
            j = 0

            for ax in axs.T.reshape(-1): # Plot up to down, rather than left to right 
                # ax -> s4
                # ax2 -> elevation
                ax2 = ax.twinx()
                
                if j < len(PRNs):
                    # Plot s4 info
                    prn_value = PRNs[j]
                    
                    # -> Get the correct freq if sbas==True
                    if sbas and prn_value[0]=='S': 
                        freq_n = self._change_frequency(const, freq)
                    else: freq_n = freq
                        
                    df3_s4 = self.get_s4(prn_value, freq_n)
                    
                    color1 = "blue" # This color is used in y axis labels, ticks and border  
                    colors1 = ["lightsteelblue", "cornflowerblue", "navy"] # These colors are used for the plots

                    for k in range(3):
                        df4_s4 = df3_s4[k+1]

                        ax.plot(df4_s4.index, df4_s4.values, '.', color=colors1[k], markersize=2)
                        ax.set_facecolor(color="lightgrey")
                        ax.axvspan(fecha_morning_first, fecha_morning_last, color="white") # strip morning/night
                    
                    # Plot elevation info
                    df3_elev = self.get_elevation(PRNs[j], freq)
                    
                    color2 = "orange"
                    ax2.plot(df3_elev.index, df3_elev.values, '.', color=color2, markersize=1)
                    
                    # Annotate the prn in the subplot
                    x_location = fecha2 + pd.Timedelta(minutes=30)
                    ax2.text(x_location, 35, self._convert2SVID(PRNs[j]), fontsize=15, weight='roman') # 0.375

                # Set axis limits 
                ax.set_xlim([fecha2, fecha2_tomorrow])
                ax.set_ylim([0,1])
                ax2.set_ylim([0,90])

                # Set ticks and tick labels 
                # Set y axis format, labels odds subplots only
                len_half_ax = len(axs.T.reshape(-1))/2

                if j >= len_half_ax: # change only for the 2nd column
                    k=j-len_half_ax

                    # Set y labels only to even subplots
                    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
                    ax.set_yticks([0,1])
                    ax2.yaxis.set_minor_locator(AutoMinorLocator(4))
                    ax2.set_yticks([0,90])

                    if k%2 == 0: 
                        ax.set_yticklabels([0,1])
                        ax2.set_yticklabels([0,90])
                    else:    
                        ax.set_yticklabels(['',''])
                        ax2.set_yticklabels(['',''])

                    # Set yellow color to the right y axis
                    for axis in ['top','bottom','left']:
                        ax.spines[axis].set_linewidth(2)
                        ax2.spines[axis].set_linewidth(2)

                    ax.spines['right'].set_color(color2)
                    ax.spines['right'].set_linewidth(2)
                    ax2.spines['right'].set_color(color2)
                    ax2.spines['right'].set_linewidth(2)
                    ax2.tick_params(axis='y', which='both', colors=color2)

                else: # apply some changes to the 1st column 
                    # remove y tick labels for elevation 
                    ax2.yaxis.set_minor_locator(AutoMinorLocator(4))
                    ax2.set_yticks([0,90])
                    ax2.set_yticklabels(['',''])

                    # set linewidth to top, bottom and right borders of the subplot
                    for axis in ['top','bottom','right']:
                        ax.spines[axis].set_linewidth(2)
                        ax2.spines[axis].set_linewidth(2)

                    # Set blue color to the left y axis
                    ax.spines['left'].set_color(color1)
                    ax.spines['left'].set_linewidth(2)
                    ax2.spines['left'].set_color(color1)
                    ax2.spines['left'].set_linewidth(2)
                    ax.tick_params(axis='y', which='both', colors=color1)

                # set x axis format 
                hours = mdates.HourLocator(interval = 2)
                ax.xaxis.set_major_locator(hours) # ticks interval: 2h
                ax.xaxis.set_minor_locator(AutoMinorLocator(2)) # minor tick division: 2
                myFmt = DateFormatter("%H")
                ax.xaxis.set_major_formatter(myFmt) # x format: hours 
                
                # set the ticks style 
                ax.xaxis.set_tick_params(width=2, length=8, which='major', direction='out')
                ax.xaxis.set_tick_params(width=1, length=4, which='minor', direction='out')
                ax.yaxis.set_tick_params(width=2, length=15, which='major', direction='inout')
                ax.yaxis.set_tick_params(width=1, length=4, which='minor', direction='out')
                ax2.yaxis.set_tick_params(width=2, length=15, which='major', direction='inout')
                ax2.yaxis.set_tick_params(width=1, length=4, which='minor', direction='out')

                # set the label ticks 
                ax.tick_params(axis='x', which='major', labelsize=12)
                ax.tick_params(axis='y', labelsize=12)
                ax2.tick_params(axis='y', labelsize=12)
                
                # set grid
                ax.grid(which='major', axis='both', ls=':', linewidth=1.2)
                ax.grid(which='minor', axis='both', ls=':', alpha=0.5)
                
                j += 1

            # Set title and axis labels 
            aux = self.get_freq_name(const, int(freq[-1]))
            frequency_name = aux["name"]
            frequency_value = aux["value"] + "MHz"
            # -> Labels
            fig.text(0.513, 0.08, 'Time UTC', ha='center', va='center', fontsize=14)
            fig.text(0.09, 0.5, 'S4', ha='center', va='center', rotation='vertical', fontsize=14, color='b')
            fig.text(0.94, 0.5, 'Elevation Angle', ha='center', va='center', rotation=-90, fontsize=14, color=color2)
            # -> Title
            fig.text(0.513, 0.895, 'S4', ha='center', va='center', fontsize=17, weight='roman')
            fig.text(0.32, 0.895, 'Jicamarca', ha='center', va='center', fontsize=17, weight='roman', color='r')
            fig.text(0.12, 0.895, fecha3, ha='left', va='center', fontsize=17, weight='roman')
            fig.text(0.9, 0.895, f"{frequency_name} | {self.get_const_name(const)}", ha='right', va='center', fontsize=17, weight='roman')
            fig.text(0.72, 0.895, frequency_value, ha='right', va='center', fontsize=17, weight='roman')
                
            # Create directory for output files
            new_directory = output_files_path + figure_name + "/plot_2/"
            if not os.path.exists(new_directory):
                os.makedirs(new_directory)    

            # Save figure as pdf
            figure_name2 = figure_name + f"_s4_{self.get_const_name(const)}_{frequency_name}.pdf"
            plt.savefig(new_directory + figure_name2, bbox_inches='tight')
            
            print(f"Plotted successfully; for const: {const}, and freq: {freq}!")
        else:
            print(f"There is only Null data; for const: {const}, and freq: {freq}!") 

        return 'Ok'

def main():
    # Specify the consts and freqs to plot 
    const_list = ['G', 'E', 'S'] #  Constelations list  
    freq_list = ['S4_sig1', 'S4_sig2', 'S4_sig3'] # Frecuencies list

    list_input_files = glob.glob(input_files_path + "*.s4")
    if len(list_input_files) > 0:
        for file_i in list_input_files:
            g = ScintillationPlot()
            g.read_s4_file(file_i)
            g.process_dataframe()
            g.filter_dataframe() # Dataframe ready to plot 

            # Plot 
            for c in const_list:
                for f in freq_list:         
                    g.plot1_s4(const=c, freq=f, sbas=True)    

            # Move input files to a permanent directory
            file_name = file_i[len(input_files_path):]
            os.rename(file_i, input_files_path_op+file_name)
    
    return 'Ok'

if __name__ == '__main__':
    main()
