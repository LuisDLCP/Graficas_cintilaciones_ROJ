#!/home/luis/anaconda3/bin/python3
#__________________________________
#           PLOT S4
#             v3.0
#----------------------------------
# This program makes some graphs of 
# 's4' variable, for each frequency
# and constellation.
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

# Declare functions
# Read s4 file as dataframe
def read_s4_file(input_file):
    df = pd.read_csv(input_file, header=None, sep="\t")
    return df

# Extract s4 info, for each frequency and constellation
def process_dataframe(df):
    # Delete columns without s4, elevation and azimuth info
    for i in df.columns:
        if i>3:
            m = (i-4)%24
            if m >= 9: del df[i]
    
    # Reindex the columns 
    df.columns = np.arange(len(df.columns)) 

    # A new df is created and filled 
    df2 = pd.DataFrame(columns=range(12))

    T = 9 # Period of dataset
    for i in range(len(df)):
        for h in range(df.iloc[i,3]): # that cell contains the number of measurements 
            m = df.iloc[i, [0, 1, 2, 4+T*h, 5+T*h, 6+T*h, 7+T*h, 8+T*h, 9+T*h, 10+T*h, 11+T*h, 12+T*h]]
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

    return df2

# Filter S4 data based on the angle and the S4 value
def filter_dataframe(df):
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
        df_aux = df[["Elevation", f"S4_sig{j}"]].apply(filter_elev_s4, axis=1, result_type="expand")
        df_aux.rename(columns = {0:f"S4_sig{j}_1", 1:f"S4_sig{j}_2", 2:f"S4_sig{j}_3"}, inplace=True)
        df = pd.concat([df, df_aux], join='inner', axis=1)

    return df    

# Plot s4 graph 01: all PRNs are in the same subplot
def plot1_s4(df, figure_name):
    # Identify the available PRNs
    prns = df["PRN"].str[0].unique() # extract the first character of each cell 
    
    # Get file UTC date
    fecha = figure_name[5:] # e.g. 200926
    fecha2 = datetime.datetime.strptime(fecha, "%y%m%d")
    fecha3 = datetime.datetime.strftime(fecha2,"%Y/%m/%d")

    fecha2_tomorrow = fecha2 + pd.DateOffset(days=1)
    fecha2_tomorrow = fecha2_tomorrow.to_pydatetime()

    # Create directory for output files
    new_directory = output_files_path + figure_name + "/plot_1/"
    if not os.path.exists(new_directory):
        os.makedirs(new_directory)

    # Plot for each PRN and freq code
    for prn in prns:
        mask = df["PRN"].str.contains(prn)
        df3 = df[mask]
        
        # Check no null columns 
        sig_n = ['S4_sig1', 'S4_sig2', 'S4_sig3'] # Extract only the columns which have freq info
        n_subplots = 0
        for sig_i in sig_n:
            if df3[sig_i].isna().sum() < len(df3):
                n_subplots += 1
                
        # Plot for each PRN
        fig,ax = plt.subplots(n_subplots, figsize=(10,4*n_subplots))
        for i in range(len(ax)):
            sig = i+1 # frequency 
            first = 0 # first graph 
            last = len(ax) - 1 # last graph
            
            # Plot data by prn and s4_sig
            prn_values = df3["PRN"].unique().tolist()
            prn_values.sort(key=lambda x: int(x[1:]))

            for value in prn_values:
                mask = df3["PRN"] == value
                df4 = df3[mask]["S4_sig" + str(sig)]
                ax[i].plot(df4.index, df4.values, '.', label= value)            
            
            #df3.groupby("PRN")["S4_sig"+str(sig)].plot(ax=ax[i], style='.')

            if i == first:
                ax[i].set_title("Jicamarca", fontsize=20, fontweight='bold')
                ax[i].set_title(fecha3, loc="left", fontsize=14, style='normal', name='Ubuntu')

            ax[i].set_title(get_sig_name(prn, sig)["name"] + f" | {get_prn_name(prn)}", loc="right",
                            fontsize=14, name = 'Ubuntu')
            ax[i].set_ylabel("S4", fontsize=14, weight='bold', color='gray')
            #
            if i == last:
                ax[i].set_xlabel("Time UTC", fontsize=14, weight='bold', color='gray')
            else:
                ax[i].set_xlabel("", fontsize=14, weight='bold', color='gray')    

            # Set axis limits 
            ax[i].set_xlim([fecha2, fecha2_tomorrow])
            ax[i].set_ylim([0,1])
            
            # Rectangle frame width 
            for axis in ['top','bottom','left','right']:
                ax[i].spines[axis].set_linewidth(1.5)

            # Ticks format 
            ax[i].xaxis.set_tick_params(width=2, length=8, which='both', direction='out')
            ax[i].yaxis.set_tick_params(width=2, length=15, direction='inout')
            ax[i].tick_params(axis='x', which='both', labelsize=12)
            ax[i].tick_params(axis='y', labelsize=12)

            myFmt = DateFormatter("%H:%M")
            ax[i].xaxis.set_major_formatter(myFmt)
            ax[i].xaxis.set_minor_formatter(myFmt)

            # Grid and legend 
            ax[i].grid(which='both', axis='both', ls=':')
            if i == last:
                ax[i].legend(loc='upper center', fancybox=True, shadow=True, 
                                    bbox_to_anchor=(0.5, -0.27), ncol=8, facecolor='whitesmoke')

        plt.subplots_adjust(hspace=0.3)
        
        # Save figure as pdf
        figure_name2 = figure_name + f"_s4_{get_prn_name(prn)}.pdf"
        plt.savefig(new_directory + figure_name2, bbox_inches='tight')

    return "Plotted succesfully!"    

# Plot s4 graph 02: there is a subplot for each PRN
def plot2_s4(df, figure_name):
    # Identify the available PRNs
    prns = df["PRN"].str[0].unique() # extract the first character of each cell 
    
    # Get file UTC date
    fecha = figure_name[5:] # e.g. 200926
    fecha2 = datetime.datetime.strptime(fecha, "%y%m%d")
    fecha3 = datetime.datetime.strftime(fecha2,"%Y/%m/%d")

    fecha2_tomorrow = fecha2 + pd.DateOffset(days=1)
    fecha2_tomorrow = fecha2_tomorrow.to_pydatetime()

    # Get UTC day range, for add a vertical span
    fecha_morning_first = fecha2 + pd.DateOffset(hours=11) 
    fecha_morning_first = fecha_morning_first.to_pydatetime()
    
    fecha_morning_last = fecha2 + pd.DateOffset(hours=23)
    fecha_morning_last = fecha_morning_last.to_pydatetime()
    fecha_morning_last

    # Create directory for output files
    new_directory = output_files_path + figure_name + "/plot_2/"
    if not os.path.exists(new_directory):
        os.makedirs(new_directory)    

    # Iterate for each prn constelation: e.g. G, E, S, R
    for prn in prns:
        mask = df["PRN"].str.contains(prn)
        df3 = df[mask]

        # Check no null columns in the frequencies
        sig_n_aux = ['S4_sig1', 'S4_sig2', 'S4_sig3'] 
        sig_n = []
        for value in sig_n_aux:
            if df3[value].isna().sum() < len(df3):
                sig_n.append(value)
        
        # Iterate for each frequency: e.g. sig1, sig2, sig3
        for sig_i in sig_n:

            # Select the available satellites: C1, C2, etc.
            prn_values_aux = df3["PRN"].unique().tolist()
            prn_values_aux.sort(key=lambda x: int(x[1:]))
            i = 0 # prn_values index
            
            # Check no null columns in the satellites
            prn_values = []
            for value in prn_values_aux:
                mask = df3["PRN"] == value
                df_test = df3[mask]
                if df_test[sig_i].isna().sum() < len(df_test): # when the column is not null 
                    prn_values.append(value)

            # Create the figure with the subplots 
            n_rows = (len(prn_values)+1)//2
            n_cols = 2
            fig, axs = plt.subplots(n_rows, n_cols, figsize=(7*n_cols,1*n_rows), sharex="col", 
                                    sharey="row", gridspec_kw={'hspace': 0, 'wspace': 0})   
            
            # Iterate for each PRN element: e.g. G1, G2, etc.
            for ax in axs.T.reshape(-1): # Plot up to down, rather than left to right 
                # ax -> s4
                # ax2 -> elevation
                ax2 = ax.twinx()
                
                if i < len(prn_values):
                    mask = df3["PRN"] == prn_values[i]
                    df_aux = df3[mask]
                    
                    # Plot elevation info
                    color2 = "orange"
                    df5 = df_aux["Elevation"]
                    ax2.plot(df5.index, df5.values, '.', color=color2, linewidth=0.1)
                    
                    # Plot s4 info
                    color1 = "blue" # This color is used in y axis labels, ticks and border  
                    colors1 = ["lightsteelblue", "cornflowerblue", "navy"] # These colors are used for the plots 
                    for k in range(3):
                        idt = "_" + str(k+1)
                        df4 = df_aux[sig_i + idt] # Select the filtered data: S4_sig1_1, S4_sig1_2, S4_sig1_3
                        ax.plot(df4.index, df4.values, '.', color=colors1[k])
                        # Plot the vertical frame 
                        ax.set_facecolor(color="lightgrey")
                        ax.axvspan(fecha_morning_first, fecha_morning_last, color="white")
        
                    # Annotate the prn number inside the subplot
                    x_location = fecha2 + pd.Timedelta(minutes=30)
                    ax2.text(x_location, 35, prn_values[i], fontsize=15, weight='roman')
                    
                # Set axis limits 
                ax.set_xlim([fecha2, fecha2_tomorrow])
                ax.set_ylim([0,1])
                ax2.set_ylim([0,90])
                
                # Set ticks and tick labels 
                # __ Set y axis format and labels; odds subplots only
                len_half_ax = len(axs.T.reshape(-1))/2
                
                if i >= len_half_ax: # change only for the 2nd column
                    j=i-len_half_ax
                    
                    # Set y labels only to even subplots
                    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
                    ax.set_yticks([0,1])
                    ax2.yaxis.set_minor_locator(AutoMinorLocator(4))
                    ax2.set_yticks([0,90])
                    
                    if j%2 == 0: 
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
                hours = HourLocator(interval = 2)
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
                                        
                # Set title and axis labels 
                aux = get_sig_name(prn, int(sig_i[-1]))
                frequency_name = aux["name"]
                frequency_value = aux["value"] + "MHz"
                
                if prn != 'S':
                    #labels
                    fig.text(0.513, 0.08, 'Time UTC', ha='center', va='center', fontsize=14)
                    fig.text(0.09, 0.5, 'S4', ha='center', va='center', rotation='vertical', 
                            fontsize=14, color='b')
                    fig.text(0.94, 0.5, 'Elevation Angle', ha='center', va='center', rotation=-90,
                            fontsize=14, color=color2)
                    # title
                    fig.text(0.513, 0.895, 'S4', ha='center', va='center', fontsize=17, weight='roman')
                    fig.text(0.32, 0.895, 'Jicamarca', ha='center', va='center', fontsize=17, 
                            weight='roman', color='r')
                    fig.text(0.12, 0.895, fecha3, ha='left', va='center', fontsize=17, weight='roman')
                    fig.text(0.9, 0.895, f"{frequency_name} | {get_prn_name(prn)}", 
                            ha='right', va='center', fontsize=17, weight='roman')
                    fig.text(0.72, 0.895, frequency_value, ha='right', va='center', fontsize=17, 
                            weight='roman')    
                else:
                    # labels
                    fig.text(0.513, -0.1, 'Time UTC', ha='center', va='center', fontsize=14)
                    fig.text(0.09, 0.5, 'S4', ha='center', va='center', rotation='vertical', 
                            fontsize=14, color=color1)
                    fig.text(0.94, 0.5, 'Elevation Angle', ha='center', va='center', rotation=-90, 
                            fontsize=14, color=color2)
                    # title 
                    fig.text(0.513, 0.95, 'S4', ha='center', va='center', fontsize=17, weight='roman')
                    fig.text(0.32, 0.95, 'Jicamarca', ha='center', va='center', fontsize=17, 
                            weight='roman', color='r')
                    fig.text(0.12, 0.95, fecha3, ha='left', va='center', fontsize=17, weight='roman')
                    fig.text(0.9, 0.95, f"{frequency_name} | {get_prn_name(prn)}", ha='right', 
                            va='center', fontsize=17, weight='roman')
                    fig.text(0.72, 0.95, frequency_value, ha='right', va='center', fontsize=17, 
                            weight='roman')
                i += 1
            
            # Save figure as pdf
            figure_name2 = figure_name + f"_s4_{get_prn_name(prn)}_{frequency_name}.pdf"
            plt.savefig(new_directory + figure_name2, bbox_inches='tight')

    return "Plotted succesfully!"

# Auxiliar functions
# Get the name for a given PRN code
def get_prn_name(prn_code):
    if prn_code == 'G': return 'GPS'
    elif prn_code == 'R': return 'GLONASS'
    elif prn_code == 'E': return 'GALILEO'
    elif prn_code == 'S': return 'SBAS'
    elif prn_code == 'C': return 'BEIDOU'
    elif prn_code == 'J': return 'QZSS'
    elif prn_code == 'I': return 'IRNSS'
    else:
        return 'Incorrect PRN code!'

# Get the frequency name for a given PRN code and Freq code
def get_sig_name(prn, sig):
    if sig == 1:
        if prn == 'G':
            return {"name":'L1CA', "value":"1575.42"}
        elif prn == 'R':
            return {"name":'L1CA', "value":"1602"} # change 
        elif prn == 'S':
            return {"name":'L1CA', "value":"1575.42"}
        elif prn == 'J':
            return {"name":'L1CA', "value":"1575.42"}
        elif prn == 'E':
            return {"name":'L1BC', "value":"1575.42"}
        elif prn == 'C':
            return {"name":'B1', "value":"1575.42"}
        elif prn == 'I':
            return {"name":'B1', "value":"1176.45"}
        else: 
            return {"name": 'null', "value": "nan"}
    elif sig == 2:
        if prn == 'G':
            return {"name":'L2C', "value":"1227.60"}
        elif prn == 'R':
            return {"name":'L2C', "value":"1246"} # change 
        elif prn == 'J':
            return {"name":'L2C', "value":"1227.60"}
        elif prn == 'E':
            return {"name":'E5a', "value":'1176.45'}
        elif prn == 'C':
            return {"name":'B2', "value":'1176.45'}
        elif prn == 'S':
            return {"name":'L5', "value":'1176.45'}
        else: 
            return {"name": 'null', "value": "nan"}
    elif sig == 3:
        if prn == 'G':
            return {"name":'L5', "value":'1176.45'}
        elif prn == 'J':
            return {"name":'L5', "value":'1176.45'}
        elif prn == 'E':
            return {"name":'E5b', "value":'1207.14'}
        elif prn == 'C':
            return {"name":'B3', "value":'1268.52'}
        else: 
            return {"name": 'null', "value": "nan"}
    else:
        return {"name": 'null', "value": "nan"}

# Main
def main():
    list_input_files = glob.glob(input_files_path + "*.s4")
    if len(list_input_files) > 0:
        for file_i in list_input_files:
            df1 = read_s4_file(file_i)
            df2_1 = process_dataframe(df1)
            df2_2 = filter_dataframe(df2_1)
            
            # Plot and save
            file_name = file_i[len(input_files_path):]
            figure_name = file_name[:-3]    
            plot1_s4(df2_1, figure_name)
            plot2_s4(df2_2, figure_name)

            # Move input files to a permanent directory
            os.rename(file_i, input_files_path_op+file_name)

    return "Ok"

if __name__ == '__main__':
    main()
