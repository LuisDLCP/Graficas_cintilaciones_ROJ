#!/home/luis/anaconda3/bin/python3

from matplotlib.dates import DateFormatter
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
    # Delete columns without s4 info
    for i in df.columns:
        if i>3:
            m = (i-4)%24
            if m == 1: del df[i]
            elif m == 2: del df[i]
            elif m >= 9: del df[i]
    
    # Reindex the columns 
    df.columns = np.arange(len(df.columns)) 

    # A new df is created and filled 
    df2 = pd.DataFrame(columns=range(10))

    T = 7 # Period of dataset
    for i in range(len(df)):
        for h in range(df.iloc[i,3]): # that cell contains the number of measurements 
            m = df.iloc[i,[0,1,2,4+T*h,5+T*h,6+T*h,7+T*h,8+T*h,9+T*h,10+T*h]]
            m.index = range(len(m))
            df2 = df2.append(m)

    df2.index = range(len(df2))

    # Change datatype: str -> int 
    for i in range(3):
        df2[i] = df2[i].astype("int")

    # Change datatype: str -> float
    for i in range(6):
        df2[4+i] = df2[4+i].astype("str").str.strip().astype("float")

    # Rename columns 
    df2.columns = ["YY", "DOY", "SOD", "PRN", "S4_sig1", "S4_sig1_corr", "S4_sig2", 
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

# Plot s4 in function of each constellation and frequency
def plot_s4(df, figure_name):
    # Identify the available PRNs
    prns = df["PRN"].str[0].unique() # extract the first character of each cell 
    
    # Get file UTC date
    fecha = figure_name[5:] # e.g. 200926
    fecha2 = datetime.datetime.strptime(fecha, "%y%m%d")
    fecha3 = datetime.datetime.strftime(fecha2,"%Y/%m/%d")

    fecha2_tomorrow = fecha2 + pd.DateOffset(days=1)
    fecha2_tomorrow = fecha2_tomorrow.to_pydatetime()

    # Create directory for output files
    new_directory = output_files_path + figure_name + "/"
    if not os.path.exists(new_directory):
        os.makedirs(new_directory)

    # Plot for each PRN and freq code
    for prn in prns:
        mask = df["PRN"].str.contains(prn)
        df3 = df[mask]
        
        # Check no null columns 
        sig_n = df3.columns[1:] # Extract only the columns which have freq info
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

            ax[i].set_title(get_sig_name(prn, sig) + f" | {get_prn_name(prn)}", loc="right",
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
        if prn == 'G' or prn == 'R' or prn == 'S' or prn == 'J':
            return 'L1CA'
        elif prn == 'E':
            return 'L1BC'
        elif prn == 'C':
            return 'B1'
        elif prn == 'I':
            return 'L5'
        else: 
            return "Insert a right code!"
    elif sig == 2:
        if prn == 'G' or prn == 'R' or prn == 'J':
            return 'L2C'
        elif prn == 'E':
            return 'E5a'
        elif prn == 'C':
            return 'B2'
        elif prn == 'S':
            return 'L5'
        else: 
            return "Insert a right code!"
    elif sig == 3:
        if prn == 'G' or prn == 'J':
            return 'L5'
        elif prn == 'E':
            return 'E5b'
        elif prn == 'C':
            return 'B3'
        else: 
            return "Insert a right code!"
    else:
        "Insert a right code!"

# Main
def main():
    list_input_files = glob.glob(input_files_path + "*.s4")
    if len(list_input_files) > 0:
        for file_i in list_input_files:
            df1 = read_s4_file(file_i)
            df2 = process_dataframe(df1)
            
            # Plot and save
            file_name = file_i[len(input_files_path):]
            figure_name = file_name[:-3]    
            status = plot_s4(df2, figure_name)

            # Move input files to a permanent directory
            os.rename(file_i, input_files_path_op+file_name)

    return status

if __name__ == '__main__':
    main()
