#!/bin/bash
# This script copies s4 files(lisn & modified lisn format) to 2 destinations

FOLDER_HOME="/home/cesar/Desktop/luisd/scripts/"
FOLDER_SRC="${FOLDER_HOME}Obtencion_cintilaciones/data_output/lisn/"
FOLDER_SRC2="${FOLDER_HOME}Obtencion_cintilaciones/data_output/lisn_rich/"
FOLDER_DST="${FOLDER_HOME}Graficas_cintilaciones/Input_data/Data_set/"
FOLDER_DST2="${FOLDER_HOME}Main/Output_files/ToUpload/"
FOLDER_AUX1="${FOLDER_HOME}Graficas_cintilaciones/Input_data/Data_procesada/"

FILES=`diff -qr ${FOLDER_SRC2} ${FOLDER_AUX1} | grep Only | grep ${FOLDER_SRC2} | awk '{print $4}'`

if [[ -z $FILES ]]
then 
  echo "There isn't any new s4 file yet!"
else
  echo "The new s4 files are:"
  echo $FILES
  for FILE in $FILES
  do
    cp -r ${FOLDER_SRC2}$FILE ${FOLDER_DST}
    cp -r ${FOLDER_SRC2}$FILE ${FOLDER_DST2}
    mv "${FOLDER_DST2}$FILE" "${FOLDER_DST2}${FILE/.s4/RICH.s4}" #rename
    cp -r ${FOLDER_SRC}$FILE ${FOLDER_DST2}
  done
  # Compress s4 files
  gzip --force ${FOLDER_SRC}*s4 
  gzip --force ${FOLDER_SRC2}*s4 

  echo "All s4 files were copied sucesfully!"
fi
