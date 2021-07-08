#!/bin/bash

FOLDER_HOME="/home/cesar/Desktop/luisd/scripts/"
FOLDER_SRC="${FOLDER_HOME}Obtencion_cintilaciones/data_output/"
FOLDER_DST="${FOLDER_HOME}Graficas_cintilaciones/Input_data/Data_set/"
FOLDER_DST2="${FOLDER_HOME}Graficas_cintilaciones/Input_data/Data_procesada/"

FILES=`diff -qr ${FOLDER_SRC} ${FOLDER_DST2} | grep Only | grep ${FOLDER_SRC} | awk '{print $4}'`

if [[ -z $FILES ]]
then 
  echo "There isn't any new file yet!"
else
  echo "The new files are:"
  echo $FILES
  for FILE in $FILES
  do
    cp -r ${FOLDER_SRC}$FILE ${FOLDER_DST}
  done
  echo "All files were copied sucesfully!"
fi
