# Temperature-Prediction
## General info
The data was provided by the LERTA company (https://www.lerta.energy/). The data was collected as part of a project to improve methods of controlling thermal comfort in buildings.
The data provided relates to an office building located in Poznań and contains a small fragment of data.
Registered:
* room temperature (for several temperature sensors located in different places) in degrees Celsius,
* degree of opening of the radiator valve, as a percentage,
* set temperature in degrees Celsius.

## Purpose of the project:
The aim of the project is to make two predictions:
* temperature values of the indicated sensor in 15 minutes,
* the value of the opening degree of the radiator valve in 15 minutes.

## Requirements:
* Python version: 3.8.6
* Pandas version: 1.2.0

## Setup:
```
$ pip install -r requirements.txt
```
To run this project, you need to pass two arguments:
path to input json file,
path to output result csv file
```
$ python main.py input_file results_file_csv
```
