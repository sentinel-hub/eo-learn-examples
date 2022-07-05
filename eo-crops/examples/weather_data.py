
import pandas as pd
from eocrops.input.meteoblue import CEHUBExtraction, CEHubFormatting

###############################################################################################################
#Read the file
input_file = pd.read_csv('./examples/layers/burkina_dataframe.csv')
input_file['coordinates'] = list(
    zip(input_file['Longitude'], input_file['Latitude'])
)

input_file['Id_location'] = input_file['Id_location'].astype(str)
input_file = input_file[input_file['Aggregation'].isin(['mean'])]
input_file[['Id_location', 'Annee']].drop_duplicates().shape

###############################################################################################################
#Step 1 : Define the query with a backbone (=units, jobs parameters) and your input file features (locations,..)
###############################################################################################################

queryBackbone = {
        "units": {
        "temperature": "C",
        "velocity": "km/h",
        "length": "metric",
        "energy": "watts"
    },
    "timeIntervalsAlignment": None,
    "runOnJobQueue": True,
    "oneTimeIntervalPerGeometry": True,
    "checkOnly": False,
    "requiresJobQueue": False,
    "geometry": {
        "type": "GeometryCollection",
        "geometries": None
    },
    "format": "csvIrregular", # best format
    "timeIntervals":  None
}


pipeline_cehub = CEHUBExtraction(api_key = '',
                                 queryBackbone = queryBackbone,
                                 ids = input_file['Id_location'].values,
                                 coordinates=  input_file['coordinates'].values,
                                 years = input_file['Annee'].values)

stat = 'mean'

query = [{"domain": "ERA5", "gapFillDomain": "NEMS4",
          "timeResolution": "daily",
          "codes": [
              {"code": 52, "level": "2 m above gnd", "aggregation": stat},  # Relative Humidity
              {"code":  11, "level": "2 m above gnd", "aggregation": stat}, # air temperature (°C)
              {"code":  32, "level": "2 m above gnd", "aggregation": stat}, # Wind Speed
              {"code": 180, "level": "sfc", "aggregation": stat}, #wind gust
              {"code":  256, "level": "sfc","aggregation": stat}, # Diffuse Shortwave Radiation
              {"code":  56, "level": "2 m above gnd","aggregation": stat}, # Vapor Pressure Deficit
              {"code":  260, "level": "2 m above gnd","aggregation": stat}, # FAO Reference Evapotranspiration,
              {"code":  261, "level": "sfc", "aggregation": stat}, # Evapotranspiration
              {"code":  52, "level": "2 m above gnd","aggregation": stat}, # Relative humidity
          ],
}]

df_output = pipeline_cehub.execute(query = query,  time_interval = ('01-01', '12-31'))
df_output.to_csv('./examples/layers/mean_meteoblue.csv', index = False)

###############################################################################################################
#Step 2 : reformat file given a resampling range (e.g. every 8 days from the 1st of January into 31 of December
###############################################################################################################

df_output = pd.read_csv('./examples/layers/mean_meteoblue.csv', skiprows=1)


pipeline_refactor = CEHubFormatting(
    input_file = input_file,
    id_column = 'Id_location',
    year_column = 'Annee',
    resample_range=('-01-01', '-12-31', 1)
)

df_mean = pipeline_refactor.execute(df_weather=df_output, stat='mean', return_pivot=False)
