"""
ds 710: Programming for Data Science
Project: Job Search Research

John Woodward

Do not rename this file
"""

# ----------------------------------------

## libraries import
# import os
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
import re # for regular expressions
import pandas as pd # data manipulation
from flashtext import KeywordProcessor # quicker than regular expressions

# prior code import
# =============================================================================
# 
# def filename_checker(ext,name):
#     '''
#     returns the filename with an extension
# 
#     Parameters
#     ----------
#     ext : str
#         extension of the filename.
#     name : str
#         descriptive filename.
# 
#     Returns
#     -------
#     return_filename : str
#         filename with extension.
# 
#     '''
#     return name if name.endswith(ext) else name + '.' + ext
# 
# =============================================================================
# ----------------------------------------

# Task 2: analyze.py

# Subtask 2.1: Read in the data and organize it

df = pd.read_csv('LIX_plus_FIVRR_LinkedIn_Job_Search.csv', 
                 encoding='latin-1')
df = df.loc[:, ['ID',
                'Link', 
                'Title', 
                'Salary_Combined', 
                'Location', 
                'Workplace type', 
                'Descrption(fivrr)']]

# split city and state
df[['City', 'State']] = df['Location'].str.split(',', n=1, expand=True)
df[['City', 'State']] = df[['City', 'State']].apply(lambda x: x.str.strip()) # trim

# set City and State to None where Workplace Type is 'Remote'
df.loc[df['Workplace type'] == 'Remote', ['City', 'State']] = None

# --------------------- Handle State, Country exceptions
mask = (df['State'].str.strip() == 'United States')
df.loc[mask, 'State'] = df[mask]['City']
df.loc[mask,'City'] = None

# df[mask]

# ----------- Handle exceptions with format (e.g., San Francisco Bay Area and Greater Boston)
mask2 = ((df['City']).notna() & pd.isna(df['State']))
df.loc[mask2, 'State'] = df.loc[mask2, 'City'].map({'San Francisco Bay Area': 'CA', 
                                                    'Greater Boston': 'MA'})
df.loc[mask2, 'City'] = df.loc[mask2, 'City'].map({'San Francisco Bay Area': 'San Francisco', 
                                                   'Greater Boston': 'Boston'})

# Discard unused location
df.drop('Location', 
        axis=1, 
        inplace=True)

# ------------------------------- Handle Salary_Combined column (hourly)

# Create boolean mask for hourly wages
mask3 = df['Salary_Combined'].notna() & df['Salary_Combined'].str.contains('hr')

# Extract salary values for hourly wages
salary_values = (df.loc[mask3, 'Salary_Combined'].str.extract(r'\$([\d.]+)\/hr\s*-\s*\$([\d.]+)\/hr'))

# Convert salary_values to numeric columns, convert hourly to yearly
salary_values = salary_values.apply(pd.to_numeric, errors='coerce')*40*52 # 40 hours per week, 52 weeks per year

# Rename columns of new DataFrame
salary_values = salary_values.rename(columns={0:'Salary_Lower', 
                                              1:'Salary_Upper'})

salary_values['Salary_UOM'] = 'yearly' # Add 'Salary_UOM' column with converted 'yearly' rates

# Update columns for hourly wages
df.loc[mask3, ['Salary_Lower', 'Salary_Upper', 'Salary_UOM']] = salary_values
#df.loc[mask3, ['Salary_Combined','Salary_Lower', 'Salary_Upper', 'Salary_UOM']]

# --------------------------------- Handle Salary_Combined column (yearly)

# Create boolean mask for hourly wages
mask4 = df['Salary_Combined'].notna() & df['Salary_Combined'].str.contains('yr')

# Extract salary values for hourly wages
salary_values = df.loc[mask4, 'Salary_Combined'].str.lower().str.extract(r'\$([\d.]+)k/yr\s*-\s*\$([\d.]+)k/yr')

# Convert salary_values to numeric columns
salary_values = salary_values.apply(pd.to_numeric, errors='coerce')*1000

# Rename columns of new DataFrame
salary_values = salary_values.rename(columns={0:'Salary_Lower', 1:'Salary_Upper'})

salary_values['Salary_UOM'] = 'yearly' # Add 'Salary_UOM' column with converted 'yearly' rates

# Update columns for hourly wages
df.loc[mask4, ['Salary_Lower', 'Salary_Upper', 'Salary_UOM']] = salary_values
#df.loc[mask4, ['Salary_Combined','Salary_Lower', 'Salary_Upper', 'Salary_UOM']]

# ------------------------------- handle any lingering states more than 2 characters

mask_lingering = df['State'].str.len() > 2

# Could be better method but for now it works
df.loc[mask_lingering, 'State'] = df.loc[mask_lingering, 'State'].map({'New York':'NY',
                                                                       'Illinois':'IL',
                                                                       'Texas Metropolitan Area':'TX'})


# ------------

# Subtask 2.2: 
# 🎯 What are the approximate salary bands by the state after adjusting for purchasing power

# cordon off the correct data for the left table for join
mask_salary = df['Salary_Combined'].notna() & df['State'].notna()
df_salary = df.loc[mask_salary, ['State','Salary_Lower', 'Salary_Upper', 'Salary_UOM']]

# read right table and correct formatting to numeric
df_purchase_power_by_state = pd.read_csv('purchase_power_by_state.csv')
df_purchase_power_by_state['Purchase_Power'] = df_purchase_power_by_state['Purchase_Power_2019'].str.replace('$', '',regex=True).astype(float)

# Use a function for associated problem
def get_salary_bands_by_state(df_salary, # left table 
                              df_purchase_power_by_state, # right table
                              count_column='n', 
                              counts_per_row=1,
                              salary_lower_column='Salary_Lower', 
                              salary_upper_column='Salary_Upper',
                              suffix_adjustment= "_Adjusted",
                              purchase_power_column='Purchase_Power',
                              salary_band_aggregate='median',
                              count_aggregate='sum',
                              sort_ascending=False):
    '''
    

    Parameters
    ----------
    df_salary : pandas df
        left join for table, contains salary ranges.
    df_purchase_power_by_state : pandas df
        right join for table, contains by state the purchase power in $.
    count_column : str, optional
        name for the count column. The default is 'n'.
    counts_per_row : int, optional
        How much a row is counted. The default is 1.
    salary_lower_column : str, optional
        Column name. The default is 'Salary_Lower'.
    salary_upper_column : str, optional
        Column name. The default is 'Salary_Upper'.
    suffix_adjustment : str, optional
        suffix to use for column names. The default is "_Adjusted".
    purchase_power_column : str, optional
        column to use when joining. The default is 'Purchase_Power'.
    salary_band_aggregate : str, optional
        aggregate to use, could be median, average, .... The default is 'median'.
    count_aggregate : str, optional
        aggregate to use, could be median, average, .... The default is 'sum'.
    sort_ascending : bool, optional
        False means descending sort, True means ascending sort. The default is False.

    Returns
    -------
    df_salary_bands : TYPE
        DESCRIPTION.

    '''
    
    # Merge dataframes on state column
    df_salary = df_salary.merge(df_purchase_power_by_state, 
                                on='State', 
                                how='inner')
    
    # Adjust column name
    salary_lower_column_adjusted = salary_lower_column+suffix_adjustment
    salary_upper_column_adjusted = salary_upper_column+suffix_adjustment
    
    # Compute adjusted salaries and add count column
    df_salary[salary_lower_column_adjusted] = round(df_salary[salary_lower_column] * df_salary[purchase_power_column])
    df_salary[salary_upper_column_adjusted] = round(df_salary[salary_upper_column] * df_salary[purchase_power_column])
    df_salary[count_column] = counts_per_row
    
    # Sort column
    sort_column=salary_upper_column+suffix_adjustment
    
    # Group by state and compute median adjusted salaries and sum of counts
    df_salary_bands = df_salary.groupby('State').agg({salary_lower_column_adjusted: salary_band_aggregate, 
                                                       salary_upper_column_adjusted: salary_band_aggregate,
                                                       count_column: count_aggregate
                                                      }).sort_values(by=sort_column, 
                                                                     ascending=sort_ascending).astype(int)
    return df_salary_bands


df_salary_bands = get_salary_bands_by_state(df_salary, 
                                            df_purchase_power_by_state)


# 🎯 and where are most non-remote jobs located?
mask_nonremote = df['Workplace type']!= 'Remote'
df_nonremote = df.loc[mask_nonremote,['State','Workplace type']]
df_nonremote['n']=1
df_nonremote_states = df_nonremote.groupby('State').agg({'n':'sum'}).sort_values(by='n', ascending=False).astype(int).head(5)

# how many remote jobs?
jobs_remote = (df['Workplace type']== 'Remote').sum()
jobs_total = len(df)

jobs_remote_proportion = jobs_remote/jobs_total

# -------------

# Subtask 2.3: 🎯 What proportion of job listings list a Ph.D.? A Masters? A Bachelors?

# Remove smart quote
df['Descrption(fivrr)'] = df['Descrption(fivrr)'].apply(lambda x: x.replace('\x92', "'"))
text = df['Descrption(fivrr)'].str.lower()


# Function to extract keywords using flashtext
def verify_keywords_flashtext(pattern
                              , text):
    '''
    Use flashtext instead of regular expression to quickly find any of the 
    keywords from the provided pattern

    Parameters
    ----------
    pattern : str list
        list of keywords.
    text : long str
        job descriptions or any long string really.

    Returns
    -------
    boolean
        True or false if keywords existed in row.

    '''
    # use flashtext to search quickly
    keyword_processor = KeywordProcessor()
    keyword_processor.add_keywords_from_list(pattern)

    # extract the matches
    matches = keyword_processor.extract_keywords(text)
    IsMatch = bool(matches)
    
    return bool(matches)
    return IsMatch

# Apply keyword searches
pattern_bs = ["bachelor's", "b.s.", "bs", "bachelor","bachelors"]
pattern_ms = ["master's", "m.s.","ms", "masters"]
pattern_phd = ["phd", "ph.d.", "doctorate"]

df['keywords_BS'] = df['Descrption(fivrr)'].apply(lambda x: verify_keywords_flashtext(pattern_bs, x))
df['keywords_MS'] = df['Descrption(fivrr)'].apply(lambda x: verify_keywords_flashtext(pattern_ms, x))
df['keywords_PHD'] = df['Descrption(fivrr)'].apply(lambda x: verify_keywords_flashtext(pattern_phd, x))

# Print the DataFrame
number_BS = df['keywords_BS'].sum()
number_MS = df['keywords_MS'].sum()
number_PHD = df['keywords_PHD'].sum()

# ---------------------------------------------------------------------------------------------------------------------------

# beginning of the Part 2

# load new libraries
from geopy.distance import geodesic as geodist # calculating distances between latitude, longitude locations
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import Point
import numpy as np


# Subtask 1.1: read City and state latitude and longitudes into a pandas df and join to the main df

# read in downloaded, open-source dataset
df_US_locations = pd.read_csv('US.txt', 
                              delimiter="\t", 
                              header=None, 
                              dtype={9:str, 11:str}) # resolving a mixed data warning

# Cleanup data for our purposes
df_US_City_State_Lat_Lon = df_US_locations[[2, 4, 5, 10]].rename(columns={2: 'City', 4: 'Latitude', 5: 'Longitude', 10: 'State'})

# remove any duplicates so we don't get duplicate rows
df_US_City_State_Lat_Lon = df_US_City_State_Lat_Lon.drop_duplicates(subset=['City', 'State'], keep='first').reindex(columns=['City', 'State', 'Latitude', 'Longitude'])

# left join Latitude and Longitude onto Primary df
df = df.merge(df_US_City_State_Lat_Lon, 
              on=['City', 'State'], 
              how='left') # the only joins that are NA are the NULL City or State, or 'St Louis' which should be 'St. Louis' but that is not in the new england area, so omitted

## Subtask 1.2: Calculate MilesFromHome distance using a function

# Use a general function that takes in city and state and outputs MilesFromHome

def calculate_distance(city1, state1, 
                       city2, state2, 
                       unit='mi'):
    '''
    
    Calculate the distance between two cities using their latitude and longitude
    
    Defaults to miles or "mi" (kilometers or "km" also an option) 
    

    Parameters
    ----------
    city1 : str
    state1 : str
    city2 : str
    state2 : str
    unit : str, optional
        distance unit. The default is 'mi'.

    Raises
    ------
    ValueError
        If the unit is not mi or km it errors.

    Returns
    -------
    distance : float
        distance between location 1 and location 2.

    '''
    if city1 is None or state1 is None or city2 is None or state2 is None:
        return None
    
    # Get the latitude and longitude of the origin city
    from_lat = df.loc[(df['City'] == city1) & (df['State'] == state1), 'Latitude'].iloc[0]
    from_lon = df.loc[(df['City'] == city1) & (df['State'] == state1), 'Longitude'].iloc[0]

    # Get the latitude and longitude of the destination city
    to_lat = df.loc[(df['City'] == city2) & (df['State'] == state2), 'Latitude'].iloc[0]
    to_lon = df.loc[(df['City'] == city2) & (df['State'] == state2), 'Longitude'].iloc[0]
    
    location1 = (from_lat, from_lon)
    location2 = (to_lat, to_lon)
    
    # Calculate the distance between the two locations using geodesic
    if unit == 'mi':
        distance = geodist(location1, location2).miles
    elif unit == 'km':
        distance = geodist(location1, location2).km
    else:
        raise ValueError("Unit must be 'mi' or 'km'")
        
    return distance

# Currying lambda function always measuring from 'Boston', 'MA' (family location)
calculate_distance_Boston = lambda city, state: calculate_distance(city, state,
                                                                   'Boston','MA')

# ---------- calculating MilesFromHome column

# Only calculate distance from Boston, MA for non-na Latitude and Longitude
lat_lon_mask = df['Latitude'].isna() | df['Longitude'].isna()
df.loc[~lat_lon_mask, 'MilesFromHome'] = df.loc[~lat_lon_mask].apply(lambda row: calculate_distance_Boston(row['City'], row['State']), axis=1)
df.loc[lat_lon_mask, 'MilesFromHome'] = np.nan

# filter for 
MilesFromHomeLimit = 250
df_jobs_newengland = df.loc[df['MilesFromHome'] < MilesFromHomeLimit, ['City','State','MilesFromHome','Latitude','Longitude']].sort_values(by=['MilesFromHome'], 
                                                    ascending=True)

# ---------- plotting in geopandas points and US map

# Geometry from latitude and longitude
geometry = [Point(xy) for xy in zip(df_jobs_newengland.Longitude, 
                                    df_jobs_newengland.Latitude)]

# GeoDataFrame (gdf)
gdf = gpd.GeoDataFrame(df_jobs_newengland, 
                       geometry=geometry)


# Read shp
usa = gpd.read_file('tl_2020_us_state.shp')

# Filter the states to only include New England states
new_england = usa.loc[usa['NAME'].isin(['Connecticut', 
                                        'Maine', 
                                        'Massachusetts', 
                                        'New Hampshire', 
                                        'New Jersey',
                                        'Rhode Island', 
                                        'Vermont',
                                        'New York'])]

# Map
fig, ax = plt.subplots(figsize=(10, 10))
new_england.plot(ax=ax, 
                 color='silver', 
                 edgecolor='black')

# Plot points
gdf.plot(ax=ax, 
         column='MilesFromHome', 
         cmap='RdYlGn_r',  # RdYlGn or Red-Yellow-Green - colorblind friendly
         legend=True, 
         legend_kwds={'shrink': 0.5},
         marker='o',
         edgecolor='black'
        )

# Customize
plt.title('Job Miles from Home')
plt.axis('off')

# Save
plt.savefig('job_miles_from_home_map.pdf', 
            bbox_inches='tight', 
            dpi=300)


# ------------- Task 2

# Subtask 2.1: Classifying skills

# =============================================================================
# # -------------- method of sampling job descriptions for keyword analysis
# ## random 
# ## seed 
# ## fix
# ## 
# np.random.seed(1) # for consistency
# 
# # Set the range of integers
# start_range = 0
# end_range = 421
# 
# # Generate an array of integers from start_range to end_range
# int_array = np.arange(start_range, end_range + 1)
# 
# # Sample 10 integers without replacement and sort
# sample = np.sort(np.random.choice(int_array, size=10, replace=False))
# 
# # Select the rows of the DataFrame that match the index of the sampled integers
# sampled_df = df.loc[sample]
# 
# sampled_df.loc[:,'Descrption(fivrr)']
# 
# #print(sampled_df.loc[sample[0],'Descrption(fivrr)'])
# =============================================================================

# --------- keywords classified

keywords = {
    'languages': ['java', 'scala', 'python', 'rust', 'go', 'swift'],
    'databases': ['sql', 'sql server', 'oracle', 'mysql', 'postgresql', 
                  'sqlite'],
    'nosql_databases': ['nosql', 'mongodb', 'cassandra', 'couchbase', 'dynamodb',
                        'mongo', 'dynamo'],
    'operating_systems': ['unix', 'linux', 'shell scripting', 'batch files', 
                          'powershell'],
    'big_data': ['big data', 'hadoop', 'mapreduce', 'hdfs', 'hive', 'emr', 
                 'stream processing frameworks', 'kafka', 'flink', 'storm', 'spark', 'apache'],
    'cloud_computing': ['cloud computing', 'aws', 'azure', 'cloud', 
                        'digitalocean'],
    'data_warehousing': ['data warehousing', 'redshift', 'bigquery', 
                         'snowflake', 'er modeling', 'dimensional modeling', 
                         'talend', 'aws glue'],
    'machine_learning': ['machine learning', 'artificial intelligence', 
                         'machine learning frameworks', 'tensorflow', 'pytorch', 
                         'scikit-learn', 'deep learning', 'natural language processing', 
                         'nlp', 'neural networks', 'text mining'],
    'software_development': ['agile', 'scrum', 'unit testing', 'performance tuning', 
                             'git', 'github'],
    'optimization': ['optimization', 'gurobi', 'cplex', 'ampl', 'coin-or', 
                     'gradient descent', 'linear programming', 'lp', 'nonlinear programming', 
                     'mixed-integer linear programming', 'milp', 'simulated annealing', 
                     'genetic algorithms', 'particle swarm optimization'],
    'visualization': ['visualization', 'statistical', 'statistics', 'r', 'plotly', 
                      'seaborn', 'ggplot', 'ggplot2', 'matplotlib', 'powerbi', 
                      'tableau', 'jmp']
}


# Subtask 2.2: keyword search applied

# Remove / slash and replace with a space
df['Descrption(fivrr)'] = df['Descrption(fivrr)'].apply(lambda x: x.replace('/', " "))

def apply_keyword_searches(df, keywords):
    '''
    
    Apply keyword searches for each category of keywords in the given DataFrame.

    Parameters
    ----------
    df : pandas df
        input dataframe
    keywords : dict
        assumed to be a dict that we loop through to create columns.

    Returns
    -------
    df : pandas df
        output dataframe with additional columns added.

    '''
    
    for category in keywords:
        pattern = keywords[category]
        col_name = f"keyword_{category}"
        df[col_name] = df["Descrption(fivrr)"].apply(lambda x: verify_keywords_flashtext(pattern, x))
    return df

df = apply_keyword_searches(df, keywords)

# -------- Summarize
total_count = jobs_total

# filter
keyword_columns = df.filter(regex='^keyword_')

# pivot
keyword_pivot = keyword_columns.astype(int).sum().reset_index()

# rename
keyword_pivot.columns = ['Keyword', 
                         'Count']

# replace
keyword_pivot['Keyword'] = keyword_pivot['Keyword'].str.replace('keyword_', '')

# sort
keyword_pivot = keyword_pivot.sort_values(by='Count', ascending=True)

# ---- Plot summary table

# horizontal bar plot
fig, ax = plt.subplots(figsize=(6, 6))
ax.barh(keyword_pivot['Keyword'], 
        keyword_pivot['Count'], 
        color='silver')

# settings
plt.tight_layout()
ax.set_xticks([])
plt.title('Keyword Matches In Job Search Description')

# add the percentage text inside the bars
bars = ax.barh(keyword_pivot['Keyword'], keyword_pivot['Count'], color='silver')
for i, bar in enumerate(bars):
    ax.text(bar.get_width() + 1, 
            i, 
            f"{bar.get_width() / total_count:.0%}", 
            va='center')

# save
plt.savefig('keyword_counts.pdf', 
            bbox_inches='tight', 
            dpi=300)

if __name__ == "__main__":
    pass
    #df[mask_lingering]
    # Debugging
    #mask_degree = (df['keywords_BS'] == False) & (df['keywords_MS'] == False) & (df['keywords_PHD'] == False)

    #test = df['Descrption(fivrr)'].apply(lambda x: extract_keywords_flashtext(["masters"], x))
    #mask_degree = (test==True)
    #df.loc[mask_degree, 'Descrption(fivrr)']
    
    
    #print(text)
    #print(text[11])#.replace('\x92',"'")
    
    #print(df_salary_bands)
    #print(df_nonremote_states)
    
    #round(jobs_remote_proportion,2)
    #print(f"Degrees Referenced\nBS:{number_BS} times or {round(number_BS/jobs_total*100)}%\nMS:{number_MS} times or {round(number_MS/jobs_total*100)}%\nPHD:{number_PHD} times or {round(number_PHD/jobs_total*100)}%")

    # Test lambda function
    #distance2 = calculate_distance_Boston('New York', 'NY')
    #print(distance2) # Output: 230.40558134507938
    
    # view
    #df.loc[~lat_lon_mask, ['City','State','Latitude','Longitude','MilesFromHome']]

    # view
    #df_jobs_newengland
