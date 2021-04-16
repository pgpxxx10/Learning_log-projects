# Stanford Open Policing Project Examining the dataset
# Throughout this course, you'll be analysing a dataset of traffic stops in Rhode Island that was collected by the Stanford Open Policing Project. Rode Island data set is only part of Stanford project for Policicing. More information available under this link https://openpolicing.stanford.edu/

# I will try for find the answer for questions below:
# Do the genders commit different violations?
# Which gender is stop and search more often by Police in Rode Island?
# Who is speeding on the Rode Island roads?
# Is drug and alcohol usage increase over a time?
# Is drug or alcohol Police stops are related to higher amount of search conducted? 
# What day time Police is usually busy in Rode Island?

# For this project I used as a guidance DataCamp project. My project is written in Sublime Text 3

# Import the libraries for this project
import pandas as pd
import matplotlib.pyplot as plt


###Data Cleaning and Exploration###

# Before beginning your analysis, it's important that you familiarize yourself with the dataset. 
# Read 'RI state data from project website into a DataFrame named ri
ri = pd.read_csv('ri_statewide_2020_04_01.csv')

# Examine the head of the DataFrame
print(ri.head())

# Count the number of missing values in each column
print(ri.isnull().sum())


# Examine the shape of the DataFrame
print(ri.shape)

# Drop the columns that will not be used in project
ri.drop(['raw_row_number', 'department_id', 'contraband_found', 'contraband_drugs',
	   'contraband_weapons', 'contraband_alcohol', 'contraband_other', 'raw_SearchResultTwo', 'raw_SearchResultThree',
	   'raw_BasisForStop', 'raw_OperatorRace', 'raw_OperatorSex',
	   'raw_ResultOfStop', 'raw_SearchResultOne', 'vehicle_model', 'vehicle_make' ], axis='columns', inplace=True)

# Examine the shape of the DataFrame (again)
print(ri.shape)


# Concatenate 'date' and 'time' (separated by a space)
combined = ri.date.str.cat(ri.time, sep=' ')

# Convert 'combined' to datetime format
ri['stop_datetime'] = pd.to_datetime(combined)

# Examine the data types of the DataFrame
print(ri.stop_datetime.dtypes)


# Set 'stop_datetime' as the index
ri.set_index('stop_datetime', inplace=True)

# Examine the index
print(ri.index)

# Examine the columns
print(ri.columns)


### Do the genders commit different violations?###
# To perform this task first datasets need to be cleaned for columne subject_sex, arrest_made, reason_for_stop. 
# Drop all rows that are missing 'subject_sex'
ri.dropna(subset=['subject_sex'], inplace=True)

# Count the number of missing values in each column (again)
print(ri.isnull().sum())

# Examine the shape of the DataFrame
print(ri.shape)

# Examine the head of the 'arrest_made' column
print(ri.arrest_made.head())

# Change the data type of 'arrest_made' to 'bool'
ri['arrest_made'] = ri.arrest_made.astype('bool')

# Check the data type of 'arrest_made' 
print(ri.arrest_made.head())


# Examining traffic reason for stop. 
# Count the unique values in 'reason_for_stop'
print(ri.reason_for_stop.value_counts())

# Express the counts as proportions
print(ri.reason_for_stop.value_counts(normalize=True))


# Comparing reason for stop by gender
# Create a DataFrame of female drivers
female = ri[ri.subject_sex == 'female']

# Create a DataFrame of male drivers
male = ri[ri.subject_sex == 'male']

# Compute the reason for stop by female drivers (as proportions)

print(f'Female reason for stop in %: \n{female.reason_for_stop.value_counts(normalize=True)}')


# Compute the reason_for_stop  by male drivers (as proportions)
print(f'Male reason for stop in %: \n{male.reason_for_stop.value_counts(normalize=True)}')

# Print size of both variables 
print(f'Female reason for stop: \n{female.shape}')
print(f'Male reason for stop: \n{male.shape}')

# Who is speeding on the Rode Island roads?
# Comparing speeding outcomes by gender
# Create a DataFrame of female drivers stopped for speeding
female_and_speeding = ri[(ri.subject_sex == 'female') & (ri.reason_for_stop == 'Speeding')]


# Create a DataFrame of male drivers stopped for speeding
male_and_speeding = ri[(ri.subject_sex == 'male') & (ri.reason_for_stop == 'Speeding')]


# Compute the stop outcomes for female drivers (as proportions)
print(f'Female Speeding \n{female_and_speeding.outcome.value_counts(normalize=True)}')

# Compute the stop outcomes for male drivers (as proportions)
print(f'Male Speeding \n{male_and_speeding.outcome.value_counts(normalize=True)}')

 

### Which gender is stop and search more often by Police in Rode Island?###
# What are the reasons for conducting search? 
#Calculating the search rate

# Check the data type of 'search_conducted'
print(ri.search_conducted.dtype)

# Calculate the search rate by counting the values
print(ri.search_conducted.value_counts(normalize=True))

# Calculate the search rate by taking the mean
print(ri.search_conducted.mean())


# Comparing search rates by gender
# Calculate the search rate for female drivers
print(ri[ri.subject_sex == 'female'].search_conducted.mean())

# Calculate the search rate for male drivers
print(ri[ri.subject_sex == 'male'].search_conducted.mean())

# Calculate the search for both groups simultaneously
print(ri.groupby('subject_sex').search_conducted.mean())

# Calculate the search rate for each combination of gender and reason for stop
print(ri.groupby(['subject_sex', 'reason_for_stop']).search_conducted.mean())


# Drugs and alcohol are very bad idea to use and drive a car. Lets check is this problem is increasing over the time.
# Count the 'reason_for_search' values
print(ri.reason_for_search.value_counts())

# Check if 'search_type' contains the string 'Odor of Drugs/Alcohol'
ri['drug_alc'] = ri.reason_for_search.str.contains('Odor of Drugs/Alcohol', na=False)

# Check the data type of drugs and alcohol as a reason for search
print(ri.drug_alc.dtype)

# Take the sum of drugs and alcohol as a reason for search
print(ri.drug_alc.sum())


# Create a DataFrame of stops in which a search was conducted
searched = ri[ri.search_conducted == True]

# Calculate the overall drug_alc rate by taking the mean of 'drug_alc'
print(searched.drug_alc.mean())

# Calculate the drugs and alcohol as a reason for search for each gender
print(searched.groupby('subject_sex').drug_alc.mean())


###Data visualisation###
# Plotting drug or alcohol related stops
# In a small portion of traffic stops, drugs are found in the vehicle during a search or driver found to drink alcohol.
# Calculate the annual rate of drug or alcohol-related stops
print(ri.drug_alc.resample('A').mean())

# Save the annual rate of drug or alcohol-related stops
annual_drug_alc_rate = ri.drug_alc.resample('A').mean()

# Create a line plot of 'annual_drug_alc_rate'
plt.plot(annual_drug_alc_rate)
plt.xlabel('Year')
plt.ylabel('Arrest Rate')
plt.title('Annual rate of drug and alcohol related stops(frequency)')

# Display the plot
plt.show()

# Lets check if drug or alcohol Police stops are related to higher amount of search conducted. 
# Comparing drug or alcohol and search rates
# Calculate and save the annual search rate
annual_search_rate = ri.search_conducted.resample('A').mean()

# Concatenate 'annual_drug_rate' and 'annual_search_rate'
annual = pd.concat([annual_drug_alc_rate, annual_search_rate], axis='columns')

# Create sub-plots from 'annual'
annual.plot(subplots=True)

# Display the sub-plot
plt.show()



### Data Visualisation####
#Calculating the hourly arrest rate. What time Police is usually busy?
# When a police officer stops a driver, a small percentage of those stops ends in an arrest. This is known as the arrest rate. 
# Calculate the overall arrest rate
print(ri.arrest_made.mean())

# Calculate the hourly arrest rate
print(ri.groupby(ri.index.hour).arrest_made.mean())

# Save the hourly arrest rate
hourly_arrest_rate = ri.groupby(ri.index.hour).arrest_made.mean()


# Visualisation of Data set. Plotting the hourly arrest rate
# Create a line plot of 'hourly_arrest_rate'
plt.plot(hourly_arrest_rate)

# Add the xlabel, ylabel, and title
plt.xlabel('Hour')
plt.ylabel('Arrest Rate')
plt.title('Arrest Rate by Time of Day')

# Display the plot
plt.show()


