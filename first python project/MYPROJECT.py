import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# setting display options to view all columns and rows to improve the analysis
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

# Loading application data to understand customers details for analysing load defaulters
data1 = pd.read_csv(r"C:\Users\Tharushi Nadeeshani\Desktop\pythonProject\application_data.csv")
data1.head(30)

# check structure of all columns
data1.info(True)

# observe statistical information of the application data of numeric columns
data1.describe()

# check for missing data of all columns
data1.isna().sum()

# check percentage of missing values in each column
data1_missing_value_percentage = 100*data1.isnull().sum()/len(data1)
# get the columns having missing value percentage more than 50 to a list
print(data1_missing_value_percentage[data1_missing_value_percentage >= 50])

# according to above observation, create list of columns with missing value percentage >50
data1_missing_value_percentage_column = data1_missing_value_percentage[data1_missing_value_percentage >= 50]
print(data1_missing_value_percentage_column)

# drop columns with percentage of missing value >50
data_1 = data1.drop(columns=data1_missing_value_percentage_column.index)
# validating the shape after dropping the columns
print(data_1.shape)

# observe value counts of OCCUPATION_TYPE column
print(data_1.OCCUPATION_TYPE.value_counts())

# plot the bar plot for all the occupation and observe the graph
data_1['OCCUPATION_TYPE'].value_counts().plot(kind='pie')

# plot the bar plot for all the occupation and observe the graph
data_1['OCCUPATION_TYPE'].value_counts().plot(kind='bar')

# It has been observed that flag_document column doesn't provide any information regarding the type of the document
# doesn't seems to be relevent with achieve the business objective,
# hence dropping these columns to improve effectiveness of the analysis

# drop the unwanted columns and storing it in new dataframe which will be used for further analysis
data_2 = data_1.drop(columns=['FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_6',
                              'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10',
                              'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14',
                              'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18',
                              'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21', 'FLAG_EMP_PHONE',
                              'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL', 'REGION_RATING_CLIENT',
                              'REGION_RATING_CLIENT_W_CITY', 'DAYS_LAST_PHONE_CHANGE', 'AMT_REQ_CREDIT_BUREAU_DAY',
                              'AMT_REQ_CREDIT_BUREAU_WEEK', 'OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT SOCIAL_CIRCLE',
                              'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE', 'DAYS LAST_PHONE_CHANGE',
                              'AMT_REQ_CREDIT BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_QRT',
                              'AMT_REQ_CREDIT_BUREAU_YEAR', 'NAME_TYPE_SUITE', 'REGION_POPULATION_RELATIVE',
                              'WEEKDAY_APPR_PROCESS_START',
                              'HOUR_APPR_PROCESS_START', 'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION',
                              'LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY',
                              'LIVE_CITY_NOT_WORK_CITY'])
print(data_2.shape)

# checking the data types for the selected columns
print(data_2.info())

# create new dataframe for target=1
data_2_target_1 = data_2[data_2['TARGET'] == 1]
print(data_2_target_1.head())

# create new dataframe for target=0
data_2_target_0 = data_2[data_2['TARGET'] == 0]
print(data_2_target_0.head())

# plot heatmap to find correlation between all numerical variables when target variable is 0
plt.figure(figsize=(25, 12))
sns.heatmap(data_2_target_0.corr(), annot=True, cmap="RdYIGn", center=0.4)
plt.title('Correlation for target variables')
plt.show()

# plot heatmap to find correlation between all numerical variables when target variable is 1
plt.figure(figsize=(25, 12))
sns.heatmap(data_2_target_1.corr(), annot=True, cmap="RdYIGn", center=0.4)
plt.title('Correlation for target variable 1')
plt.show()

# plot graph for Days_birth column for analysis
plt.figure(figsize=(20, 10))

plt.subplot(1, 2, 1)
ax1 = sns.boxplot(y=data_2_target_1['DAYS_BIRTH'])
plt.title('Clients with repay difficulties')

plt.subplot(1, 2, 2)
ax2 = sns.boxplot(y=data_2_target_0['DAYS_BIRTH'])
plt.title('Clients without repay difficulties')
plt.show()

plt.subplot(1, 2, 1)
ax3 = sns.boxplot(y=data_2_target_1['AMT_INCOME_TOTAL'])
plt.title('Clients with repay difficulties')

plt.subplot(1, 2, 2)
ax4 = sns.boxplot(y=data_2_target_0['AMT_INCOME_TOTAL'])
plt.title('Clients without repay difficulties')
plt.show()

# plot NAME CONTRACT_TYPE, CODE GENDER, Owning Cars and FLAG_OWN_REALTY columns for analysis
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Making a plot with 2 subplots
contract_type = data_2.NAME_CONTRACT_TYPE.value_counts()
code_genders = data_2.CODE_GENDER.value_counts()
owns_cars_flag = data_2.FLAG_OWN_CAR.value_counts()
own_reality_flag = data_2.FLAG_OWN_REALTY.value_counts()

fig, ax = plt.subplots(2, 2, figsize=(12, 10))
fig.tight_layout(h_pad=7, w_pad=13)

sns.barplot(x=contract_type.index, y=contract_type, ax=ax[0][0])
ax[0][0].set_title("Distribution for CONTRACT_TYPE", fontsize=12)
ax[0][0].set_xticklabels(['Cash loans', 'Revolving loans'])
sns.barplot(x=code_genders.index, y=code_genders, ax=ax[0][1])
ax[0][1].set_title("Distribution for Genders", fontsize=12)
ax[0][1].set_xticklabels(['F', 'M', 'XNA'])
sns.barplot(x=owns_cars_flag.index, y=owns_cars_flag, ax=ax[1][0])
ax[1][0].set_title("Distribution for Customers Owning Cars", fontsize=12)
ax[1][0].set_xticklabels(['No', 'Yes'])
sns.barplot(x=own_reality_flag.index, y=own_reality_flag, ax=ax[1][1])
ax[1][1].set_title("Distribution for Customers Owning House", fontsize=12)
ax[1][1].set_xticklabels(['Yes', 'No'])

plt.show()

# function to plot boxplot of Numerical vs Categorical variable


def box_plot_numerical_categorical(num_var, cat_var):
    plt.figure(figsize=(17, 6))

    plt.subplot(1, 2, 2)
    ax_plot = sns.boxplot(data=data_2_target_1, y=num_var, x=cat_var)
    plt.title('Clients with payment difficulties')
    plt.xticks(rotation=90)
    plt.show()
    return ax_plot

# plot the Distributions for 'AMT_CREDIT' vs 'NAME_CONTRACT_TYPE'


box_plot_numerical_categorical('AMT_CREDIT', 'NAME_CONTRACT_TYPE')

# plot the Distributions for 'AMT_CREDIT' vs 'NAME_INCOME_TYPE'
box_plot_numerical_categorical('AMT_CREDIT', 'NAME_INCOME_TYPE')

# Categorical-categorical variables
# make function to plot categorical-categorical variables


def box_plot_categorical_categorical(cat1, cat2):
    plt.figure(figsize=(18, 8))

    plt.subplot(1, 2, 2)
    ax_plot = sns.boxplot(data=data_2_target_1, x=cat1, hue=cat2)
    plt.title('Clients with payment difficulties')
    plt.legend(loc='best')
    plt.xticks(rotation=90)
    plt.show()
    return ax_plot

# plot the Distributions for 'OCCUPATION_TYPE' vs 'CODE_GENDER'


box_plot_categorical_categorical('OCCUPATION_TYPE', 'CODE_GENDER')

# plot the Distributions for 'FLAG_OWN_REALTY' vs 'CODE_GENDER'
box_plot_categorical_categorical('FLAG_OWN_REALTY', 'CODE_GENDER')

# plot the Distributions for 'OCCUPATION_TYPE' vs 'NAME_FAMILY_STATUS'
box_plot_categorical_categorical('OCCUPATION_TYPE', 'NAME_FAMILY_STATUS')
