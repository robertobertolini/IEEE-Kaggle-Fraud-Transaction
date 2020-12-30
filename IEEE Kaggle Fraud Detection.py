#!/usr/bin/env python
# coding: utf-8

# # Roberto Bertolini
# # IEEE Fraud Detection

# ## Part 1 - Examination of Fraudulent vs Non-Fraudulent Transactions

# In[1]:


# Read in the training data sets and merge them together

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train_transaction = pd.read_csv('train_transaction.csv') # (590540,394)
train_identity = pd.read_csv('train_identity.csv') # (144233,41) # contains device info

print('Train_Transaction Shape: ',train_transaction.shape,'   Train_Identity Shape: ',train_identity.shape)

# Merge the training data together
train = pd.merge(train_transaction,train_identity, how='outer',on='TransactionID')

print('Merged Training Data Shape: ',train.shape)


# In[2]:


# For exploratory data analysis, I will select and explore the following variables:

quest_1_5 = ['TransactionID','isFraud','DeviceType','DeviceInfo',
             'TransactionDT','TransactionAmt','ProductCD',
             'card4','card6','P_emaildomain','R_emaildomain',
             'addr1','addr2','dist1','dist2']

train_q1_5 = train[quest_1_5] # Create subset of the training data


# https://stackoverflow.com/questions/10369681/how-to-plot-bar-graphs-with-same-x-coordinates-side-by-side-dodged

# In[3]:


# Visualization of missing values in the training data.

# Generic function for plotting a side-by-side bar chart
def sideBarChart(data1,data2,limits,legend_1,legend_2,y_label,graph_title,y_step):
    fig = plt.figure() # Create matplotlib figure
    bar_width = 0.3 # Bar width
    ax = fig.add_subplot(111) # Create matplotlib axes
    data1.plot(kind='bar', color='green', ax=ax, position=1, width=bar_width,label=legend_1)
    data2.plot(kind='bar', color='blue', ax=ax,  position=0, width=bar_width,label=legend_2)
    plt.legend([legend_1,legend_2],loc='best') # Add legend   
    ax.set_ylabel(y_label) # Set the y-label
    ax.set_title(graph_title) # Set the graph title
    ax.set_ylim(limits) # Set the y-limits
    ax.set_yticks(np.arange(limits[0], limits[1], step=y_step)) # Arrange the y ticks

train_fraud = train_q1_5[train_q1_5["isFraud"]==1] # Select fraudulent cases from the training data
train_no_fraud = train_q1_5[train_q1_5["isFraud"]==0] # Select non-fraudulent cases from the training data

sideBarChart(pd.DataFrame(train_fraud.isnull().sum()*100/len(train_fraud))[0],
                pd.DataFrame(train_no_fraud.isnull().sum()*100/len(train_no_fraud))[0],
               [0,110],'Fraud','No Fraud','Percentage','Percentage of Missing Data - Fraud and Not Fraud',10)


# There is a large percentage of data missing in the training data: for non-fraudulent and fraudulent transactions, the variable "dist2" has the highest percentage of missing data at 90% and 80%, respectively. There are four times as many missing data values for the variables "addr1" and "addr2" for fraudulent cases, compared to non-fraudulent cases. 

# ## Part 1A: Distribution of the values for the limited set of predictors for fraudulent transactions

# ## Device Type

# In[5]:


# Create a generic function to plot a bar chart and display the frequency on top of the bar. I include the missing
# values as their own bar because they are prevalent in the categorical predictors of the dataset

def createBarChart(variable,x_axis_labels,title,y_limits,dataset):
    x_position = np.arange(len(x_axis_labels)) # Create the xticks
    plt.bar(x_position,dataset[variable].value_counts(dropna=False), align='center') # Include missing values 
    plt.xticks(x_position, x_axis_labels,rotation=90) # Rotate the x-ticks
    plt.ylabel('Frequency') # Y-label
    plt.title(title) # Title
    plt.ylim(y_limits) # Y-limits

# Add the frequency count on the top of each barplot
    for i in range(len(x_axis_labels)):
        plt.text(x = x_position[i], 
                 y = dataset[variable].value_counts(dropna=False)[i]+200, 
                 s = dataset[variable].value_counts(dropna=False)[i],
                 size = 13, ha = 'center')
    plt.show()
    
createBarChart('DeviceType',('NA', 'Desktop','Mobile'),'Fraud: Device Type',[0,11000],train_fraud)


# The majority of entries, 9452 (45.7%), in the Device Type variable for fraudulent transactions, are missing. Desktop and mobile devices account for 5657 (27.3%) and 5554 (26.9%) fradulent transactions, respectively. 

# ## DeviceInfo

# In[6]:


# There were many miscellaneous categories so I only selected the top four categories (including missing entries) and 
# made the remaining categories "Other"

device = train_fraud['DeviceInfo'].value_counts(dropna = False)[:4] # Select the top 4 categories
common_device = list(device) # Convert this into a list
mod_device = train_fraud['DeviceInfo'].apply(lambda x: x if x in device else 'Other') # Make all other categories "Other"
mod_device = pd.DataFrame({'Id':mod_device.index, 'DeviceInfo':mod_device.values})

# Plot the bar chart
createBarChart('DeviceInfo',('NA', 'Other', 'Windows','iOS Device', 'MacOS'),'Fraud: Device Info',[0,13000],mod_device)


# Like the variable Device Type, the majority of fraudulent entries are missing information for Device Info [12,056 (58.3%)]. Windows, iOS Device, and MacOS account for 3,121 (15.1%), 1240 (6.0%), and 278 (1.3%) of fraudulent transactions, respectively.

# ## ProductCD

# In[7]:


createBarChart('ProductCD',('W','C','H','R','S'),'Fraud: Product CD',[0,10000],train_fraud)


# W and C were the products with the highest number of fraudulent transactions: 8969 (43.4%) and 8008 (38.8%) transactions, respectively.
# Product S has the lowest frequency of fraudulent transactions [686 (3.3%)]

# ## card4 

# In[8]:


createBarChart('card4',('Visa', 'Mastercard', 'Discover','American Express', 'NaN'),'Fraud: Card 4',[0,15000],train_fraud)


# Visa had the highest frequency of fraudulent transactions [13373 (64.7%)] followed by Mastercard [ 6496 31.4%)]. American Express had the lowest number of fraudulent transactions [239 (1.2%)]. 

# ## card6

# In[9]:


createBarChart('card6',('Debit','Credit','NaN'),'Fraud: Card 6',[0,12000],train_fraud)


# The frequency of debit and credit fraudulent transactions are roughly equal: 10,674 (51.7%) and 9,950 (48.2%) for debit and credit, respectively. 

# ## P_emaildomain and R_emaildomain

# There was one issue I found with the P_emaildomain address in the training data. Some of the responses only had gmail instead of gmail.com. This was corrected below.

# In[10]:


train_fraud['P_emaildomain'][train_fraud[train_fraud['P_emaildomain']=='gmail'].index] = "gmail.com"


# In[11]:


# For both email domains, I selected only the top 6 email addresses and grouped the rest into another category entitled 'Other'

# Extract the top 6 P_emaildomain addresses and group the rest into the category Other
p_domain = train_fraud['P_emaildomain'].value_counts(dropna = False)[:7]
p_domain_common = list(p_domain)
mod_p_domain = train_fraud['P_emaildomain'].apply(lambda x: x if x in p_domain else 'Other')

# Extract the top 6 R_emaildomain addresses and group the rest into the category Other
r_domain = train_fraud['R_emaildomain'].value_counts(dropna = False)[:7]
r_domain_common = list(r_domain)
mod_r_domain = train_fraud['R_emaildomain'].apply(lambda x: x if x in p_domain else 'Other')

# Plot both email addresses on the same plot using a side-by-side bar chart

sideBarChart(mod_p_domain.value_counts(),mod_r_domain.value_counts(),[0,11000],'P Email','R Email','Frequency',
                'Email Domain: Fraud Transactions',1000)


# gmail.com is the most common email domain for the fraudulent transactions, followed by hotmail.com and yahoo.com for both the P and R email domain.

# ## Dist 1 and Dist 2

# In[12]:


# Check to see whether any observations contain entries in both variables dist1 and dist2.
dist_na = train_fraud[['dist1','dist2']].notna()
dist_na = dist_na[((dist_na['dist1'] == True) & (dist_na['dist2'] == True))]
#dist_na.shape # (0,2): it just prints the column names so no samples exhibit this property


# In[13]:


# I will only plot the non-missing entries for distance 1 and distance 2 since they are numerical quantities. 

# Extract the non-missing entries of dist1
dist1_notna = pd.DataFrame(train_fraud['dist1'].notna())
dist1_notna = dist1_notna[dist1_notna['dist1']==True]
dist1_not_na_index = train_fraud[train_fraud.index.isin(dist1_notna.index)] # Create another dateframe

# Extract the non-missing entries of dist2
dist2_notna = pd.DataFrame(train_fraud['dist2'].notna())
dist2_notna = dist2_notna[dist2_notna['dist2']==True]
dist2_not_na_index = train_fraud[train_fraud.index.isin(dist2_notna.index)] # Create another dataframe


# The distances are significantly skewed to the right. Moreover, some distance values are 0. Therefore, the transformation log(1+dist) and log(1+dist2) was applied to reduce skeweness and since the log of 0 is not defined. 

# In[14]:


dist1_not_na_index['dist1'] = np.log(1 + dist1_not_na_index['dist1']) # Perform the log transformation
dist2_not_na_index['dist2'] = np.log(1 + dist2_not_na_index['dist2']) # Perform the log transformation


# In[15]:


# Semi-log plot of distance 1 and distance 2 as subplots. 

# Create a 2 x 1 matrix for the plots
fig, axes = plt.subplots(2, 1)

# Log(Distance 1 + 1)
dist1_not_na_index.hist('dist1', bins=50, ax=axes[0])
axes[0].set_ylabel('Frequency')
axes[0].set_title('Fraud: Distance 1')
axes[0].set_xlabel('Log(Distance 1 + 1)')

# Log(Distance 2 + 1)
dist2_not_na_index.hist('dist2', bins=50, ax=axes[1])
axes[1].set_ylabel('Frequency')
axes[1].set_title('Fraud: Distance 2')
axes[1].set_xlabel('Log(Distance 2 + 1)')

# Customize plot specifications including the amount of horizontal space and the size
fig.subplots_adjust(hspace=.3)
fig.set_size_inches(10, 10)


# There is a higher frequency of log(dist1+1) values that lie between 2 and 4 compared to log(dist2+1) values at this same interval. Moreover, there is a higher frequency of log(dist2 +1) values that lie between 5 and 6 compared to log(dist1+1)

# # Address 1

# In[16]:


# Bucketed Bar Chart for Address 1 Fraudulent Transactions. The buckets were used because addresses are discrete quantities.

bins= [100,200,300,400,500,600]
plt.hist(train_fraud['addr1'], bins=bins, edgecolor="k")

plt.xticks(bins)
plt.xlabel('Address 1')
plt.ylabel('Frequency')
plt.title('Fraud: Address 1')
plt.show()


# The majority of fraudulent transactions are have an address1 between [200,300]. The least amount lies between [500,600]

# # Address 2

# In[17]:


# Bar chart for addr2

dist_addr2 = train_fraud
dist_addr2 ['addr2'] = dist_addr2 ['addr2'].astype(str) # Convert the variable into a string

# Extract the top three addr2's and make the rest fall into the Other category.
eve= dist_addr2 ['addr2'].value_counts()[:3]
common_eve = list(eve)
eve_mod= dist_addr2 ['addr2'].apply(lambda x: x if x in eve else 'Other')

# Create a separate dataframe just with the addr2 information
eve_mod = pd.DataFrame({'Id':eve_mod.index, 'addr2':eve_mod.values})
eve_mod['addr2'].value_counts()

# Plot a bar chart
createBarChart('addr2',('"87"', 'NaN', '"60"','Other'),'Fraud: Address_2',[0,14000],eve_mod)


# The most frequency address2 is "87" encompassing 12,477 (60.4%) of the data. 

# ## Part 1A Observations
# 
# The majority of entries, 9452 (45.7%), in the Device Type variable for fraudulent transactions, are missing. Desktop and mobile devices account for 5657 (27.3%) and 5554 (26.9%) fradulent transactions, respectively. 
# 
# Like the variable Device Type, the majority of fraudulent entries are missing information for Device Info [12,056 (58.3%)]. Windows, iOS Device, and MacOS account for 3,121 (15.1%), 1240 (6.0%), and 278 (1.3%) of fraudulent transactions, respectively.
# 
# W and C were the products with the highest number of fraudulent transactions: 8969 (43.4%) and 8008 (38.8%) transactions, respectively.
# 
# Product S has the lowest frequency of fraudulent transactions [686 (3.3%)]
# 
# The frequency of debit and credit fraudulent transactions are roughly equal: 10,674 (51.7%) and 9,950 (48.2%) for debit and credit, respectively. 
# 
# gmail.com is the most common email domain for the fraudulent transactions, followed by hotmail.com and yahoo.com for both the P and R email domain.
# 
# The most frequency address2 is "87" encompassing 12,477 (60.4%) of the data. 

# https://matplotlib.org/3.1.0/gallery/subplots_axes_and_figures/subplots_demo.html

# # Part 1B: Distribution of fraudulent and non-fraudulent transactions

# In[19]:


# Plot the kernel density estimation. This is a non-parametric representation of the probability density function

sns.kdeplot(np.log(train_fraud['TransactionAmt'])) # Log(fraudulent transaction amount)
sns.kdeplot(np.log(train_no_fraud['TransactionAmt'])) # Log(non-fraudulent transaction amount)
plt.xlabel("Log(Transaction Amount)") # X-label
plt.ylabel("Kernel Density Estimate") # Y-label
plt.title("Distribution of Transaction Amount for Fraud and Not Fraud Transactions") # Title
plt.legend(['Fraud', 'Not Fraud']) # Legend specifications


# ## Part 1B Observations
# 
# The distribution of the logarithm of the transaction amount was plotted for fraudulent (blue) and non-fraudulent data (orange). 
# 
# The distribution of the fraudulent transactions has a slighter higher kernel density estimate for the non-fraudulent transactions when log(transaction amount) > 6. 
# 
# The distribution of non-fraudulent transactions appears bimodal while the fraudulent transactions is unimodal.

# ## Part 2 - Transaction Frequency

# "87" is the most frequent address2. I will convert the time to UTC via Python's datetime commands to try and discern the times when the unlawful transactions occurred. I will focus on the hour the transaction took place.

# In[20]:


import datetime

# The default time is elapsed from 1/1/1970 UTC (I do not care about the year)
unix_time = '1970-01-01'
unix_time = datetime.datetime.strptime(unix_time, "%Y-%m-%d")

# Convert the fraudulent transaction dates into proper dates and extract the hour
# and day of the month that the transaction took place.

train_transaction_fraud = train_fraud # Create another dataframe

# Convert the transaction date to a comprehensible date time in UTC time and extract the hour
# and day of the transaction
train_transaction_fraud["Date"] = train_transaction_fraud['TransactionDT'].apply(lambda x: (unix_time + datetime.timedelta(seconds=x)))
train_transaction_fraud['Hour'] = train_transaction_fraud['Date'].dt.hour # Extract the hour
train_transaction_fraud['Day'] = train_transaction_fraud['Date'].dt.day # Extract the day

# In this case all hours are rounded down (for example: 00:00-00:59 all take place in 00:00 hours)

# Extract the non-missing entries
train_transaction_nnull_fraud = train_transaction_fraud[train_transaction_fraud['addr2'].notnull()]

# Extract all entries with the 87 country code, the most frequent field in address2
train_transaction_addr2_87_fraud = train_transaction_nnull_fraud[train_transaction_nnull_fraud['addr2']=='87.0']

# Apply the same procedure to the non-fraudulent transactions
train_transaction_no_fraud = train_no_fraud # Create a separate dataframe
train_transaction_no_fraud["Date"] = train_transaction_no_fraud['TransactionDT'].apply(lambda x: (unix_time + datetime.timedelta(seconds=x)))
train_transaction_no_fraud['Hour'] = train_transaction_no_fraud['Date'].dt.hour # Extract the hour
train_transaction_no_fraud['Day'] = train_transaction_no_fraud['Date'].dt.day # Extract the day

# Extract the non-missing entries
train_transaction_nnull_no_fraud = train_transaction_no_fraud[train_transaction_no_fraud['addr2'].notnull()]

# Extract all entries with the 87 country code, the most frequent field in address2
train_transaction_addr2_87_no_fraud = train_transaction_nnull_no_fraud[train_transaction_nnull_no_fraud['addr2']==87.0]

# Create the x-axis time ticks
time_ticks = ['00:00','01:00','02:00','03:00','04:00','05:00','06:00','07:00',
          '08:00','09:00','10:00','11:00','12:00','13:00','14:00','15:00',
          '16:00','17:00','18:00','19:00','20:00','21:00','22:00','23:00']


# In[21]:


# Plot a line graph of the fradulent transactions
plt.figure(figsize=(5,5)) # Size of figure
plt.plot(time_ticks,np.bincount(train_transaction_addr2_87_fraud['Hour'])[0:24],marker='o', color='b') # Tick marks
plt.xlabel('Hour (00:00 - 23:00) UTC') # X-label
plt.ylabel('Frequency') # Y-label
plt.xticks(time_ticks, rotation='vertical') # X-ticks
plt.yticks(np.arange(0,1100,100)) # Y-ticks
plt.title('Transaction Amounts by Time of Date: Fraud') # Title
plt.grid(b=True) # Display the grid
plt.show()


# In[22]:


# Non-fraudulent transactions
plt.figure(figsize=(5,5)) # Size of figure
plt.plot(time_ticks,np.bincount(train_transaction_addr2_87_no_fraud['Hour'])[0:24],marker='o', color='g') # Tick marks
plt.xlabel('Hour (00:00 - 23:00) UTC') # X-label
plt.ylabel('Frequency') # Y-label
plt.xticks(time_ticks, rotation='vertical') # X-ticks
plt.yticks(np.arange(0,40000,5000)) # Y-ticks
plt.title('Transaction Amounts by Time of Date: No Fraud') # Title
plt.grid(b=True) # Display grid
plt.show()


# ## Part 2 Solution
# 
# Based on both line charts, the most frequent transactions (fraudulent and non- fraudulent) occur between the hours of 15:00 to 01:00 UTC time. This corresponds to 11:00 A.M. - 8:00 P.M. EST.

# ## Part 3 - Product Code

# For this part, I tried to identify what the least and most expensive products were (we were not provided this information in the corpus).

# In[23]:


# Plot the frequency of each item by fraudulent and non-fraudulent transactions
createBarChart('ProductCD',('W','C','H','R','S'),'Fraud: Product CD',[0,10000],train_fraud) # Fraudulent
createBarChart('ProductCD',('W','C','H','R','S'),'No Fraud: Product CD',[0,500000],train_no_fraud) # Non-Fraudulent


# W is the most common transaction encompassing 8969 (43.4%) and 430,701 (75.6%) fraudulent and non-fraudulent transactions, respectively.
# 
# For each product, I examined the proportion of fraudulent and non-fraudulent transactions

# I used the melt command in Python to reshape the data into a table so I could plot a stacked bar chart of the proportions.
# 
# https://www.geeksforgeeks.org/python-pandas-melt//

# In[24]:


# Use the melted command from pandas
melted_df = pd.melt(train_q1_5, id_vars=['isFraud'], value_vars=['ProductCD']) # Melt the datafrmae
no_fraud_tab = melted_df[melted_df['isFraud']==0]['value'].value_counts() # Count the number of non-fraud transactions
fraud_tab = melted_df[melted_df['isFraud']==1]['value'].value_counts() # Count the number of fraud transactions

tab_concat = pd.concat([no_fraud_tab, fraud_tab], axis=1) # Combine fraudulent and non-fraudulent sums for each product

# Divide to yield the proportion of fraudulent and non-fraudulent  transactions for each product
tab_concat = tab_concat.apply(lambda x: x / x.sum(),axis=1)
tab_concat


# In[25]:


# Stacked bar chart quantifying the proportion of fraudulent and non-fraudulent transactions for each product
# see table above)

x_tick_marks = ['C','H','R','S','W'] # Set the x labels

p1 = plt.bar(x_tick_marks,tab_concat.iloc[:,0])
p2 = plt.bar(x_tick_marks, tab_concat.iloc[:,1],bottom=tab_concat.iloc[:,0])

plt.ylabel('Proportion')
plt.title('Proportion of Fraduelent and Non-Fraduelent Transactions')
plt.legend((p1[0], p2[0]), ('No Fraud', 'Fraud'),loc='best')
plt.show()

# The proportion of products with the highest fraud is C (11.7%) followed by S (5.9%)


# While W had the highest frequency of transactions, the product which had the highest percentage of fraudulent transactions are C (11.7%) followed by S (5.9%)

# Finally, I created side-by-side boxplots to show the distribution of log(transaction amount) for each product, broken down by fraudulent and non-fraudulent transactions.

# In[26]:


# Use seaborn to create a family of boxplots depicting the fraudulent and non-fraudulent transactions for each product
sns.set(style="whitegrid")
ax = sns.boxplot(x="ProductCD", y=np.log(train_q1_5["TransactionAmt"]), hue = 'isFraud',data= train_q1_5) # Seaborn boxplot
ax.set_xlabel('Product CD') # Set the x label
ax.set_ylabel('Log(Transaction Amount +1)') # Set the y label
ax.set_title('Boxplot of Log(Transaction Amount +1) by Fraud/Transactions') # Set the title
legend = ax.legend() # Create the legend
legend.get_texts()[0].set_text('No Fraud')
legend.get_texts()[1].set_text('Fraud')


# ## Question 3 Summary and Answer 
# Based on the distribution of log transaction amount, product R has a slighter higher median log(transaction amount) compared to the other products. Along with product H, product R's transaction amounts exhibit the least amount of variability. C appears to be the least expensive product due to the prevalent amount of outliers for both fraudulent and non-fraudulent transactions at the bottom of the plot. 
# 
# I believe R is most expensive product. While W is the most frequently purchased product (across both fraudulent and non-fraudulent transactions), the variance of W's transaction amount is high suggesting that some of its products are most expensive and others. Compared to W, R has much less variability in the transaction amount.  
# 
# I believe that C is the least expensive product. Not only does it have the highest percentage of frauds (11.7%) but based on the boxplot, has lower transactions amounts compared to the other products. This would imply that fraudulent transactions tend to be smaller amounts.

# ## Part 4 - Correlation Coefficient

# I calculated the correlation coefficient between the fraudulent and non-fraudulent transactions. Therefore, I will need to perform the same data manipulation as I did in Question 2 to convert the variable TransactionDT into UTC time. I will compute the correlation coefficient between the hour that the transaction was performed and the transaction amount. 

# In[27]:


# Convert TransactionDT to UTC time for all fraudulent and non-fraudulent transactions
unix_time = '1970-01-01'
unix_time = datetime.datetime.strptime(unix_time, "%Y-%m-%d")
train_all = train_q1_5
train_all["Date"] = train_all['TransactionDT'].apply(lambda x: (unix_time + datetime.timedelta(seconds=x)))
train_all['Hour'] = train_all['Date'].dt.hour
train_all['Day'] = train_all['Date'].dt.day


# In[28]:


# To visualize the distribution of the transaction amount, I will 
# create boxplots depicting the log of the transaction amount.

sns.set(style="whitegrid")
ax = sns.boxplot(x="Hour", y=np.log(train_q1_5["TransactionAmt"]), data=train_q1_5) # Boxplot
ax.set_xlabel('Hour [00:00-23:00] UTC') # X-label
ax.set_ylabel('Log(Transaction Amount)') # Y-label
ax.set_xticklabels(time_ticks,rotation=90) # Rotate tick marks
ax.set_title('Boxplot of Log(Transaction Amount) by Hour for all Transactions') # Title

# Print the correlation coefficient between hour and transaction amount
print(np.corrcoef(train_q1_5['Hour'],train_q1_5['TransactionAmt']))


# The boxplots indicate that the median log(transaction amount) is uniform across all hours of the day. Moreover, the correlation coefficient is close to 0 (0.04), indicating no association between the transaction amount and time of the day of the purchase.

# ## Part 5 - Time Series Plot

# I chose to generate a time series plot showing the percentage of fraudulent transactions averaged over the total amount of transactions per day for the training data. 
# 
# Since we do not know the year of the transactions, I will start the plot at 1-1-1970

# In[29]:


import plotly.graph_objects as go
import matplotlib.dates as dates

startdate = datetime.datetime.strptime('1970-01-01', '%Y-%m-%d')
train_q5 = train_q1_5
train_q5['TransactionDT'] = train_q5['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds = x)))
q5 = pd.DataFrame(train_q5.set_index('TransactionDT').resample('D').mean()['isFraud'])
q5['TransactionDT'] = q5.index
q5['isFraud'] = q5['isFraud']*100

fig = go.Figure([go.Scatter(x=q5['TransactionDT'], y=q5['isFraud'])])
fig.update_layout(title_text='Percentage of Fraudulent Transactions per Day for the Training Data',
                 xaxis=go.layout.XAxis(title=go.layout.xaxis.Title(text="Transaction Date")),
                yaxis=go.layout.YAxis(title=go.layout.yaxis.Title(text="Percentage of Fraudulent Transactions")))
fig.show()


# Note that this graph allows the user to toggle over a specific date and extract the percentage of fraudulent transactions.

# The largest amount of fraudulent transactions occur on March 1 (7%). January 26 had the lowest percentage of fradulent transactions (1.09%).
# 
# #### Note: I am using the year 1970 based on the unix UTC time since we are not provided with the year by Kaggle.

# ## Part 6 - Prediction Model

# The overview of my data mining pipeline is presented below: 
# 
# (1) Only variables with less than 50% percent of data were considered. In my initial two models, I excluded all M and V predictors in order to work with a small subset of the data and then applied this 50% cut-point. For models three and four, I considered all predictors initially and then removed variables with > 50% missing data.
# 
# (2) The training data (train_transaction and train_identity files) were merged and divided into training and validation sets encompassing 70% and 30% of the data, respectively. 
# 
# (3) Missing data were imputed the mice package in Python https://www.statsmodels.org/devel/generated/statsmodels.imputation.mice.MICEData.html. This package uses multiple imputation to fill in missing entries instead of simply using a single imputation such as the mean or median.
# 
# (4) The training and validation sets were standardized by centering and rescaling predictors to have a mean of 0 and variance of 1 using https://scikit-learn.org/stable/modules/preprocessing.html
# 
# (4b) In my second attempt, I used an oversampling technique entitled SMOTE (Synthetic Minority Oversampling Technique) to balance the number of fraudulent and non-fraudulent transactions in the training data by generating synthetic data points via a nearest-neighbor algorithm https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.SMOTE.html. This technique did not improve my Kaggle score substantially so it was only applied to the second model.
# 
# (5) Machine learning algorithms were applied to the training set to build my model:
#     - Model 1: Logistic regression with 5-fold cross-validation
#     - Model 2: Logistic regression with 5-fold cross-validation and SMOTE
#     - Model 3: Random forest (no cross-validation)
#     - Model 4: XGBClassifier (no cross-validation) * Best model
# 
# (6) Each model's performance on the validation set was evaluated using the accuracy, AUC (area under the ROC curve), sensitivity, specificity, and precision metrics for classification (see below)
# 
# (7) The model was applied to the testing data provided by Kaggle. The probabilites of a fraudulent transaction were used for Kaggle submission. 

# In[30]:


# Import the training and testing sets again

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import the training data

train_transaction = pd.read_csv('train_transaction.csv') # (590540,394)
train_identity = pd.read_csv('train_identity.csv') # (144233,41) 

print('Train_Transaction Shape: ',train_transaction.shape,'   Train_Identity Shape: ',train_identity.shape)

# Merge the training sets together
train = pd.merge(train_transaction,train_identity, how='outer',on='TransactionID')

print('Merged Training Data Shape: ',train.shape)

# Import the testing data

test_transaction = pd.read_csv('test_transaction.csv') # (590540,394)
test_identity = pd.read_csv('test_identity.csv') # (144233,41) - contains device info

print('Test_Transaction Shape: ',test_transaction.shape,'   Test_Identity Shape: ',test_identity.shape)

# Merge the training sets together
test = pd.merge(test_transaction,test_identity, how='outer',on='TransactionID')

print('Merged Test Data Shape: ',test.shape)


# In[31]:


# Remove all V and M columns and select all columns with less than 
# 50% missing data.

# Calculate the percentage of missing data for all the variables
list_of_cols = pd.DataFrame(train.isnull().sum()*100/len(train))

# Remove columns that start with V or M corresponding to the M and V variables
filter_cols = [col for col in train if col.startswith('V') | col.startswith('M')]
train_subset = train.drop(filter_cols,axis=1) # Drop these columns

# Calculate the percentage of missing data in all columns excluding the
# M and V columns. 

list2 = pd.DataFrame(train_subset.isnull().sum()*100/len(train_subset))
eve = list2[list2[0]<50] # Remove columns with less than 50% missing data
eve_index = pd.DataFrame(eve.index)
train_subset = train[eve_index[0]]
train_subset.shape # Now, we have 35 variables (34 predictors + isFraud category)


# In[32]:


# We will only use the top 6 email addresses for P_emaildomain (R_emaildomain
# did not meet the 50% missing data cut-off criteria)

train_subset['P_emaildomain'][train_subset[train_subset['P_emaildomain']=='gmail'].index] = "gmail.com"
p_domain = train_subset['P_emaildomain'].value_counts()[:7] # Select the top 6 email addresses
p_domain_common = list(p_domain)
mod_p_domain = train_subset['P_emaildomain'].apply(lambda x: x if x in p_domain else 'Other')

train_subset['P_emaildomain'] = train_subset.values


# In[33]:


# For the variable, TransactionDT, I will just use the hour of the
# transaction after it is converted to UTC time

import datetime
unix_time = '1970-01-01' 
unix_time = datetime.datetime.strptime(unix_time, "%Y-%m-%d")

train_transaction_fraud = train_subset
train_transaction_fraud["Date"] = train_transaction_fraud['TransactionDT'].apply(lambda x: (unix_time + datetime.timedelta(seconds=x)))
train_transaction_fraud['Hour'] = train_transaction_fraud['Date'].dt.hour
train_transaction_fraud['Day'] = train_transaction_fraud['Date'].dt.day

train_subset['TransactionDT'] = train_transaction_fraud['Hour']
train_subset = train_subset.iloc[:,0:len(train_subset.columns)-3]


# In[34]:


# Convert the categorical predictors into indicator variables, dropping the
# first indicator alphabetically for each variable to use as a baseline for my logistic
# regression model. 

# Note: I tried performing this step all at once but I continued to
# receive a memory error in Python.

# ProductCD
one_hot_productcd = pd.get_dummies(train_subset['ProductCD'],drop_first=True)
train_subset = train_subset.drop('ProductCD',axis=1)
train_subset = train_subset.join(one_hot_productcd)

# Card4
one_hot_card4 = pd.get_dummies(train_subset['card4'],drop_first=True)
train_subset = train_subset.drop('card4',axis=1)
train_subset = train_subset.join(one_hot_card4)

# Card6
one_hot_card6 = pd.get_dummies(train_subset['card6'],drop_first=True)
train_subset = train_subset.drop('card6',axis=1)
train_subset = train_subset.join(one_hot_card6)


# In[35]:


# I had to drop the P_emaildomain category since I received a memory error when creating indicator variables using 
# the top 6 domains and condensing the rest into the 'Other' category

# I also delete the TransactionID column since this is used for organizing the data corpora. 

train_subset = train_subset.drop('P_emaildomain',axis=1) # 1: denotes column 
train_subset =  train_subset.drop('TransactionID', 1) # 1: denotes column


# In[36]:


# Split the training data into a training and validation set encompassing 
# 70% and 30% of the data, respectively.

from sklearn.model_selection import train_test_split
train_dummy = train_subset
X_train, X_valid = train_test_split(train_dummy,test_size=0.3, random_state=42)


# In[37]:


# Use mice imputation to impute the training and validation datasets
import statsmodels.imputation.mice as mice

imp_X_train = mice.MICEData(X_train)
imputed_X_train = imp_X_train.data

imp_X_valid = mice.MICEData(X_valid)
imputed_X_valid = imp_X_valid.data


# In[38]:


# Extract the dependent variable and store it in a separate variable
# before deleting it. This is because I will standardize the independent predictors only

train_dependent = imputed_X_train.iloc[:,0]
valid_dependent = imputed_X_valid.iloc[:,0]
imputed_X_train = imputed_X_train.drop('isFraud', 1)
imputed_X_valid = imputed_X_valid.drop('isFraud', 1)


# In[39]:


# Center and re-scale the data so each independent predictor has a mean
# of 0 and variance 1

from sklearn import preprocessing

scal_imp_X_train = pd.DataFrame(preprocessing.scale(imputed_X_train))
scal_imp_X_valid = pd.DataFrame(preprocessing.scale(imputed_X_valid))


# In[40]:


# Perform SMOTE. This is only applied to the training data during my second attempt.

from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=2008)
X_smote,y_smote = sm.fit_resample(scal_imp_X_train,train_dependent)
scal_imp_X_train = X_smote
train_dependent = y_smote


# In[41]:


# Perform logistic regression with 5-fold cross-validation

from sklearn.linear_model import LogisticRegressionCV
clf=LogisticRegressionCV(cv=5, random_state=0,max_iter = 500).fit(scal_imp_X_train,train_dependent)
train_res=clf.predict_proba(scal_imp_X_train)
valid_res=clf.predict_proba(scal_imp_X_valid)


# In[42]:


# Extract the probability outcomes that were used to assign each 
# prediction to a binary outcome 

p = clf.predict(scal_imp_X_train)
p_prob = clf.predict_proba(scal_imp_X_train)
p_prob = p_prob[:, 1]


# In[43]:


# Generate predictions on the validation data
predictions = clf.predict(scal_imp_X_valid)


# In[44]:


# Print the confusion matrix and AUC
from sklearn import metrics
cm = metrics.confusion_matrix(valid_dependent, predictions)

from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
roc=roc_auc_score(valid_dependent, clf.predict_proba(scal_imp_X_valid)[:,1])

print('Confusion Matrix: ',cm, "   The AUC: ", roc)


# In[45]:


# Print the accuracy as instructed by the homework prompt
print('Accuracy of logistic regression classifier on validation set: {:.2f}'.format(clf.score(scal_imp_X_valid, valid_dependent)))


# # 1st Attempt
# The accuracy of the first model was very high: 0.970. Upon examining the confusion matrix, I noticed that this metric is misleading due to the large class imbalance. 
# 
# Accuracy: 0.97
# 
# AUC (area under the receiver operator characteristic curve): 0.78 
# 
# Sensitivity: 0.99 
# 
# Specificity: 0.57 Very low correct classification for fraudulent transactions
# 
# Precision: 0.97
# 
# Kaggle Score: 0.8277 
# 
# # 2nd Attempt
# Because of the large disparity in classes, I applied the oversampling technique entitled SMOTE with the hopes of improving my specificity and my Kaggle score.
# 
# While this method improved the specificity and correctly predicted more fradulent transactions than my first attempt, the AUC and my Kaggle scored remained relatively similar. Moreover, my accuracy decreased. Because of the small increase in my Kaggle score, I decided to not incorporate SMOTE into any further prediction attempts. 
# 
# Accuracy: 0.75
# 
# AUC: 0.79
# 
# Sensitivity: 0.75
# 
# Specificity: 0.68 (A much better improvement)
# 
# Precision: 0.98
# 
# Kaggle Score: 0.8279 (Surprisingly, this did not improve much)

# ## Model 3 and Model 4

# In[46]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train_transaction = pd.read_csv('train_transaction.csv') # (590540,394)
train_identity = pd.read_csv('train_identity.csv') # (144233,41) - contains device info

print('Train_Transaction Shape: ',train_transaction.shape,'   Train_Identity Shape: ',train_identity.shape)

# Merge the training sets together
train = pd.merge(train_transaction,train_identity, how='outer',on='TransactionID')

print('Merged Training Data Shape: ',train.shape)


test_transaction = pd.read_csv('test_transaction.csv') # (590540,394)
test_identity = pd.read_csv('test_identity.csv') # (144233,41) - contains device info

print('Test_Transaction Shape: ',test_transaction.shape,'   Test_Identity Shape: ',test_identity.shape)

# Merge the training sets together
test = pd.merge(test_transaction,test_identity, how='outer',on='TransactionID')

print('Merged Test Data Shape: ',test.shape)


# In[47]:


# Remove columns with more than 50% missing data

list_of_cols = pd.DataFrame(train.isnull().sum()*100/len(train))
eve = list_of_cols[list_of_cols[0]<50]
eve_index = pd.DataFrame(eve.index)
train_subset = train[eve_index[0]]
train_subset.shape


# In[48]:


# Identify the categorical predictors: the ones that are not integers or floats
train_subset.select_dtypes(exclude=['int','float']).columns


# In[49]:


# TransactionID: This is only used for organizational purposes so I will remove it
train_subset =  train_subset.drop('TransactionID', axis=1)


# In[50]:


# isFraud: I will temporarily store this variable in case I need it later. 

train_dependent = train_subset['isFraud']


# In[51]:


# ProductCD: use one hot encoding to convert this categorical predictor into a numeric predictor, remove the first
# product alphabetically as the reference

one_hot_productcd = pd.get_dummies(train_subset['ProductCD'],drop_first=True)
train_subset = train_subset.drop('ProductCD',axis=1)
train_subset = train_subset.join(one_hot_productcd)


# In[52]:


# Card4 - one hot encoding, dropping the baseline
one_hot_card4 = pd.get_dummies(train_subset['card4'],drop_first=True)
train_subset = train_subset.drop('card4',axis=1) # Drop original variable
train_subset = train_subset.join(one_hot_card4)


# In[53]:


# Card6 - one hot encoding, dropping the baseline

one_hot_card6 = pd.get_dummies(train_subset['card6'],drop_first=True)
train_subset = train_subset.drop('card6',axis=1) # Drop original variable
train_subset = train_subset.join(one_hot_card6)


# In[54]:


# Convert TransactionDT to UTC time. I will only use the hour that the
# transaction was recorded

import datetime
unix_time = '1970-01-01' # I will just use the default time (I am not concerned about the year)
unix_time = datetime.datetime.strptime(unix_time, "%Y-%m-%d")

train_transaction_fraud = train_subset
train_transaction_fraud["Date"] = train_transaction_fraud['TransactionDT'].apply(lambda x: (unix_time + datetime.timedelta(seconds=x)))
train_transaction_fraud['Hour'] = train_transaction_fraud['Date'].dt.hour
train_transaction_fraud['Day'] = train_transaction_fraud['Date'].dt.day

train_subset['TransactionDT'] = train_transaction_fraud['Hour']
train_subset = train_subset.iloc[:,0:len(train_subset.columns)-3]


# In[55]:


# P-emaildomain: use the top 6 categories and collapse the rest into the other category

train_subset['P_emaildomain'][train_subset[train_subset['P_emaildomain']=='gmail'].index] = "gmail.com"
p_domain = train_subset['P_emaildomain'].value_counts()[:7]
p_domain_common = list(p_domain)
mod_p_domain = train_subset['P_emaildomain'].apply(lambda x: x if x in p_domain else 'Other')


# In[56]:


# Perform one-hot encoding on the modified P_emaildomain category

train_subset['P_emaildomain'] = mod_p_domain

one_hot_p = pd.get_dummies(train_subset['P_emaildomain'],drop_first=True)
train_subset = train_subset.drop('P_emaildomain',axis=1)
train_subset = train_subset.join(one_hot_p)


# In[57]:


# M1: one-hot encoding
one_hot_M1 = pd.get_dummies(train_subset['M1'],drop_first=True)
train_subset = train_subset.drop('M1',axis=1)
train_subset = train_subset.join(one_hot_M1)
train_subset = train_subset.drop(train_subset.columns[223],axis=1)


# In[58]:


# M2: one-hot encoding
one_hot_M2 = pd.get_dummies(train_subset['M2'],drop_first=True)
train_subset['M2'] = one_hot_M2['T']


# In[59]:


# M3: one-hot encoding
one_hot_M3 = pd.get_dummies(train_subset['M3'],drop_first=True)
train_subset['M3'] = one_hot_M3['T']


# In[60]:


# Drop M4 column
train_subset =  train_subset.drop('M4', axis=1)


# In[61]:


# Drop M6 column
one_hot_M6 = pd.get_dummies(train_subset['M6'],drop_first=True)
#train_subset = train_subset.drop('M2',axis=1)
train_subset['M6'] = one_hot_M3['T']


# In[62]:


# Divide the data into training and validation sets that encompass
# 70% and 30% of the data, respectively

from sklearn.model_selection import train_test_split
train_dummy = train_subset
X_train, X_valid = train_test_split(train_dummy,test_size=0.3, random_state=42)


# In[63]:


# Use mice imputation to impute the training and validation datasets
import statsmodels.imputation.mice as mice

imp_X_train = mice.MICEData(X_train)
imputed_X_train = imp_X_train.data

imp_X_valid = mice.MICEData(X_valid)
imputed_X_valid = imp_X_valid.data


# In[64]:


#Extract the fraud column and drop it from the data frame

train_dependent = imputed_X_train.iloc[:,0]
valid_dependent = imputed_X_valid.iloc[:,0]

imputed_X_train = imputed_X_train.drop('isFraud', 1)
imputed_X_valid = imputed_X_valid.drop('isFraud', 1)


# In[65]:


# Center and standardize the independent predictors

from sklearn import preprocessing

scal_imp_X_train = pd.DataFrame(preprocessing.scale(imputed_X_train))
scal_imp_X_valid = pd.DataFrame(preprocessing.scale(imputed_X_valid))


# In[66]:


# Perform XGBClassifier or Random Forest

from sklearn import model_selection
from sklearn import ensemble
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier

# Random Forest

# rfc = ensemble.RandomForestClassifier()
# rfc.fit(scal_imp_X_train,train_dependent)

# XGBClassifier
rfc=XGBClassifier()
rfc.fit(pd.DataFrame(scal_imp_X_train),train_dependent)


# In[67]:


# Assess the predictive accuracy of the model on the validation set
rfc_predict = rfc.predict(scal_imp_X_valid)


# In[68]:


# Display the confusion matrix
from sklearn import metrics
cm = metrics.confusion_matrix(valid_dependent, rfc_predict)
print(cm)


# In[69]:


# Compute the AUC

from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
roc=roc_auc_score(valid_dependent, rfc.predict_proba(scal_imp_X_valid)[:,1]) 
print(roc)


# In[70]:


# Compute the accuracy
print('Accuracy of the random forest classifier on test set: {:.2f}'.format(rfc.score(scal_imp_X_valid, valid_dependent)))


# # 3rd Attempt
# 
# I used the machine learning algorithm Random Forest to make predictions using the same data mining pipeline as described previously. SMOTE was not applied. 
# 
# Accuracy: 0.980
# 
# AUC: 0.868
# 
# Sensitivity: 0.998
# 
# Specificity: 0.417
# 
# Precision: 0.979
# 
# Kaggle Score: 0.8178 (surprisingly, my Kaggle score decreased)  
# 
# # 4th Attempt
# 
# I applied the machine learning algorithm XGBClassifer. This improved my score significantly. This is my final model for the competition. 
# 
# Accuracy: 0.97
# 
# AUC: 0.89
# 
# Sensitivity: 0.99
# 
# Specificity: 0.28
# 
# Precision: 0.97
# 
# Kaggle Score: 0.8720

# ## Part 7 - Final Result

# In[ ]:


# Now, I will manipulate the testing data and get it ready to make predictions


# In[ ]:


# First, extract the same features that were used to build the model for the training and validation sets
eve_index_nofraud = eve_index.drop(eve_index.index[1])
test_subset = test[eve_index_nofraud[0]]


# In[ ]:


# Extract the transaction id and delete it from the dataframe
test_transactionid = test_subset['TransactionID']
test_subset =  test_subset.drop('TransactionID', axis=1)


# In[ ]:


# ProductCD: use one hot encoding to convert this categorical predictor into a numeric predictor, remove the first
# product alphabetically as the reference
one_hot_productcd = pd.get_dummies(test_subset['ProductCD'],drop_first=True)
test_subset = test_subset.drop('ProductCD',axis=1)
test_subset = test_subset.join(one_hot_productcd)


# In[ ]:


# card4: one hot encoding
one_hot_productcard4 = pd.get_dummies(test_subset['card4'],drop_first=True)
test_subset = test_subset.drop('card4',axis=1)
test_subset = test_subset.join(one_hot_productcard4)


# In[ ]:


# card6: one hot encoding
one_hot_card6 = pd.get_dummies(test_subset['card6'],drop_first=True)
test_subset = test_subset.drop('card6',axis=1)
test_subset = test_subset.join(one_hot_card6)


# In[ ]:


# Convert TransactionDT into UTC time. I will only keep the hour the
# transaction was made
import datetime
unix_time = '1970-01-01' # I will just use the default time (I am not concerned about the year)
unix_time = datetime.datetime.strptime(unix_time, "%Y-%m-%d")

test_transaction_fraud = test_subset
test_transaction_fraud["Date"] = test_transaction_fraud['TransactionDT'].apply(lambda x: (unix_time + datetime.timedelta(seconds=x)))
test_transaction_fraud['Hour'] = test_transaction_fraud['Date'].dt.hour
test_transaction_fraud['Day'] = test_transaction_fraud['Date'].dt.day

test_subset['TransactionDT'] = test_transaction_fraud['Hour']
test_subset = test_subset.iloc[:,0:len(test_subset.columns)-3]


# In[ ]:


# For p_emaildomain select only the top 6 email domains and the condense the 
# remaining categories into other (the top 6 turn out to be the same for both 
# the training and testing sets)
p_domain = test_subset['P_emaildomain'].value_counts()[:7]
p_domain_common = list(p_domain)
mod_p_domain = test_subset['P_emaildomain'].apply(lambda x: x if x in p_domain else 'Other')

test_subset['P_emaildomain'] = mod_p_domain

one_hot_p = pd.get_dummies(test_subset['P_emaildomain'],drop_first=True)
test_subset = test_subset.drop('P_emaildomain',axis=1)
test_subset = test_subset.join(one_hot_p)


# In[ ]:


# M1 Hot Encoding
one_hot_M1 = pd.get_dummies(test_subset['M1'],drop_first=True)
test_subset = test_subset.drop('M1',axis=1)
test_subset = test_subset.join(one_hot_M1)


# In[ ]:


# M2 Hot Encoding
one_hot_M2 = pd.get_dummies(test_subset['M2'],drop_first=True)
test_subset['M2'] = one_hot_M2['T']


# In[ ]:


# M3 Hot Encoding
one_hot_M3 = pd.get_dummies(test_subset['M3'],drop_first=True)
test_subset['M3'] = one_hot_M3['T']


# In[ ]:


# Drop M4
test_subset =  test_subset.drop('M4', axis=1)


# In[ ]:


# M6 Hot Encoding
one_hot_M6 = pd.get_dummies(test_subset['M6'],drop_first=True)
test_subset['M6'] = one_hot_M3['T']
test_dummy = test_subset


# In[ ]:


# Perform mice imputation as well as standardize and scale the data
imp_X_predictions_test = mice.MICEData(test_dummy)
imputed_X_predictions_test = imp_X_predictions_test.data


# In[ ]:


scal_imp_X_predictions_test = pd.DataFrame(preprocessing.scale(imputed_X_predictions_test))
final_pred = rfc.predict(scal_imp_X_predictions_test)


# In[ ]:


# Extract the probabilities for Kaggle submission
p_prob_final = rfc.predict_proba(scal_imp_X_predictions_test)
p_prob_final = p_prob_final[:, 1]


# In[ ]:


# Kaggle submission
# final_submit = pd.DataFrame({'TransactionID': test_transactionid,'isFraud': p_prob_final})
# final_submit
# final_submit.to_csv('sample_submission_923_4.csv')


# Score: 0.8720

# Number of entries: 4
