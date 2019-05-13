import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.feature_selection import SelectFromModel

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split


data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')



def get_combined_data():
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    targets = train.Loan_Status
    train.drop('Loan_Status', 1, inplace=True)
    combined = train.append(test)
    combined.reset_index(inplace=True)
    combined.drop(['index', 'Loan_ID'], inplace=True, axis=1)
    return combined



combined = get_combined_data()
combined.describe()



def impute_gender():
    global combined
    combined['Gender'].fillna('Male', inplace=True)

def impute_martial_status():
    global combined
    combined['Married'].fillna('Yes', inplace=True)


def impute_employment():
    global combined
    combined['Self_Employed'].fillna('No', inplace=True)


def impute_loan_amount():
    global combined
    combined['LoanAmount'].fillna(combined['LoanAmount'].mean(), inplace=True)


print(combined['Credit_History'].value_counts())

def impute_credit_history():
    global combined
    combined['Credit_History'].fillna(1.0, inplace=True)


impute_gender()

impute_martial_status()

impute_employment()

impute_loan_amount()

impute_credit_history()




def process_gender():
    global combined
    combined['Gender'] = combined['Gender'].map({'Male':1,'Female':0})

def process_martial_status():
    global combined
    combined['Married'] = combined['Married'].map({'Yes':1,'No':0})

def process_dependents():
    global combined
    combined['Singleton'] = combined['Dependents'].map(lambda d: 1 if d=='1' else 0)
    combined['Small_Family'] = combined['Dependents'].map(lambda d: 1 if d=='2' else 0)
    combined['Large_Family'] = combined['Dependents'].map(lambda d: 1 if d=='3+' else 0)
    combined.drop(['Dependents'], axis=1, inplace=True)

def process_education():
    global combined
    combined['Education'] = combined['Education'].map({'Graduate':1,'Not Graduate':0})

def process_employment():
    global combined
    combined['Self_Employed'] = combined['Self_Employed'].map({'Yes':1,'No':0})

def process_income():
    global combined
    combined['Total_Income'] = combined['ApplicantIncome'] + combined['CoapplicantIncome']
    combined.drop(['ApplicantIncome','CoapplicantIncome'], axis=1, inplace=True)




def process_loan_amount():
    global combined
    combined['Debt_Income_Ratio'] = combined['Total_Income'] / combined['LoanAmount']


def process_loan_term():
    global combined
    combined['Very_Short_Term'] = combined['Loan_Amount_Term'].map(lambda t: 1 if t<=60 else 0)
    combined['Short_Term'] = combined['Loan_Amount_Term'].map(lambda t: 1 if t>60 and t<180 else 0)
    combined['Long_Term'] = combined['Loan_Amount_Term'].map(lambda t: 1 if t>=180 and t<=300  else 0)
    combined['Very_Long_Term'] = combined['Loan_Amount_Term'].map(lambda t: 1 if t>300 else 0)
    combined.drop('Loan_Amount_Term', axis=1, inplace=True)

def process_credit_history():
    global combined
    combined['Credit_History_Bad'] = combined['Credit_History'].map(lambda c: 1 if c==0 else 0)
    combined['Credit_History_Good'] = combined['Credit_History'].map(lambda c: 1 if c==1 else 0)
    combined['Credit_History_Unknown'] = combined['Credit_History'].map(lambda c: 1 if c==2 else 0)
    combined.drop('Credit_History', axis=1, inplace=True)

def process_property():
    global combined
    property_dummies = pd.get_dummies(combined['Property_Area'], prefix='Property')
    combined = pd.concat([combined, property_dummies], axis=1)
    combined.drop('Property_Area', axis=1, inplace=True)

process_gender()

process_martial_status()

process_dependents()

process_education()

process_employment()

process_income()

process_loan_amount()

process_loan_term()

process_credit_history()

process_property()

def remove_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out


from sklearn.preprocessing import MinMaxScaler

def feature_scaling(df):
    df -= df.min()
    df /= df.max()
    return df

combined['LoanAmount'] = feature_scaling(combined['LoanAmount'])
combined['Total_Income'] = feature_scaling(combined['Total_Income'])
combined['Debt_Income_Ratio'] = feature_scaling(combined['Debt_Income_Ratio'])

#combined['Total_Income']=remove_outlier(combined,'Total_Income')
#combined['LoanAmount']=remove_outlier(combined,'LoanAmount')

'''
b=combined['LoanAmount'] 
import matplotlib.pyplot as plt
plt.hist(a)
plt.show()

plt.hist(b)
plt.show()



scaler = MinMaxScaler()
combined[['LoanAmount', 'Total_Income','Debt_Income_Ratio']] = scaler.fit_transform(combined[['LoanAmount', 'Total_Income','Debt_Income_Ratio']])


import matplotlib.pyplot as plt
plt.hist(x)
plt.show()

Index([u'Gender', u'Married', u'Education', u'Self_Employed', u'LoanAmount',
       u'Singleton', u'Small_Family', u'Large_Family', u'Total_Income',
       u'Debt_Income_Ratio', u'Very_Short_Term', u'Short_Term', u'Long_Term',
       u'Very_Long_Term', u'Credit_History_Bad', u'Credit_History_Good',
       u'Credit_History_Unknown', u'Property_Rural', u'Property_Semiurban',
       u'Property_Urban'],
      dtype='object')

'''

#training the data 


from sklearn.decomposition import PCA


def recover_train_test_target():
    global combined, data_train
    print(combined.isnull().sum())
    targets = data_train['Loan_Status'].map({'Y':1,'N':0})
    train = combined.head(614)
    test = combined.iloc[614:]
    return train, test, targets


train, test, targets = recover_train_test_target()

from sklearn.linear_model import LogisticRegression
clf =svm.SVC()
clf.fit(train,targets)  
output=clf.predict(test)

my_submission = pd.DataFrame({'Loan_ID':data_test['Loan_ID'], 'Loan_Status': output})
my_submission['Loan_Status']=my_submission['Loan_Status'].map({1:'Y',0:'N'})
my_submission.to_csv('submission.csv', index=False)

def compute_score(clf, X, y, scoring='accuracy'):
    xval = cross_val_score(clf, X, y, cv = 5, scoring=scoring)
    return np.mean(xval)
print(compute_score(clf, train, targets, scoring='accuracy'))
