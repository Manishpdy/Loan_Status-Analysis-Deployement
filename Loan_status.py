#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# # importing and readind my data set in train and test variable.

# In[2]:


test = pd.read_csv("test.csv")
train = pd.read_csv("train.csv")


# In[3]:


test


# In[4]:


#train


# # droping loan_status column and saving them into y variable.

# In[5]:


y = train["Loan_Status"]
train.drop(columns="Loan_Status",axis=1 , inplace=True)


# In[6]:


#y


# In[7]:


#train


# In[ ]:





# # Finding Null value and replacing them on train dataset

# In[ ]:





# In[8]:


df = train


# In[9]:


#df


# In[10]:


#df.info()


# In[11]:


#df.head()


# # Here we are checking nun value

# In[12]:


#df.isna().sum()


# # Here we are checking value count of gender column

# In[13]:


#df["Gender"].value_counts()


# # Here we are filling male value in gender columns using fillna.

# In[14]:


df["Gender"] = df["Gender"].fillna("Male")


# In[15]:


#df["Gender"]


# In[16]:


#df["Gender"].value_counts()


# # Here we are checking value count of Married column

# In[17]:


#df["Married"].value_counts()


# # Here we are filling Yes value in Married columns using fillna.

# In[18]:


df["Married"] = df["Married"].fillna("Yes")


# In[19]:


#df["Married"].value_counts()


# # Here we are checking value count of Dependents column

# In[20]:


#df["Dependents"].value_counts()


# # Here we are filling 0 value in Dependents columns using fillna.

# In[21]:


df["Dependents"] = df["Dependents"].fillna("0")


# In[22]:


#df["Dependents"].value_counts()


# # Here we are checking value count of Self_Employed column

# In[23]:


#df["Self_Employed"].value_counts()


# # Here we are filling No value in Self_Employed columns using fillna.

# In[24]:


df["Self_Employed"] = df["Self_Employed"].fillna("No")


# In[25]:


#df["Self_Employed"].value_counts()


# # Here we are checking value count of LoanAmount column

# In[26]:


#df["LoanAmount"].value_counts()


# # Here we are filling mean value in LoanAmount columns using fillna.

# In[27]:


df["LoanAmount"] = df["LoanAmount"].fillna(np.mean(df["LoanAmount"]))


# In[28]:


#df["LoanAmount"].value_counts()


# In[29]:


#sns.distplot(df["LoanAmount"])


# # Here we are checking value count of Loan_Amount_Term column

# In[30]:


#df["Loan_Amount_Term"].value_counts()


# In[31]:


#sns.distplot(df["Loan_Amount_Term"])


# In[32]:


df["Loan_Amount_Term"] = df["Loan_Amount_Term"].fillna(np.median(float(360)))


# In[33]:


#df["Loan_Amount_Term"].value_counts()


# In[34]:


#df.isna().sum()


# In[35]:


#df["Credit_History"].value_counts()


# In[36]:


#sns.distplot(df["Credit_History"])


# In[37]:


df["Credit_History"] = df["Credit_History"].fillna(df["Credit_History"].median())


# In[38]:


#df["Credit_History"].value_counts()


# In[39]:


#df.head()


# In[40]:


#df.info()


# In[41]:


#df.isna().sum()


# # Data Visulization:-

# In[42]:


#df.columns


# In[43]:


#df.head()


# In[44]:


# gender_counts = df['Gender'].value_counts()

# # Create bar plot
# plt.figure(figsize=(8, 6))
# gender_counts.plot(kind='bar', color=['blue', 'pink'])
# plt.xlabel('Gender')
# plt.ylabel('Count')
# plt.title('Gender Distribution')
# plt.xticks(rotation=0)  # Rotate x-axis labels if needed
# plt.show()


# In[45]:


# education_counts = df['Education'].value_counts()

# # Create a bar plot
# plt.figure(figsize=(8, 6))
# education_counts.plot(kind='bar', color=['blue', 'orange'])
# plt.xlabel('Education')
# plt.ylabel('Count')
# plt.title('Number of People by Education Level')
# plt.xticks(rotation=0)  # Rotate x-axis labels if needed
# plt.show()


# In[46]:


# gender_education_counts = df.groupby(['Gender', 'Education',]).size().unstack(fill_value=0)

# # Create a grouped bar plot
# plt.figure(figsize=(8, 6))
# gender_education_counts.plot(kind='bar', stacked=False, color=['blue', 'orange',])
# plt.xlabel('Gender')
# plt.ylabel('Count')
# plt.title('Number of People by Gender and Education Level')
# plt.xticks(rotation=0)  # Rotate x-axis labels if needed
# plt.legend(title='Education')
# plt.show()


# In[47]:


# gender_education_selfemployed_counts = df.groupby(['Gender','Self_Employed']).size().unstack(fill_value=0)

# # Create a grouped bar plot
# plt.figure(figsize=(8, 6))
# gender_education_selfemployed_counts.plot(kind='bar', stacked=False, color=['red','green'])
# plt.xlabel('Gender')
# plt.ylabel('Count')
# plt.title('Number of People by Gender, Education, and Self-Employment')
# plt.xticks(rotation=0)  # Rotate x-axis labels if needed
# plt.legend(title='Self-Employment')
# plt.show()


# In[48]:


# property_counts = df['Property_Area'].value_counts()

# # Create a pie chart
# plt.figure(figsize=(8, 6))
# plt.pie(property_counts, labels=property_counts.index, autopct='%1.1f%%', colors=['skyblue', 'lightgreen', 'lightcoral'])
# plt.title('Distribution of Property Area')
# plt.show()


# # Converting catergorical data to numerical data using lable incoder

# In[49]:


#df.head()


# In[50]:


from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
df["Gender"] = lb.fit_transform(df["Gender"])
df["Married"] = lb.fit_transform(df["Married"])
df["Education"] = lb.fit_transform(df["Education"])
df["Self_Employed"] = lb.fit_transform(df["Self_Employed"])
df["Property_Area"] = lb.fit_transform(df["Property_Area"])
df["Dependents"] = lb.fit_transform(df["Dependents"])


# In[51]:


#df


# In[52]:


df.drop(columns=["Loan_ID"],axis=1,inplace=True)


# In[53]:


#df.head()


# In[54]:


#y


# In[55]:


x=df


# In[56]:


#x


# In[57]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x = scaler.fit_transform(x)


# In[58]:


#x


# # Finding Null value and replacing them on test dataset

# In[ ]:





# In[59]:


df1 = test


# In[60]:


#df1


# In[61]:


#df1.isna().sum()


# In[62]:


#df1["Gender"].value_counts()


# In[63]:


df1["Gender"] = df1["Gender"].fillna("Male")


# In[64]:


#df1["Gender"].value_counts()


# In[65]:


#df1["Dependents"].value_counts()


# In[66]:


df1["Dependents"] = df1["Dependents"].fillna("0")


# In[67]:


#df1["Dependents"].value_counts()


# In[68]:


#df1["Self_Employed"].value_counts()


# In[69]:


df1["Self_Employed"] = df1["Self_Employed"].fillna("No")


# In[70]:


#df1["Self_Employed"].value_counts()


# In[71]:


#df1["LoanAmount"].value_counts()


# In[72]:


df1["LoanAmount"] = df1["LoanAmount"].fillna(df1["LoanAmount"].median())


# In[73]:


#df1["LoanAmount"].value_counts()


# In[74]:


#df1["Loan_Amount_Term"].value_counts()


# In[75]:


df1["Loan_Amount_Term"] = df1["Loan_Amount_Term"].fillna(df1["Loan_Amount_Term"].median())


# In[76]:


#df1["Loan_Amount_Term"].value_counts()


# In[77]:


#df1["Credit_History"].value_counts()


# In[78]:


df1["Credit_History"] = df1["Credit_History"].fillna(df1["Credit_History"].median())


# In[79]:


#df1["Credit_History"].value_counts()


# In[80]:


#df1.isna().sum()


# In[81]:


#df1.head()


# In[82]:


df1.drop(columns=["Loan_ID"],axis=1,inplace=True)


# In[83]:


#df1.head()


# In[84]:


from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
df1["Gender"] = lb.fit_transform(df1["Gender"])
df1["Married"] = lb.fit_transform(df1["Married"])
df1["Education"] = lb.fit_transform(df1["Education"])
df1["Self_Employed"] = lb.fit_transform(df1["Self_Employed"])
df1["Property_Area"] = lb.fit_transform(df1["Property_Area"])
df1["Dependents"] = lb.fit_transform(df1["Dependents"])


# In[85]:


#df.head()


# In[86]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df1 = scaler.fit_transform(df1)


# In[ ]:





# In[87]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2,random_state=1)


# In[88]:


df.shape


# # Performing KNeighborsClassifier Model:-

# In[89]:


# from sklearn.neighbors import KNeighborsClassifier
# neigh = KNeighborsClassifier(n_neighbors=3)
# neigh.fit(xtrain,ytrain)
# pred = neigh.predict(xtest)
# pred



# In[90]:


#finding accuracy score


# In[91]:


# from sklearn.metrics import accuracy_score
# accuracy_score(pred,ytest)


# # Performing LogisticRegression Model:-

# In[94]:


from sklearn.linear_model import LogisticRegression
lg = LogisticRegression()
lg.fit(xtrain,ytrain)
pred1 = lg.predict(xtest)


# In[95]:


from sklearn.metrics import accuracy_score
accuracy_score(pred1,ytest)


# In[ ]:




# In[ ]:


import pickle
pickle_out = open("classifier.pkl",'wb')
pickle.dump(lg,pickle_out)
pickle_out.close()
import streamlit as st
from PIL import Image

def prediction(Gender,Married,Dependents,Education,Self_Employed,ApplicantIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term,Credit_History,Property_Area):
    prediction =lg.predict(
    [[Gender,Married,Dependents,Education,Self_Employed,ApplicantIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term,Credit_History,Property_Area]])
    print(prediction)
    return prediction

# this is the main function in which we define our webpage
def main():
    #Giving the webpage title
    st.title("LoanStatus Prediction")
    
    html_temp = """<div style ="background-color:green;padding:13px">
    <h1 style ="color:black;text-align:center;">Loan Status Prediction App</h1>
    </div>  
    """
    st.markdown(html_temp,unsafe_allow_html = True)
    Gender = st.number_input("Gender")
    Married = st.number_input("Married")
    Dependents = st.number_input("Dependents")
    Education = st.number_input("Education")
    Self_Employed = st.number_input("Self_Employed")
    ApplicantIncome = st.number_input("ApplicantIncome")
    CoapplicantIncome = st.number_input("CoapplicantIncome")
    LoanAmount = st.number_input("LoanAmount")
    Loan_Amount_Term = st.number_input("Loan_Amount_Term")
    Credit_History = st.number_input("Credit_History")
    Property_Area = st.number_input("Property_Area")
    result =""
    
    if st.button("predict"):
        result = prediction(Gender,Married,Dependents,Education,Self_Employed,ApplicantIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term,Credit_History,Property_Area)
    st.success("The Output is {}".format(result))


if __name__=="__main__":
    main()



