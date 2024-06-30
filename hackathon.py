import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder 
from sklearn.preprocessing import StandardScaler
x_train=pd.read_csv(r'C:\Users\akhte\OneDrive\Desktop\Summer Analytics\Hackathon\sa2022\train.csv')
x_test=pd.read_csv(r'C:\Users\akhte\OneDrive\Desktop\Summer Analytics\Hackathon\sa2022\test.csv')
c1=x_test['id']
df=x_train.copy()
def impute_rating(cols):
    norating=cols[0]
    f5=cols[1]
    if pd.isnull(f5):
        return norating*0.315
    else:
        return f5
df['star_5f']=df[['norating1','star_5f']].apply(impute_rating,axis=1)
def impute_rating2(cols):
    f3=cols[0]
    f5=cols[1]
    if pd.isnull(f5):
        return f3*4.2
    else:
        return f5
df['star_5f']=df[['star_3f','star_5f']].apply(impute_rating,axis=1)
def impute_rating(cols):
    norating=cols[0]
    f4=cols[1]
    if pd.isnull(f4):
        return norating*0.23
    else:
        return f4
df['star_4f']=df[['norating1','star_5f']].apply(impute_rating,axis=1)
#print(df['star_4f'].isna().value_counts())
df=df.dropna(subset=['star_5f','star_4f','star_3f','maincateg'],how='any')
y_train=df['price1']   
df2=df.drop(['id','norating1','noreviews1','price1','title','Offer %'],axis=1)
df3=df2.copy()
df3=pd.get_dummies(df3,columns=['maincateg','platform'],drop_first=True)
sc=StandardScaler()
df5=df3.copy()
df3=sc.fit_transform(df3)
lr=LinearRegression()
lr.fit(df3,y_train)
df6=x_test.copy()
df6['maincateg']=df6['maincateg'].fillna('Women')
df6['platform']=df6['platform'].fillna('Flipkart')
df6=pd.get_dummies(df6,columns=['maincateg','platform'],drop_first=True)
df6=df6.drop(['id','title','norating1','noreviews1'],axis=1)
df7=df6.copy()
df7['Rating']=df7['Rating'].fillna(4)
df7['actprice1']=df7['actprice1'].fillna(df7['actprice1'].mean())
df7['star_5f']=df7['star_5f'].fillna(df7['star_5f'].mean())
df7['star_4f']=df7['star_4f'].fillna(df7['star_4f'].mean())
df7['star_3f']=df7['star_3f'].fillna(df7['star_3f'].mean())
df7['star_2f']=df7['star_2f'].fillna(df7['star_2f'].mean())
df7['star_1f']=df7['star_1f'].fillna(df7['star_1f'].mean())
df7['fulfilled1']=df7['fulfilled1'].fillna(1)
df7=sc.transform(df7)
pred=lr.predict(df7)
pred2=pd.Series(pred)
header=['id,price1']
sub=pd.DataFrame({'id':c1,'price1':pred})
print(sub.head())
sub.to_csv('pred4.csv',index=False)






