import pandas as pd
import joblib
import numpy as np

df=pd.read_csv('images/Car (2).csv')
 
df=df.drop(columns='torque')
df=df.drop(columns='i')
df=df.drop(columns='name')


df['km_driven']=df['km_driven'].astype('float64')
df['km_driven']=df['km_driven'].astype('float64')
df['engine']=df['engine'].replace({'CC':''},regex=True)
df['engine']=df['engine'].replace({' ':''},regex=True)

df['max_power']=df['max_power'].replace({'.bhp':''},regex=True)
df = df.replace(r'^\s*$', np.nan, regex=True)
df=df.dropna()

Y=df['selling_price']
X=df.drop(columns='selling_price')

#fuel---------------------------------------------------
X['fuel']=X['fuel'].str.lower()
print(X['fuel'].unique())
from sklearn.preprocessing import LabelEncoder
li=LabelEncoder()
X['fuel']=li.fit_transform(X['fuel'])
#X['fuel']=X['transmission'].astype('float64')
joblib.dump(li,'fuel.joblib')
#-------------------------------------------------------

#seller_type--------------------------------------------
from sklearn.preprocessing import LabelEncoder
X['seller_type']=X['seller_type'].str.lower()
le=LabelEncoder()
X['seller_type']=le.fit_transform(X['seller_type'])
X['seller_type']=X['seller_type'].astype('float64')
joblib.dump(le,'seller_type.joblib')
#-------------------------------------------------------

#transmission-------------------------------------------
from sklearn.preprocessing import LabelEncoder
X['transmission']=X['transmission'].str.lower()
la=LabelEncoder()
X['transmission']=la.fit_transform(X['transmission'])
X['transmission']=X['transmission'].astype('float64')
joblib.dump(la,'transmission.joblib')
#--------------------------------------------------------

#mileage-------------------------------------------------

X['mileage']=X['mileage'].replace({'kmpl':''},regex=True)
X['mileage']=X['mileage'].replace({'km/kg':''},regex=True)
X['mileage']=X['mileage'].astype('float64')
#-----------------------------------------------------------

#owner-----------------------------------------------------
X['owner']=X['owner'].str.lower()
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[5])],remainder='passthrough')
X=ct.fit_transform(X)
joblib.dump(ct,'onehot.joblib')

from sklearn.preprocessing import StandardScaler as SC
sc=SC()#Transformers
X=sc.fit_transform(X)
joblib.dump(sc,'scaler.joblib')

from sklearn.model_selection import train_test_split as tts
X_train,X_test,Y_train,Y_test=tts(X,Y,test_size=0.1,random_state=2)

#Training Linear Regression
from sklearn.svm import SVR as LR
regressor=LR(C=1000000)#Estimator
regressor.fit(X_train,Y_train)

Y_pred=regressor.predict(X_test)

#Metrics

from sklearn.metrics import r2_score
print(r2_score(Y_test,Y_pred))

joblib.dump(regressor,'regressor.joblib')