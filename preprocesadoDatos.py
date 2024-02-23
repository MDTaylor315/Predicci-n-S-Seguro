import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


df=pd.read_csv('seguro.txt')

print(df.isnull().sum())
#Llenamos los datos faltantes con el promedio de edad_casa
pedad = df['edad'].mean()
df['edad'].fillna(pedad, inplace=True)


#Llenamos los datos faltantes con la moda
moda_region = df['region'].mode()[0]
df['region'].fillna(moda_region, inplace=True)
print(df.isnull().sum())

moda_sexo = df['sexo'].mode()[0]
df['sexo'].fillna(moda_sexo, inplace=True)
print(df.isnull().sum())

moda_fumador = df['fumador'].mode()[0]
df['fumador'].fillna(moda_fumador, inplace=True)
print(df.isnull().sum())
#ONE-HOT
#Hacemos de "calidad_materiales" variables dummy c1|c2|c3
df_dummies = pd.get_dummies(df['region'], prefix='r', drop_first=True)
df_dummies2 = pd.get_dummies(df['sexo'], prefix='sex', drop_first=True)
df_dummies3 = pd.get_dummies(df['fumador'], prefix='fuma', drop_first=True)
# Concatenamos las variables dummy
df = pd.concat([df, df_dummies], axis=1)
df = pd.concat([df, df_dummies2], axis=1)
df = pd.concat([df, df_dummies3], axis=1)

df = df.drop(['region'], axis=1)
df = df.drop(['sexo'], axis=1)
df = df.drop(['fumador'], axis=1)

pd.set_option('display.max_columns', None)

# Mostrar el DataFrame resultante
print(df.head())




X = df[['edad','sex_male','imc','hijos','fuma_yes','r_southwest','r_southeast','r_northwest']]

y = df['prima_seguro']



#Dividimos los datos de entrenamiento y de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



#Utilizamos un objeto StandardScaler para escalar características
scaler = StandardScaler()
#fit_transform entrena al transformador 
X_train_scaled = scaler.fit_transform(X_train)
#transform usa el transformador en los datos con datos de prueba
X_test_scaled = scaler.transform(X_test)

#se encarga de estandarizar los datos


model = LinearRegression()
#Se ajusta el modelo de regresion lineal con los datos de entrenamiento
model.fit(X_train_scaled, y_train)


#Guarda una serie de predicciones en base a los datos de prueba
y_pred = model.predict(X_test_scaled)




from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Supongamos que 'y_pred' son las predicciones del precio de la casa y 'y_test' son los valores reales.

# Calcular el Error Absoluto Medio (MAE)
mae = mean_absolute_error(y_test, y_pred)
print(f'Error Absoluto Medio (MAE): {mae}')

# Calcular el Error Cuadrático Medio (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f'Error Cuadrático Medio (MSE): {mse}')

# Calcular la Raíz del Error Cuadrático Medio (RMSE)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f'Raíz del Error Cuadrático Medio (RMSE): {rmse}')

# Calcular el Coeficiente de Determinación (R^2)
r2 = r2_score(y_test, y_pred)
print(f'Coeficiente de Determinación (R^2): {r2}')

# Calcular el Error Porcentual Absoluto Medio (MAPE)
mape = mean_absolute_error(y_test, y_pred) / (y_test.mean()) * 100
print(f'Error Porcentual Absoluto Medio (MAPE): {mape}%')



nueva_fila = {'edad': 49,
              'sex_male': 1,
              'imc' : 26.35,
              'hijos' : 2,
              'fuma_yes': 0,
              'r_southwest':1,
              'r_southeast': 0,
              'r_northwest': 0
              }

# Convertir la nueva fila en un DataFrame
nueva_fila_df = pd.DataFrame([nueva_fila])

# Concatenar la nueva fila al DataFrame original
df_nuevo = pd.concat([df, nueva_fila_df], ignore_index=True)

# Mostrar el DataFrame resultante con la nueva fila
print(df_nuevo)

# Seleccionar las variables independientes (X) para la nueva fila
X_nuevo = nueva_fila_df[['edad','sex_male','imc','hijos','fuma_yes','r_southwest','r_southeast','r_northwest']]

# Escalar las variables independientes de la nueva fila
X_nuevo_scaled = scaler.transform(X_nuevo)

# Realizar la predicción para la nueva fila
precio_predicho_nuevo = model.predict(X_nuevo_scaled)

print(f'Precio predicho para la nueva fila: {precio_predicho_nuevo[0]}')

import numpy as np
import matplotlib.pyplot as plt

# Calcular residuos
residuos = y_test - y_pred

# Gráfico de Residuos
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuos)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicciones')
plt.ylabel('Residuos')
plt.title('Gráfico de Residuos')
plt.show()
