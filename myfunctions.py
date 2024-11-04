from IPython.display import Markdown, display
import pandas as pd
import numpy as np
from IPython.display import Markdown, display
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from functools import reduce
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split     # Para dividir el dataset en entrenamiento y prueba
from sklearn.preprocessing import StandardScaler         # Para normalizar o escalar los datos
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report  # Métricas de evaluación
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error


# Llama a todas las liberías necesarias
def myimports(): 
    from IPython.display import Markdown, display
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    from functools import reduce
    import matplotlib.pyplot as plt
    import seaborn as sns

    from sklearn.preprocessing import OneHotEncoder
    from sklearn.model_selection import train_test_split     # Para dividir el dataset en entrenamiento y prueba
    from sklearn.preprocessing import StandardScaler         # Para normalizar o escalar los datos
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report  # Métricas de evaluación
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error

    # Modelos de Machine Learning en scikit-learn
    from sklearn.linear_model import LogisticRegression      # Para Regresión Logística

    # Importaciones para los nuevos modelos
    from sklearn.linear_model import Lasso, Ridge
    from sklearn.svm import SVR
    from sklearn.neural_network import MLPRegressor
    

    # Para XGBoost
    import xgboost as xgb


    # Si vas a trabajar con clustering:
    from sklearn.cluster import KMeans                       # Para Clustering con K-means

    # Si vas a trabajar con reducción de dimensionalidad:
    from sklearn.decomposition import PCA

    from numpy import isnan
    from pandas import read_csv
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    
    return print ('Tus liberías se han importado exitosamente')

 
# Convierte la columna Periodo en 4 digitos y tipo String
def Periodo_to4digits (df, columna_Periodo):
    df[columna_Periodo]= df[columna_Periodo].astype(str).str[:4]
    return df[columna_Periodo].sample(5)

# Indica cuantos nulos y valores únicos tiene una columna
def Nulls_Uniques (df, column):
    markdown_text1 = (
    f"<p style='font-size:18px;'>Este campo tiene {df[column].isnull().sum()} registros nulos  y  {df[column].nunique()} valores únicos </p>"
    )
    
    markdown_text2 = (
    f"<p style='font-size:18px;'>Sus valores únicos son:</p>"
    )

    d= pd.DataFrame(df[column].value_counts())

    display(Markdown(markdown_text1))
    display(Markdown(markdown_text2))
    display(d)

# Cambia el nombre de una columna fácilmente.
def rename_col(df, col_name, col_new_name):
    df.rename(columns={col_name: col_new_name}, inplace=True)
    markdown_text = (
        f"<p style='font-size:18px;'>"
        f"Ejecutamos una función personalizada para cambiar el nombre de la columna "
        f"<code>{col_name}</code> por <code>{col_new_name}</code> para evitar posibles errores en los procesos posteriores."
        f"</p>"
    )
    display(Markdown(markdown_text))

def myText(cadena_de_texto):
    markdown_text = f"<p style='font-size:16px;'>{cadena_de_texto}</p>"
    display(Markdown(markdown_text))


def reset_metrics(metricas_modelos,):
    metricas_modelos = pd.DataFrame(columns=['Modelo', 'MAE', 'RMSE', 'R2'])
    return metricas_modelos

def del_row(dataframe,row_number):
    # Verificamos si el índice existe en el DataFrame
    if row_number in dataframe.index:
        dataframe = dataframe.drop(row_number)
    else:
        print(f"El índice {row_number} no existe en el DataFrame.")
    return dataframe


metricas_modelos = pd.DataFrame(columns=['Modelo', 'MAE', 'RMSE', 'R2'])

def gen_metrics(y_train, y_train_pred, y_test,y_pred,metricas_modelos,nombre_del_modelo, color):

    # Calculamos las métricas
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2_test= r2_score(y_test, y_pred)
    r2_train = r2_score(y_train, y_train_pred)
    desempeno_train = r2_train * 100
    desempeno_test = r2_test * 100




    nueva_fila = pd.DataFrame({
        'Modelo': [nombre_del_modelo],
        'MAE': [mae],
        'RMSE': [rmse],
        'R2': [r2_test],
        'Desempeño_Train(%)': [desempeno_train],
        'Desempeño_Test(%)': [desempeno_test]
    })

    # Concatenamos el DataFrame original con la nueva fila
    metricas_modelos = pd.concat([metricas_modelos, nueva_fila], ignore_index=True)

    # Imprimimos las métricas
    print(f"Métricas del Modelo de: {nombre_del_modelo} ")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2_test:.4f}")
    print(f"Desempeño_Train(%): {desempeno_train:.4f}")
    print(f"Desempeño_Test(%): {desempeno_test:.4f}")

    #Genera grafica

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color=color)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Línea de referencia
    plt.xlabel('Valores Reales')
    plt.ylabel('Valores Predichos')
    plt.title('Regresión Lineal: Valores Reales vs. Predichos')
    plt.show()
    return metricas_modelos