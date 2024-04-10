import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pmdarima.arima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Cargar los datos
reservas = pd.read_excel("db/datos_reservas.xlsx")

# Crear un DataFrame con un índice de tiempo mensual
fecha_inicio = '2018-01-01'
fecha_fin = '2023-12-01'
fechas = pd.date_range(start=fecha_inicio, end=fecha_fin, freq='MS')
serie_reservas = reservas.melt(var_name='Año', value_name='Reservas').sort_values('Año')['Reservas']

# Crear el DataFrame para el análisis ARIMA
df_arima_reservas = pd.DataFrame({'Reservas': serie_reservas}, index=fechas)
df_arima_reservas.index.name = 'Tiempo'

# Función para encontrar los parámetros ARIMA óptimos
def encontrar_parametros_arima(df, column_name):
    modelo = auto_arima(df[column_name], max_p=100, max_d=5, max_q=100, seasonal=True, m=12, trace=True)
    return modelo.order, modelo.seasonal_order

# Hallar parámetros para ARIMA
parametros_orden, parametros_orden_estacional = encontrar_parametros_arima(df_arima_reservas, 'Reservas')

# Aplicar SARIMA
model_sarima_reservas = SARIMAX(df_arima_reservas['Reservas'], order=parametros_orden, seasonal_order=parametros_orden_estacional)
modelo_fit_sarima_reservas = model_sarima_reservas.fit()

# Realizar predicciones
predicciones_sarima_reservas = modelo_fit_sarima_reservas.get_forecast(steps=12)
predicciones_sarima_reservas = predicciones_sarima_reservas.predicted_mean

# Gráfico de las reservas bancarias por año
plt.figure(figsize=(15, 8))
sns.lineplot(data=df_arima_reservas.reset_index(), x='Tiempo', y='Reservas', marker='o', palette=sns.color_palette("husl", len(df_arima_reservas.columns)))

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.xticks(rotation=45)
plt.title('Reservas Bancarias por Año')
plt.xlabel('Mes')
plt.ylabel('Reservas en millones de quetzales')

# Mover la leyenda fuera del gráfico, a la parte superior derecha
plt.legend(title='Año', loc='upper left', bbox_to_anchor=(1, 1))

plt.savefig("imagenes/proyecciones_sarima/prediccion_reservas_sarima.png")
plt.show()
