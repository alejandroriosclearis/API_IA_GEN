import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# === 1. Cargar CSVs ===
orders_df = pd.read_csv("orders.csv", sep=';')
details_df = pd.read_csv("orders_details.csv", sep=';')

# === 2. Preparar y unir datos ===
# Renombrar columnas para unir y limpiar
orders_df = orders_df.rename(columns={'reference_id': 'ticket_id', 'reference_date': 'fecha'})
details_df = details_df.rename(columns={
    'ticket-reference_id': 'ticket_id',
    'product-reference_id': 'producto',
    'quantity': 'cantidad'
})

# Asegurarse de que las fechas sean tipo datetime
orders_df['fecha'] = pd.to_datetime(orders_df['fecha'])

# Hacer join
df = pd.merge(details_df, orders_df[['ticket_id', 'fecha']], on='ticket_id', how='left')

# Agrupar ventas totales por día
ventas_diarias = df.groupby('fecha').agg({'cantidad': 'sum'}).reset_index()
ventas_diarias.rename(columns={'fecha': 'ds', 'cantidad': 'y'}, inplace=True)

# === 3. Entrenar modelo Prophet ===
model = Prophet()
model.fit(ventas_diarias)

# === 4. Hacer predicción a 30 días vista ===
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# === 5. Visualizar predicción ===
model.plot(forecast)
plt.title("Predicción de demanda total diaria (30 días)")
plt.xlabel("Fecha")
plt.ylabel("Cantidad de productos vendidos")
plt.tight_layout()
plt.show()

# === 6. Mostrar resumen de tendencia ===
cambio = forecast['yhat'].iloc[-1] - forecast['yhat'].iloc[-30]
tendencia = "positiva" if cambio > 0 else "negativa"
print(f"\nTendencia esperada: {tendencia}")
print(f"Cambio estimado en demanda en 30 días: {cambio:.2f} unidades")
