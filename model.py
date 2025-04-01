from prophet import Prophet
import pandas as pd

def unir_datos(orders_df, details_df):
    # Renombramos columnas para facilitar el join
    orders_df = orders_df.rename(columns={
        'reference_id': 'ticket_id',
        'reference_date': 'fecha'
    })
    details_df = details_df.rename(columns={
        'ticket-reference_id': 'ticket_id',
        'product-reference_id': 'producto',
        'quantity': 'cantidad'
    })

    # Convertimos la fecha
    orders_df['fecha'] = pd.to_datetime(orders_df['fecha'])

    # Hacemos el join
    df_merged = pd.merge(details_df, orders_df[['ticket_id', 'fecha']], on='ticket_id', how='left')

    return df_merged


def predecir_demanda(df_merged):
    df = df_merged.copy()

    # Agrupar ventas por fecha
    ventas_diarias = df.groupby('fecha').agg({'cantidad': 'sum'}).reset_index()
    ventas_diarias.rename(columns={'fecha': 'ds', 'cantidad': 'y'}, inplace=True)

    # Entrenar modelo Prophet
    model = Prophet()
    model.fit(ventas_diarias)

    # Predecir los próximos 30 días
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    tendencia = forecast[['ds', 'yhat']].tail(30).to_dict(orient='records')
    resumen = generar_resumen(forecast)

    return {
        "prediccion": tendencia,
        "resumen": resumen
    }

def generar_resumen(forecast):
    crecimiento = forecast['yhat'].iloc[-1] - forecast['yhat'].iloc[-30]
    if crecimiento > 0:
        tendencia = "positiva"
    else:
        tendencia = "negativa"
    return f"La tendencia de ventas para los próximos 30 días es {tendencia}, con un cambio estimado de {crecimiento:.2f} unidades."
