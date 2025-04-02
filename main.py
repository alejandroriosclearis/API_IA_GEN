from fastapi import FastAPI
from fastapi.responses import Response
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from io import BytesIO

app = FastAPI()

def preparar_datos():
    orders_df = pd.read_csv("orders (3).csv", sep=';', encoding='utf-8-sig')
    details_df = pd.read_csv("orders_details (3).csv", sep=';', encoding='utf-8-sig')

    # Limpiar nombres de columnas
    orders_df.columns = orders_df.columns.str.strip()
    details_df.columns = details_df.columns.str.strip()

    # Renombrar columnas clave
    orders_df = orders_df.rename(columns={
        'reference_id': 'ticket_id',
        'reference_date': 'fecha'
    })

    details_df = details_df.rename(columns={
        'ticket-reference_id': 'ticket_id',
        'product-reference_id': 'producto',
        'quantity': 'cantidad'
    })

    # Convertir fechas y unir
    orders_df['fecha'] = pd.to_datetime(orders_df['fecha'])
    df = pd.merge(details_df, orders_df[['ticket_id', 'fecha']], on='ticket_id', how='left')

    # Agrupar por fecha
    ventas_diarias = df.groupby('fecha').agg({'cantidad': 'sum'}).reset_index()
    ventas_diarias.rename(columns={'fecha': 'ds', 'cantidad': 'y'}, inplace=True)
    ventas_diarias.dropna(subset=['ds', 'y'], inplace=True)

    return ventas_diarias


@app.get("/predecir")
def predecir():
    ventas_diarias = preparar_datos()
    model = Prophet()
    model.fit(ventas_diarias)
    future = model.make_future_dataframe(periods=90)
    forecast = model.predict(future)

    tendencia = forecast[['ds', 'yhat']].tail(90).to_dict(orient='records')
    cambio = forecast['yhat'].iloc[-1] - forecast['yhat'].iloc[-90]
    resumen = "positiva" if cambio > 0 else "negativa"

    return {
        "resumen": f"Tendencia {resumen}, cambio estimado en 3 meses: {cambio:.2f} unidades.",
        "prediccion": tendencia
    }

@app.get("/grafico")
def mostrar_grafico():
    ventas_diarias = preparar_datos()

    if ventas_diarias.empty:
        return {"error": "No hay datos suficientes para entrenar el modelo"}

    model = Prophet()
    model.fit(ventas_diarias)

    future = model.make_future_dataframe(periods=90)
    forecast = model.predict(future)

    # Graficar
    fig = model.plot(forecast)
    ax = fig.gca()
    ax.set_xlabel("Fecha del pedido")
    ax.set_ylabel("Cantidad total de productos vendidos")
    ax.set_title("Predicción de demanda para los próximos 3 meses")

    buf = BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)

    return Response(content=buf.getvalue(), media_type="image/png")
