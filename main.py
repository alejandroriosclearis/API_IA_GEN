from fastapi import FastAPI, UploadFile, File
import pandas as pd
from model import unir_datos, predecir_demanda

app = FastAPI()

@app.post("/predecir/")
async def predecir(
    orders_file: UploadFile = File("orders.csv"),
    details_file: UploadFile = File("orders_details.csv")
):
    orders_df = pd.read_csv(orders_file.file, sep=';')
    details_df = pd.read_csv(details_file.file, sep=';')

    merged_df = unir_datos(orders_df, details_df)
    resultado = predecir_demanda(merged_df)

    return resultado
