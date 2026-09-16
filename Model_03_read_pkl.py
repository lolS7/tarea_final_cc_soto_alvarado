"""Clasificación por consola a partir de las estadísticas disponibles."""
import json
import pickle

import pandas as pd

from Model_01 import BASE, COLUMNAS, NUMERICAS


def cargar_modelo():
    ruta = BASE / 'model' / 'model.pkl'
    if not ruta.exists():
        ruta = BASE / 'modelo_goles.pkl'
    if not ruta.exists():
        raise FileNotFoundError('Falta model/model.pkl. Ejecute python train.py.')
    with ruta.open('rb') as archivo:
        return pickle.load(archivo)


def predecir(modelo, valores):
    entrada = pd.DataFrame([valores], columns=COLUMNAS)
    probabilidades = modelo.predict_proba(entrada)[0]
    return {'rango_goles_registrados': str(modelo.predict(entrada)[0]),
            'probabilidades': {str(clase): float(p) for clase, p in zip(modelo.classes_, probabilidades)}}


if __name__ == '__main__':
    modelo = cargar_modelo()
    valores = {}
    for columna in COLUMNAS:
        while True:
            texto = input(f'{columna} (vacío = dato faltante): ').strip()
            try:
                valor = float(texto) if columna in NUMERICAS and texto else texto or None
                if columna in NUMERICAS and valor is not None:
                    import math
                    if not math.isfinite(valor) or valor < 0:
                        raise ValueError()
                valores[columna] = valor
                break
            except ValueError:
                print('Ingrese un número finito no negativo.')
    print(json.dumps(predecir(modelo, valores), indent=2, ensure_ascii=False))
