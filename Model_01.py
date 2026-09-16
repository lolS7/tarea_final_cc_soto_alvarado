"""Clasificación del rango de goles registrados usando el CSV disponible."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = Path(__file__).resolve().parent
DATOS_PREDETERMINADOS = BASE / 'dataset - 2020-09-24 (1).csv'
NUMERICAS = ['Age', 'Appearances', 'Assists', 'Shots']
CATEGORICAS = ['Position']
COLUMNAS = NUMERICAS + CATEGORICAS
CLASES = ['0', '1-9', '10+']


def preparar_datos(ruta):
    datos = pd.read_csv(ruta)
    requeridas = ['Goals'] + COLUMNAS
    faltantes = sorted(set(requeridas) - set(datos.columns))
    if faltantes:
        raise ValueError('Faltan columnas: ' + ', '.join(faltantes)
                         + '. Use el CSV original con sus encabezados.')
    if datos['Goals'].isna().any():
        raise ValueError('Goals no puede ser nulo porque define la clase objetivo.')
    if 'Name' in datos and datos['Name'].duplicated().any():
        raise ValueError('Hay nombres duplicados: revise los jugadores antes de dividir los datos.')
    for columna in NUMERICAS + ['Goals']:
        datos[columna] = pd.to_numeric(datos[columna], errors='raise')
        presentes = datos[columna].dropna()
        if (~np.isfinite(presentes)).any() or (presentes < 0).any():
            raise ValueError(f'{columna} debe contener números finitos no negativos.')
    if (datos['Goals'] % 1 != 0).any():
        raise ValueError('Goals debe ser un conteo entero.')
    # Goals es sólo la etiqueta; nunca entra en las características.
    datos['Target'] = pd.cut(datos['Goals'], [-1, 0, 9, np.inf], labels=CLASES).astype(str)
    return datos.reset_index(drop=True), len(datos)


def pipeline(clasificador):
    numeros = Pipeline([
        ('imputacion', SimpleImputer(strategy='median', keep_empty_features=True)),
        ('escala', StandardScaler()),
    ])
    categorias = Pipeline([
        ('imputacion', SimpleImputer(strategy='most_frequent', keep_empty_features=True)),
        ('codificacion', OneHotEncoder(handle_unknown='ignore')),
    ])
    return Pipeline([
        ('preprocesamiento', ColumnTransformer([
            ('numericas', numeros, NUMERICAS), ('categoricas', categorias, CATEGORICAS)
        ])),
        ('clasificador', clasificador),
    ])


def evaluar(modelo, datos):
    real = datos['Target']
    prediccion = modelo.predict(datos[COLUMNAS])
    return {
        'macro_f1': float(f1_score(real, prediccion, labels=CLASES, average='macro', zero_division=0)),
        'reporte': classification_report(real, prediccion, labels=CLASES, output_dict=True, zero_division=0),
        'matriz_confusion': confusion_matrix(real, prediccion, labels=CLASES).tolist(),
    }


def entrenar(ruta):
    datos, originales = preparar_datos(ruta)
    if set(datos['Target']) != set(CLASES) or datos['Target'].value_counts().min() < 10:
        raise ValueError('Se necesitan al menos 10 ejemplos de cada clase: 0, 1-9 y 10+.')
    desarrollo, test = train_test_split(datos, test_size=0.2, stratify=datos['Target'], random_state=42)
    train, valid = train_test_split(desarrollo, test_size=0.25, stratify=desarrollo['Target'], random_state=42)
    candidatos = {
        'regresion_logistica': LogisticRegression(class_weight='balanced', random_state=42, max_iter=2000),
        'bosque_aleatorio': RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                                                  class_weight='balanced', random_state=42, n_jobs=1),
    }
    modelos, validacion = {}, {}
    for nombre, estimador in candidatos.items():
        modelo = pipeline(estimador)
        modelo.fit(train[COLUMNAS], train['Target'])
        modelos[nombre] = modelo
        validacion[nombre] = evaluar(modelo, valid)
    elegido = max(validacion, key=lambda nombre: validacion[nombre]['macro_f1'])
    desarrollo = pd.concat([train, valid])
    modelo = modelos[elegido]
    modelo.fit(desarrollo[COLUMNAS], desarrollo['Target'])
    baseline = pipeline(DummyClassifier(strategy='most_frequent'))
    baseline.fit(desarrollo[COLUMNAS], desarrollo['Target'])
    metricas = {
        'objetivo': 'Rango de goles registrados en el CSV (no goles futuros)', 'clases': CLASES,
        'columnas_entrada': COLUMNAS, 'filas_originales': originales,
        'distribucion_clases': {str(k): int(v) for k, v in datos['Target'].value_counts().items()},
        'division': '60% entrenamiento, 20% validación, 20% prueba; estratificada, semilla 42',
        'filas': {'entrenamiento': len(train), 'validacion': len(valid), 'prueba': len(test)},
        'seleccionado': elegido, 'validacion': validacion,
        'prueba': evaluar(modelo, test), 'baseline_prueba': evaluar(baseline, test),
        'advertencia': 'Clasifica goles registrados; no estima goles de la siguiente temporada. Probabilidades no calibradas.',
    }
    return modelo, metricas


def argumentos(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DATOS_PREDETERMINADOS,
                        help='CSV de jugadores; por defecto se lee dataset - 2020-09-24 (1).csv junto al script')
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = argumentos()
    _, resultado = entrenar(args.data)
    print(json.dumps(resultado, indent=2, ensure_ascii=False))
