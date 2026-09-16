"""Entrenar y guardar un único PKL con preprocesamiento y clasificador."""
import hashlib
import json
import pickle
import platform

import numpy
import pandas
import sklearn

from Model_01 import BASE, argumentos, entrenar


def guardar_modelo(ruta):
    modelo, metricas = entrenar(ruta)
    metricas['entorno'] = {'python': platform.python_version(), 'scikit-learn': sklearn.__version__,
                           'pandas': pandas.__version__, 'numpy': numpy.__version__}
    metricas['datos_sha256'] = hashlib.sha256(ruta.read_bytes()).hexdigest()
    serializado = pickle.dumps(modelo, protocol=5)
    (BASE / 'model').mkdir(exist_ok=True)
    (BASE / 'model' / 'model.pkl').write_bytes(serializado)
    # Compatibilidad con los comandos y archivos de las entregas anteriores.
    (BASE / 'modelo_goles.pkl').write_bytes(serializado)
    (BASE / 'metricas.json').write_text(json.dumps(metricas, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Modelo guardado en {BASE / "model" / "model.pkl"}')
    print(json.dumps(metricas['prueba'], indent=2, ensure_ascii=False))
    return modelo, metricas


if __name__ == '__main__':
    guardar_modelo(argumentos().data)
