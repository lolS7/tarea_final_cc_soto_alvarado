"""Pruebas del CSV real, ausencia de fuga de etiqueta, PKL y API."""
import pickle
import unittest
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from Model_01 import COLUMNAS, DATOS_PREDETERMINADOS, argumentos, entrenar, preparar_datos
from Model_03_read_pkl import predecir
from Model_04_FASTAPI import app


class PruebasModelo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.datos, _ = preparar_datos(DATOS_PREDETERMINADOS)
        cls.modelo, cls.metricas = entrenar(DATOS_PREDETERMINADOS)

    def test_csv_automatico_y_objetivo(self):
        self.assertEqual(argumentos([]).data, DATOS_PREDETERMINADOS)
        self.assertNotIn('Goals', COLUMNAS)
        self.assertNotIn('Minutes', COLUMNAS)
        for goles, etiqueta in zip(self.datos.Goals, self.datos.Target):
            self.assertEqual(etiqueta, '0' if goles == 0 else '1-9' if goles < 10 else '10+')
        self.assertEqual(sum(self.metricas['filas'].values()), len(self.datos))
        self.assertEqual(list(self.modelo.feature_names_in_), COLUMNAS)

    def test_pickle_y_datos_faltantes(self):
        modelo = pickle.loads(pickle.dumps(self.modelo))
        entrada = self.datos[COLUMNAS].iloc[:3]
        np.testing.assert_allclose(modelo.predict_proba(entrada), self.modelo.predict_proba(entrada))
        resultado = predecir(modelo, {c: None if c != 'Position' else 'Nueva' for c in COLUMNAS})
        self.assertAlmostEqual(sum(resultado['probabilidades'].values()), 1)
        self.assertIn('rango_goles_registrados', resultado)

    def test_reproducibilidad(self):
        repetido, metricas = entrenar(DATOS_PREDETERMINADOS)
        self.assertEqual(self.metricas, metricas)
        np.testing.assert_allclose(repetido.predict_proba(self.datos[COLUMNAS]),
                                   self.modelo.predict_proba(self.datos[COLUMNAS]))

    def test_api(self):
        with patch('Model_04_FASTAPI.cargar_modelo', return_value=self.modelo):
            with TestClient(app) as cliente:
                self.assertEqual(cliente.get('/health').status_code, 200)
                self.assertEqual(cliente.get('/', follow_redirects=False).headers['location'], '/docs')
                ejemplo = cliente.get('/openapi.json').json()['components']['schemas']['Jugador']['examples'][0]
                respuesta = cliente.post('/prediccion/', json=ejemplo)
                self.assertEqual(respuesta.status_code, 200)
                self.assertAlmostEqual(sum(respuesta.json()['probabilidades'].values()), 1)
                for entrada in [{'Position': 'Forward', 'Shots': -1}, {'Position': 'Forward', 'Goals': 8}]:
                    self.assertEqual(cliente.post('/prediccion/', json=entrada).status_code, 422)
        with patch('Model_04_FASTAPI.cargar_modelo', side_effect=FileNotFoundError):
            with TestClient(app) as cliente:
                self.assertEqual(cliente.get('/health').status_code, 503)


if __name__ == '__main__':
    unittest.main()
