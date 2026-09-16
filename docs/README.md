# E7 — Evidencia de ejecución en localhost

La prueba se ejecutó el 16 de septiembre de 2026 (UTC) con Python 3.12.14 y un proceso real de Uvicorn que sirvió `app.main:app` en `127.0.0.1:8001`. Se usa un puerto separado para no interrumpir el servidor del usuario en 8000.

Archivos de evidencia:

- `solicitud.json`: JSON enviado al modelo.
- `evidencia_localhost.json`: fecha UTC, comando, versión, SHA-256 del PKL, solicitudes y respuestas reales.
- `uvicorn.log`: salida del proceso y estados HTTP de las solicitudes.
- `pruebas.txt`: salida de las pruebas automatizadas.

Resultados observados:

| Solicitud | Estado esperado y observado | Resultado |
|---|---|---|
| GET /health | 200 | Modelo cargado |
| POST /prediccion/ con solicitud.json | 200 | Rango 1-9; probabilidades suman 1 |
| POST /prediccion/ con Shots = -1 | 422 | Entrada rechazada por validación |

Desde la raíz del proyecto, con el entorno activado:

```powershell
python verificar_localhost.py --port 8001
python -m unittest -v test_modelo.py
```

El verificador inicia su propio servidor, realiza solicitudes HTTP y detiene exclusivamente ese proceso al terminar. La evidencia corresponde al PKL identificado por su hash, no a una simulación con TestClient. Las pruebas unitarias complementarias sí usan TestClient.

Esta evidencia acredita ejecución local. No acredita publicación en GitHub ni despliegue Cloud. La URL y el historial real del equipo siguen pendientes para E8.
