# Clasificación de goles registrados

## Entregables obligatorios E1–E8

**Use el contenido de `Modelo_Tarea_Final` como raíz del repositorio y del servicio.** Así `train.py`, `Procfile`, `runtime.txt`, `requirements.txt`, `app/` y `model/` quedan en las rutas exigidas. Los scripts anteriores siguen disponibles para compatibilidad.

| ID | Entregable verificable | Estado |
|---|---|---|
| E1 | [train.py](train.py): `python train.py` | Implementado y ejecutado |
| E2 | [model/model.pkl](model/model.pkl): pipeline entrenado completo | Generado y probado |
| E3 | [requirements.txt](requirements.txt): dependencias con versiones fijadas | Disponible |
| E4 | [runtime.txt](runtime.txt): Python 3.12.14 | Coincide con el intérprete de entrenamiento |
| E5 | [Procfile](Procfile): comando de arranque | Disponible |
| E6 | [app/main.py](app/main.py): API FastAPI | Probada con Uvicorn |
| E7 | [docs/evidencia_localhost.json](docs/evidencia_localhost.json), [log](docs/uvicorn.log) y [guía](docs/README.md) | HTTP 200 y 422 verificados |
| E8 | URL GitHub e historial de commits del equipo | Pendiente: no se proporcionó URL y la carpeta no tiene historial Git |

No se ha publicado en GitHub ni generado un historial atribuido al equipo. Para cerrar E8 se necesita el repositorio real y los commits de sus integrantes.

### Inicio con la estructura de entrega

Desde esta carpeta, usando el entorno virtual del proyecto:

```powershell
.\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Para reproducir la evidencia HTTP sin detener su servidor del puerto 8000:

```powershell
.\.venv\Scripts\python.exe verificar_localhost.py --port 8001
```

El `Procfile` usa sintaxis de shell POSIX para tomar `$PORT` del proveedor (8000 por defecto). En PowerShell use el comando anterior. `runtime.txt` declara la versión; cada proveedor debe admitirla o configurarse explícitamente. El funcionamiento remoto no se ha probado.

El entrenamiento usa automáticamente `dataset - 2020-09-24 (1).csv` situado junto a los scripts. No necesita archivos adicionales, `PlayerID`, `Season` ni `Minutes`.

## Objetivo y datos

Clasifica `Goals` en tres rangos: **0**, **1–9** y **10 o más**. Se basa en goles registrados en el CSV; **no predice la siguiente temporada**, porque el archivo no contiene esa información.

Las entradas son `Age`, `Appearances`, `Assists`, `Shots` y `Position`. `Goals` sólo define la etiqueta y no se incluye como entrada. Tampoco se usan sus desgloses ni `Goals per match`, para evitar entregar la respuesta al modelo. Las estadísticas deben tener la misma cobertura que las del CSV; no se deben mezclar acumulados con datos de una sola temporada.

El archivo contiene 571 jugadores. Hay valores faltantes en Age y Shots: se imputan con la mediana aprendida exclusivamente durante el ajuste. Los goles objetivo no pueden faltar. No se inventan temporadas ni etiquetas futuras.

## Ejecutar

Desde `Modelo_Tarea_Final`, con Python 3.12 y el entorno virtual activado:

```powershell
python -m pip install -r requirements.txt
python Model_02.py
python -m uvicorn Model_04_FASTAPI:app --host 127.0.0.1 --port 8000
```

`train.py` y `Model_02.py` entrenan y guardan `model/model.pkl`, una copia compatible `modelo_goles.pkl` y `metricas.json`. La API carga primero `model/model.pkl`. El PKL incluye imputación, escalamiento, codificación de categorías y clasificador. El entrenamiento no se repite con cada solicitud de la API. Si reentrena, reinicie Uvicorn.

La lectura del CSV y las salidas se resuelven respecto de la carpeta de los scripts, incluso si se ejecutan desde otro directorio. Opcionalmente puede usar `python Model_02.py --data "otro.csv"` con el mismo esquema.

- `Model_01.py`: preprocesamiento, comparación y evaluación; puede ejecutarse sin argumentos para revisar resultados sin guardar.
- `Model_02.py`: entrenamiento y guardado del pipeline completo.
- `Model_03_read_pkl.py`: ingreso de características por consola y clasificación.
- `Model_04_FASTAPI.py`: recepción de JSON y clasificación con el PKL.

## JSON para la API

Abra http://127.0.0.1:8000, que redirige a `/docs`. Seleccione **POST /prediccion/** → **Try it out**, ingrese este JSON y pulse **Execute**:

```json
{
  "Age": 25,
  "Appearances": 30,
  "Assists": 3,
  "Shots": 60,
  "Position": "Forward"
}
```

`Position` es obligatorio: `Goalkeeper`, `Defender`, `Midfielder` o `Forward`. Los números son no negativos; Appearances, Assists y Shots son enteros. Puede omitir números desconocidos o enviarlos como `null`; el modelo los imputa. No envíe `Goals`, `Minutes`, `PlayerID` ni `Season`.

La respuesta contiene `rango_goles_registrados` y `probabilidades` para `0`, `1-9` y `10+`. Las probabilidades no están calibradas ni representan garantías. El nombre anterior `rango_goles_siguiente_temporada` fue reemplazado para reflejar el objetivo real.

También puede enviar una solicitud desde PowerShell:

```powershell
$cuerpo = @{
    Age = 25; Appearances = 30; Assists = 3
    Shots = 60; Position = 'Forward'
} | ConvertTo-Json
Invoke-RestMethod -Uri 'http://127.0.0.1:8000/prediccion/' -Method Post -ContentType 'application/json' -Body $cuerpo
```

Uvicorn recibe el JSON por HTTP, no mediante una pregunta en la terminal. `/health` confirma si el PKL está cargado. Un cuerpo inválido recibe 422; si falta el PKL, se devuelve 503 con instrucciones para entrenar.

## Evaluación y reproducibilidad

Se divide el CSV de forma estratificada en aproximadamente 60% entrenamiento, 20% validación y 20% prueba, con semilla 42. Se comparan regresión logística y bosque aleatorio por macro-F1 de validación. El mejor se reajusta con entrenamiento + validación y se evalúa en prueba, junto a una referencia de clase mayoritaria. El conjunto de prueba no se incorpora al modelo guardado.

`metricas.json` contiene reportes, matrices de confusión (orden 0, 1-9, 10+), modelo seleccionado, versiones y SHA-256 del CSV. Las métricas evalúan clasificación dentro de este conjunto de jugadores, no pronóstico temporal. Sólo hay 571 registros y los patrones de faltantes pueden depender de la posición; los resultados pueden cambiar en otra población.

Para Cloud use las mismas versiones de Python y bibliotecas que en entrenamiento. El PKL contiene componentes estándar de scikit-learn y debe proceder de una fuente de confianza. Incluya el PKL generado al subir la carpeta a GitHub. No incluya `.venv`.

```bash
docker build -t modelo-goles .
docker run --rm -p 8000:8000 modelo-goles
```

El Dockerfile usa Python 3.12. El despliegue Docker/Cloud no se ha verificado en un proveedor remoto.

## Pruebas

```powershell
python -m pip install -r requirements-dev.txt
python -m unittest -v test_modelo.py
```

`historico_plantilla.csv` es una plantilla del enfoque anterior y no se utiliza en este entrenamiento.
