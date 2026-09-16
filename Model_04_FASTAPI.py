"""API de clasificación del rango de goles registrados."""
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, ConfigDict, Field

from Model_03_read_pkl import cargar_modelo, predecir


@asynccontextmanager
async def lifespan(app):
    try:
        app.state.modelo = cargar_modelo()
    except FileNotFoundError:
        app.state.modelo = None
    yield


app = FastAPI(title='Clasificación de goles registrados', lifespan=lifespan)


class Jugador(BaseModel):
    model_config = ConfigDict(
        extra='forbid', allow_inf_nan=False,
        json_schema_extra={'examples': [{
            'Age': 25, 'Appearances': 30, 'Assists': 3,
            'Shots': 60, 'Position': 'Forward',
        }]},
    )
    Age: float | None = Field(default=None, ge=0)
    Appearances: int | None = Field(default=None, ge=0)
    Assists: int | None = Field(default=None, ge=0)
    Shots: int | None = Field(default=None, ge=0)
    Position: Literal['Goalkeeper', 'Defender', 'Midfielder', 'Forward']


@app.get('/', include_in_schema=False)
def inicio():
    return RedirectResponse(url='/docs')


@app.get('/health')
def health():
    if app.state.modelo is None:
        raise HTTPException(status_code=503, detail='Modelo pendiente de entrenamiento con datos reales.')
    return {'status': 'ok'}


@app.post('/prediccion/')
def predecir_goles(jugador: Jugador):
    if app.state.modelo is None:
        raise HTTPException(status_code=503, detail='Falta model/model.pkl; ejecute train.py y reinicie la API.')
    return predecir(app.state.modelo, jugador.model_dump())
