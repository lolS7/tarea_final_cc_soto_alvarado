"""E7: inicia Uvicorn, prueba HTTP real y conserva evidencia verificable."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

BASE = Path(__file__).resolve().parent


def solicitar(url, cuerpo=None):
    request = Request(url, data=json.dumps(cuerpo).encode() if cuerpo is not None else None,
                      headers={'Content-Type': 'application/json'})
    try:
        respuesta = urlopen(request, timeout=3)
    except HTTPError as error:
        respuesta = error
    with respuesta:
        return {'status': respuesta.status, 'body': json.loads(respuesta.read())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8001)
    args = parser.parse_args()
    # No reutilizar por accidente otro servidor ya iniciado por el usuario.
    with socket.socket() as comprobacion:
        comprobacion.bind(('127.0.0.1', args.port))
    docs = BASE / 'docs'
    docs.mkdir(exist_ok=True)
    comando = [sys.executable, '-m', 'uvicorn', 'app.main:app', '--host', '127.0.0.1', '--port', str(args.port)]
    with (docs / 'uvicorn.log').open('w', encoding='utf-8') as log:
        proceso = subprocess.Popen(comando, cwd=BASE, stdout=log, stderr=subprocess.STDOUT)
        try:
            url = f'http://127.0.0.1:{args.port}'
            for _ in range(60):
                if proceso.poll() is not None:
                    raise RuntimeError('Uvicorn no inició; revise docs/uvicorn.log.')
                try:
                    salud = solicitar(url + '/health')
                    break
                except (URLError, TimeoutError):
                    time.sleep(0.25)
            else:
                raise RuntimeError('El servidor no respondió a tiempo.')
            entrada = json.loads((docs / 'solicitud.json').read_text(encoding='utf-8'))
            prediccion = solicitar(url + '/prediccion/', entrada)
            invalida = solicitar(url + '/prediccion/', {'Position': 'Forward', 'Shots': -1})
            assert salud['status'] == 200, salud
            assert prediccion['status'] == 200, prediccion
            assert invalida['status'] == 422, invalida
            assert abs(sum(prediccion['body']['probabilidades'].values()) - 1) < 1e-9
            assert prediccion['body']['rango_goles_registrados'] in ['0', '1-9', '10+']
            evidencia = {'fecha_utc': datetime.now(timezone.utc).isoformat(),
                         'python': sys.version, 'comando': comando, 'url': url,
                         'modelo_sha256': hashlib.sha256((BASE / 'model/model.pkl').read_bytes()).hexdigest(),
                         'solicitud': entrada, 'health': salud, 'prediccion': prediccion,
                         'validacion_entrada_negativa': invalida, 'resultado': 'PASS'}
            (docs / 'evidencia_localhost.json').write_text(
                json.dumps(evidencia, indent=2, ensure_ascii=False), encoding='utf-8')
            print(json.dumps(evidencia, indent=2, ensure_ascii=False))
        finally:
            proceso.terminate()
            try:
                proceso.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proceso.kill()
                proceso.wait()


if __name__ == '__main__':
    main()
