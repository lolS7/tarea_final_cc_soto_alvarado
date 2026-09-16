"""E1: entrenamiento reproducible. Ejecutar: python train.py."""
from Model_01 import argumentos
from Model_02 import guardar_modelo


if __name__ == '__main__':
    guardar_modelo(argumentos().data)
