# Early-intent-detection-MindSpore-Huawei-

[![MindSpore](https://img.shields.io/badge/MindSpore-2.3.0-red)](https://www.mindspore.cn/)
[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)


Exploración de herramienta MindSpore para un caso práctico de aplicación electrónica. 

## Introducción

Mientras realizaba la capacitación del HCIA.AI de Huawei; me topé con MindSpore; y la verdad se me hizo interesante la propuesta y la comunidad en torno a este framework ... así tomé la iniciativa de aplicarlo a un campo de electrónica para averiguar el impacto que tendría en un caso práctico; así entonces trato de crear un sistema de detección temprana de intenciones de movimiento basado en señales **EMG (electromiografía)** de un sensor basado en mediciones del fabricante **NinaPro** (https://ninapro.hevs.ch/). Eventualmente incorporo **MindSpore** con objeto de predecir movimientos específicos con alta precisión, lo que tiene aplicaciones en prótesis inteligentes y rehabilitación; lo cual estaría muy interesante averiguar a profundidad el alcanse real de este tipo de tecnologías prácticas. 

## Objetivos

- **Detección temprana**: Predecir intenciones de movimiento antes de su ejecución completa
- **Clasificación precisa**: Identificar movimientos a partir de señales EMG
- **Implementación eficiente**: Utilizar MindSpore para optimizar el proceso

## Preprocesamiento de Datos

### Estructura del Dataset
El archivo `S10_E1_A1.mat` contiene:
- **EMG**: 149,919 muestras × 16 canales
- **Stimulus**: 149,919 etiquetas de movimiento
- **Frecuencia**: 2000 Hz de muestreo

### Proceso de Ventaneo
```python
fs = 2000  # Hz - frecuencia de muestreo
window_size = 400  # 200 ms (0.2 * 2000)
step_size = 100    # 50 ms (0.05 * 2000) - 75% de solapamiento