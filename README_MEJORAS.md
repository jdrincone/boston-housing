# Propuestas de Mejora (MLOps) – Boston Housing

Este documento propone mejoras técnicas y de proceso para llevar el proyecto a nivel productivo con prácticas MLOps. Incluye monitoreo integral (logs, métricas de cara al negocio, drift), gobierno de features (feature store), manejo de nulos y un plan de alertas.

---

## 1) Observabilidad integral (Logs, Métricas, Trazas)

Tener visibilidad end to end del pipeline, la API y el modelo.
- Propuesta de stack:
  - Logs:  Grafana, DataDog.
  - Métricas de app/modelo: Dashboard para revisar Drift, comportamiento de los feature importan más
    relevantes.

### 1.1 Tipos de logs a capturar (estructura y ejemplos)
- Logs de aplicación API (nivel INFO/WARN/ERROR):
- Logs de entrenamiento (pipeline):

Estandarizar con formato JSON.

### 1.2 Métricas a exponer
- API:
  - http_requests_total{endpoint,method,status}
  - http_request_duration_seconds_bucket{endpoint}
  - api_errors_total{endpoint,error_type}
- Modelo:
  - model_inference_latency_seconds
  - model_predictions_total
  - model_imputations_total
  - Precios muy bajo o muy altos


---

## 2) Feature Store

- Garantizar coherencia de features entre entrenamiento y serving, versionado, trazabilidad y reutilización.


---

## 3) Monitoreo de métricas de negocio y de ML

- MAE sobre ventana de backtesting/producción (rolling 7/30 días) ¿Cada cuanto es mejor un reentreno, como se 
degrandan los features en el tiempo).
- Tasa de solicitudes servidas con latencia < 200 ms.
- % de predicciones con residuo absoluto > umbral (ej. > 5 unidades MEDV).
- Porcentajes de solicitudes donde la variable más importante esta por fuera del rango normal.
- Desempeño: R2, RMSE, MAE (rolling, por segmento si es viable).
- Estabilidad de entrada: distribución y KS-test por feature clave (RM, LSTAT, NOX, TAX, PTRATIO, etc.).
- Salud del pipeline: tiempos por etapa, tasas de error por job.

---

## 4) Drift (datos y rendimiento)

- Herramientas: Evidently AI o whylogs + jobs programados (GitHub Actions/cron/K8s CronJob).
- Tipos:
  - Data Drift: cambios en la distribución de features.
  - Target Drift: cambios en MEDV (cuando se dispone del real posteriormente).
  - Prediction Drift: cambios en la distribución de predicciones.
---

## 5) Manejo de valores nulos (features importantes)

Principios:
- Consistencia entre train y serve.
- Métricas de imputación monitoreadas (conteo y tasa por feature).

Estrategias recomendadas por feature relevante (ejemplos; validar con análisis):
- RM (habitaciones), LSTAT, NOX, TAX, PTRATIO:
  - Imputación robusta con mediana entrenada (ya implementada con `SimpleImputer(strategy="median")`).
- CHAS, RAD (categóricas/índices discretos):
---

## 6) Alertas (operativas y de negocio)

Tipos y reglas (ejemplos):
- Disponibilidad API
- Errores
- Modelo/Negocio
---

## 7) Registro de modelos

- MLflow Model Registry (o alternativa) para:
  - Versionado de experimentos y parámetros.
  - Ciclo de vida del modelo: Staging → Production (aprobaciones).
- Integrar con CI/CD para promover automáticamente si pasa umbrales (guardrails) en backtesting/validación.

---

## 9) CI/CD reforzado (train + deploy + monitoreo)

- Jobs separados:
  - CI: lint, tests, `dvc pull`, validaciones de datos rápidas.
  - Retrain (cron + manual): `dvc repro --force`, publicar métricas y reportes, `dvc push` (si se usa remoto).
  - LLamar los datos de entrada desde la api de kaggel y no desde datos estaticos en local.

---


