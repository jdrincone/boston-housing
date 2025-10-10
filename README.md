# Boston Housing Price Prediction API

Pipeline **MLOps end-to-end** para entrenar y desplegar un modelo de **regresión** sobre el dataset de **Boston Housing**. Incluye versionado de datos con **DVC**, entrenamiento reproducible con **AutoML (FLAML)**, artefactos versionados y una **API** en **FastAPI** lista para producción con base de datos PostgreSQL.

---

## 🚀 Tecnologías

- **API**: FastAPI + Uvicorn  
- **Base de datos**: PostgreSQL + SQLAlchemy
- **Contenerización**: Docker + Docker Compose
- **Versionado**: Git + DVC  
- **Gestión de entorno**: `uv` (wrapper de pip/venv)  
- **Modelado**: scikit-learn, FLAML (AutoML), SHAP, XGBoost
- **Testing**: pytest + httpx
- **Linting**: ruff

---

## ⚙️ Prerrequisitos

- Python **3.11+**
- Git
- Docker & Docker Compose
- `uv` → `pip install uv`
- (Opcional) DVC remoto configurado si vas a `dvc pull` desde storage

---

## 📦 Clonar y configurar

```bash
git clone https://github.com/jdrincone/boston-housing.git
cd boston-housing

# Crear y activar entorno
uv venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate           # Windows

# Instalar dependencias
uv pip install -r requirements.txt
```

---

## 🧠 Pipeline de entrenamiento

Los datos y artefactos se versionan con DVC.

### 1) Descargar datos con DVC
```bash
dvc pull
```
Si es la primera vez, asegúrate de tener configurado el remoto de DVC (S3).

### 2) Reproducir el pipeline completo
```bash
dvc repro
```

Al finalizar, tendrás:
- **Modelo**: `models/best_pipeline.pkl`
- **Reportes**: `reports/` (métricas, SHAP plots, feature importance)
- **Métricas**: `reports/metrics.json`
- **Logs**: `reports/main.log`

Para ver el grafo del pipeline: `dvc dag`

### 3) Reentreno del modelo
```bash
dvc repro --force
```

---

## 🏗️ Arquitectura del proyecto

```
boston-housing/
├── app/                          # API FastAPI
│   ├── main.py                   # Aplicación principal con endpoints
│   ├── schemas.py                # Modelos Pydantic para validación
│   └── database.py               # Configuración de PostgreSQL
├── src/                          # Código fuente del ML pipeline
│   ├── config.py                 # Configuración y rutas
│   ├── data_manager.py           # I/O de modelos y métricas
│   ├── pipeline.py               # Pipeline de ML con FLAML
│   └── train.py                  # Script de entrenamiento
├── scripts/                      # Scripts de preparación
│   └── prepare_data.py           # División train/backtest
|  └── backtesting.py               # Evaluar el rendimiento del modelo
├── data/                         # Datos (DVC tracked)
│   ├── HousingData.csv           # Dataset original
│   ├── train_data.csv            # Datos de entrenamiento
│   └── backtest_data.csv         # Datos de backtesting
├── models/                       # Modelos entrenados (DVC tracked)
│   └── best_pipeline.pkl         # Pipeline completo serializado
├── reports/                      # Reportes y métricas (DVC tracked)
│   ├── metrics.json              # Métricas del modelo
│   ├── shap_summary.png          # Análisis SHAP
│   ├── feature_importance.png    # Importancia de features
│   ├── automl_summary.txt        # Resumen de AutoML
│   └── main.log                  # Logs de entrenamiento
├── tests/                        # Tests unitarios
│   ├── test_api.py               # Tests de la API
│   └── test_training.py          # Tests del pipeline
├── dvc.yaml                      # Configuración del pipeline DVC
├── params.yaml                   # Parámetros del modelo
├── docker-compose.yml            # Orquestación de servicios
├── Dockerfile                    # Imagen de la API
└── requirements.txt              # Dependencias Python
```

---

## 🚀 Despliegue con Docker

### Levantar los servicios completos
```bash
docker-compose up --build
```

Este comando:
- Construye la imagen de la API
- Levanta PostgreSQL en el puerto 5432
- Levanta la API en el puerto 8000
- Configura la conexión entre servicios

### Acceder a la API
- **API**: http://localhost:8000
- **Documentación**: http://localhost:8000/docs
- **Base de datos**: localhost:5432

---
## 📊 Características del modelo

### Features utilizadas
- **CRIM**: Tasa de criminalidad per cápita
- **ZN**: Proporción de terreno residencial zonificado
- **INDUS**: Proporción de acres de negocio no minorista
- **CHAS**: Variable dummy del río Charles
- **NOX**: Concentración de óxidos nítricos
- **RM**: Número promedio de habitaciones por vivienda
- **AGE**: Proporción de unidades ocupadas construidas antes de 1940
- **DIS**: Distancias ponderadas a cinco centros de empleo de Boston
- **RAD**: Índice de accesibilidad a autopistas radiales
- **TAX**: Tasa de impuesto a la propiedad
- **PTRATIO**: Ratio alumno-profesor por ciudad
- **B**: Proporción de afroamericanos por ciudad
- **LSTAT**: % de estatus socioeconómico bajo

### Pipeline de ML
1. **Imputación**: Valores faltantes con mediana
2. **Escalado**: StandardScaler
3. **AutoML**: FLAML con búsqueda automática de hiperparámetros
4. **Métricas**: R², MSE, RMSE, MAE
5. **Explicabilidad**: SHAP y feature importance

---


### Health Check
```bash
curl http://localhost:8000/
```

### Predicción
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "CRIM": 0.02731,
       "ZN": 0.0,
       "INDUS": 7.07,
       "CHAS": 0,
       "NOX": 0.469,
       "RM": 6.421,
       "AGE": 78.9,
       "DIS": 4.9671,
       "RAD": 2,
       "TAX": 242,
       "PTRATIO": 17.8,
       "B": 396.9,
       "LSTAT": 9.14
     }'
```

### Respuesta
```json
{
  "prediction": 24.5
}
```

---

## 📈 Monitoreo y logs

- **Logs de entrenamiento**: `reports/main.log`
- **Métricas del modelo**: `reports/metrics.json`
- **Análisis SHAP**: `reports/shap_summary.png`
- **Importancia de features**: `reports/feature_importance.png`
- **Resumen AutoML**: `reports/automl_summary.txt`

---

## 🧪 Testing

```bash
# Ejecutar todos los tests
pytest

# Tests específicos
pytest tests/test_api.py
pytest tests/test_training.py

```

---

## 🔄 CI/CD Pipeline

El proyecto incluye un pipeline de **CI/CD** automatizado con **GitHub Actions** que se ejecuta en cada push y pull request.

### Configuración del Pipeline

El archivo `.github/workflows/ci.yml` define el pipeline que incluye:

#### **Triggers**
- **Push** a las ramas `main` y `develop`
- **Pull Requests** hacia la rama `main`

#### **Servicios**
- **PostgreSQL 13**: Base de datos de prueba con health checks
- **Ubuntu Latest**: Sistema operativo del runner

#### **Pasos del Pipeline**

1. **Checkout del código**
   ```yaml
   - Checkout repository (actions/checkout@v4)
   ```

2. **Configuración del entorno Python**
   ```yaml
   - Set up Python 3.11 (actions/setup-python@v5)
   - Install uv and dependencies
   ```

3. **Configuración de AWS para DVC**
   ```yaml
   - Configure AWS Credentials (aws-actions/configure-aws-credentials@v4)
   - Pull DVC tracked data
   ```

4. **Linting y validación**
   ```yaml
   - Lint code with Ruff
   ```

5. **Testing**
   ```yaml
   - Run Tests with Pytest (con PostgreSQL de prueba)
   ```

### Variables de Entorno Requeridas

Para que el pipeline funcione correctamente, se necesitan configurar los siguientes **secrets** en GitHub:

- `AWS_ACCESS_KEY_ID`: Clave de acceso de AWS para DVC
- `AWS_SECRET_ACCESS_KEY`: Clave secreta de AWS para DVC

### Configuración de Secrets

1. Ve a **Settings** → **Secrets and variables** → **Actions**
2. Agrega los secrets necesarios:
   - `AWS_ACCESS_KEY_ID`
   - `AWS_SECRET_ACCESS_KEY`

### Estado del Pipeline

El pipeline valida:
- ✅ **Linting**: Código limpio con Ruff
- ✅ **Tests**: Todos los tests unitarios pasan
- ✅ **DVC**: Datos y modelos se descargan correctamente
- ✅ **Base de datos**: Conexión a PostgreSQL funcional

---

## 📊 Backtesting del Modelo

El proyecto incluye un sistema de **backtesting** que permite evaluar el rendimiento del modelo en datos no vistos mediante llamadas a la API en producción.

### Funcionalidad del Backtesting

El script `scripts/backtesting.py` realiza las siguientes operaciones:

1. **Carga de datos**: Lee el archivo `data/backtest_data.csv` (5% de los datos originales)
2. **Llamadas a la API**: Envía cada registro a `http://localhost:8000/predict`
3. **Comparación**: Compara predicciones vs valores reales
4. **Reporte**: Genera un informe detallado con métricas

### Ejecutar Backtesting

```bash
# Asegúrate de que la API esté corriendo
docker-compose up --build

# En otra terminal, ejecuta el backtesting
python scripts/backtesting.py
```

### Archivos Generados

- **`reports/backtest_report.csv`**: Reporte detallado con predicciones vs valores reales
- **`reports/backtest.log`**: Logs del proceso de backtesting

### Estructura del Reporte

El archivo `backtest_report.csv` contiene:
- `id`: Identificador del registro
- `actual_value`: Valor real del precio de la vivienda
- `predicted_value`: Predicción del modelo
- `payload_sent`: Datos enviados a la API
- `error`: Errores si los hay (ej: valores fuera de rango JSON)


### Validación de la API

El backtesting valida:
- ✅ **Conectividad**: La API responde correctamente
- ✅ **Formato de datos**: Validación de esquemas Pydantic
- ✅ **Predicciones**: Valores numéricos válidos
- ✅ **Manejo de errores**: Valores fuera de rango JSON
- ✅ **Logging**: Registro detallado de cada operación