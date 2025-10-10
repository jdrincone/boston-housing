# Boston Housing Price Prediction API

Pipeline **MLOps end-to-end** para entrenar y desplegar un modelo de **regresión** sobre el dataset de **Boston Housing**. Incluye versionado de datos con **DVC**, entrenamiento reproducible, artefactos versionados y una **API** en **FastAPI** lista para producción.

---

## 🚀 Tecnologías

- **API**: FastAPI + Uvicorn  
- **Contenerización**: Docker  
- **Versionado**: Git + DVC  
- **Gestión de entorno**: `uv` (wrapper de pip/venv)  
- **Modelado**: scikit-learn, FLAML (AutoML), SHAP  

---

## ⚙️ Prerrequisitos

- Python **3.11+**
- Git
- Docker
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

## 🧠 Pipeline de entrenamiento

Los datos y artefactos se versionan con DVC.
```
1) Descargar datos con DVC
dvc pull
```
Si es la primera vez, asegúrate de tener configurado el remoto de DVC (S3/local/GDrive).
2) Reproducir el pipeline completo
```
dvc repro
```
Al finalizar, tendrás:

Modelo/pipeline: models/best_pipeline.pkl (o models/best_model.pkl según tu configuración)

Transformadores/feature names: transformers/…

Reportes: reports/ (si aplica)

Métricas: metrics/metrics.json

Para ver el grafo del pipeline: dvc dag

Probar la API local (sin Docker)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

boston-housing/
├── app/
│   ├── main.py                 # FastAPI app
│   └── schemas.py              # Pydantic schemas
├── src/
│   ├── config.py               # Rutas/constantes
│   ├── data_manager.py         # I/O de modelo, métricas
│   ├── data_ingestion.py       # Descarga/copia de datos
│   ├── data_cleaning.py        # Limpieza
│   ├── feature_engineering.py  # FE + selección + escalado
│   └── train_automl.py         # Entrenamiento (FLAML/AutoML)
├── data/
│   ├── raw/                    # DVC tracked
│   └── processed/              # DVC tracked
├── models/                     # DVC tracked (artefactos)
├── transformers/               # DVC tracked (scaler, features)
├── metrics/
│   └── metrics.json
├── dvc.yaml
├── params.yaml
├── requirements.txt
└── README.md


Levanta los servicios:
Desde la raíz del proyecto, ejecuta el siguiente comando:

Bash

docker-compose up --build
Este comando construye la imagen de la API y levanta los contenedores de la API y la base de datos.

La API estará disponible en http://localhost:8000.


Reentreno del modelo
dvc repro --force
