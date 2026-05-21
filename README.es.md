# Clasificador Random Forest — Predicción de Diabetes

> Clasificador de ensemble aplicado al dataset Pima Indians Diabetes, construido sobre el mismo pipeline de EDA que el proyecto de Árbol de Decisión — un `RandomForestClassifier(n_estimators=60)` alcanza una **precisión del 76,0%** sin ningún ajuste de hiperparámetros, superando el 71,4% del árbol individual optimizado y demostrando la ventaja del ensemble.

---

## Problema

Predecir si un paciente tiene diabetes basándose en medidas diagnósticas. El mismo contexto clínico que el proyecto de Árbol de Decisión: una comparación de modelo interpretable vs. poderoso sobre datos idénticos — la pregunta central siendo cuánta precisión gana el ensemble sobre una única frontera de decisión.

## Dataset

- **Fuente:** Dataset Pima Indians Diabetes (768 filas × 9 características)
- **Target:** `Outcome` — 1 = diabetes, 0 = no diabetes (distribución de clases 65% / 35%)
- **Características usadas tras limpieza:** Glucose, BMI, Age, Pregnancies (top 4 por SelectKBest f_classif)

El pipeline completo de EDA y preprocesamiento es idéntico al proyecto de Árbol de Decisión:

| Paso | Acción |
|---|---|
| Eliminar columnas imposibles | Insulin (48,7% ceros) y SkinThickness (29,6% ceros) eliminadas |
| Imputación de ceros | Imputación con mediana estratificada por grupo en Glucose, BloodPressure, BMI |
| Capping de outliers | Método IQR en Pregnancies, DiabetesPedigreeFunction, Age |
| Escalado | StandardScaler |
| Selección de características | SelectKBest (f_classif, k=4) → Glucose, BMI, Age, Pregnancies |
| División | 80/20 estratificada (614 entrenamiento / 154 prueba) |

## Resultados del Modelo

| Modelo | Precisión | Notas |
|---|---|---|
| Árbol de Decisión individual (optimizado, max_depth=5) | 71,4% | GridSearchCV, validación cruzada 10 pliegues |
| **Random Forest (n_estimators=60)** | **76,0%** | Parámetros por defecto, sin ajuste |

Un Random Forest con configuración por defecto supera a un Árbol de Decisión cuidadosamente optimizado en **+4,6 puntos porcentuales**. Este es el efecto del ensemble en la práctica: 60 árboles entrenados cada uno en una muestra bootstrap aleatoria y un subconjunto aleatorio de características, cuyos votos mayoritarios promedian los errores individuales.

## Conclusiones Clave

- **Los ensembles superan a los árboles individuales sin ajuste:** El Random Forest alcanza mayor precisión que el Árbol de Decisión optimizado con GridSearchCV sin ningún ajuste de hiperparámetros. La diversidad de 60 árboles crecidos independientemente es más valiosa que el ajuste perfecto de uno solo.
- **Bootstrap + aleatoriedad de características = varianza reducida:** Cada árbol del bosque ve una muestra aleatoria diferente de los datos de entrenamiento y un subconjunto aleatorio de características en cada división. Esto descorrelaciona los árboles — sus errores no coinciden — por lo que el voto del ensemble es más fiable que cualquier árbol individual.
- **Precisión vs. interpretabilidad:** El Árbol de Decisión individual puede graficarse e inspeccionarse — un médico puede seguir el camino exacto de características a predicción. El Random Forest es una caja negra de 60 árboles. La ganancia de precisión del 4,6% tiene un coste real en explicabilidad.

## Stack Tecnológico

`Python` · `scikit-learn` · `pandas` · `NumPy` · `Matplotlib` · `Seaborn`

## Ejecutar Localmente

```bash
git clone https://github.com/matthewkane-ml/ML_RandomForest_MTK.git
cd ML_RandomForest_MTK
pip install -r requirements.txt
jupyter notebook src/RandomForest.ipynb
```

El modelo entrenado se guarda en `models/` mediante `pickle`.

## Próximos Pasos

- Ejecutar `GridSearchCV` en el Random Forest (`n_estimators`, `max_depth`, `min_samples_split`, `max_features`) para encontrar el techo de precisión con un ajuste adecuado
- Graficar las **importancias de características** del bosque — ¿en cuál de las 4 características se apoya más el ensemble? Comparar con la estructura de divisiones del árbol individual
- Añadir **valores SHAP** para predicciones individuales — esto recupera parcialmente la interpretabilidad del ensemble sin sacrificar precisión

---

**Autor:** Matthew Kane — [LinkedIn](https://www.linkedin.com/in/thomas-k-392094410/) · [Portafolio GitHub](https://github.com/matthewkane-ml)
