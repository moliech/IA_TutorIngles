"""
Modulo de Modelamiento Predictivo con Scikit-Learn
Proyecto: IA_TutorIngles (lexiaAdapt)
Requisito 5 de los Lineamientos del Proyecto Final (Clase 12)
"""

import os 
import joblib 
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

def ejecutar_entrenamiento_comparativo():
    print("=" * 70)
    print(" SISTEMA PREDICTIVO SCIKIT-LEARN IA_TUTORINGLES (CLASE 12)")
    print("=" * 70)


    ruta_datos = os.path.join(
        os.path.dirname(__file__), "..", "data", "proyecto_limpio.csv"
    )

    if not os.path.exists(ruta_datos):
        raise FileNotFoundError(f"No se encontro el dataset en {ruta_datos}")

    df = pd.read_csv(ruta_datos)
    print(f"[+] Dataset cargado correctamente: {len(df)} registros.")

    df["clase_objetivo"] = df["nivel_cefr"].map(
        {"A1": 0, "A2": 0, "B1": 1, "B2": 1}
    )


   features = [
       "conteo_palabras",
       "dificultad_lexica",
       "tiempo_lectura_est_min",
       "velocidad_lectura_ppm"

    ]

    X = df[features]
    y = df["clase_objetivo"]


    print(f"[+] Variables de entrada: {features}")
    print(f"[+] Distribucion de clases: {dict(y.value_counts())} (0: Basico, 1: Avanzado)")

    X_train, X_test, y_train, y_test = train_test_split(
        X,y, test_size=0.25, random_state=42, stratify=y
    )


    print(f"[+] Participacion completada. train={len(X_train)} filas, Test={len(X_test)} filas.")

    cv = StratifiedKFold(n_splits=5,  shuffle=True, random_state=42)

    candidatos = {
        "Regresion logistica": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=42))
         ]),

    "Random Forest": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=50, max_depth=4, random_state=42
        ))
    ]),
    
    "Red Neuronal (MLPClassifier)": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(16, 8),
            max_iter=800,
            random_state=42
         ))
       ])

     }

    print("\n" + "-" * 70)
    print(f"{'modelo':<30} | {'CV Accuracy (media' + chr(0x00B1) + ' Std)':<25}")
    print("-" * 70)


    resultados = {}

    for nombre,pipe in candidatos.items():
        scores = cross_val_score(
             pipe, X_train, y_train, cv=cv, scoring="accuracy"
         )
      
        resultados[nombre] = {
            "cv_mean": scores.mean(),
            "cv_std": scores.std(),
            "pipeline": pipe
         }

        print(f"{nombre:<30} | {scores.mean():.4f} (chr(0x00B1) {scores.std():.4f})")

     mejor_nombre = max(
          resultados, key=lambda k: resultados[k]["cv_mean"]
      )


      mejor_pipeline = resultados[mejor_nombre]["pipeline"]

      print("-" * 70)
      print(f"[{chr(0x2605)}] mejor modelo seleccionado: {mejor_nombre}")

      mejor_pipeline.fit(X_train, y_train)
      y_pred = mejor_pipeline.predict(X_test)

      acc = accuracy_score(y_test, y_pred)
      prec = precision_score(y_test, y_pred, zero_division=0)
      rec = recall_score(y_test, y_pred, zero_division=0)
      f1 = f1_score(y_test, y_pred, zero_division=0)
      
      print("\n" + "=" * 70)
      print(f"  METRICAS EN CONJUNTO DE PRUEBA (DATOS NO VISTOS: {len(y_test)} CASOS)")
      print ("=" * 70)

      print(f"Exactitud (Accuracy): {acc * 100:.2f}%")
      print(f"Precision (Precision): {prec * 100:.2f}%")
      print(f"exhaustividad (Recall): {rec * 100:.2f}%")
      print(f"Puntaje f1 (f1-score): {f1 * 100:.2f}%")
      
      print("\nInforme Detallado de clasificacion:")
      print(classification_report(
          y_test,
          y_pred,
          target_names=["Basico (A1/A2)", "Avanzado  (B1/B2)"]
      ))

      ruta_graficos = os.path.join(
          os.path.dirname(__file__), "..", "graficos"
      )

      os.makedirs(ruta_graficos, exist_ok=True)

      ruta_matriz_png = os.path.join(
          ruta_graficos, "matriz_confusion_sklearn.png"
      )

      cm = confusion_matrix(y_test, y_pred)

      disp = ConfusionMatrixDisplay(
          confusion_matrix=cm,
          display_labels=["Basico", "Avanzado"]
      )

      fig, ax = plt.subplots(figsize=(6, 5))
      disp.plot(cmap="Blues", ax=ax, values_format="d")

      plt.title(
          f"Matriz de Confusion - {mejor_nombre}\n"
          "(Conjunto de prueba: 10 lecturas)",
          fontsize=11,
          fontweight="bold",
          pad=12
      )

      plt.tight_layout()
      plt.savefig(ruta_matriz_png, dpi=300)
      plt.close()

      print(f"[+] Grafico exportado: {ruta_matriz_png")


      ruta_modelo_joblib = os.path.join(
          os.path.dirname(__file__), "modelo_final.joblib"
      )

      joblib.dump(mejor_pipeline, ruta_modelo_joblib)

      print(f"[+] Modelo entrenado guardado con exito en: {ruta_modelo_joblib}")
      print("=" * 70)

if __name__ == "__main__":
    ejecutar_entrenamiento_comparativo()
