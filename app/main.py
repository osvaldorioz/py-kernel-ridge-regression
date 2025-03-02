from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
from typing import List
import matplotlib
import matplotlib.pyplot as plt
from krr_module import KernelRidgeRegression
import json

matplotlib.use('Agg')  # Usar backend no interactivo
app = FastAPI()

# Generación de datos de ejemplo
def generate_data(n_samples: int):
    X = np.linspace(0, 2 * np.pi, n_samples)
    y = np.sin(X)
    return X.reshape(-1, 1), y.reshape(-1, 1)

# Definir el modelo para el vector
class VectorF(BaseModel):
    vector: List[float]
    
@app.post("/kernel-ridge-regression")
def calculo(num_samples: int, alpha: float, gamma: float):
    output_file = 'kernel_ridge_regression.png'

    from krr_module import KernelRidgeRegression

    # Generar datos de ejemplo
    np.random.seed(42)
    X = np.linspace(-3, 3, num_samples).reshape(-1, 1)
    y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

    # Entrenar el modelo
    #alpha = 0.1
    #gamma = 0.5
    krr = KernelRidgeRegression(alpha)
    krr.fit(X, y, gamma)

    y_pred = krr.predict(X, gamma)

    # Graficar resultados
    plt.figure(figsize=(10, 4))

    # Gráfica de dispersión
    plt.subplot(1, 2, 1)
    plt.scatter(X, y, label="Datos reales", color='blue', alpha=0.6)
    plt.plot(X, y_pred, label="Predicción", color='red')
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("Regresión con KRR")
    plt.legend()

    # Gráfica de pérdida (error cuadrático medio)
    mse = np.mean((y - y_pred) ** 2)
    plt.subplot(1, 2, 2)
    plt.bar(["MSE"], [mse], color='orange', width=0.4)
    plt.ylabel("Error")
    plt.title("Error de Predicción")
    plt.text(0, mse / 2, f"{mse:.5f}", ha='center', va='center', fontsize=12, color='blue')

    plt.tight_layout()
    #plt.show()

    plt.savefig(output_file)
    plt.close()
    
    j1 = {
        "Grafica": output_file
    }
    jj = json.dumps(str(j1))

    return jj

@app.get("/kernel-ridge-regression-graph")
def getGraph(output_file: str):
    return FileResponse(output_file, media_type="image/png", filename=output_file)