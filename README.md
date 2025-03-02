### Algoritmo Kernel Ridge Regression (KRR)  

Kernel Ridge Regression (KRR) es una técnica de regresión que combina Ridge Regression con métodos de kernels. Se basa en la siguiente ecuación:  

![imagen](https://github.com/user-attachments/assets/b08c9274-3346-47fc-b082-90d2c4dd9d59)


El uso de kernels permite que KRR modele relaciones no lineales en los datos.  

---

### Implementación en el Código  

1. **Generación de datos**  
   - Se crean datos de prueba con ruido usando una función seno.  

2. **Entrenamiento del modelo**  
   - Se inicializa la clase `KernelRidgeRegression` con un valor de \( \alpha \).  
   - Se entrena el modelo con `fit(X, y, gamma)`, donde `gamma` es el parámetro de ancho para el kernel RBF.  

3. **Predicción**  
   - Se usa `predict(X, gamma)` para estimar los valores de salida.  

4. **Visualización de resultados**  
   - Se genera una gráfica de dispersión con los datos reales y la predicción del modelo.  
   - Se calcula el error cuadrático medio (MSE) y se muestra en una gráfica de barras.  

Esta implementación permite visualizar cómo KRR ajusta una función no lineal a los datos de entrada. 
