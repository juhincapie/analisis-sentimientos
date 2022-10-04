# analisis-sentimientos
# Caso de uso

El archivo `data.txt` contiene una serie de comentarios sobre productos
de la tienda de amazon, los cuales están etiquetados como positivos (1), negativos (0)
o indterminados (NULL).

Haciendo uso de Naive Bayes, realice análisis de sentimientos

# Pasos a Seguir

1. Cargue los datos haciendo uso de la librería de pandas, indicando que la columna 0 corresponde al mensaje y la columna 1 corresponde a su etiqueta.
2. Obtenga los grupos de mensajes (etiquetados y sin etiquetar), definiendo como variable independependiente el mensaje y como dependiente la etiqueta
3. Preparar conjunto de datos: Partione la data, de tal manera que obtenga la data de entrenamiento para variable independiente y dependiente, a partir de los grupos de datos etiquetados.Es de resaltar que, la semilla del generador de números aleatorios debe ser 12345 y debe usar el 10% de patrones para la muestra de prueba
4. Cunstruya un analizador de palabras: crear _stemmer_, compilar un _vectorizer_, crear instancia a partir del _vectorizer_ haciendo uso de build_analyzer, ejecutar el analizador
5. Crear una instancia de CountVectorizer que use el analizador de palabras antes mencionado: Esta instancia debe retornar una matriz binaria. El límite superior para la frecuencia de palabras es del 100% y un límite inferior de 5 palabras. Solo deben analizarse palabras conformadas por letras.
6. Crear pipeline que contenga el CountVectorizer mencionado en el ítem anterior y el modelo de BernoulliNB, con el objetivo de definir el estimador de GridSearchCV previo al entrenamiento.
7. Definir un diccionario de parámetros para el GridSearchCV (param_grid). Se deben considerar 10 valores entre 0.1 y 1.0 para el parámetro alpha de BernoulliNB.
8. Definir una instancia de GridSearchCV con el pipeline y el diccionario de parámetros. Use cv=5, y "accuracy" como métrica de evaluación
9. Buscar la mejor combinación de regresores: entrenar instancia de GridSearchCV
10. Evaluar el modelo con los datos de entrenamiento y de prueba usando la matriz de confusión de sklearn.metrics
11. Pronosticar la polaridad del sentimiento para los datos
