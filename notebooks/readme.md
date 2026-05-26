## ¡Equipo!

En esta sección tendremos nuestro notebook final  +  una carpeta de tests
Semanticamente no es correcto, pero la idea principal de esta carpeta es que vayan testeando cada una de las secciones definidas en el cuaderno principal
> La idea aca es que vayan haciendo un cuaderno x cada sección para despues implementar el .ipynb final que es el 01 (Para esta entrega)
---
por otro lado, tienen una carpeta `src` en la cual tendremos distintos tipos de códigos, ya sea:
- Tratamiendo de datos
- Pipelines de modelos
- Archivos de entrenamiento
- Archivo dedicado a la evaluacion de los modelos
- Vistas y plots relevantes para el proceso de entrenamiento

Esta carpeta se realizó netamente para mantener un orden modular en el proyecto y dejar lo más legible posible el notebook principal
> MUY IMPORTANTE: no se puede como tal importar un data set completo, la librería nos ofrece importar los splits oficiales como train, validation y test. Tienen que ver como juntarlos todos, almenos el seba para el eda. ya para el entrenamiento lo bueno es que tienen separado los conjuntos, asi que teóricamente no deberian tener problema y debiera ser dentro de todo sencillo. Porfa leanse el enunciado tambien