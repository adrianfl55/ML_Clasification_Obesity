datos = read.csv("ObesityDataSet_raw_and_data_sinthetic.csv")
datos
datos <- datos[1:499, ]
datos
datos_numericos <- datos[ , 2:4]
datos_numericos
target <- datos[ , 17]
# Cambia los colores de las cajas
colores <- c("red", "green", "blue")

# Cambia los nombres de las cajas
nombres_cajas <- c("Edad", "Altura", "Peso")

# Crea el boxplot
boxplot(datos_numericos, 
        main = "Boxplot de los datos numéricos", 
        xlab = "Variables", 
        ylab = "Valores", 
        col = colores, 
        names = nombres_cajas)

# Obtén el conteo de personas por clase
personas_por_clase <- table(target)

# Imprime el resultado
print(personas_por_clase)

# Crea un gráfico de barras

colores <- c("blue", "red", "green", "purple", "orange", "yellow", "black")

# Crear el gráfico de barras
barplot(personas_por_clase, 
        main = "Personas por Clase", 
        xlab = "Clase", 
        ylab = "Número de Personas", 
        col = colores,
        legend = c("Peso insuficiente", "Peso normal", "Obesidad Tipo I", 
                   "Obesidad Tipo II", "Obesidad Tipo III", "Sobrepeso Nivel I",
                   "Sobrepeso Nivel II"),
        args.legend = list(cex = 0.8))  # Ajustar el tamaño de la leyenda

########## GRÁFICO RNA TOPOLOGIAS ############

########## GRÁFICO SVM DIAGRAMA DE BARRAS ######################

########## GRÁFICO DTC LINEAS ######################

########## GRÁFICO KNN LINEAS ######################


