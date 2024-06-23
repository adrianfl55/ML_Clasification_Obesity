datos = read.csv("ObesityDataSet_raw_and_data_sinthetic.csv")
datos
datos <- datos[1:499, ]
datos
datos_numericos <- datos[ , 2:4]
datos_numericos
target <- datos[ , 17]

# Boxplot
colores <- c("red", "green", "blue")
nombres_cajas <- c("Edad", "Altura", "Peso")
boxplot(datos_numericos, main = "Boxplot de los datos numéricos", xlab = "Variables", 
        ylab = "Valores", col = colores, names = nombres_cajas)

personas_por_clase <- table(target)
print(personas_por_clase)

# Gráfico de barras
colores <- c("blue", "red", "green", "purple", "orange", "yellow", "black")
barplot(personas_por_clase, main = "Personas por Clase", xlab = "Clase", 
        ylab = "Número de Personas", col = colores,
        legend = c("Peso insuficiente", "Peso normal", "Obesidad Tipo I", 
                   "Obesidad Tipo II", "Obesidad Tipo III", "Sobrepeso Nivel I",
                   "Sobrepeso Nivel II"),
        args.legend = list(cex = 0.8))  # Ajustar el tamaño de la leyenda


