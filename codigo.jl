# Importamos las librerías necesarias
using CSV
using DataFrames
using Random
using Statistics
using Plots
using StatsPlots
using MLBase
using JLD2

# Incluimos el archivo firmas.jl para usar las funciones previas
include("firmas_def.jl")

# Leemos el archivo CSV y lo convertimos en un DataFrame
datos = CSV.read("ObesityDataSet_raw_and_data_sinthetic.csv", DataFrame)

# Usamos únicamente las 500 primeras filas
datos = datos[1:499, :]
datos_original = copy(datos)
datos_original

############################################################## DESCRIPCIÓN DE LA BASE DE DATOS ################################################################################

# Columnas de la base de datos
Gender = unique(datos[:, 1]) # Female y Male
Age = unique(datos[:, 2]) 
Height = unique(datos[:, 3]) 
Weight = unique(datos[:, 4]) 
family_history_with_overweight = unique(datos[:, 5]) # yes, no
FAVC = unique(datos[:, 6]) # yes, no
FCVC = unique(datos[:, 7]) # 1, 2, 3
NCP = unique(datos[:, 8]) # 1, 3, 4
CAEC = unique(datos[:, 9]) # Sometimes, Frequently, Always, no
SMOKE = unique(datos[:, 10]) # yes, no
CH20 = unique(datos[:, 11]) # 1, 1.152736, 2, 3
SCC = unique(datos[:, 12]) # yes, no
FAF = unique(datos[:, 13]) # 0, 0.319156, 1, 2, 3
TUE = unique(datos[:, 14]) # 0, 1, 2
CALC = unique(datos[:, 15]) # no, Sometimes, Frequently, Always
MTRANS = unique(datos[:, 16]) # Public_Transportation, Walking, Automobile, Motorbike, Bike
NObeyesdad = unique(datos[:, 17]) # Insufficient_Weight, Normal_Weight, Overweight_Level_I, Overweight_Level_II, Obesity_Type_I, Obesity_Type_II, Obesity_Type_III

# Patrones y características
patrones = size(datos, 1)
atributos = size(datos, 2) - 1
# entradas = names(datos[1, 1:atributos])
clases = unique(datos[:, end])

datos = Matrix(datos)
inputs = datos[:, 1:atributos]
targets = datos[:, end]
inputsNumericas = inputs[:, [2, 3, 4]]

# Estadísticas de las columnas numéricas
minimos = minimum(inputsNumericas, dims = 1)
maximos = maximum(inputsNumericas, dims = 1)
medias = mean(inputsNumericas, dims = 1)
desviaciones = std(inputsNumericas, dims = 1)

# Cuantiles
Q1 = mapslices(x -> quantile(x, 0.25), inputsNumericas, dims = 1)
Q2 = mapslices(x -> quantile(x, 0.5), inputsNumericas, dims = 1)
Q3 = mapslices(x -> quantile(x, 0.75), inputsNumericas, dims = 1)
IQR = Q3 - Q1

personas_por_clase = countmap(datos[:, end])

############################################################## PREPROCESADO DE LOS DATOS ################################################################################
# Procesamos los datos -> Nuestro target es Obesidad
# Columnas categóricas numéricas 7, 8, 11, 13, 14
categorica7 = Int.(oneHotEncoding(inputs[:, 7]))
categorica8 = Int.(oneHotEncoding(inputs[:, 8]))
categorica11 = Int.(oneHotEncoding(inputs[:, 11]))
categorica13 = Int.(oneHotEncoding(inputs[:, 13]))
categorica14 = Int.(oneHotEncoding(inputs[:, 14]))

# Columnas binarias 1, 5, 6, 10, 12
binaria1 = Int.(oneHotEncoding(inputs[:, 1]))
binaria5 = Int.(oneHotEncoding(inputs[:, 5]))
binaria6 = Int.(oneHotEncoding(inputs[:, 6]))
binaria10 = Int.(oneHotEncoding(inputs[:, 10]))
binaria12 = Int.(oneHotEncoding(inputs[:, 12]))

# Columnas categóricas de texto 9, 15, 16 y targets
categorica9 = Int.(oneHotEncoding(inputs[:, 9]))
categorica15 = Int.(oneHotEncoding(inputs[:, 15]))
categorica16 = Int.(oneHotEncoding(inputs[:, 16]))

# ¿ Normalizamos los datos ?
# Comprobamos si hay valores atípicos
atipicos = []
for i in eachcol(inputsNumericas)
    Q1 = quantile(i, 0.25)
    Q3 = quantile(i, 0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    push!(atipicos, i[(i .< limite_inferior) .| (i .> limite_superior)])
end

# Efectivamente los tenemos
for i in atipicos
    println(i)
end

medias_atipicos = [isempty(col) ? NaN : mean(col) for col in atipicos]

# Hacemos un plot con los datos atípicos
scatter(atipicos, legend = false)

# Observamos algunos valores atípicos, por lo que aplicamos la normalización de media 0:
inputsNumericas = convert(AbstractMatrix{Float32}, inputsNumericas) # Necesitamos convertirlas a Float32 para poder aplicar Normalización (antes era Any)
inputsNumericas = normalizeZeroMean(inputsNumericas) 
round(mean(inputsNumericas))
round(std(inputsNumericas))

inputsFinal = [binaria1 inputsNumericas binaria5 binaria6 categorica7 categorica8 categorica9 binaria10 categorica11 binaria12 categorica13 categorica15 categorica16]
inputsFinal
# println(inputsFinal[1,:])
# inputsFinal = convert(AbstractMatrix{Real}, inputsFinal)
# println(typeof(inputsFinal))

# Fijamos una semilla
Random.seed!(1234)
crossValidationIndices = crossvalidation(targets, 10) 

############################################################## ALGORITMOS ################################################################################
# 2. SVM
println("SVM")

if isfile(joinpath(pwd(), "svm.jld2"))
    resultado = load("svm.jld2", "resultados")
    precision_svm = resultado["precision"]
    tasasdeerror_svm = resultado["tasadeerror"]
    sensibilidad_svm = resultado["sensibilidad"]
    especificidad_svm = resultado["especificidad"]
    VPP_svm = resultado["vpp"]
    VPN_svm = resultado["vpn"]
    F1_svm = resultado["f1"]
    hiperparametros_svm = resultado["hiperparametros"]
else
    configuraciones = 8
    precision_svm = Array{Tuple{Float32, Float32}, 1}(undef, configuraciones)
    tasasdeerror_svm = Array{Tuple{Float32, Float32}, 1}(undef, configuraciones)
    sensibilidad_svm = Array{Tuple{Float32, Float32}, 1}(undef, configuraciones)
    especificidad_svm = Array{Tuple{Float32, Float32}, 1}(undef, configuraciones)
    VPP_svm = Array{Tuple{Float32, Float32}, 1}(undef, configuraciones)
    VPN_svm = Array{Tuple{Float32, Float32}, 1}(undef, configuraciones)
    F1_svm = Array{Tuple{Float32, Float32}, 1}(undef, configuraciones)
    hiperparametros_svm = []
    hiperparametro1 = Dict("C" => 0.1, "kernel" => "rbf", "gamma" => 2, "degree" => 3, "coef0" =>0.0);
    push!(hiperparametros_svm, hiperparametro1)
    hiperparametro2 = Dict("C" => 1, "kernel" => "rbf", "gamma" => 2, "degree" => 3, "coef0" =>0.0);
    push!(hiperparametros_svm, hiperparametro2)
    hiperparametro3 = Dict("C" => 10, "kernel" => "rbf", "gamma" => 2, "degree" => 3, "coef0" =>0.0);
    push!(hiperparametros_svm, hiperparametro3)
    hiperparametro4 = Dict("C" => 100, "kernel" => "rbf", "gamma" => 2, "degree" => 3, "coef0" =>0.0);
    push!(hiperparametros_svm, hiperparametro4)
    hiperparametro5 = Dict("C" => 0.1, "kernel" => "linear", "gamma" => 2, "degree" => 3, "coef0" =>0.0);
    push!(hiperparametros_svm, hiperparametro5)
    hiperparametro6 = Dict("C" => 1, "kernel" => "linear", "gamma" => 2, "degree" => 3, "coef0" =>0.0);
    push!(hiperparametros_svm, hiperparametro6)
    hiperparametro7 = Dict("C" => 10, "kernel" => "linear", "gamma" => 2, "degree" => 3, "coef0" =>0.0);
    push!(hiperparametros_svm, hiperparametro7)
    hiperparametro8 = Dict("C" => 100, "kernel" => "linear", "gamma" => 2, "degree" => 3, "coef0" =>0.0);
    push!(hiperparametros_svm, hiperparametro8)

    for i in 1:configuraciones
        precision, tasadeerror, sensibilidad, especificidad, vpp, vpn, f1 = modelCrossValidation(:SVC, hiperparametros_svm[i] , inputsFinal, targets, crossValidationIndices)
        precision_svm[i] = precision
        tasasdeerror_svm[i] = tasadeerror
        sensibilidad_svm[i] = sensibilidad
        especificidad_svm[i] = especificidad
        VPP_svm[i] = vpp
        VPN_svm[i] = vpn
        F1_svm[i] = f1
    end
    resultados = Dict(
        "precision" => precision_svm,
        "tasadeerror" => tasasdeerror_svm,
        "sensibilidad" => sensibilidad_svm,
        "especificidad" => especificidad_svm,
        "vpp" => VPP_svm,
        "vpn" => VPN_svm,
        "f1" => F1_svm,
        "hiperparametros" => hiperparametros_svm
    )
    save("svm.jld2", "resultados", resultados)
end

# 3. DecisionTreeClassifier
println("DecisionTreeClassifier")

if isfile(joinpath(pwd(), "dtc.jld2"))
    resultado = load("dtc.jld2", "resultados")
    precision_dtc = resultado["precision"]
    tasasdeerror_dtc = resultado["tasadeerror"]
    sensibilidad_dtc = resultado["sensibilidad"]
    especificidad_dtc = resultado["especificidad"]
    VPP_dtc = resultado["vpp"]
    VPN_dtc = resultado["vpn"]
    F1_dtc = resultado["f1"]
    hiperparametros_dtc = resultado["hiperparametros"]
else

    valores = 8
    precision_dtc = Array{Tuple{Float32, Float32}, 1}(undef, valores)
    tasasdeerror_dtc = Array{Tuple{Float32, Float32}, 1}(undef, valores)
    sensibilidad_dtc = Array{Tuple{Float32, Float32}, 1}(undef, valores)
    especificidad_dtc = Array{Tuple{Float32, Float32}, 1}(undef, valores)
    VPP_dtc = Array{Tuple{Float32, Float32}, 1}(undef, valores)
    VPN_dtc = Array{Tuple{Float32, Float32}, 1}(undef, valores)
    F1_dtc = Array{Tuple{Float32, Float32}, 1}(undef, valores)
    hiperparametros_dtc = []

    for i in 1:valores
        modelHyperparameters = Dict("max_depth" => i*i) 
        push!(hiperparametros_dtc, modelHyperparameters)

        precision, tasadeerror, sensibilidad, especificidad, vpp, vpn, f1 = modelCrossValidation(:DecisionTreeClassifier, modelHyperparameters, inputsFinal, targets, crossValidationIndices)
        
        precision_dtc[i] = precision
        tasasdeerror_dtc[i] = tasadeerror
        sensibilidad_dtc[i] = sensibilidad
        especificidad_dtc[i] = especificidad
        VPP_dtc[i] = vpp
        VPN_dtc[i] = vpn
        F1_dtc[i] = f1
    end
    resultados = Dict(
        "precision" => precision_dtc,
        "tasadeerror" => tasasdeerror_dtc,
        "sensibilidad" => sensibilidad_dtc,
        "especificidad" => especificidad_dtc,
        "vpp" => VPP_dtc,
        "vpn" => VPN_dtc,
        "f1" => F1_dtc,
        "hiperparametros" => hiperparametros_dtc
    )
    save("dtc.jld2", "resultados", resultados)
end

# 4. KNN  
println("kNN")

if isfile(joinpath(pwd(), "knn.jld2"))
    resultado = load("knn.jld2", "resultados")
    precision_knn = resultado["precision"]
    tasasdeerror_knn = resultado["tasadeerror"]
    sensibilidad_knn = resultado["sensibilidad"]
    especificidad_knn = resultado["especificidad"]
    VPP_knn = resultado["vpp"]
    VPN_knn = resultado["vpn"]
    F1_knn = resultado["f1"]
    hiperparametros_knn = resultado["hiperparametros"]
else
    valores = 8
    precision_knn = Array{Tuple{Float32, Float32}, 1}(undef, valores)
    tasasdeerror_knn = Array{Tuple{Float32, Float32}, 1}(undef, valores)
    sensibilidad_knn = Array{Tuple{Float32, Float32}, 1}(undef, valores)
    especificidad_knn = Array{Tuple{Float32, Float32}, 1}(undef, valores)
    VPP_knn = Array{Tuple{Float32, Float32}, 1}(undef, valores)
    VPN_knn = Array{Tuple{Float32, Float32}, 1}(undef, valores)
    F1_knn = Array{Tuple{Float32, Float32}, 1}(undef, valores)
    hiperparametros_knn = []

    for i in 1:valores
        modelHyperparameters = Dict("n_neighbors" => i)
        push!(hiperparametros_knn, modelHyperparameters)

        precision, tasadeerror, sensibilidad, especificidad, vpp, vpn, f1 = modelCrossValidation(:KNeighborsClassifier, modelHyperparameters, inputsFinal, targets, crossValidationIndices)

        precision_knn[i] = precision
        tasasdeerror_knn[i] = tasadeerror
        sensibilidad_knn[i] = sensibilidad
        especificidad_knn[i] = especificidad
        VPP_knn[i] = vpp
        VPN_knn[i] = vpn
        F1_knn[i] = f1
    end

    resultados = Dict(
            "precision" => precision_knn,
            "tasadeerror" => tasasdeerror_knn,
            "sensibilidad" => sensibilidad_knn,
            "especificidad" => especificidad_knn,
            "vpp" => VPP_knn,
            "vpn" => VPN_knn,
            "f1" => F1_knn,
            "hiperparametros" => hiperparametros_knn
        )
    save("knn.jld2", "resultados", resultados)
end

# 1. RR.NN.AA
println("RR.NN.AA")

if isfile(joinpath(pwd(), "rna.jld2"))
    resultado = load("rna.jld2", "resultados")
    precision_rna = resultado["precision"]
    tasasdeerror_rna = resultado["tasadeerror"]
    sensibilidad_rna = resultado["sensibilidad"]
    especificidad_rna = resultado["especificidad"]
    VPP_rna = resultado["vpp"]
    VPN_rna = resultado["vpn"]
    F1_rna = resultado["f1"]
    topologias = resultado["topologias"]
else
    arquitecturas = 8
    topologias = []
    precision_rna = Array{Tuple{Float32, Float32}, 1}(undef, arquitecturas)
    tasasdeerror_rna = Array{Tuple{Float32, Float32}, 1}(undef, arquitecturas)
    sensibilidad_rna = Array{Tuple{Float32, Float32}, 1}(undef, arquitecturas)
    especificidad_rna = Array{Tuple{Float32, Float32}, 1}(undef, arquitecturas)
    VPP_rna = Array{Tuple{Float32, Float32}, 1}(undef, arquitecturas)
    VPN_rna = Array{Tuple{Float32, Float32}, 1}(undef, arquitecturas)
    F1_rna = Array{Tuple{Float32, Float32}, 1}(undef, arquitecturas)
    hiperparametros_rna = []

    rango = 1:10
    for i in 1:arquitecturas
        if i in 1:4
            topology = [rand(rango.^2)] # Una única capa oculta
        else
            topology = rand(rango.^2, 2) # Dos capas ocultas
        end
        push!(topologias, topology)

        modelHyperparameters = Dict(
        "topology" => topologias[i], 
        "numExecutions" => 50,  # Número de ejecuciones para cada arquitectura
        "maxEpochs" => 1000,
        "minLoss" => 0.0,
        "learningRate" => 0.01,
        "validationRatio" => 0,
        "maxEpochsVal" => 20
        )
        push!(hiperparametros_rna, modelHyperparameters)

        precision, tasadeerror, sensibilidad, especificidad, vpp, vpn, f1 = modelCrossValidation(:ANN, modelHyperparameters, inputsFinal, targets, crossValidationIndices)

        precision_rna[i] = precision
        tasasdeerror_rna[i] = tasadeerror
        sensibilidad_rna[i] = sensibilidad
        especificidad_rna[i] = especificidad
        VPP_rna[i] = vpp
        VPN_rna[i] = vpn
        F1_rna[i] = f1  
    end
    resultados = Dict(
        "precision" => precision_rna,
        "tasadeerror" => tasasdeerror_rna,
        "sensibilidad" => sensibilidad_rna,
        "especificidad" => especificidad_rna,
        "vpp" => VPP_rna,
        "vpn" => VPN_rna,
        "f1" => F1_rna,
        "topologias" => topologias
    )
    save("rna.jld2", "resultados", resultados)
end

############################################################## RESULTADOS ################################################################################

# 2. SVM
println("SVM")
println("Media de precisiones: ", [x[1] for x in precision_svm])
println("Desviación de precisiones: ", [x[2] for x in precision_svm])
println("Media de tasas de error: ", [x[1] for x in tasasdeerror_svm])
println("Desviación de tasas de error: ", [x[2] for x in tasasdeerror_svm])
println("Media de sensibilidad: ", [x[1] for x in sensibilidad_svm])
println("Desviación de sensibilidad: ", [x[2] for x in sensibilidad_svm])
println("Media de especificidad: ", [x[1] for x in especificidad_svm])
println("Desviación de especificidad: ", [x[2] for x in especificidad_svm])
println("Media de VPP: ", [x[1] for x in VPP_svm])
println("Desviación de VPP: ", [x[2] for x in VPP_svm])
println("Media de VPN: ", [x[1] for x in VPN_svm])
println("Desviación de VPN: ", [x[2] for x in VPN_svm])
println("Media de F1: ", [x[1] for x in F1_svm])
println("Desviación de F1: ", [x[2] for x in F1_svm])


# 3. DecisionTreeClassifier
println("DecisionTreeClassifier")
println("Media de precisiones: ", [x[1] for x in precision_dtc])
println("Desviación de precisiones: ", [x[2] for x in precision_dtc])
println("Media de tasas de error: ", [x[1] for x in tasasdeerror_dtc])
println("Desviación de tasas de error: ", [x[2] for x in tasasdeerror_dtc])
println("Media de sensibilidad: ", [x[1] for x in sensibilidad_dtc])
println("Desviación de sensibilidad: ", [x[2] for x in sensibilidad_dtc])
println("Media de especificidad: ", [x[1] for x in especificidad_dtc])
println("Desviación de especificidad: ", [x[2] for x in especificidad_dtc])
println("Media de VPP: ", [x[1] for x in VPP_dtc])
println("Desviación de VPP: ", [x[2] for x in VPP_dtc])
println("Media de VPN: ", [x[1] for x in VPN_dtc])
println("Desviación de VPN: ", [x[2] for x in VPN_dtc])
println("Media de F1: ", [x[1] for x in F1_dtc])
println("Desviación de F1: ", [x[2] for x in F1_dtc])

# 4. KNN
println("kNN")
println("Media de precisiones: ", [x[1] for x in precision_knn])
println("Desviación de precisiones: ", [x[2] for x in precision_knn])
println("Media de tasas de error: ", [x[1] for x in tasasdeerror_knn])
println("Desviación de tasas de error: ", [x[2] for x in tasasdeerror_knn])
println("Media de sensibilidad: ", [x[1] for x in sensibilidad_knn])
println("Desviación de sensibilidad: ", [x[2] for x in sensibilidad_knn])
println("Media de especificidad: ", [x[1] for x in especificidad_knn])
println("Desviación de especificidad: ", [x[2] for x in especificidad_knn])
println("Media de VPP: ", [x[1] for x in VPP_knn])
println("Desviación de VPP: ", [x[2] for x in VPP_knn])
println("Media de VPN: ", [x[1] for x in VPN_knn])
println("Desviación de VPN: ", [x[2] for x in VPN_knn])
println("Media de F1: ", [x[1] for x in F1_knn])
println("Desviación de F1: ", [x[2] for x in F1_knn])

# 1. RR.NN.AA
println("RR.NN.AA")
println(topologias)
println("Media de precisiones: ", [x[1] for x in precision_rna])
println("Desviación de precisiones: ", [x[2] for x in precision_rna])
println("Media de tasas de error: ", [x[1] for x in tasasdeerror_rna])
println("Desviación de tasas de error: ", [x[2] for x in tasasdeerror_rna])
println("Media de sensibilidad: ", [x[1] for x in sensibilidad_rna])
println("Desviación de sensibilidad: ", [x[2] for x in sensibilidad_rna])
println("Media de especificidad: ", [x[1] for x in especificidad_rna])
println("Desviación de especificidad: ", [x[2] for x in especificidad_rna])
println("Media de VPP: ", [x[1] for x in VPP_rna])
println("Desviación de VPP: ", [x[2] for x in VPP_rna])
println("Media de VPN: ", [x[1] for x in VPN_rna])
println("Desviación de VPN: ", [x[2] for x in VPN_rna])
println("Media de F1: ", [x[1] for x in F1_rna])
println("Desviación de F1: ", [x[2] for x in F1_rna])

# Hacemos un barplot donde vemos como varían las especificides de cada modelo para cada algoritmo
nombres = ["Modelo 1", "Modelo 2", "Modelo 3", "Modelo 4", "Modelo 5", "Modelo 6", "Modelo 7", "Modelo 8"]
especificidades_svm = [x[1] for x in especificidad_svm]
bar(nombres, especificidades_svm, title = "Especificidad media para SVM", xlabel = "Modelo", ylabel = "Especificidad", color = "blue", label = "SVM")

especificidades_dtc = [x[1] for x in especificidad_dtc]
bar(nombres, especificidades_dtc, title = "Especificidad media para DTC", xlabel = "Modelo", ylabel = "Especificidad", color = "green", label = "DTC")

especificidades_knn = [x[1] for x in especificidad_knn]
bar(nombres, especificidades_knn, title = "Especificidad media para KNN", xlabel = "Modelo", ylabel = "Especificidad", color = "red", label = "KNN")

especificidades_rna = [x[1] for x in especificidad_rna]
bar(nombres, especificidades_rna, title = "Especificidad media para RNA", xlabel = "Modelo", ylabel = "Especificidad", color = "yellow", label = "RR.NN.AA")

# Seleccionamos el mejor modelo para cada algoritmo
function seleccionar_mejormodelo(precision, especificidad)
    indices = []
    maximos = []
    for (i, valor) in enumerate(especificidad)
        if isempty(maximos) || valor > maximum(maximos)
            maximos = [valor]
            indices = [i]
        elseif valor == maximum(maximos)
            push!(maximos, valor)
            push!(indices, i)
        end
    end

    if length(maximos) > 1
        indice_maxima_precision = indices[argmax(precision[indices])]
        return indice_maxima_precision
    else 
        return indices[1]
    end
end

precisiones_svm = [x[1] for x in precision_svm]
precisiones_dtc = [x[1] for x in precision_dtc]
precisiones_knn = [x[1] for x in precision_knn]
precisiones_rna = [x[1] for x in precision_rna]

sensibilidades_svm = [x[1] for x in sensibilidad_svm]
sensibilidades_dtc = [x[1] for x in sensibilidad_dtc]
sensibilidades_knn = [x[1] for x in sensibilidad_knn]
sensibilidades_rna = [x[1] for x in sensibilidad_rna]

# 2. SVM
mejor_indice_svm = seleccionar_mejormodelo(precisiones_svm, especificidades_svm)

# 3. DecisionTreeClassifier
mejor_indice_dtc = seleccionar_mejormodelo(precisiones_dtc, especificidades_dtc)

# 4. KNN 
mejor_indice_knn = seleccionar_mejormodelo(precisiones_knn, especificidades_knn)

# 1. RR.NN.AA
mejor_indice_rna = seleccionar_mejormodelo(precisiones_rna, especificidades_rna)

# Juntamos los mejores resultados de cada algoritmo en un mismo gráfico
nombres = ["Modelo $mejor_indice_rna ANN", "Modelo $mejor_indice_svm SVM", "Modelo $mejor_indice_dtc DTC", "Modelo $mejor_indice_knn KNN"]
colores = [:yellow, :blue, :green, :red]
mejores_especificidades = [especificidades_rna[mejor_indice_rna], especificidades_svm[mejor_indice_svm], especificidades_dtc[mejor_indice_dtc], especificidades_knn[mejor_indice_knn]]
bar(nombres, mejores_especificidades, title = "Especificidades del mejor resultado", xlabel = "Modelo", ylabel = "Especificidad", color = colores)

# Mostramos un diagrama de cajas para las métricas de los 4 algoritmos
# Para eso modificamos la función modelCrossValidation para que devuelva todas las métricas sin hacer la media ni la desviación típica
# La modificamos en el archivo firmas_soluciones.jl
# Y cogemos el modelo que mejor nos ha funcionado para cada algoritmo

# 2. SVM
println("SVM")
mejor_precision_svm, _, mejor_especificidad_svm, _, _, _ = modelCrossValidationModificada(:SVC, hiperparametros_svm[mejor_indice_svm], inputsFinal, targets, crossValidationIndices)

# 3. DecisionTreeClassifier
println("DecisionTreeClassifier")
mejor_precision_dtc, _, mejor_especificidad_dtc, _, _, _ = modelCrossValidationModificada(:DecisionTreeClassifier, hiperparametros_dtc[mejor_indice_dtc], inputsFinal, targets, crossValidationIndices)

# 4. KNN  
println("kNN")
mejor_precision_knn, _, mejor_especificidad_knn, _, _, _ = modelCrossValidationModificada(:KNeighborsClassifier, hiperparametros_knn[mejor_indice_knn], inputsFinal, targets, crossValidationIndices)

# 1. RR.NN.AA
println("RR.NN.AA")
mejor_precision_rna, _, mejor_especificidad_rna, _, _, _ = modelCrossValidationModificada(:ANN, hiperparametros_rna[mejor_indice_rna], inputsFinal, targets, crossValidationIndices)

boxplot(["SVM", "DTC", "KNN"], [mejor_especificidad_svm, mejor_especificidad_dtc, mejor_especificidad_knn], title = "Especificidad del mejor modelo de los 4 algoritmos", xlabel = "Algoritmo", ylabel = "Especificidad")


# Seleccionamos el mejor algoritmo de los 4
#=
function seleccionar_mejoralgoritmo(precisiones, especificidades, indices, orden, umbral_precision, umbral_especificidad)
    puntuaciones = [umbral_precision * precision + umbral_especificidad * especificidad for (precision, especificidad) in zip(precisiones, especificidades)]
    indice_mejor_algoritmo = argmax(maximum(puntuaciones))
    return orden[indice_mejor_algoritmo], indices[indice_mejor_algoritmo]
end

mejores_precisiones = [precision_rna[mejor_indice_rna], precisiones_svm[mejor_indice_svm], precisiones_dtc[mejor_indice_dtc], precisiones_knn[mejor_indice_knn]]
mejores_indices_modelo = [mejor_indice_rna, mejor_indice_svm, mejor_indice_dtc, mejor_indice_knn]
orden = ["SVM", "DTC", "KNN", "RNA"]
umbral_precision = umbral_especificidad = 0.5

seleccionar_mejoralgoritmo(mejores_precisiones, mejores_especificidades, mejores_indices_modelo, orden, umbral_precision, umbral_especificidad)
=#
