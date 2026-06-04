# Student Dropout Prediction con Spark

Este proyecto analiza la desercion estudiantil durante el primer anio usando Apache Spark. El objetivo es procesar el archivo `tbl_desercion_estudiantil_primer_anio.csv`, preparar los datos y ejecutar un flujo de trabajo distribuido para explorar el comportamiento de abandono academico.

El dataset contiene 7,517 registros, ademas de una fila de encabezado.

El notebook `desercion_estudiantil_spark.ipynb` esta pensado como version de prueba para Kaggle. El codigo principal para ejecucion en Spark esta en `desercion_estudiantil_spark.py`.

## Ejecucion en la imagen Docker de la U

Para correr el proyecto, basta con mover el archivo CSV y el archivo Python a la imagen Docker que contiene Hadoop y Spark proporcionada por la universidad. Luego, copia el CSV al HDFS dentro de `input` y envia el script a Spark con estos comandos:

```bash
hadoop fs -put tbl_desercion_estudiantil_primer_anio.csv /user/root/input/

spark-submit /root/desercion_estudiantil_spark.py hdfs:///user/root/input/tbl_desercion_estudiantil_primer_anio.csv
```

## Archivos principales

- `tbl_desercion_estudiantil_primer_anio.csv`: dataset de entrada.
- `desercion_estudiantil_spark.py`: script para ejecutar el procesamiento en Spark.
- `desercion_estudiantil_spark.ipynb`: notebook de pruebas para Kaggle.