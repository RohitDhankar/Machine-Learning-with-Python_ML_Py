
# See the Terminal Prints Log Files ---> results_terminal_prints_1.log 
# DATA SOURCE -- https://archive.ics.uci.edu/dataset/352/online+retail
# conda activate dbfs_env

import findspark
findspark.init()
import pyspark
import random
from pyspark.sql import SparkSession

# from datetime import datetime, date
# import pandas as pd
# from pyspark.sql import Row
from pyspark.sql import SparkSession
#import pyspark.pandas as ps #TODO -- Upgrade Spark to 3.2
import pandas as pd
from pyspark.sql.functions import *
from pyspark.sql.types import *
from memory_profiler import profile

## TEMP -- Dont Repeat 
# xlsx_path = "./data_dir/online_retail.xlsx"
# df_xlsx = pd.read_excel(xlsx_path, index_col=0) 
# print(df_xlsx.info(verbose=True))
# df_xlsx.to_csv("./data_dir/retail_data.csv")

# spark = SparkSession.builder.getOrCreate()
# See the Terminal Prints Log Files ---> results_terminal_prints_1.log 

def read_data():
  """
  """
  spark = SparkSession.builder.appName("app_retail_analysis_1").config("spark.memory.offHeap.enabled","true").config("spark.memory.offHeap.size","5g").getOrCreate()
  print("--type--spark--",spark)
  #pyspark.pandas.read_excel(   #df_xlsx = ps.read_excel(xlsx_path)
  df_rtl = spark.read.csv("./data_dir/retail_data.csv",header=True,escape="\"")
  #print("--type--df_xlsx-",type(df_rtl)) # <class 'pyspark.sql.dataframe.DataFrame'>
  df_rtl.show(5,0)
  print("-[INFO]-df_rtl.count()-",df_rtl.count()) # 541909
  return df_rtl , spark


def retail_analysis(df_rtl , spark):
  """
  https://www.databricks.com/blog/2021/10/04/pandas-api-on-upcoming-apache-spark-3-2.html
  """
  print("-[INFO]-df_rtl--CustomerID').distinct().count(--->",df_rtl.select('CustomerID').distinct().count()) # 4373
  #
  print("  "*90)
  df_rtl.groupBy('Country').agg(countDistinct('CustomerID').alias('country_count')).orderBy(desc('country_count')).show()
  print("  "*90)
  df_1 = df_rtl.groupBy('Country').agg(countDistinct('CustomerID').alias('country_count')).orderBy(desc('country_count'))
  print("----df_1--->/n",df_1.show(30,0))
  print("  "*90)
  spark.sql("set spark.sql.legacy.timeParserPolicy=LEGACY")
  #df_2 = df_rtl.withColumn('date',to_timestamp("InvoiceDate", 'yy/MM/dd HH:mm'))
  df_2 = df_rtl.withColumn('date',to_timestamp("InvoiceDate", 'yyyy-MM-dd HH:mm:ss')) #
  #'yyyy-MM-dd HH:mm:ss' --- https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.functions.to_timestamp.html
  print("----df_2--->/n",df_2.show(10,0))
  print("  "*90)
  df_2.select(max("date")).show()
  print("  "*30)
  df_2.select(min("date")).show()
  #
  #df_2 = df_2.withColumn('
  df_2.select(lit(5).alias('height')).withColumn('spark_user', lit(True)).take(1)
  # Add LITERAL VAlue Columns -- Non Calculated - HardCoded Values 
  # https://spark.apache.org/docs/3.1.3/api/python/reference/api/pyspark.sql.functions.lit.html
  print("----df_2--->/n",df_2.show(10,0))

@profile
def run_analysis():
  """
  """
  df_rtl , spark = read_data()
  retail_analysis(df_rtl , spark)

if __name__ == "__main__":
  run_analysis()
