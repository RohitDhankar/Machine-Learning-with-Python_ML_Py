
# See the Terminal Prints Log Files ---> results_terminal_prints_1.log 
## DATA SOURCE -- https://archive.ics.uci.edu/dataset/352/online+retail


##https://spark.apache.org/docs/latest/api/python/getting_started/index.html
#<pyspark.sql.session.SparkSession object at 0x7fccd4a86eb0>

# Changed in -- ~/.bashrc 
# /usr/lib/jvm/java-8-openjdk-amd64/jre/
## conda activate dbfs_env

import findspark
findspark.init()
import pyspark
import random

# # sc = pyspark.SparkContext(appName="Pi")
# # num_samples = 10 #100000000

# # def inside(p):
# #     #
# #     x, y = random.random(), random.random()
# #     return x*x + y*y < 1

# # count = sc.parallelize(range(0, num_samples)).filter(inside).count()

# # pi = 4 * count / num_samples
# # print(pi)


# # sc = pyspark.SparkContext(appName="Daily_Show_Test1")
 
# # print(sc) #

# # #sc.stop()


from pyspark.sql import SparkSession

# spark = SparkSession.builder.getOrCreate()
# print(spark) #<pyspark.sql.session.SparkSession object at 0x7f8185824e80>

# from datetime import datetime, date
# import pandas as pd
# from pyspark.sql import Row

# df = spark.createDataFrame([
#     Row(a=1, b=2., c='string1', d=date(2000, 1, 1), e=datetime(2000, 1, 1, 12, 0)),
#     Row(a=2, b=3., c='string2', d=date(2000, 2, 1), e=datetime(2000, 1, 2, 12, 0)),
#     Row(a=4, b=5., c='string3', d=date(2000, 3, 1), e=datetime(2000, 1, 3, 12, 0))
# ])
# print("    "*90)
# print(df) ##DataFrame[a: bigint, b: double, c: string, d: date, e: timestamp]
# print("    "*90)
# print(df.head(3))
# """
# [Row(a=1, b=2.0, c='string1', d=datetime.date(2000, 1, 1), e=datetime.datetime(2000, 1, 1, 12, 0)), Row(a=2, b=3.0, c='string2', d=datetime.date(2000, 2, 1), e=datetime.datetime(2000, 1, 2, 12, 0)), Row(a=4, b=5.0, c='string3', d=datetime.date(2000, 3, 1), e=datetime.datetime(2000, 1, 3, 12, 0))]
# """
# print("    "*90)
# print(type(df))
# print("    "*90)
# df.show()
# df.printSchema()
# print("    "*90)
# #
# pd_df = df.toPandas()
# print(pd_df.info(verbose=True))
# print("    "*90)
# #https://stackoverflow.com/questions/32977360/pyspark-changing-type-of-column-from-date-to-string
# #
# # SHOW Summary 
# df.select("a", "b", "c").describe().show()
# print("    "*90)
# #
# df.take(2)
# #
# df.tail(2)
# print("    "*90)
# #
## DATA DIR -- online_retail

from pyspark.sql import SparkSession
#import pyspark.pandas as ps #TODO -- Upgrade Spark to 3.2
import pandas as pd
from pyspark.sql.functions import *
from pyspark.sql.types import *

## TEMP -- Dont Repeat 
# xlsx_path = "./data_dir/online_retail.xlsx"
# df_xlsx = pd.read_excel(xlsx_path, index_col=0) 
# print(df_xlsx.info(verbose=True))
# df_xlsx.to_csv("./data_dir/retail_data.csv")

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
  return df_rtl


def retail_analysis(df_rtl):
  """
  https://www.databricks.com/blog/2021/10/04/pandas-api-on-upcoming-apache-spark-3-2.html
  """
  print("-[INFO]-df_rtl--CustomerID').distinct().count(--->",df_rtl.select('CustomerID').distinct().count()) # 4373
  df_rtl.groupBy('Country').agg(countDistinct('CustomerID').alias('country_count')).show()



if __name__ == "__main__":
    df_rtl = read_data()
    retail_analysis(df_rtl)
    


