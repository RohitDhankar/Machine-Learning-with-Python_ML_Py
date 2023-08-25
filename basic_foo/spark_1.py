


##https://spark.apache.org/docs/latest/api/python/getting_started/index.html
#<pyspark.sql.session.SparkSession object at 0x7fccd4a86eb0>

# Changed in -- ~/.bashrc 
# /usr/lib/jvm/java-8-openjdk-amd64/jre/
## conda activate dbfs_env

import findspark
findspark.init()
import pyspark
import random

# sc = pyspark.SparkContext(appName="Pi")
# num_samples = 10 #100000000

# def inside(p):
#     #
#     x, y = random.random(), random.random()
#     return x*x + y*y < 1

# count = sc.parallelize(range(0, num_samples)).filter(inside).count()

# pi = 4 * count / num_samples
# print(pi)


# sc = pyspark.SparkContext(appName="Daily_Show_Test1")
 
# print(sc) #

# #sc.stop()


from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
print(spark) #<pyspark.sql.session.SparkSession object at 0x7f8185824e80>

from datetime import datetime, date
import pandas as pd
from pyspark.sql import Row

df = spark.createDataFrame([
    Row(a=1, b=2., c='string1', d=date(2000, 1, 1), e=datetime(2000, 1, 1, 12, 0)),
    Row(a=2, b=3., c='string2', d=date(2000, 2, 1), e=datetime(2000, 1, 2, 12, 0)),
    Row(a=4, b=5., c='string3', d=date(2000, 3, 1), e=datetime(2000, 1, 3, 12, 0))
])
print("    "*90)
print(df) ##DataFrame[a: bigint, b: double, c: string, d: date, e: timestamp]
print("    "*90)
print(df.head(3))
"""
[Row(a=1, b=2.0, c='string1', d=datetime.date(2000, 1, 1), e=datetime.datetime(2000, 1, 1, 12, 0)), Row(a=2, b=3.0, c='string2', d=datetime.date(2000, 2, 1), e=datetime.datetime(2000, 1, 2, 12, 0)), Row(a=4, b=5.0, c='string3', d=datetime.date(2000, 3, 1), e=datetime.datetime(2000, 1, 3, 12, 0))]
"""
print("    "*90)
print(type(df))
print("    "*90)
df.show()
df.printSchema()
print("    "*90)
#
pd_df = df.toPandas()
print(pd_df.info(verbose=True))
print("    "*90)
#https://stackoverflow.com/questions/32977360/pyspark-changing-type-of-column-from-date-to-string
#
# SHOW Summary 
df.select("a", "b", "c").describe().show()
print("    "*90)
#
df.take(2)
#
df.tail(2)
print("    "*90)
#







"""
ase) dhankar@dhankar-1:/$ 
(base) dhankar@dhankar-1:/$ update-alternatives --config java
There are 2 choices for the alternative java (providing /usr/bin/java).

  Selection    Path                                            Priority   Status
------------------------------------------------------------
  0            /usr/lib/jvm/java-11-openjdk-amd64/bin/java      1111      auto mode
* 1            /usr/lib/jvm/java-11-openjdk-amd64/bin/java      1111      manual mode
  2            /usr/lib/jvm/java-8-openjdk-amd64/jre/bin/java   1081      manual mode

Press <enter> to keep the current choice[*], or type selection number: 
(base) dhankar@dhankar-1:/$ 

"""
"""
/opt/spark/bin/spark-class: line 71: /usr/lib/jvm/jdk1.8.0_291//bin/java: cannot execute binary file: Exec format error
/opt/spark/bin/spark-class: line 96: CMD: bad array subscript
Traceback (most recent call last):
  File "spark_1.py", line 6, in <module>
    sc = pyspark.SparkContext(appName="Pi")
  File "/opt/spark/python/pyspark/context.py", line 144, in __init__
    SparkContext._ensure_initialized(self, gateway=gateway, conf=conf)
  File "/opt/spark/python/pyspark/context.py", line 331, in _ensure_initialized
    SparkContext._gateway = gateway or launch_gateway(conf)
  File "/opt/spark/python/pyspark/java_gateway.py", line 108, in launch_gateway
    raise Exception("Java gateway process exited before sending its port number")
Exception: Java gateway process exited before sending its port number
"""


# /etc/environment

# /etc/environment

# export JAVA_HOME=/usr/lib/jvm/jdk1.8.0_291/


# ## JAVA_PATH -- for SCALA - IntelliJIDEA Projects -- JUL21
# export JAVA_HOME=/usr/lib/jvm/jdk1.8.0_291
# export PATH=$PATH:$JAVA_HOME/bin
# export PATH=$JAVA_HOME/bin:$PATH

"""
## SO -- https://stackoverflow.com/questions/37190934/spark-cluster-master-ip-address-not-binding-to-floating-ip

23/08/24 19:53:57 WARN Utils: Your hostname, dhankar-1 resolves to a loopback address: 127.0.1.1; using 192.168.1.2 instead (on interface enp2s0)
23/08/24 19:53:57 WARN Utils: Set SPARK_LOCAL_IP if you need to bind to another address
23/08/24 19:53:57 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
Using Spark's default log4j profile: org/apache/spark/log4j-defaults.properties
Setting default log level to "WARN".
To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).
<SparkContext master=local[*] appName=Daily_Show_Test1>
"""