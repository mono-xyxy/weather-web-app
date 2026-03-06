from pyspark.sql import SparkSession
from pyspark.sql.functions import col, rand, randn, current_timestamp, expr, round as spark_round
import sys

def main(scale="million"):
    # 1. Initialize Spark Session
    # Setting master to 'yarn' allows it to run across the cluster nodes.
    # enableHiveSupport() connects Spark to the Hive metastore for SQL queries.
    print(f"Initializing Spark Session for scale: {scale}...")
    spark = SparkSession.builder \
        .appName("SyntheticWeatherDataGenerator") \
        .config("spark.sql.catalogImplementation", "hive") \
        .config("spark.sql.parquet.compression.codec", "snappy") \
        .enableHiveSupport() \
        .getOrCreate()

    # Determine number of records based on scale
    if scale == "million":
        num_records = 500_000 # Reduced for local test
        partitions = 10
    elif scale == "half-billion":
        num_records = 500_000_000 # 10 nodes setup
        partitions = 2000
    elif scale == "billion":
        num_records = 2_000_000_000 # 50 nodes setup
        partitions = 10000
    else:
        num_records = 1_000_000
        partitions = 20

    print(f"Generating {num_records} records with {partitions} partitions...")

    # 2. Synthetic Generator (Distributed)
    # Using spark.range creates a distributed DataFrame automatically spread across the cluster.
    # This avoids bottlenecks of generating data on a single Node.
    df = spark.range(0, num_records, 1, numPartitions=partitions)

    # 3. Feature Engineering & Realistic Data Generation
    # Using Spark SQL functions to generate realistic data in parallel.
    # We simulate data across 10,000 different weather stations over time.
    generated_df = df.withColumn("weather_station_id", (rand() * 10000).cast("int")) \
        .withColumn("timestamp", current_timestamp() - expr("INTERVAL 1 MINUTE") * col("id")) \
        .withColumn("temp_celsius", spark_round(randn() * 12 + 25, 2)) \
        .withColumn("humidity_percent", spark_round(rand() * 60 + 40, 2)) \
        .withColumn("wind_speed_kmh", spark_round(rand() * 30 + 5, 2)) \
        .withColumn("precipitation_mm", spark_round(expr("CASE WHEN rand() > 0.8 THEN rand() * 20 ELSE 0 END"), 2)) \
        .withColumn("pressure_hpa", spark_round(randn() * 10 + 1013, 2))

    # Calculate boolean conditions for AI Dataset target labels
    generated_df = generated_df.withColumn("is_extreme_weather", 
        expr("temp_celsius > 40 OR temp_celsius < -10 OR wind_speed_kmh > 100 OR precipitation_mm > 50")
    )

    # 4. Save locally using Pandas (bypasses Hadoop winutils requirement on Windows)
    local_csv_path = f"synthetic_weather_output_{scale}.csv"
    print(f"Converting to Pandas and writing to local CSV: {local_csv_path}...")
    
    # Convert distributed dataframe to local pandas dataframe for easy saving without Hadoop
    pd_df = generated_df.toPandas()
    pd_df.to_csv(local_csv_path, index=False)
    
    print(f"Successfully generated and saved {len(pd_df)} records!")
    
    # 5. Prepare AI Training Dataset example
    print("\nSample Data for AI Training:")
    print(pd_df.head(5).to_string())

    spark.stop()

if __name__ == "__main__":
    scale_arg = sys.argv[1] if len(sys.argv) > 1 else "million"
    main(scale_arg)
