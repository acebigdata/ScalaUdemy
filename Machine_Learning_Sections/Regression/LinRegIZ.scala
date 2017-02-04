//import Linear Regression
import org.apache.spark.ml.regression.LinearRegression

//Set the error reporting
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

//Start a simple Spark Session
import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder().getOrCreate()

//use spark to read in the Ecommerce Customers csv file
val data = (spark.read.option("header","true")
            .option("inferSchema","true")
            .format("csv")
            .load("Ecommerce Customers"))

data.printSchema()

//print out an example row
// val col_names = data.columns
// val first_row = data.head(1)(0)
//
// for (counter_i <- Range(1,col_names.length)){
//   println(col_names(counter_i))
//   println(first_row(counter_i))
//   println("\n")
// }

//Setting up Dataframe for ML
//Need to be in the form of two columns
//("label","features")

//Import VectorAssembler and Vectors
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors

val df = (data.select(data("Yearly Amount Spent").as("label"),
          $"Avg Session Length",
          $"Time on App",
          $"Time on Website",
          $"Length of Membership"))

//An assembler converts the input values to a vector
val assembler = new VectorAssembler().setInputCols(Array("Avg Session Length", "Time on App", "Time on Website", "Length of Membership")).setOutputCol("features")

//Use the assember to transform Dataframe to two columns: label and features
val output = assembler.transform(df).select($"label", $"features")

//create a linear regression model
val lr = new LinearRegression()

//fit the model to the data and call this model lrModel
val lrModel = lr.fit(output)

//print the coefficients and intercept for linear regression
println(s"Coff: ${lrModel.coefficients}, Intercept: ${lrModel.intercept}")

//Summarize the model
val trainingSummary = lrModel.summary

//Show the residuals, the RMSE, the MSE, and the R-square values
trainingSummary.residuals.show()
println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
println(s"MSE: ${trainingSummary.meanSquaredError}")
println(s"R2: ${trainingSummary.r2}")
