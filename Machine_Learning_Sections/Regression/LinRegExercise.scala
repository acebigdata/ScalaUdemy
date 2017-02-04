////////////////////////////////////////////
//// LINEAR REGRESSION EXERCISE ///////////
/// Complete the commented tasks below ///
/////////////////////////////////////////

// Import LinearRegression
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}

// Optional: Use the following code below to set the Error reporting
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)


// Start a simple Spark Session
import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder().getOrCreate()
// Use Spark to read in the Ecommerce Customers csv file.
val data = spark.read.option("header","true").option("inferSchema","true").format("csv").load("Ecommerc Customers")

// Print the Schema of the DataFrame
data.printSchema()

// Print out an example Row
// Various ways to do this, just
// choose whichever way you prefer
val colnames = data.columns
val first_row = data.head(1)(0)
for (col_name <- Range(1,colnames.length)){
  println(colnames(col_name))
  println(first_row(col_name))
  println("\n")
}

////////////////////////////////////////////////////
//// Setting Up DataFrame for Machine Learning ////
//////////////////////////////////////////////////

// A few things we need to do before Spark can accept the data!
// It needs to be in the form of two columns
// ("label","features")

// Import VectorAssembler and Vectors
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors

// Rename the Yearly Amount Spent Column as "label"
// Also grab only the numerical columns from the data
// Set all of this as a new dataframe called df
val df = data.select(data("Price").as("label"),$"Avg Area Income",$"Avg Area House Age",$"Avg Area Number of Rooms",$"Area Population")

// An assembler converts the input values to a vector
// A vector is what the ML algorithm reads to train a model

// Use VectorAssembler to convert the input columns of df
// to a single output column of an array called "features"
// Set the input columns from which we are supposed to read the values.
// Call this new object assembler
val assembler = new VectorAssembler().setInputCols(Array("Avg Area Income","Avg Area House Age","Avg Area Number of Rooms","Area Population")).setOutputCol("features")

// Use the assembler to transform our DataFrame to the two columns: label and features
val output = assembler.transform(df).select($"label",$"features")

// Create a Linear Regression Model object
val lr = new LinearRegression()

// Fit the model to the data and call this model lrModel
val lrModel = lr.fit(output)

// Print the coefficients and intercept for linear regression
println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

// Summarize the model over the training set and print out some metrics!
// Use the .summary method off your model to create an object
// called trainingSummary
val trainingSummary = lrModel.summary
println(s"numIterations: ${trainingSummary.totalIterations}")
println(s"objectiveHistory: ${trainingSummary.objectiveHistory.toList}")

trainingSummary.residuals.show()

// Show the residuals, the RMSE, the MSE, and the R^2 Values.
println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
println(s"MSE: ${trainingSummary.meanSquaredError}")
println(s"r2: ${trainingSummary.r2}")


// Great Job!
