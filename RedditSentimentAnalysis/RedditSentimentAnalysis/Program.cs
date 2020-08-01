using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using static Microsoft.ML.DataOperationsCatalog;

namespace RedditSentimentAnalysis
{
    class Program
    {
        static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "sneakers_labelled.txt");

        static void Main(string[] args)
        {

            RedditScraper rs = new RedditScraper();
            List<string> commentList = rs.ScrapeCommentsSinglePost("/r/sneakers");


            MLContext mlContext = new MLContext();

            TrainTestData splitDataView = LoadData(mlContext);

            ITransformer model = BuildAndTrainModel(mlContext, splitDataView.TrainSet);

            Evaluate(mlContext, model, splitDataView.TestSet);

            UseModelWithSingleItem(mlContext, model, commentList);

            UseModelWithBatchItems(mlContext, model, commentList);
        }

        public static TrainTestData LoadData(MLContext mlContext)
        {
            IDataView dataView = mlContext.Data.LoadFromTextFile<SentimentData>(_dataPath, hasHeader: false);
            TrainTestData splitDataView = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
            return splitDataView;
        }

        public static ITransformer BuildAndTrainModel(MLContext mlContext, IDataView splitTrainSet)
        {
            var estimator = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(SentimentData.SentimentText))
                .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));

            Console.WriteLine("=============== Create and Train the Model ===============");
            var model = estimator.Fit(splitTrainSet);
            Console.WriteLine("=============== End of training ===============");
            Console.WriteLine();

            return model;
        }

        public static void Evaluate(MLContext mlContext, ITransformer model, IDataView splitTestSet)
        {
            Console.WriteLine("=============== Evaluating Model accuracy with Test data===============");
            IDataView predictions = model.Transform(splitTestSet);

            //evaluates the model
            CalibratedBinaryClassificationMetrics metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");

            //Displays the metrics 
            Console.WriteLine();
            Console.WriteLine("Model quality metrics evaluation");
            Console.WriteLine("--------------------------------");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");      // proportion of correct preditctions in the test set
            Console.WriteLine($"Auc: {metrics.AreaUnderRocCurve:P2}");  // how confident the model is correctly flassifying the positivie and negative classes
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");        // measure of balance between precision and recall
            Console.WriteLine("=============== End of model evaluation ===============");
        }

        /// <summary>
        ///  -- Creates a single comment of test data.
        ///  -- Predicts sentiment based on test data.
        ///  -- Displays the predicted results.
        /// </summary>
        /// <param name="mlContext"></param>
        /// <param name="model"></param>
        private static void UseModelWithSingleItem(MLContext mlContext, ITransformer model, List<string> commentList)
        {
            PredictionEngine<SentimentData, SentimentPrediction> predictionFunction = mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);

            SentimentData sampleStatement = new SentimentData
            {
                SentimentText = commentList[0]
            };

            var resultPrediction = predictionFunction.Predict(sampleStatement);

            Console.WriteLine();
            Console.WriteLine("=============== Prediction Test of model with a single sample and test dataset ===============");

            Console.WriteLine();
            Console.WriteLine($"Sentiment: {resultPrediction.SentimentText} | Prediction: {(Convert.ToBoolean(resultPrediction.Prediction) ? "Positive" : "Negative")} | Probability: {resultPrediction.Probability} ");

            Console.WriteLine("=============== End of Predictions ===============");
            Console.WriteLine();
        }

        /// <summary>
        ///  -- Predicts sentiment based on test data.
        ///  -- Combines test data and predictions for reporting.
        ///  -- Displays the predicted results.
        /// </summary>
        /// <param name="mlContext"></param>
        /// <param name="model"></param>
        public static void UseModelWithBatchItems(MLContext mlContext, ITransformer model, List<string> commentList)
        {
            List<SentimentData> listConversion = new List<SentimentData>();

            foreach (var comment in commentList)
            {
                listConversion.Add(new SentimentData { SentimentText = comment });
            }

            IEnumerable<SentimentData> sentiments = listConversion;

            IDataView batchComments = mlContext.Data.LoadFromEnumerable(sentiments);

            IDataView predictions = model.Transform(batchComments);

            // Use model to predict whether comment data is Positive (1) or Negative (0).
            IEnumerable<SentimentPrediction> predictedResults = mlContext.Data.CreateEnumerable<SentimentPrediction>(predictions, reuseRowObject: false);

            //Console.WriteLine();
            Console.WriteLine("\n=============== Prediction Test of loaded model with multiple samples ===============");

            foreach (SentimentPrediction prediction in predictedResults)
            {
                Console.WriteLine($"Sentiment: {prediction.SentimentText} | Prediction: {(Convert.ToBoolean(prediction.Prediction) ? "Positive" : "Negative")} | Probability: {prediction.Probability} ");
                Console.WriteLine("\n\n");
            }
            Console.WriteLine("=============== End of predictions ===============");

        }
    }
}
