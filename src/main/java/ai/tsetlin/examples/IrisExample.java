package ai.tsetlin.examples;

import ai.graphics.AccuracyPlotCanvas;
import ai.record.AnyRecord;
import ai.record.CSVInterface;
import ai.record.CategoryLabel;
import ai.record.Evolutionize;
import ai.tsetlin.GraphAttentionLearning;
import ai.tsetlin.GraphEncoder;
import javafx.application.Application;
import javafx.scene.Scene;
import javafx.stage.Stage;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Random;
import java.util.logging.Logger;

public class IrisExample extends Application {

    //Logger
    Logger logger = Logger.getLogger(IrisExample.class.getName());


    private ArrayList<AnyRecord> records;
    //create two sets of random records
    ArrayList<AnyRecord> train_set = new ArrayList<AnyRecord>();
    ArrayList<AnyRecord> test_set = new ArrayList<AnyRecord>();
    CategoryLabel label;
    int[][] Xi;
    int[] Y;

    Evolutionize evolution = new Evolutionize(1, 1);
    private double[] accuracy;
    private AccuracyPlotCanvas accuracyPlotCanvas = new AccuracyPlotCanvas();

    public void pullData() throws IOException, IllegalAccessException {


        CSVInterface csv = new CSVInterface("data/iris.csv",4);
        AnyRecord anyrecord = csv.createRecord();

        String[] fields = anyrecord.getField_names();

        for(int i = 0; i < fields.length; i++) {
            System.out.println(fields[i]);
        }



        HashMap<String, Integer> categoryMap = new HashMap<String, Integer>();
        categoryMap.put("setosa", 0);
        categoryMap.put("versicolor", 1);
        categoryMap.put("virginica", 2);

        label = new CategoryLabel(categoryMap);
        records = csv.getAllRecords();

        evolution.initiate(anyrecord, 10);
        for(int i = 0; i < records.size(); i++) {
            evolution.addValue(records.get(i));
        }
        evolution.fit();
        evolution.initiateConvolutionEncoder();


        int train_set_size = (int) (records.size() * .7);
        long seed = 12345L;
        Random random = new Random(seed);
        //shuffle the records
        Collections.shuffle(records, random);



        for(int i = 0; i < train_set_size; i++) {
            AnyRecord record = records.get(i);
            train_set.add(record);
        }
        for(int i = train_set_size; i < records.size(); i++) {
            test_set.add(records.get(i));
        }

        Xi = new int[train_set.size()][];
        Y = new int[train_set.size()];
        //create samples from train set
        for(int i = 0; i < train_set.size(); i++) {
            AnyRecord r = train_set.get(i);
            evolution.add(r);
            Xi[i] = evolution.get_last_sample();
            Y[i] = (int)label.getLabel(r.getLabel_name());
        }


    }




    public void compute(int epochs) throws IOException, IllegalAccessException {

        GraphEncoder encoder = new GraphEncoder(evolution.getEncoderDimension());
        int nClauses = 3;
        int nClasses = 3;

        int max_specificity = 30;
        boolean boost = false;
        int LF = (int) (.80* encoder.getFeatureDimension());

        int threshold = (int) Math.sqrt((nClauses / 2f) * LF);
        //int threshold = nClauses * LF;

        int max_literals = LF;

        //log the parameters

        GraphAttentionLearning model = new GraphAttentionLearning(encoder, nClauses, nClasses,threshold, max_specificity,  boost, LF, max_literals);

        int epocs = epochs;
        accuracy = new double[epocs];

        for(int e = 0; e < epocs; e++) {

            model.fit(Xi, Y, true);

            int correct = 0;
            for(AnyRecord record : test_set) {

                int mylabel = (int)label.getLabel(record.getLabel_name());
                evolution.add(record);
                int[] xi = evolution.get_last_sample();

                int predicted = model.predict(xi);

                //System.out.println("Predicted: " + predicted + " Actual: " + mylabel);

                if(predicted == mylabel) {
                    correct++;
                }
            }
            logger.info("Epoch: " + e + " Accuracy: " + (correct / (float)test_set.size()));
            accuracy[e] = (correct / (float)test_set.size());
        }
    }



    public void compute(int epochs, int nClauses, int S, float fuzzyiness, int thresholdMult, boolean boostme) throws IOException, IllegalAccessException {


        GraphEncoder encoder = new GraphEncoder(evolution.getEncoderDimension());

        int nClasses = 3;

        boolean boost = boostme;
        int LF = (int) (fuzzyiness* encoder.getFeatureDimension());

        int threshold = (int) Math.sqrt((nClauses / 2f) * LF) * thresholdMult;
        //int threshold = nClauses * LF;

        int max_literals = LF;


        GraphAttentionLearning model = new GraphAttentionLearning(encoder, nClauses, nClasses,threshold, S,  boost, LF, max_literals);

        int epocs = epochs;
        accuracy = new double[epocs];

        for(int e = 0; e < epocs; e++) {

            model.fit(Xi, Y, true);

            int correct = 0;
            for(AnyRecord record : test_set) {

                int mylabel = (int)label.getLabel(record.getLabel_name());
                evolution.add(record);
                int[] xi = evolution.get_last_sample();

                int predicted = model.predict(xi);

                //System.out.println("Predicted: " + predicted + " Actual: " + mylabel);

                if(predicted == mylabel) {
                    correct++;
                }
            }
            logger.info("Epoch: " + e + " Accuracy: " + (correct / (float)test_set.size()));
            accuracy[e] = (correct / (float)test_set.size());
        }

    }



    @Override
    public void start(Stage stage) throws Exception {

        int epochs = 500;


        pullData();
        compute(epochs, 3, 30, .8f, 1, true);

        accuracyPlotCanvas.plotTimeSeriesData(accuracy);


        Scene scene = new Scene(accuracyPlotCanvas.getvBox());
        scene.getStylesheets().add(getClass().getClassLoader().getResource("css/Chart.css").toExternalForm());

        stage.setScene(scene);
        stage.show();

    }

    public static void main(String[] args) {
        launch(args);
    }
}


