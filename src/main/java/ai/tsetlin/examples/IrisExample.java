package ai.tsetlin.examples;

import ai.record.AnyRecord;
import ai.record.CSVInterface;
import ai.record.CategoryLabel;
import ai.record.Evolutionize;
import ai.tsetlin.GraphAttentionLearning;
import ai.tsetlin.GraphEncoder;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Random;
import java.util.logging.Logger;

public class IrisExample {

    public IrisExample() throws IOException, IllegalAccessException {

        //Logger
        Logger logger = Logger.getLogger(IrisExample.class.getName());


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

        CategoryLabel label = new CategoryLabel(categoryMap);

        ArrayList<AnyRecord> records = csv.getAllRecords();

        Evolutionize evolution = new Evolutionize(1, 1);
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

        //create two sets of random records
        ArrayList<AnyRecord> train_set = new ArrayList<AnyRecord>();
        ArrayList<AnyRecord> test_set = new ArrayList<AnyRecord>();

        for(int i = 0; i < train_set_size; i++) {
            AnyRecord record = records.get(i);
            train_set.add(record);
        }
        for(int i = train_set_size; i < records.size(); i++) {
            test_set.add(records.get(i));
        }

        int[][] Xi = new int[train_set.size()][];
        int[] Y = new int[train_set.size()];
        //create samples from train set
        for(int i = 0; i < train_set.size(); i++) {
            AnyRecord r = train_set.get(i);
            evolution.add(r);
            Xi[i] = evolution.get_last_sample();
            Y[i] = (int)label.getLabel(r.getLabel_name());
        }









        GraphEncoder encoder = new GraphEncoder(evolution.getEncoderDimension());






        int nClauses = 10;
        int nClasses = 3;

        int max_specificity = 40;
        boolean boost = true;
        int LF = (int) (.80* encoder.getFeatureDimension());

        int threshold = (int) Math.sqrt((nClauses / 2f) * LF);
        //int threshold = nClauses * LF;

        int max_literals = LF;

        //log the parameters
        logger.info("nClauses: " + nClauses);
        logger.info("nClasses: " + nClasses);
        logger.info("max_specificity: " + max_specificity);
        logger.info("LF: " + LF);
        logger.info("threshold: " + threshold);
        logger.info("max_literals: " + max_literals);
        //log number of features
        logger.info("number_of_features: " + encoder.getFeatureDimension());

        GraphAttentionLearning model = new GraphAttentionLearning(encoder, nClauses, nClasses,threshold, max_specificity,  boost, LF, max_literals);

        for(int e = 0; e < 100; e++) {

            model.fit(Xi, Y, true);

            int correct = 0;
            for(AnyRecord record : test_set) {

                int mylabel = (int)label.getLabel(record.getLabel_name());
                evolution.add(record);
                int[] xi = evolution.get_last_sample();

                int predicted = model.predict(xi);

                System.out.println("Predicted: " + predicted + " Actual: " + mylabel);

                if(predicted == mylabel) {
                    correct++;
                }
            }
            logger.info("Epoch: " + e + " Accuracy: " + (correct / (float)test_set.size()));

        }


    }

    public static void main(String[] args) throws IOException, IllegalAccessException {
        IrisExample example = new IrisExample();
    }
}
