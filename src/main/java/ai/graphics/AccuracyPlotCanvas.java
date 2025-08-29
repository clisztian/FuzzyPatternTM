package ai.graphics;

import com.jfoenix.controls.JFXSlider;
import javafx.geometry.Insets;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.XYChart;
import javafx.scene.control.CheckBox;
import javafx.scene.effect.DropShadow;
import javafx.scene.layout.Background;
import javafx.scene.layout.BackgroundFill;
import javafx.scene.layout.CornerRadii;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;

public class AccuracyPlotCanvas {

    private LineChart<Number, Number> lineChart;
    private VBox vbox;

    private JFXSlider threshSlider = new JFXSlider(1, 10, 1);
    //slider for number of clauses
    private JFXSlider clauseSlider = new JFXSlider(1, 20, 5);
    private JFXSlider fuzzySlider = new JFXSlider(30, 100, 80);
    private JFXSlider specificitySlider = new JFXSlider(2, 50, 10);
    private CheckBox boosBox = new CheckBox();


    private void setupSlider(JFXSlider slider) {
        slider.setShowTickLabels(false);
        slider.setShowTickMarks(false);
        slider.setBlockIncrement(1);
        slider.setMajorTickUnit(1);
        slider.setMinorTickCount(0);
    }


    public AccuracyPlotCanvas() {


        setupSlider(clauseSlider);
        setupSlider(fuzzySlider);
        setupSlider(specificitySlider);
        setupSlider(threshSlider);
        boosBox.setSelected(true);


        NumberAxis xAxis = new NumberAxis();
        NumberAxis yAxis = new NumberAxis();
        xAxis.setLabel("Epocs");
        yAxis.setLabel("Accuracy");

        yAxis.setLowerBound(0);
        yAxis.setUpperBound(1.02);

        lineChart = new LineChart<>(xAxis, yAxis);
        lineChart.setTitle("Time Series Data");
        lineChart.setAnimated(false);

        lineChart.setBackground(new Background(new BackgroundFill(Color.DARKGRAY.darker().darker().darker(), CornerRadii.EMPTY, Insets.EMPTY )));
        lineChart.getStylesheets().add(getClass().getClassLoader().getResource("css/Chart.css").toExternalForm());

        vbox = new VBox();
        vbox.getChildren().addAll(lineChart);
        vbox.setSpacing(10);
        VBox.setVgrow(lineChart, javafx.scene.layout.Priority.ALWAYS);
        vbox.setBackground(new Background(new BackgroundFill(Color.TRANSPARENT, CornerRadii.EMPTY, Insets.EMPTY)));
        vbox.setPadding(new Insets(10, 10, 10, 10));
        vbox.getStylesheets().add(getClass().getClassLoader().getResource("css/Chart.css").toExternalForm());


    }

    public void plotTimeSeriesData(double[] ts) {



        DropShadow dropShadow = new DropShadow();
        dropShadow.setRadius(5.0);
        dropShadow.setOffsetX(3.0);
        dropShadow.setOffsetY(3.0);
        dropShadow.setColor(Color.color(0.2, 0.2, 0.2));


        XYChart.Series<Number, Number> series = new XYChart.Series<>();
        for(int i = 0; i < ts.length; i++) {
            series.getData().add(new XYChart.Data<>(i, ts[i]));
        }
        lineChart.getData().setAll(series);

    }

    public VBox getvBox() {
        return vbox;
    }
}
