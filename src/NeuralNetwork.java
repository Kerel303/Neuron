import java.util.ArrayList;
import java.util.List;

public class NeuralNetwork<T extends Trainable> {
    
    private List<NeuronLayer<T>> layers = new ArrayList<>();

    // Konstruktor
    NeuralNetwork(){

    }

    // Dodawanie warstwy neuronowej
    public void addLayer(NeuronLayer<T> layer){
        layers.add(layer);
    }

    // Forward pass przez wszystkie warstwy (bez uczenia w hidden layers)
    public double[] forward(double[] input) {
        double[] activations = input;
        for (NeuronLayer<T> layer : layers) {
            double[] layerOutput = new double[layer.getClasses().size()];
            int i = 0;
            for (String clazz : layer.getClasses()) {
                Neuron neuron = layer.neurons.get(clazz); // korzystamy z warstwy
                layerOutput[i] = neuron.predict(activations);
                i++;
            }
            activations = layerOutput; // wyjście staje się wejściem kolejnej warstwy
        }
        return activations;
    }

    // Klasyfikacja: wybór neuronu z największą aktywacją w ostatniej warstwie
    public String classify(T t) {
        double[] output = forward(t.getInput());
        List<String> classes = layers.get(layers.size() - 1).getClasses();
        double max = Double.NEGATIVE_INFINITY;
        String bestClass = null;
        for (int i = 0; i < classes.size(); i++) {
            if (output[i] > max) {
                max = output[i];
                bestClass = classes.get(i);
            }
        }
        return bestClass;
    }

    // Trening pojedynczej warstwy wyjściowej (reszta warstw można implementować z propagacją wsteczną)
    public void trainOutputLayer(List<T> data, int epochs) {
        if (layers.isEmpty()) {
            return;
        }
        NeuronLayer<T> outputLayer = layers.get(layers.size() - 1);
        outputLayer.train(data, epochs);
    }

    // Test celności na całej sieci
    public double accuracy(List<T> testData) {
        int correct = 0;
        for (T t : testData) {
            if (classify(t).equals(t.getLabel())) correct++;
        }
        return (double) correct / testData.size();
    }


}
