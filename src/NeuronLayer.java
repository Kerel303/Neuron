import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class NeuronLayer<T extends Trainable> {
    private Map<String, Neuron> neurons = new LinkedHashMap<>();
    private List<String> classes = new ArrayList<>();
    private int inputSize = 0;
    private double learningRate = 0.25; 

    // Konstruktor
    NeuronLayer(List<String> classes, int inputSize){
        if(classes == null){
            throw new IllegalArgumentException("Lista klas jest pusta. Anulowanie tworzenia warstwy neuronów");
        }
        this.classes = classes;
        this.inputSize = inputSize;
        for(String clazz : classes){
            neurons.put(clazz, new Neuron(inputSize, learningRate));
        }
    }

    // Metody

    // Uczenie perceptronów
    public void train(List<T> data, int epochs){
        if(data == null){
            throw new IllegalArgumentException("Lista danych jest pusta");
        }
        if(data.get(0).getInput().length != inputSize){
            throw new IllegalArgumentException("Zły rozmiar wektora wejściowego");
        }
        for(int e = 0; e < epochs; e++){
            TeachNeurons(data);
            // Opcjonalne do dodania: early stopping w następnej linijce
            //if(accuracy(data) > 0.95) break;
        }
    }
    // Opcjonalne do dodania: early stopping
    private void TeachNeurons(List<T> listToTeach){
        List<T> shuffled = new ArrayList<>(listToTeach);
        Collections.shuffle(shuffled);
        for (String targetClass : neurons.keySet()){
            Neuron n = neurons.get(targetClass);
            for (T t : shuffled){
                double expected = t.getLabel().equals(targetClass) ? 1.0 : 0.0;
                n.train(t.getInput(), expected);
            }
        }
    }

    // Klasyfikacja perceptronów
    public String classify(T t){
        double maxScore = Double.NEGATIVE_INFINITY;
        String bestClass = null;
        double[] data = t.getInput();

        for(Map.Entry<String, Neuron> entry : neurons.entrySet()){
            String clazz = entry.getKey();
            Neuron n = entry.getValue();
            
            double score = n.predict(data);

            if(score > maxScore){
                maxScore = score;
                bestClass = clazz;
            }
        }
        if(bestClass == null){
            throw new IllegalStateException("Brak klasyfikacji - sprawdź dane");
        }
        return bestClass;
    }

    // Test celności warstwy perceeptronów
    public double accuracy(List<T> testData){
        int correct = 0;
        for(T t : testData){
            if(classify(t).equals(t.getLabel())){
                correct++;
            }
        }
        return (double) correct / testData.size();
    }

    // Możliwość obserwowania jak spada cross-entropy w trakcie uczenia
    public double datasetLoss(List<T> data){
        double sum = 0;
        for(T t : data){
            for(String clazz : classes){
                Neuron n = neurons.get(clazz);
                double real = t.getLabel().equals(clazz) ? 1.0 : 0.0;
                sum += n.loss(t.getInput(), real);
            }
        }
        return sum / (data.size() * classes.size());
    }


    // Gettery
    public List<String> getClasses(){
        return classes;
    }
    public int getInputSize(){
        return inputSize;
    }
}
