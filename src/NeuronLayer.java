import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class NeuronLayer<T extends Trainable> {
    private Map<String, Neuron> perceptrons = new LinkedHashMap<>();
    private List<String> classes = new ArrayList<>();
    private int inputSize = 0;
    private double alphaLearningConstant = 0.25; 

    // Konstruktor
    NeuronLayer(List<String> classes, int inputSize){
        this.classes = classes;
        this.inputSize = inputSize;
        for(String clazz : classes){
            perceptrons.put(clazz, new Neuron(inputSize, alphaLearningConstant));
        }
    }

    // Metody

    // Uczenie perceptronów
    public void train(List<T> data, int epochs){
        if(data.get(0).getInput().length != inputSize){
            throw new IllegalArgumentException("Zły rozmiar wektora wejściowego");
        }
        for(int e = 0; e < epochs; e++){
            TeachPerceptrons(data);
            // Opcjonalne do dodania: early stopping w następnej linijce
            //if(accuracy(data) > 0.95) break;
        }
    }
    // Opcjonalne do dodania: early stopping
    private void TeachPerceptrons(List<T> listToTeach){
        List<T> shuffled = new ArrayList<>(listToTeach);
        Collections.shuffle(shuffled);
        for (String targetClass : perceptrons.keySet()){
            Neuron p = perceptrons.get(targetClass);
            for (T t : shuffled){
                int expected = t.getLabel().equals(targetClass) ? 1 : 0;
                p.learn(t.getInput(), expected);
            }
        }
    }

    // Klasyfikacja perceptronów
    public String classify(T t){
        double maxScore = Double.NEGATIVE_INFINITY;
        String bestClass = null;
        double[] data = t.getInput();

        for(Map.Entry<String, Neuron> entry : perceptrons.entrySet()){
            String clazz = entry.getKey();
            Neuron p = entry.getValue();
            double sum = 0;
            double[] weights = p.getWeights();
            
            for(int j = 0; j < weights.length - 1; j++){
                sum += data[j] * weights[j];
            }
            sum += weights[weights.length - 1]; // bias

            if(sum > maxScore){
                maxScore = sum;
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

    // Gettery
    public List<String> getClasses(){
        return classes;
    }
    public int getInputSize(){
        return inputSize;
    }
}
