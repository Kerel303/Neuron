public class Neuron {
    private double[] weights;
    private double bias;
    private double learningRate;
    //Przyjmujemy długość wektora wag, próg , próg włączenia neuronu, oraz stałą uczenia
    Neuron(int inputSize, double learningRate){
        this.weights = new double[inputSize];
        this.bias = (Math.random()*2)-1;
        this.learningRate = learningRate;
        for(int i = 0; i < weights.length; i++){
            weights[i] = (Math.random()*2)-1;
        }
    }

    // Funkcja aktywacji
    private double sigmoid(double x){
        if (x >= 0) {
            double z = Math.exp(-x);
            return 1.0 / (1.0 + z);
        } else {
            double z = Math.exp(x);
            return z / (1.0 + z);
        }
    }


    // Forward pass
    public double predict(double[] data){
        if(data.length != this.weights.length){
            throw new IllegalArgumentException("Zła długość wektora wejściowego");
        }

        double sum = 0;
        
        for(int i = 0; i < this.weights.length; i++){
            sum += data[i] * weights[i];
        }

        sum += bias;

        return sigmoid(sum);
    }

    // Klasyfikacja
    public int classify(double[] data){
        return predict(data) >= 0.5 ? 1 : 0;
    }


    // Uczenie neuronu
    public void train(double[] data, double realAnswer){
        if(data.length != this.weights.length){
            throw new IllegalArgumentException("Zła długość wektora wejściowego");
        }

        double output = predict(data);
        double error = output - realAnswer;// Zmieniony znak!

        // aktualizacja wag
        for(int i = 0; i < this.weights.length; i++){
            weights[i] -= error * learningRate * data[i];
        }

        // aktualizacja biasu
        bias -= error * learningRate;

    }

    // Funkcja błędu
    public double loss(double[] data, double realAnswer){
        double pred = predict(data);
        return crossEntropy(pred, realAnswer);
    }

    public double crossEntropy(double predicted, double realAnswer){
        double epsilon = 1e-15; // żeby uniknąć log(0)

        predicted = Math.max(epsilon, Math.min(1 - epsilon, predicted));

        return -(realAnswer * Math.log(predicted) + (1 - realAnswer) * Math.log(1 - predicted));
    }


    // Sprawdzanie, czy neuron się uczy
    public double datasetLoss(double[][] inputs, double[] targets){
        double sum = 0;

        for(int i = 0; i < inputs.length; i++){
            sum += loss(inputs[i], targets[i]);
        }

        return sum / inputs.length;
    }



    // Gettery
    public double[] getWeights() {
        return weights;
    }
    public double getBias(){
        return bias;
    }
}
