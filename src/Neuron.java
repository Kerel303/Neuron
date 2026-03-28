public class Neuron {
    public class Perceptron {
    private double[] weights;
    private double activationThreshold;
    private double alphaLearningConstant;
    private double betaLearningConstant = 0;
    //Przyjmujemy długość wektora wag, próg , próg włączenia perceptronu, oraz stałą uczenia alfa
    Perceptron(int LengthOfTheWeightVector, double activationThreshold, double alphaLearningConstant){
        weights = new double[LengthOfTheWeightVector];
        for(int i = 0; i < LengthOfTheWeightVector; i++){
            // Początkowe wartości wag są losowane od (-1) do (1)
            weights[i] = (Math.random()*2)-1;
        }
        this.activationThreshold = activationThreshold;
        this.alphaLearningConstant = alphaLearningConstant;
    }
    //Przyjmujemy długość wektora wag, próg , próg włączenia perceptronu, oraz stałą uczenia alfa i stałą uczenia beta dla progu aktywacji
    Perceptron(int LengthOfTheWeightVector, double activationThreshold, double alphaLearningConstant, double betaLearningConstant){
        weights = new double[LengthOfTheWeightVector];
        for(int i = 0; i < LengthOfTheWeightVector; i++){
            // Początkowe wartości wag są losowane od (-1) do (1)
            weights[i] = (Math.random()*2)-1;
        }
        this.activationThreshold = activationThreshold;
        this.alphaLearningConstant = alphaLearningConstant;
        this.betaLearningConstant = betaLearningConstant;
    }

    int classify(double[] data){
        if(data.length != this.weights.length){
            throw new IllegalArgumentException("Zła długość wektora wejściowego");
        }

        double sum = 0;
        
        for(int i = 0; i < this.weights.length; i++){
            sum += data[i]*weights[i];
        }
        
        if(sum >= this.activationThreshold){
            return 1;
        }else{
            return 0;
        }
    }

    void learn(double[] data, int realAnswer){
        if(data.length != this.weights.length){
            throw new IllegalArgumentException("Zła długość wektora wejściowego");
        }

        int output = classify(data);
        int error = realAnswer - output;
        for(int i = 0; i < this.weights.length; i++){
            weights[i] = weights[i] + error*alphaLearningConstant*data[i];
        }
        if(betaLearningConstant != 0){
            activationThreshold = activationThreshold - betaLearningConstant * error;
        }else{
            activationThreshold = activationThreshold - alphaLearningConstant * error;
        }
    }


    // Gettery
    public double[] getWeights() {
        return weights;
    }
    public double getActivationThreshold() {
        return activationThreshold;
    }
    public double getAlphaLearningConstant(){
        return alphaLearningConstant;
    }
}

}
