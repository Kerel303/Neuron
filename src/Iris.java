public class Iris implements Trainable{
    double sepalLength;
    double sepalWidth;
    double petalLength;
    double petalWidth;
    String type;

    Iris(double sepalLength, double sepalWidth, double petalLength, double petalWidth, String type){
        this.sepalLength = sepalLength;
        this.sepalWidth = sepalWidth;
        this.petalLength = petalLength;
        this.petalWidth = petalWidth;
        this.type = type;
    }

    @Override
    public String toString(){
        return sepalLength + " " + sepalWidth + " " + petalLength + " " + petalWidth +  " " + type;
    }

    @Override
    public double[] getInput() {
        return new double[]{sepalLength, sepalWidth, petalLength, petalWidth};
    }

    @Override
    public String getLabel() {
        return this.type;
    }
}
