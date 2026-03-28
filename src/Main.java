import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

public class Main {

    static boolean debugMode = false;
    static List<Iris> listToTeach = new ArrayList<>();
    static List<Iris> listToTest = new ArrayList<>();

    static NeuronLayer<Iris> layer;

    static int iterationNumber = 500;

    public static void main(String[] args) {
        getData();

        List<String> classes = List.of(
                "Iris-setosa",
                "Iris-versicolor",
                "Iris-virginica"
        );

        layer = new NeuronLayer<>(classes, 4);

        layer.train(listToTeach, iterationNumber);

        testAccuracy();

    }

    static void testAccuracy(){
        double accuracy = layer.accuracy(listToTest) * 100;
        System.out.println("Celność warstwy perceptronów: " + accuracy + "%");

        if(debugMode){
            System.out.println("\n=== DEBUG: szczegółowe wyniki ===");
            for(Iris iris : listToTest){
                String predicted = layer.classify(iris);
                System.out.println("Real: " + iris.type + " | Pred: " + predicted);
            }
        }
    }



    // Pobranie danych z pliku
    static void getData(){
        try(BufferedReader br = new BufferedReader(new FileReader("src/Iris.txt"))){
            String line;
            int counter = 1;
            while ((line = br.readLine()) != null) {
                String[] data = line.split(",");

                Iris iris = new Iris(
                        Double.parseDouble(data[0]),
                        Double.parseDouble(data[1]),
                        Double.parseDouble(data[2]),
                        Double.parseDouble(data[3]),
                        data[4]
                );
                if(counter <= 3){
                    listToTeach.add(iris);
                    if(debugMode){
                        System.out.println(iris.type + " Teach");
                    }
                    counter++;
                }else if(counter < 5){
                    listToTest.add(iris);
                    if(debugMode){
                        System.out.println(iris.type + " Test");
                    }
                    counter++;
                }else if(counter == 5){
                    listToTest.add(iris);
                    if(debugMode){
                        System.out.println(iris.type + " Test");
                    }
                    counter = 1;
                }

            }
        }catch (Exception e){
            System.out.println(e.getMessage());
        }
    }
}
