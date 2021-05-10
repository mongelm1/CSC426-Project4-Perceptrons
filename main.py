from Perceptron import *
import matplotlib.pyplot as plt
from random import random as from_0_to_1, shuffle

def training_data(filename: str) -> list:
    import csv
    with open(filename, newline='') as iris_file:
        data = csv.reader(iris_file)
        examples = [([*map(float,inputs)], training_label) for (*inputs, training_label) in data]
    return examples

def Perceptron_Training_Rule(p: Perceptron, n: float, target_flower: str, iris_examples: list, taskname: str) -> None:

    plottitle = f"{taskname}_{target_flower}" # title for plot which graphs results of learning
    outputfilename = f"{taskname}_stats.txt"
    open(outputfilename, 'w') # open empty file to store results
    outputfile = open(outputfilename, 'a') # epoch stats file, stores results of learning, final results at end of file

    print(f"LEARNING PROBLEM: '{target_flower}'", file=outputfile)

    errors = [True] # put one element in errors so process starts
    epoch = 0 # counter for number of times perceptron is trained on the dataset
    errors_per_epoch = [] # number of errors accumulated from each epoch
    while errors:
        errors = []
        
        print('\tWEIGHTS: <', *[f"w{i}: {w:.3f}," for i,w in enumerate(p.weights)], '>', file=outputfile)
        
        for i, (measurements, flower) in enumerate(iris_examples): # for d in D

            # classification of flower as pos or neg with respect to target
            t = 1 if flower==target_flower else -1
            # perceptron's classification of flower based on current weights
            o = p.o(*measurements)

            # add misclassified examples to list of errors
            if t != o: errors.append(f"d_{i} = {(measurements, flower)}, t = {t}, o = {o}")

            # Update rule
            updated_bias = p.weights[0] + (n * (t - o) * 1)
            updates = [p.weights[i+1] + (n * (t - o) * measurements[i]) for i in range(len(p.weights[1:]))] 
            p.weights = (updated_bias,) + tuple(updates)

        # print errors to output file
        print(f"\tEPOCH #{epoch}, {len(errors)} errors:", file=outputfile)
        for misclassified_example in errors:
            print("\t\t"+str(misclassified_example), file=outputfile)
        print(file=outputfile)

        num_errors_in_this_epoch = len(errors) # number of errors collected from this epoch
        errors_per_epoch.append(num_errors_in_this_epoch) # list of number of errors collected in each epoch

        # if no errors, then target was learned perfectly,
        # or if number of errors per epoch stays the same for many (10) epochs,
        # then learning has stabilized, end alg and print results
        if errors==[] or errors_per_epoch.count(num_errors_in_this_epoch) > 10: 
            print(f"RESULTS FOR '{target_flower}'", file=outputfile)
            print(f"\tepoch # = {epoch}", file=outputfile)
            print(f"\t# errors = {num_errors_in_this_epoch}", file=outputfile)
            print('\tWEIGHTS: <', *[f"w{i}: {w:.3f}," for i,w in enumerate(p.weights)], '>', file=outputfile)

            # plot epoch number (x) vs number of errors per epoch (y)
            x = list(range(epoch+1))
            y = errors_per_epoch
            plt.plot(x, y, '-ok') # scatter plot connecting points with lines
            plt.title(plottitle)
            plt.xlabel('Epoch #')
            plt.ylabel('# Errors per Epoch')
            plt.xlim(0)
            plt.ylim(0)
            plotfilename = f"{plottitle}.pdf"
            plt.savefig(plotfilename)
            plt.clf() # clear plot after plot is saved

            print(f"{outputfilename} and {plotfilename} created")

            return

        epoch += 1 # keep track of number of epochs

def run_learning_problems(weights: list, examples: list, taskname: str) -> None:
    # Run perceptron training rule on each flower class

    print(f"Performing {taskname}...")

    # LP 1
    p = Perceptron(*weights)
    Perceptron_Training_Rule(p, n, 'Iris-setosa', examples, taskname)
    # LP 2
    p = Perceptron(*weights)
    Perceptron_Training_Rule(p, n, 'Iris-versicolor', examples, taskname)
    # LP 3
    p = Perceptron(*weights)
    Perceptron_Training_Rule(p, n, 'Iris-virginica', examples, taskname)

    print(f"Task {taskname} complete.")

iris_examples = training_data('iris.data')
n = 0.001
num_inputs = len(iris_examples[0][0])

#### TASK 2 - all initial weights are 0
weights_all_0 = [0 for i in range(1+num_inputs)]
run_learning_problems(weights_all_0, iris_examples, 'T2')

#### TASK 3
# T 3.1 - all initial weights are 1
weights_all_1 = [1 for i in range(1+num_inputs)]
run_learning_problems(weights_all_1, iris_examples, 'T3.1')
# T 3.2 - all initial weights are randomized in bounds [0,1]
weights_all_0_to_1 = [from_0_to_1() for i in range(1+num_inputs)]
run_learning_problems(weights_all_0_to_1, iris_examples, 'T3.2')
# T 3.3 - repeat T3.2 with new random weights
weights_all_0_to_1 = [from_0_to_1() for i in range(1+num_inputs)]
run_learning_problems(weights_all_0_to_1, iris_examples, 'T3.3')

#### TASK 4 ####
# T 4.1 - repeat T2 with input data randomly shuffled
shuffle(iris_examples)
weights_all_0 = [0 for i in range(1+num_inputs)]
run_learning_problems(weights_all_0, iris_examples, 'T4.1')
# T 4.2 - repeat T4.2 with data shuffled again
shuffle(iris_examples)
weights_all_0 = [0 for i in range(1+num_inputs)]
run_learning_problems(weights_all_0, iris_examples, 'T4.2')