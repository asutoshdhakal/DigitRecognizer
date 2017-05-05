import numpy as np
import pandas as pd
import random as random
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn import metrics
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import matplotlib.pyplot as plt
import pylab

train = pd.read_csv('train.csv')
# create target
y = np.array(train['label'], dtype=str)
# create input array
X = np.array(train.drop('label', axis=1), dtype=float)
# normalize
X = X / 255
# select 75% for training, we will use the other 25% to test
train_size = int(X.shape[0] * .75)
plt.figure()
pca = PCA(n_components=50)
training_data = pca.fit_transform(X[:train_size], y[:train_size])

def evaluate(x):
    if x[0] < 1:
        return -1,
    decimal = 0
    for digit in x:
        decimal = decimal*2 + int(digit)
    clf = MLPClassifier(solver='sgd', alpha=1e-5, activation='relu', max_iter=2500, hidden_layer_sizes=(decimal,), random_state=1)
    clf.fit(training_data, y[:train_size].ravel())
    predicted = clf.predict(pca.transform(X[train_size:]))
    actual = y[train_size:]
    matrix = metrics.confusion_matrix(actual, predicted)
    sumMatrix = 0
    for i in range(0,10):
        sumMatrix = sumMatrix + matrix[i][i]
    return (sumMatrix,)



creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
population_size = 20
toolbox = base.Toolbox()
toolbox.register("bit", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.bit, n=6)
toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=population_size)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=.5)
CXPB, MUTPB, n_gen = .5, .2, 20
pop = toolbox.population()
toolbox.register("select", tools.selTournament, tournsize=3)

averageFit = [1 for x in range(n_gen)]
bestFit = [1 for x in range(n_gen)]
allFitness = []

for g in range(n_gen):
    print("Generation " + str(g) + "\n")
    # Select the next generation individuals
    offspring = toolbox.select(pop, len(pop))
    # Clone the selected individuals
    offspring = map(toolbox.clone, offspring)
    
    # Apply crossover on the offspring
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < CXPB:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

# Apply mutation on the offspring
for mutant in offspring:
    if random.random() < MUTPB:
        toolbox.mutate(mutant)
            del mutant.fitness.values

sumFit = 0
    bestInd = 0
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    print fitnesses
    length = len(fitnesses)
    totalFitness = 0
    nonVal = 0
    for fitness in fitnesses:
        if(fitness[0] == -1):
            nonVal = nonVal + 1
        else:
            totalFitness = totalFitness + fitness[0]
div = (length - nonVal)
    if(div > 0):
        averageFitness = float(totalFitness)/div
        #         print averageFitness
        #         print length
        #         print nonVal
        #         print totalFitness
        allFitness.append(averageFitness)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
        sumFit = sumFit + fit[0]
        if(fit[0] > bestInd):
            bestInd = fit[0]

averageFit[g] = float(sumFit)/len(offspring)
    bestFit[g] = bestInd
    
    # The population is entirely replaced by the offspring
    pop[:] = offspring

# print(averageFit)
# print(bestFit)
# print allFitness
x = np.arange(0, n_gen, 1)
y = allFitness
plt.plot(x, y)
pylab.show()



#from sklearn import metrics

#predicted = clf.predict(pca.transform(X[train_size:]))
#actual = y[train_size:]
#print(metrics.classification_report(actual, predicted))
#print(metrics.confusion_matrix(actual, predicted))
