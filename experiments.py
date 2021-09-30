import argparse
import os
import sys
import numpy as np
import matplotlib as mpl
mpl.use('pdf')

import matplotlib.pyplot as plt
from cycler import cycler
import pandas as pd
import seaborn as sns

#Plot configurations
sns.set(style="white")

plt.rc('font', family='serif', serif='Times')
'''plt.rc('font', size=10)
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rc('axes', labelsize=8)
plt.rc('lines', linewidth=1)
plt.rc('lines', markersize=2)
plt.rc('legend', fontsize=6)'''

plt.rc('font', size=10)
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rc('axes', labelsize=8)
plt.rc('lines', linewidth=1)
plt.rc('lines', markersize=2)
plt.rc('legend', fontsize=6)

# width as measured in inkscape
width = 3.491
height = width / 1.618

#Class to represent a set of objects from the same class in the retrieval experiment
class DatasetClass:
    def __init__(self):
        self.name = ''
        self.parentClass = -1
        self.models = []

#Class that encapsulates the methods for computing retrieval metrics
class Evaluator:
    def __init__(self):
        self.classes = []
        self.classesQuery = []
        self.classObject = []
        self.classQuery = []
        self.distanceMatrix = None
        self.numObjects = -1
        self.numQueryObjects = -1

    #Parses the classification file for the target set
    def parseClassificationTarget(self,filename):
        with open(filename, 'rt') as f:
            text = f.readlines()
            
            firstLine = text[1].split()
            numClasses = int(firstLine[0])
            self.numObjects = int(firstLine[1])
            self.classObject = [0 for i in range(self.numObjects)]
            
            initIndex = 3
            for cl in range(numClasses):
                headerClass = text[initIndex].split()
                newClass = DatasetClass()
                newClass.name = headerClass[0]
                newClass.parentClass = int(headerClass[1])
                numObjectsClass = int(headerClass[2])

                initIndex = initIndex + 1

                for mod in range(numObjectsClass):
                    newClass.models.append(int(text[initIndex + mod]))
                    self.classObject[int(text[initIndex+mod])] = cl
                
                initIndex = initIndex + numObjectsClass
                
                self.classes.append(newClass)
                initIndex = initIndex + 1

    # Parses the classification file for the query set                            
    def parseClassificationQuery(self, filename):
        with open(filename, 'rt') as f:
            text = f.readlines()
            
            firstLine = text[1].split()
            numClasses = int(firstLine[0])
            self.numQueryObjects = int(firstLine[1])
            self.classQuery = [0 for i in range(self.numQueryObjects)]
            
            initIndex = 3
            for cl in range(numClasses):
                headerClass = text[initIndex].split()
                newClass = DatasetClass()
                newClass.name = headerClass[0]
                newClass.parentClass = int(headerClass[1])
                numObjectsClass = int(headerClass[2])

                initIndex = initIndex + 1

                for mod in range(numObjectsClass):
                    newClass.models.append(int(text[initIndex + mod]))
                    self.classQuery[int(text[initIndex+mod])] = cl
                
                initIndex = initIndex + numObjectsClass
                initIndex = initIndex + 1
                self.classesQuery.append(newClass)

    #Parses a distance matrix stored in a text file
    def parseDistanceMatrix(self, filename):
        self.distanceMatrix = np.loadtxt(filename)

    #Compute the DCG metric given a binary retrieval list, where 1's appear in locations of relevant objects
    def computeDCG(self,result):
        dcg = []
        for i, val in enumerate(result):
            dcg.append((2**val-1)/(np.log2(i + 2)))
        return sum(dcg)

    #Compute the NDCG metric for a given binary retrieval list
    def computeNDCG(self, result):
        perfectResult = sorted(result, reverse=True)
        return self.computeDCG(result)/self.computeDCG(perfectResult)

    #Compute all the metrics for a given model. "value" is the number of bins in recall plots
    def computeMetricsPerModel(self, model, value):
        recall_precision = [0 for i in range(value + 1)] #Array with precision values per recall

        #Init values for metrics
        NN = 0.0
        FT = 1.0
        ST = 1.0
        MAP = 0.0
        NDCG = 0.0

        results = []    # List with information about retrieval list (each element is a dictionary)
        for i in range(self.numObjects):
            result = dict()

            # Important information to store: 
            #   - the id (original position)
            #   - the object class
            #   - The distance to query

            result['id'] = i
            result['class'] = self.classObject[i]
            result['distance'] = self.distanceMatrix[model][i]
            results.append(result)

        def compareDistance(e):
            return e['distance']
        
        # Sort the retrieval list by distance
        results.sort(key=compareDistance)

        relevantRetrieved = 0
        numRetrieved = 0
        queryClass = self.classQuery[model] #Get the class of the query object
        nearestNeighborClass = results[0]['class'] #Get the class of the nearest neighbor object
        numModels = len(self.classes[queryClass].models) #Get the number of target objects in the class
        rankedRelevantList = [1 if results[i]['class'] == queryClass else 0 for i in range(self.numObjects)]

        NDCG = self.computeNDCG(rankedRelevantList)

        #This table stores the corresponding precision and recall in every relevant object
        precisionRecallTable = np.zeros((numModels, 2), dtype=np.float)
        
        while relevantRetrieved < numModels:
            if results[numRetrieved]['class'] == queryClass:
                if numRetrieved == 0:   #If the first retrieved is relevant, NN is 1.0
                    NN = 1.0
                
                #Get recall, precision values in this relevant
                rec = (relevantRetrieved + 1)/numModels
                prec =  (relevantRetrieved + 1)/(numRetrieved+1)
                
                # 
                precisionRecallTable[relevantRetrieved,0] = rec * 100
                precisionRecallTable[relevantRetrieved,1] = prec
                MAP = MAP + prec
                relevantRetrieved = relevantRetrieved + 1
            
            if numRetrieved == (numModels-1):
                FT = (relevantRetrieved+1)/(numRetrieved+1)
            
            if numRetrieved == (2*numModels-1):
                ST = (relevantRetrieved+1)/(numRetrieved+1)
            
            numRetrieved = numRetrieved + 1
        
        MAP = MAP/numModels

        # Interpolation procedure
        index = numModels - 2
        recallValues = 100 - (100/value)
        maxim = precisionRecallTable[index+1][1]
        pos = value
        recall_precision[pos] = maxim

        pos = pos - 1

        while index >= 0:
            if int(precisionRecallTable[index][0]) >= recallValues:
                if precisionRecallTable[index][1] > maxim:
                    maxim = precisionRecallTable[index][1]
                index = index - 1
            else:
                recall_precision[pos] = maxim
                recallValues = recallValues - 100/value
                pos = pos - 1
        
        while pos>=0:
            recall_precision[pos] = maxim
            pos = pos - 1
        
        # The result is returned in a dictionary
        resultModel = dict()
        resultModel['pr'] = [recall_precision[i] for i in range(value+1)]
        resultModel['NN'] = NN
        resultModel['FT'] = FT
        resultModel['ST'] = ST
        resultModel['MAP'] = MAP
        resultModel['NDCG'] = NDCG
        resultModel['queryClass'] = queryClass
        resultModel['nnClass'] = nearestNeighborClass
        resultModel['rankedList'] = rankedRelevantList

        return resultModel
    
    #Compute the consolidated metrics for a given class
    def computeMetricsPerClass(self, clas, value):
        models = self.classesQuery[clas].models
        
        resultClass = dict()
        resultClass['pr'] = [0.0 for i in range(value+1)]
        resultClass['NN'] = 0.0
        resultClass['FT'] = 0.0
        resultClass['ST'] = 0.0
        resultClass['MAP'] = 0.0
        resultClass['NDCG'] = 0.0

        for model in models:
            result = self.computeMetricsPerModel(model, value)

            resultClass['pr'] = [resultClass['pr'][i] + result['pr'][i] for i in range(value + 1)]
            resultClass['NN'] = resultClass['NN'] + result['NN']
            resultClass['FT'] = resultClass['FT'] + result['FT']
            resultClass['ST'] = resultClass['ST'] + result['ST']
            resultClass['MAP'] = resultClass['MAP'] + result['MAP']
            resultClass['NDCG'] = resultClass['NDCG'] + result['NDCG']
        
        resultClass['pr'] = [(resultClass['pr'][i]/len(models)) for i in range(value + 1)]
        resultClass['NN'] = resultClass['NN']/len(models)
        resultClass['FT'] = resultClass['FT']/len(models)
        resultClass['ST'] = resultClass['ST']/len(models)
        resultClass['MAP'] = resultClass['MAP']/len(models)
        resultClass['NDCG'] = resultClass['NDCG']/len(models)

        return resultClass

    #Computes the metrics for the entire set, but class-averaged
    def computeMetricsAvgClass(self, value):
        numClasses = len(self.classesQuery)

        resultAvgClass = dict()
        resultAvgClass['pr'] = [0.0 for i in range(value+1)]
        resultAvgClass['NN'] = 0.0
        resultAvgClass['FT'] = 0.0
        resultAvgClass['ST'] = 0.0
        resultAvgClass['MAP'] = 0.0
        resultAvgClass['NDCG'] = 0.0
        

        for cl in range(numClasses):
            result = self.computeMetricsPerClass(cl, value)

            resultAvgClass['pr'] = [resultAvgClass['pr'][i] + result['pr'][i] for i in range(value + 1)]
            resultAvgClass['NN'] = resultAvgClass['NN'] + result['NN']
            resultAvgClass['FT'] = resultAvgClass['FT'] + result['FT']
            resultAvgClass['ST'] = resultAvgClass['ST'] + result['ST']
            resultAvgClass['MAP'] = resultAvgClass['MAP'] + result['MAP']
            resultAvgClass['NDCG'] = resultAvgClass['NDCG'] + result['NDCG']
            

        resultAvgClass['pr'] = [(resultAvgClass['pr'][i]/numClasses) for i in range(value + 1)]
        resultAvgClass['NN'] = resultAvgClass['NN']/numClasses
        resultAvgClass['FT'] = resultAvgClass['FT']/numClasses
        resultAvgClass['ST'] = resultAvgClass['ST']/numClasses
        resultAvgClass['MAP'] = resultAvgClass['MAP']/numClasses
        resultAvgClass['NDCG'] = resultAvgClass['NDCG']/numClasses
        
        return resultAvgClass

    #Compute the metrics for the entire dataset, averaged by object
    def computeMetricsAll(self, value):
        resultAll = dict()
        resultAll['pr'] = [0.0 for i in range(value+1)]
        resultAll['NN'] = 0.0
        resultAll['FT'] = 0.0
        resultAll['ST'] = 0.0
        resultAll['MAP'] = 0.0
        resultAll['NDCG'] = 0.0
        CM = np.zeros((len(self.classes), len(self.classes)))

        ranking = np.zeros((self.numQueryObjects, self.numObjects))
        listRanking = list()

        for i in range(self.numQueryObjects):
            
            result = self.computeMetricsPerModel(i, value)

            resultAll['pr'] = [resultAll['pr'][i] + result['pr'][i] for i in range(value + 1)]
            resultAll['NN'] = resultAll['NN'] + result['NN']
            resultAll['FT'] = resultAll['FT'] + result['FT']
            resultAll['ST'] = resultAll['ST'] + result['ST']
            resultAll['MAP'] = resultAll['MAP'] + result['MAP']
            resultAll['NDCG'] = resultAll['NDCG'] + result['NDCG']
            CM[result['queryClass']][result['nnClass']] = CM[result['queryClass']][result['nnClass']] + 1
            listRanking.append(result['rankedList'])

        resultAll['pr'] = [(resultAll['pr'][i]/self.numQueryObjects) for i in range(value + 1)]
        resultAll['NN'] = resultAll['NN']/self.numQueryObjects
        resultAll['FT'] = resultAll['FT']/self.numQueryObjects
        resultAll['ST'] = resultAll['ST']/self.numQueryObjects
        resultAll['MAP'] = resultAll['MAP']/self.numQueryObjects
        resultAll['NDCG'] = resultAll['NDCG']/self.numQueryObjects
        resultAll['CM'] = CM

        cnt = 0
        for cl in self.classesQuery:
            for idx in cl.models:
                ranking[cnt] = np.asarray(listRanking[idx], dtype=np.int8)
                cnt = cnt + 1

        resultAll['rankedList'] = ranking


        return resultAll

#Class to represent a participant method in the retrieval experiment. One method can have several runs
class Method:
    def __init__(self):
        self.path = ''
        self.name = ''
        self.ext = []
        self.setupNames = [] #Names of the runs
        self.resultSetup = [] #Results per run
        self.matrices = []
        self.loadedMatrices = False

    #Performs the evaluation, given an evaluator and parameters for the evaluation
    def performEvaluation(self, evaluator, type='all', value=None, numBins=10):
        self.resultSetup.clear()

        #Load matrices for the first time
        if not self.loadedMatrices:
            for i, name in enumerate(self.setupNames):
                matrix = np.loadtxt(os.path.join(self.path, self.name + '.' + name + '.' + self.ext[i]))
                print(f'{self.name} - {name}: {matrix.shape}')
                self.matrices.append(matrix)
            self.loadedMatrices = True

        #For each run
        for i, name in enumerate(self.setupNames):
            if type=='all':
                evaluator.distanceMatrix = self.matrices[i]
                result = evaluator.computeMetricsAll(numBins)
                self.resultSetup.append(result)

            if type == 'class':
                evaluator.distanceMatrix = self.matrices[i]
                result = evaluator.computeMetricsPerClass(value, numBins)
                self.resultSetup.append(result)
            
            if type == 'avg_class':
                evaluator.distanceMatrix = self.matrices[i]
                result = evaluator.computeMetricsAvgClass(numBins)
                self.resultSetup.append(result)

    #Selects the best run in terms of MAP
    def selectBestPerformance(self):
        maxim = 0.0
        bestResult = None
        bestName = None

        for res, name in zip(self.resultSetup, self.setupNames):
            if res['MAP'] > maxim:
                maxim = res['MAP']
                bestResult = res
                bestName = name

        return bestResult, bestName

#Class that represents the entire experiment
class Experiment:
    def __init__(self, path, outputPath, evaluator, type='all', value=None, numBins=10, reportMethods = False):

        self.styles = []

        self.files = sorted(os.listdir(path))
        self.methods = dict()
        self.listMethods = []
        self.numBins = numBins
        self.evaluator = evaluator
        self.outputPath = outputPath
        self.path = path
        self.reportMethods = reportMethods
        self.type = type
        self.value = value

        print(self.files)
        for name in self.files:
            A = name.split('.')
            self.listMethods.append(A[0])

        self.listMethods = set(self.listMethods)
        self.listMethods = sorted(self.listMethods)

        for elem in self.listMethods:
            self.methods[elem] = Method()
            self.methods[elem].name = elem
            self.methods[elem].path = path

        for name in self.files:
            A = name.split('.')
            self.methods[A[0]].setupNames.append(A[1])
            self.methods[A[0]].ext.append(A[2])

        for k,v in self.methods.items():
            print(k)

        #Perform the evaluation for each method according to the parameters
        for key in self.methods:
            self.methods[key].performEvaluation(self.evaluator, type=type, value=value, numBins=self.numBins)

    #Styles for plots
    def defineStyles(self):
        last_cycler = plt.rcParams['axes.prop_cycle']
        colors = list()
        for d in last_cycler:
            colors.append(d["color"])

        self.styles.append(dict(marker='x',color=colors[0],linestyle='--'))
        self.styles.append(dict(marker='o',color=colors[1],linestyle='-'))
        self.styles.append(dict(marker='+',color=colors[2],linestyle='-.'))
        self.styles.append(dict(marker='s',color=colors[3],linestyle=':'))
        self.styles.append(dict(marker='8',color=colors[4],linestyle='--'))
        self.styles.append(dict(marker='*',color=colors[5],linestyle='-'))
        self.styles.append(dict(marker='v',color=colors[0],linestyle='-.'))
        self.styles.append(dict(marker='p',color=colors[1],linestyle=':'))
        self.styles.append(dict(marker='D',color=colors[2],linestyle='--'))
        self.styles.append(dict(marker='.',color=colors[3],linestyle='-'))

    #Plot precision-recall curves for every run in methods
    def generateRecallPrecisionPlotByMethod(self):
        X = np.linspace(0.0, 1.0, num=self.numBins+1)

        for (key, v) in self.methods.items():
            fig, ax = plt.subplots()
            #fig.subplots_adjust(left=.15, bottom=.2, right=0.9, top=.97)

            for i, (run, res) in enumerate(zip(v.setupNames, v.resultSetup)):
                pr = np.asarray(res['pr'])
                plt.plot(X, pr, label= run, **self.styles[i])

            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

            # Put a legend to the right of the current axis
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.ylim(0.0,1.0)
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            
            fig.set_size_inches(width, height)

            namefile = v.name + '.pdf'
            if self.type == 'class':
                namefile = v.name + '_' + self.evaluator.classesQuery[self.value].name + '.pdf'

            fig.savefig(os.path.join(self.outputPath, namefile + '.pdf'))
            
    #Plot precision-recall curves, the best run per method
    def generateRecallPrecisionPlot(self):
        X = np.linspace(0.0, 1.0, num=self.numBins+1)
        fig, ax = plt.subplots()
        fig.subplots_adjust(left=.15, bottom=.2, right=0.9, top=.97)
        
        for i, (key,v) in enumerate(self.methods.items()):
            res, nameSetup = self.methods[key].selectBestPerformance()
            res = np.asarray(res['pr'])
            plt.plot(X, res, label=self.methods[key].name + '('+nameSetup+')', **self.styles[i])
               
        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0-0.05*box.height, box.width, box.height*0.9])

        # Put a legend to the right of the current axis
        
        if self.type=='all' or self.type=='avg_class':
            ax.legend(loc='upper center', bbox_to_anchor=(0.45, 1.2), ncol = 4)
        
        plt.ylim(0.0,1.0)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.grid()
        
        fig.set_size_inches(width, width)

        namefile = 'recall_precision.pdf'
        if self.type == 'class':
            namefile = 'recall_precision_' + self.evaluator.classesQuery[self.value].name + '.pdf'

        fig.savefig(os.path.join(self.outputPath, namefile))
        
    #Generate a Latex table with the results
    def generateLatexTable(self):
        namefile = 'table.tex'
        if self.type == 'class':
            namefile = 'table_' + self.evaluator.classesQuery[self.value].name + '.tex'

        with open(os.path.join(self.outputPath, namefile), 'wt') as f:
            f.write('\\begin{table}\n')
            f.write('\\centering\n')
            f.write('\\begin{tabular}{| c | c | c | c | c | c |}\n')
            f.write('\\hline\n')
            f.write('Methods & NN & FT & ST & mAP & NDCG\\\\ \\hline\n')

            for i,key in enumerate(self.methods):
                name = self.methods[key].name
                for run, res in zip(self.methods[key].setupNames, self.methods[key].resultSetup):
                    f.write(f'{name} ({run}) & {res["NN"]:.4f} & {res["FT"]:.4f} & {res["ST"]:.4f} & {res["MAP"]:.4f} & {res["NDCG"]:.4f} \\\\ \\hline\n')

            f.write('\\end{tabular}\n')
            f.write('\\caption{Table}\n')
            f.write('\\end{table}\n')

    #Generate the results in a plain text file
    def generateTextTable(self):
        namefile = 'table.txt'
        if self.type == 'class':
            namefile = 'table_' + self.evaluator.classesQuery[self.value].name + '.txt'

        with open(os.path.join(self.outputPath, namefile), 'wt') as f:
            f.write('Methods \t NN \t FT \t ST \t mAP \t NDCG\n')

            for i,key in enumerate(self.methods):
                name = self.methods[key].name
                for run, res in zip(self.methods[key].setupNames, self.methods[key].resultSetup):
                    f.write(f'{name} ({run}) \t {res["NN"]:.4f} \t {res["FT"]:.4f} \t {res["ST"]:.4f} \t {res["MAP"]:.4f} \t {res["NDCG"]:.4f} \n')

    #Generates a confusion matrix for the experiment
    def plotConfussionMatrix(self, name, nameSetup, matrix):
        nameClasses = [cl.name for cl in self.evaluator.classes]
        data = matrix
        sum_per_row = data.sum(axis=1)
        dataNorm = data / sum_per_row[:,np.newaxis]
            
        f, ax = plt.subplots()
        f.subplots_adjust(left=.25, bottom=.25)
        heatmap = sns.heatmap(dataNorm, cbar=False, cmap='viridis', vmin=0.0, vmax=1.0,square=True, linewidths=.5, xticklabels=nameClasses, yticklabels=nameClasses)
        heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation = 90, fontsize=14)
        heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation = 0, fontsize=14)
        plt.title(name + '('+nameSetup+')')
        f.set_size_inches(width, width)
        f.savefig(os.path.join(self.outputPath, f'cm_{name}_{nameSetup}.pdf'))

        plt.close('all')
    
    #Generates a set of confusion matrices depending on the configuration
    def generateConfussionMatrices(self):
        for i, (key,v) in enumerate(self.methods.items()):
            name = v.name
            
            if not self.reportMethods: #Plot the best configuration only
                res, nameSetup = v.selectBestPerformance()
                self.plotConfussionMatrix(name, nameSetup, res['CM'])
            else: #Plot all the configurations
                for run, res in zip(v.setupNames, v.resultSetup):
                    self.plotConfussionMatrix(name, run, res['CM'])
           


parser = argparse.ArgumentParser()
parser.add_argument('--inputFolder', type=str, default='', help='Folder path with distance matrices')
parser.add_argument('--outputFolder', type=str, default='', help='Folder path for output')
parser.add_argument('--target', type=str, default='', help='Classification file with targets')
parser.add_argument('--query', type=str, default='', help='Classification file with queries')
parser.add_argument('--granularity', type=str, default='', help='Option to make analysis in all the dataset, a given class or a model. Allowed values: all, avg_class, class')
parser.add_argument('--idGranularity', type=int, default=-1, help='If granularity is class or model, this parameter specifies which to choose')
parser.add_argument('--numBinsRecall', type=int, default=10, help='Number of recall values in the precision recall plots')
parser.add_argument('--methodByMethod', action='store_true')

opt = parser.parse_args()

#Basic check of parameters
if not os.path.isdir(opt.inputFolder):
    sys.exit('Folder ' + opt.inputFolder + ' does not exist')

if not os.path.exists(opt.target):
    sys.exit('File ' + opt.target + ' does not exist')

if not os.path.exists(opt.query):
    sys.exit('File ' + opt.query + ' does not exist')

if opt.granularity != 'all' and opt.granularity != 'class' and opt.granularity != 'avg_class':
    sys.exit('Granularity must be: all, avg_class or class')

if opt.granularity == 'class':
    if opt.idGranularity is None:
        sys.exit('Granularity requires idGranularity to be set')


#Load information about the methods and distance matrices
eval = Evaluator()
eval.parseClassificationTarget(opt.target)
eval.parseClassificationQuery(opt.query)

exper = Experiment(opt.inputFolder, opt.outputFolder, eval, type=opt.granularity,value=opt.idGranularity, numBins=opt.numBinsRecall, reportMethods=opt.methodByMethod)

if opt.granularity == 'all':
    exper.generateConfussionMatrices()

with plt.style.context('seaborn-colorblind'):
    exper.defineStyles()
    if opt.methodByMethod:
        exper.generateRecallPrecisionPlotByMethod()
    else:
        exper.generateRecallPrecisionPlot()
    
exper.generateLatexTable()
exper.generateTextTable()
plt.close('all')
