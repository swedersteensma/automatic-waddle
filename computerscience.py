import numpy as np
import pandas as pd
import math as mt
import os
import json
import string
from random import shuffle
from collections import Counter
from itertools import combinations
import matplotlib.pyplot as plt 

os.chdir("/Users/swedersteensma/Desktop/Computer_Science")

#%% Load in the Data
#open and load the dataset
with open('TVs-all-merged.json') as f:
    data = json.load(f)

#%% Cleaning of the data 
def removecapitals(x):
    result = x.lower()
    return result   

def replaceinches(x):
    result = x.replace('"', "inch") and x.replace(' "', "inch") and x.replace('inches', 'inch') and x.replace(' inch', 'inch') and x.replace('-inch', 'inch') and x.replace('hertz', 'hz') and x.replace('-hz', 'hz') and x.replace(' hz', 'hz') and x.replace(' lbs', 'lbs')   
    return result

def removeinterpunction(x):
    result = x.translate(str.maketrans("", "", string.punctuation))
    return result
    
def removeWords(x, deletewords):
    stringList = x.split()
    result = list(set(stringList)-set(deletewords)) 
    return(result)

def cleanstring(x) :
    result1 = removecapitals(x) 
    result2 = replaceinches(result1)
    result3 = removeinterpunction(result2) 
    deletewords = ['newegg', 'neweggcom','amazon', 'amazoncom', 'bestbuycom', 'best', 'buy', 'bestbuy', 'com', 'tv', 'the']
    result4 = removeWords(result3, deletewords) 
    return result4

#%% Extracting the model words 
def alphanum(x):
    toberemoved = []
    for word in x:
        if (any(chr.isalpha() for chr in word) and any(chr.isdigit() for chr in word)) == False:
            toberemoved.append(word)
    
    for word in toberemoved: 
        x.remove(word)
    
    return x

#%% Transforming data into dataframe and applying data cleaning 
# pre processing: http://ethen8181.github.io/machine-learning/clustering_old/text_similarity/text_similarity.html
DataFrame = pd.DataFrame()
keys  = []
shop = []
title = [] #let op: hierna wordt deze gecombineerd met de featuresMap
featuresMap = []

for key in data.keys():
    for i in range(len(data[key])):
        keys.append(key)
        title.append(cleanstring(data[key][i]['title']))
        shop.append(data[key][i]['shop'])
        featlist = []
        values = [cleanstring(item)[0] for item in data[key][i]['featuresMap'].values()]
        for value in values:
            featlist.append(value)
        featuresMap.append(featlist)

#%% Extract the brands from the featuresmap (WIP) 
contains = 'Brand'
brandsfeatures = []
for key in data.keys():
    for i in range(len(data[key])):
            if (contains in data[key][i]['featuresMap'].keys()) == True:
                brandsfeatures.append(data[key][i]['featuresMap']['Brand'])
            else: 
                brandsfeatures.append(0)

#%% Adding brands from title and featuresMap to the list 
alltvbrands = list(Counter(brandsfeatures))
alltvbrands.remove(0)
alltvbrandsclean = []

for i in range(len(alltvbrands)):
    alltvbrandsclean.append(removecapitals(alltvbrands[i]))
                   
titlebrands = {}
for i in range(len(title)):
    titlebrands[i] = 0
    tit = title[i]
    for brand in alltvbrandsclean:
        if brand in tit:
            titlebrands[i] = brand
            break

titlebrandslist = list(titlebrands.values())
brandscombined = []

for i in range(len(titlebrandslist)): 
    if titlebrandslist[i] != 0:
        brandscombined.append(titlebrandslist[i])
    else: 
        brandscombined.append(brandsfeatures[i])


#%% Combine the titles and features into one list and remove all non letter/number words
combinedtitlefeature = []
for i in range(len(title)):
    combinedtitlefeature.append(title[i]+featuresMap[i])


titlewithmodelwords =[]
for i in range(len(combinedtitlefeature)):
    results = alphanum(combinedtitlefeature[i])
    results = list(dict.fromkeys(results))
    titlewithmodelwords.append(results)    

#%% Adding brands to the cleaned titlewithmodelwords
for i in range(len(titlewithmodelwords)):
    titlewithmodelwords[i].append(brandscombined[i])

title = titlewithmodelwords
    

#%% Create Dataframe 
DataFrame['keys'] = keys
DataFrame['shop'] = shop 
DataFrame['title'] = title
#DataFrame['featuresMap'] = featuresMap


                
          
#%% Count words in title 
counttitle = pd.Series(Counter([y for x in DataFrame['title'] for y in x]))
countdf = counttitle.to_frame()
countdf.columns = ["frequency"]

#%%
upperthreshold = 1000
thresholdmet=[]
for ind in countdf.index:
    if countdf["frequency"][ind] > upperthreshold or countdf["frequency"][ind] == 1:
        thresholdmet.append(ind)
        
#%% Removing words that meet the threshold conditions 
cleaned_list = pd.Series.tolist(DataFrame['title'])
delete = thresholdmet

for i in range(len(cleaned_list)):
    for word in cleaned_list[i]:
        if word in delete:
            cleaned_list[i].remove(word)                    

DataFrame['title'] = cleaned_list            
       
# might want to add feature words according to same principles: modelwords + brandname  
# add all modelwords from feature words + add brand name for each 
#%% Prepare Data for Shingling
single_words_list = [] #11699

for i in range(len(cleaned_list)):
    for word in cleaned_list[i]:
        single_words_list.append(word)
    
# remove duplicates 
removedup = []
for i in single_words_list:
    if i not in removedup:
        removedup.append(i) 
   

#%% Creating Binary Matrix
N = len(removedup)
M = len(cleaned_list)
binary_matrix = np.zeros((N,M))

for i in range(M):
    binary_vector = np.zeros(len(removedup))
    for word in cleaned_list[i]:
        if word in removedup: 
            index = removedup.index(word) 
            binary_vector[index] = 1
    binary_matrix[:,i] = binary_vector
         
        
#%% Shuffle indices randomly
def hashing(binary_vector):
    createlist = list(range(len(binary_vector)))
    shuffle(createlist)
    return createlist

def adding_hashes(binary_vector, iterations):
    # function for building multiple minhash vectors
    permutedindices = []
    for i in range(iterations):
        permutedindices.append(hashing(binary_vector))
    return permutedindices

# iterations = round(len(binary_matrix)/2) 
iterations = 1000 
indices_shuffeld = np.array(adding_hashes(binary_vector, iterations))
indices_shuffeld = np.transpose(indices_shuffeld)

#%% Signature matrix 
signature_matrix = np.zeros((iterations, len(binary_matrix[0])))

for k in range(iterations): 
    for j in range(len(binary_matrix[0])):
        for i in range(len(indices_shuffeld)): 
            index = np.where(indices_shuffeld[:,k] == i)[0][0] 
            if binary_matrix[index, j] == 1:
                signature_matrix[k,j] = indices_shuffeld[index][k]
                break
            
#%% predefining required functions for LSH 
iterations1 = iterations 
diffBandsResults = pd.DataFrame()
listrecall = []
listprecision = []
listrecallLSH = []
listprecisionLSH = []
listcandidatepairs = []
listF1 = []
listF1_LSH = []

#%%
#for j in range(6,20):
#2 moet naar 20 en iterations moet naar wat erboven staat, range moet naar wat hierboven staat

#bands1 = [10, 15, 20, 25, 30, 35, 40, 45, 48, 50, 52, 55, 60, 75, 100, 110, 105, 115, 125, 150, 160, 175, 190, 200, 210, 225, 230, 250, 275, 300, 310, 325, 350, 375, 400, 425, 450, 475, 490, 500, 525, 550, 575, 600, 625, 650, 675, 700, 725, 750, 775, 800]
bands1 = [10, 20, 25, 40, 50, 100, 125, 200, 200, 200, 200, 250, 250, 250,250,250, 500,500,500,500]

for b in bands1:
    print(b)
    def numberofbands(iterations, b):
        while iterations%b != 0:
            b -= 1 
        return round(b)
    
    bands = numberofbands(iterations, b)
    
    # split 
    def bandsSplit(signature_vector, bands):
        assert len(signature_vector) % bands == 0 
        # assert False, "It is not possible to equally divide this vector into this amount of bands"
        split = np.array_split(signature_vector, bands)
        split = np.array(split)
        return np.transpose(split)
    #make k buckets as large as possible see slides 
    #adjust code to use the elements of a vector after each other as the hessian function 
    
    def hashFunction(splittedVector):
        hashvalue = ''
        for i in range(len(splittedVector)):
            hashvalue += str(round(splittedVector[i]))
        return hashvalue
    
    #def assignToBucket(hashvalue):
        
        
    #%% Looping over the whole signature_matrix
    assignedBuckets = {}
    amountshuffles = 10 
    # reduce possibility of FN --> als er een blokje anders is in je band komt het nu in een andere bucket
    # dat is zonde, door de bands the shuffelen is de kans groter dat er een match komt 
    
    for k in range(amountshuffles):
        np.random.shuffle(signature_matrix)
      
        for i in range(len(signature_matrix[0])):
            split = bandsSplit(signature_matrix[:,i], bands)
            for j in range(bands):
                hashvalue = hashFunction(split[:,j])
                if hashvalue in assignedBuckets: 
                    assignedBuckets[hashvalue] += [i] 
                else:
                    assignedBuckets[hashvalue] = [i]
    
    #%% Delete all zero keys 
    # for key in assignedBuckets.keys():    
    #     if len(assignedBuckets[key]) == 1:
    #         assignedBuckets.pop(key, None)
    
    zeros = round(len(signature_matrix)/bands)
    keydelete = ''
    for i in range(zeros):
        keydelete += str(0)
    
    if keydelete in assignedBuckets: 
        del assignedBuckets[keydelete]
        
    #%% Delete all single bucket keys from the dictonary as these do not represent a candidate pair 
    removekeys = []
    for key in assignedBuckets:
        if len(assignedBuckets[key]) ==1:
            removekeys.append(key)
            
    for key in removekeys:
        del assignedBuckets[key]
    
    #%% Create new dictonary with all possible candidate pairs resulting from LSH 
    candidatepairs = {}
    
    for key in assignedBuckets:
        pairs =list(combinations(assignedBuckets[key],2))
        for i in range(len(pairs)):
            if pairs[i][0] > pairs[i][1]:
                pairs[i] = (pairs[i][1],pairs[i][0])
        candidatepairs[pairs[i]] = 1
    
    candidatepairs = list(candidatepairs)
    listcandidatepairs.append(len(candidatepairs))
    
    
    #%% Jaccard Similarity to evaluate pairs 
    sim = np.zeros(len(candidatepairs))
    
    def jac_sim(binvector1, binvector2):
        intersection = np.logical_and(binvector1, binvector2)
        union = np.logical_or(binvector1, binvector2)
        sim = intersection.sum() / union.sum()
        return sim
    
    
    for i in range(len(candidatepairs)):
        binvector1 = binary_matrix[:, candidatepairs[i][0]]
        binvector2 = binary_matrix[:, candidatepairs[i][1]]
        sim[i] = jac_sim(binvector1, binvector2)
    
    
    #%% Performance evaluation     
    pairsmeetingthreshold = []
    threshold = 0.7
    
    for i in range(len(sim)):
        if sim[i] >= threshold:
            pairsmeetingthreshold.append(1)
        else:
            pairsmeetingthreshold.append(0)
    
    #compute the amount of pairs found 
    numberpairs = sum(pairsmeetingthreshold)
    totalcomparisons = len(pairsmeetingthreshold)
    
    #%% Calculating True Positives 
    chosenpairs = []
    for i in range(len(pairsmeetingthreshold)):
        if pairsmeetingthreshold[i] == 1:
            chosenpairs.append(candidatepairs[i])
    
    listoftruepositives = []
    for i in list(DataFrame.index):
        for j in list(DataFrame.index):
            if i>=j:
                continue 
            else:
                if DataFrame['keys'][i] == DataFrame['keys'][j]:
                    pair = tuple([i,j])
                    listoftruepositives.append(pair)
                    
    #%% Calculating F score
    truepositives = len(listoftruepositives) - len((set(listoftruepositives) - set(chosenpairs)))                                         
    falsepositives = numberpairs - truepositives #wij zeggen: horen bij elkaar, maar dat is niet zo 
    counttotalpairs = len(listoftruepositives)
    falsenegatives = counttotalpairs - truepositives #I have not found these
    
    precision = truepositives/(truepositives+falsepositives)
    listprecision.append(precision)
    recall = truepositives/(falsenegatives+truepositives)
    listrecall.append(recall)
    
    F1 = 2*precision*recall/(precision+recall) 
    listF1.append(F1)
    #%% F1score after LSH
    truepositivesLSH = len(listoftruepositives) - len((set(listoftruepositives) - set(candidatepairs))) 
    falsepositivesLSH = len(candidatepairs) - truepositivesLSH
    falsenegativesLSH = counttotalpairs - truepositivesLSH
    
    precisionLSH = truepositivesLSH/(truepositivesLSH+falsepositivesLSH) #dichtheid is niet hoog: hoeveel goed
    listprecisionLSH.append(precisionLSH)
    recallLSH = truepositivesLSH/(falsenegativesLSH+truepositivesLSH) #zoveel pairs vinden we
    listrecallLSH.append(recallLSH)
    
    F1_LSH = 2*precisionLSH*recallLSH/(precisionLSH+recallLSH)  
    listF1_LSH.append(F1_LSH)

#%%
print("The amount of pairs found is " + str(numberpairs)) 
print("Out of " + str(totalcomparisons) + " possible pairs")
print('The amount of true positives equals ' + str(truepositives))
print('The amount of false positives equals ' + str(falsepositives))

print('The F1 score after Jaccard equals ' + str(F1))  
print('The F1 score after LSH equals ' + str(F1_LSH))   

#F1, hoeveel wordt eruit gefilterd, accuracy? 

#%%
totalpossiblepairs = 1624*1623/2 
diffBandsResults['F1'] = listF1
diffBandsResults['F1_LSH'] = listF1_LSH 
diffBandsResults['precision'] = listprecision
diffBandsResults['recall'] = listrecall 
diffBandsResults['precisionLSH'] = listprecisionLSH
diffBandsResults['recallLSH'] = listrecallLSH 
fraction = [pair / totalpossiblepairs for pair in listcandidatepairs]
#fraction = [len(listcandidatepairs) / totalpossiblepairs for totalpossiblepairs in listcandidatepairs]
diffBandsResults['fraction'] = fraction

#%%
diffBandsResults.plot.scatter(x = 'fraction', y = 'recallLSH', color = 'blue', label = 'complete')
plt.grid()

diffBandsResults.plot.scatter(x = 'fraction', y = 'precisionLSH', color = 'red')
plt.grid()

# diffBandsResults.plot.scatter(x = 'fraction', y = 'recall')
# plt.grid()

# diffBandsResults.plot.scatter(x = 'fraction', y = 'precision')
# plt.grid()

diffBandsResults.plot.scatter(x = 'fraction', y = 'F1')
plt.grid()

# diffBandsResults.plot.scatter(x = 'fraction', y = 'F1_LSH')
# plt.grid()

#meer FN dan FP 
#%%
diffBandsResults_without = pd.read_csv('test')
diffBandsResults_without.plot(kind = 'scatter', x = 'fraction', y = 'recallLSH', color = 'red', label = 'reduced')


 