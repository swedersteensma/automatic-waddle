# automatic-waddle
Automated Duplicate Detection through Product Titles and Features using LSH

In this project a duplicate detection algorithm is programmed to find duplicate pairs in a dataset of TV's.
This helps users to identify similar pairs in order to better compare different products. 
Computation time is diminished by applying LSH to find possible candidate pairs, the correctness of these pairs is then evaluated through Jaccard Similarity. 

In the first part of the code the data is cleaned. Different functions are used for that. 
Afterwords, the words are selected which are to be used for the LSH step. In this case, words containing alphabetic/numeric characters are selected as well as the brands.  

Users can adjust the modelwords selected and the cleaning procedure, by calling or not calling in the functions made and by adjusting threshold values. 
After the eventual model words list is created, this is used to preform LSH and Jaccard Similarity and to get a performance of the detection algorithm using an F1-score. 
