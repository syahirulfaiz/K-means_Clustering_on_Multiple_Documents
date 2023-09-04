# K-means_Clustering_on_Multiple_Documents

Kmeans Clustering on Multiple Documents (Word Embedding) 

The task is to cluster words to 4 categories: animals, countries, fruits and veggies. The words are splitted in four different files. The first column in each line is a word followed by 300 features (word embedding) describing the meaning of that word.

The task compares the performance, on various distance measure

### Eucledian Distance ###

$$eucledian\\_ distance(x_1,x_2)=\sqrt{\Sigma_{i=1}^N(x_1^{(i)}-x_2^{(i)})^2}$$

### Manhattan Distance ###

$$manhattan\\_ distance(x_1,x_2)=\Sigma_{i=1}^N\mid x_1^{(i)}-x_2^{(i)}\mid$$


### Cosine Similarity Distance ###

$$cosine\\_ similarity(x_1,x_2)=\frac{x_1^\intercal x_2}{||x_1||||x_2||}$$

Where 
$$x_1^\intercal x_2 = \Sigma_{i=1}^N x_1^{(i)}x_2^{(i)}$$
and 
$$||x|| = \sqrt{\Sigma_{i=1}^N x_{(i)}^2}$$


### Result ###

we consider F-score as our primary evaluation measurement. We order them by f-score, then Precision, and then Recall.  The best setting for each setting, as shown in the table below:

|**Distance Measurement**|$L_2$ ***normalisation?***|**k**|**precision**|**recall**|**f-score**|
| - | - | - | - | - | - |
|Euclidean Distance|No|4|0\.909148|0\.91003|0\.909589|
|Euclidean Distance|Yes|2|0\.651405|1|0\.78891|
|Manhattan Distance|No|9|0\.98674|0\.827875|0\.900354|
|Manhattan Distance|Yes|4|0\.972262|0\.971874|0\.972068|
|Cosine Similarity|No|2|0\.651404787|1|0\.78891|

*Table: Comparison Table of Precision, Recall, and F-Score among several settings*

As we can see, the best setting for this experiment is using the Manhattan Distance measure, with the $L_2$ normalisation, in k=4.
