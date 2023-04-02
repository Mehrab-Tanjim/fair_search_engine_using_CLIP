# Fair Image Search Engine using CLIP


This is a search engine using CLIP image and text embedding. The basic mechanism for image search is simple: calculate the cosine similarity scores between CLIP embedding of the text query and images in the database.

There are two modes of image search.

1. Base Search: In this mode, we just calculate the cosine similarity and sort from highest to lowest to show the most relevant results on the top.

2. Fair Search: In the base mode, the top results may not show diversity. To rectify that, the Algorithm 2 from the paper is implemented: "Using Image Fairness Representations in Diversity-Based Re-ranking for Recommendations". The main idea of the algorithm is to penalize the relevance score if the top retreived results does not maximize the diversity for the chosen demographics.


### Results:

These are the search results for the query "Faces with eyeglasses". For the image repository, I used CelebA-HQ.

#### Before Search:
<img  src="https://github.com/Mehrab-Tanjim/fair_search_engine_using_CLIP/blob/master/results/plots/visualization%20of%20images%20before%20search.jpg" height="400"> 


#### Base Search:
<img  src="https://github.com/Mehrab-Tanjim/fair_search_engine_using_CLIP/blob/master/results/plots/visualization%20of%20images%20after%20base%20search.jpg" height="400">

#### Fair Search:
<img  src="https://github.com/Mehrab-Tanjim/fair_search_engine_using_CLIP/blob/master/results/plots/visualization%20of%20images%20after%20fmmr%20search.jpg" height="400">
