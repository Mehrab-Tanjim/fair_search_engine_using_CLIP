import numpy as np
import torch 

class SearchEngine(object):
    def __init__(self, relevance_scores, demography_scores) -> None:
        super().__init__()

        self.relevance_scores = relevance_scores
        self.demography_scores = demography_scores

    def base_engine(self):

        return torch.argsort(torch.Tensor(self.relevance_scores.flatten()), descending=True).numpy()


    def fmmr(self, image_item_set: set, k: int, lamda: float):

        reranked_results = []
       
        for j in range(k):
            # image_item_set = list(set(image_item_set).difference(set(reranked_results))) #reordered everytime

            repeated_scores = np.repeat(np.expand_dims(self.demography_scores[reranked_results], axis=1), len(image_item_set), axis=1) # num_item_added x num_item_remaining x 2 
            
            diversity_scores =  np.sum(-abs(repeated_scores - self.demography_scores[image_item_set]), axis=-1) if reranked_results else 0
            i_j = np.argmax(lamda  * self.relevance_scores[image_item_set] - (1-lamda) * np.max(diversity_scores, axis=0))
                        # num_item_added x num_item_remaining x 2 - num_item_remaining x 2 -> # num_item_added x num_item_remaining x 1 - max(1 x num_item_remaining x 1)

            reranked_results.append(image_item_set[i_j])

            del image_item_set[i_j]

        return reranked_results
