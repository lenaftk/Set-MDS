from set_mds_fast import (
    distance_matrix
)

def set_mds(d_goal,
            n_components=2,
            ):
    print(d_goal)
    return -1

class SetMDS():
    def __init__(self,
                 n_components=2,
                ):
        self.n_components = n_components


    @timemethod
    def fit_transform(self, X, init=None):
        X = X.astype(np.float64)    ##o X einai tipou float64
        X = check_array(X)          # elegxos gia na einai ola ta stoixeia tou X coble( oxi Inf)
        d_goal = (X if self.dissimilarity == 'precomputed'  #X=X an exw idi to X upologismeno
                  else distance_matrix(X))   # alliws upologise to distance matrix tou X

    #TREXEI PSMDS
        self.embedding_ = set_mds(
            d_goal,
            init=init,
            n_components=self.n_components
        )
        return self.embedding_

    def fit(self, X, init=None):   ##
        self.fit_transform(X, init=init)
        return self
