import numpy as np
# when to use class
# If you need to store data and manipulate it over time, a class encapsulates everything cleanly

class VectorNormalization:
    def __init__(self, xvect):
        self.x = np.asarray(xvect, dtype=np.float64)

    def max_min_norm(self):
        xmax = np.max(self.x, axis=0)
        xmin = np.min(self.x, axis=0)
        diff = xmax - xmin
        # if diff == 0:
        #     diff = 1.0
        diff = np.where(diff==0.0, 1.0, diff)
        x_norm = (self.x - xmin) / diff # what will happen if diff is 0
        # basically if x is an identical matrix then xmax and xmin are going to be same.
        # so make it 1 if diff is 0 then all the values of identical matrix will be 1.
        return x_norm
    
    def zscore_norm(self):
        x_mean = np.mean(self.x, axis=0)
        x_std = np.std(self.x, axis=0)
        x_std = np.where(x_std == 0.0, 1.0, x_std)
        z_norm = (self.x - x_mean) / x_std
        return z_norm
    
    def count_zero_pixels(self):
        count = len(np.where(self.x == 0)[0])
        return count

## formula
#x_norm = (x - xmin)/ (xmax - xmin)
# Type hinting
# what is the type for input. It is a numpy array
# what is the type for return value? again a numpy array np.ndarray

def norm_matrix(x:np.ndarray)->np.ndarray:
    xmax = np.max(x, axis=0)
    xmin = np.min(x, axis=0)
    diff = xmax - xmin
    # if diff == 0:
    #     diff = 1.0
    diff[diff==0] = 1
    x_norm = (x - xmin) / diff # what will happen if diff is 0
    # basically if x is an identical matrix then xmax and xmin are going to be same.
    # so make it 1 if diff is 0 then all the values of identical matrix will be 1.
    return x_norm

def z_norm(x:np.ndarray)-> np.ndarray:
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    z_norm = (x - x_mean) / x_std
    return z_norm


def main():
    feat_vect = [[[1, 0], [10, 20]], [[2, 5], [10, 0]], [[12, 1]]]
    feat_norm = [norm_matrix(f) for f in feat_vect if len(f) > 1] # list comprehension
    print(feat_norm)
    feat_znorm = [z_norm(f) for f in feat_vect if len(f)> 1]
    print(feat_znorm)
    feat_vect = [f for f in feat_vect if len(f) == 2]
    vect_norm = VectorNormalization(feat_vect)
    print(vect_norm.max_min_norm())
    print(vect_norm.zscore_norm())
    print(vect_norm.count_zero_pixels())

if __name__ == "__main__":
    main()