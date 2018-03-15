import tensorflow as tf
import math, numpy as np
import matplotlib.pyplot as plt


def plot_data(centroids, data, n_samples):
    colour = plt.cm.rainbow(np.linspace(0, 1, len(centroids)))
    for i, centroid in enumerate(centroids):
        samples = data[n_samples*i: n_samples*(i+1)]
        plt.scatter(samples[:,0], samples[:,1], c=colour[i], s=1)
        plt.plot(centroid[0], centroid[1], marker='x', markersize=10, color='k', mew=5)
        plt.plot(centroid[0], centroid[1], marker='x', markersize=5, color='r', mew=2)
        

def all_distances(a, b):
    diff = tf.squared_difference(tf.expand_dims(a,0), tf.expand_dims(b,1))
    diff_sum = tf.reduce_sum(diff, axis=2)
    return diff_sum
        
    
class Kmeans(object):
    
    def __init__(self, data, n_clusters):
        self.n_data, self.n_dim = data.shape
        self.n_clusters = n_clusters
        self.data = data
        self.v_data = tf.Variable(data)
        self.n_samples = self.n_data // n_clusters
    
    def run(self):
        tf.global_variables_initializer().run()
        initial_centroids = self.find_initial_centroids(self.n_clusters).eval()
        
        current_centroids = tf.Variable(initial_centroids)
        #tf.global_variables_initializer().run()
        nearest_indices = self.assign_nearest_indices(current_centroids)
        updated_centroids = self.update_centroids(nearest_indices)
        # Begin main algorithm
        tf.global_variables_initializer().run()
        c = initial_centroids
        for i in range(10):
            c2 = current_centroids.assign(updated_centroids).eval()
            if np.allclose(c, c2): break
            c = c2
        return c2
    
    
    def find_initial_centroids(self, k):
        # first time, randomly assign first centroid
        arr_index = tf.random_uniform([1], minval=0, maxval=k, dtype=tf.int32)
        arr = tf.expand_dims(self.v_data[tf.squeeze(arr_index)], dim=0) #shape: (1, 2)
        
        initial_centroids = []
        for i in range(k):
            # count all distance with centroids
            dist= all_distances(self.v_data, arr) 
            # choose the closest distance with any centroid
            dist_min=tf.reduce_min(dist, 0)
            # pick the index of the farthest distance (with any centroid)
            farthest_index = tf.argmax(dist_min, axis=0)
            # pick the fartheset point by index
            farthest_point = self.v_data[farthest_index]
            # add farthest_point into initial_centroids arrays
            initial_centroids.append(farthest_point)
            arr = tf.stack(initial_centroids)
        return arr
    
    
    def assign_nearest_indices(self, v_centroids):
        return tf.argmin(all_distances(self.v_data, v_centroids), axis=0)
    
    
    def update_centroids(self, nearest_indices):
        partitions = tf.dynamic_partition(data=self.v_data, 
                                          partitions=tf.to_int32(nearest_indices), 
                                          num_partitions=self.n_clusters)
        return tf.concat([tf.expand_dims(tf.reduce_mean(p, 0), dim=0) for p in partitions], axis=0)


    
    
    