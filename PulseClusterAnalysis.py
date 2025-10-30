import h5py
import numpy as np
from dtaidistance import dtw
import matplotlib.pyplot as plt

# Import PulseDB Subset
path = 'VitalDB_CalBased_Test_Subset.mat'

# -------- Helper Functions --------------------------------------------------------------------------------------------

# Compute correlation coefficent for two segments inputs
def correlation(x, y):
    return np.corrcoef(x,y)[0,1]

# Apply Z-Score Normalization onto an input segment
def normalize_segment(segment):
    return (segment - np.mean(segment)) / np.std(segment)

# Reduces size of input segment for calculations
def reduce_size(segment, factor):
    return segment[::factor]

# -------- 1. Divide-and-Conquer Clustering of Time Series Segments ----------------------------------------------------

# Calculate center most segment in cluster
def median_pivot_centroid(cluster):
    # Generate centroid of cluster
    centroid = np.mean(cluster, axis=0)
    corr = []

    # Compare each segment to centroid to find center most segment
    for segment in cluster:
        corr.append(correlation(segment, centroid))
    p_index = np.argmax(corr)

    # Return center most segment for use as pivot in split_cluster
    return cluster[p_index]

# Split an input cluster into two sub-clusters based on it's pivot correlation and an input threshold
def split_cluster(cluster, indices, threshold):
    # Select center most cluster segment to use as pivot
    pivot = median_pivot_centroid(cluster)
    cluster1, cluster2, idx1, idx2 = [], [], [], []

    # For each segment in the cluster check its correlation to the pivot
    for segment, i in zip(cluster, indices):
        # If correlation is over or equal to the threshold, place in sub-cluster 1
        if correlation(pivot, segment) >= threshold:
            cluster1.append(segment)
            idx1.append(i)
        # If correlation is less than the threshold, place in sub-cluster 2
        else:
            cluster2.append(segment)
            idx2.append(i)

    # Return new sub-clusters
    return (cluster1, idx1), (cluster2, idx2)

def cluster_segments(segments, indices, threshold=0.8, min_size=5):
    # Stop clustering if cluster is below min_size
    if len(segments) <= min_size:
        return [(segments, indices)]

    # Split current segments/cluster into 2 sub-clusters based on a given threshold
    #  - Clustering by if segment correlation to center-most segment is above/below threshold
    (c1,idx1), (c2, idx2) = split_cluster(segments, indices, threshold)

    # If split fails or if results are the same, stop splitting
    if len(c1) == len(segments) or len(c2) == len(segments) or not c1 or not c2:
        return [(segments,indices)]

    clusters = []
    # Recursive calls to continue to cluster new sub-clusters until segments can no longer be clustered
    clusters += cluster_segments(c1, idx1, threshold, min_size)
    clusters += cluster_segments(c2, idx2, threshold, min_size)

    # Return a list of the new clusters
    return clusters

# -------- 2. Closest Pair of Time Series Within Clusters --------------------------------------------------------------

# Find pair of sgements with a cluster most similar based on DTW distance
def closest_pair(cluster, indices):
    n = len(cluster)
    # If cluster consists of 1 point return empty pair
    if n < 2:
        return (None, None), -1

    # Place holder variables for comparing
    min_dist = float('inf')
    best_pair = (None, None)

    # Compare each segment to another using DTW distance
    for i in range(n):
        for j in range(i + 1, n):
            # Reduce clusters by a factor of 50 to drastically improve execution time
            cluster_i_reduced = reduce_size(cluster[i], 50)
            cluster_j_reduced = reduce_size(cluster[j], 50)
            # Check DTW distance between both points
            d = dtw.distance(cluster_i_reduced, cluster_j_reduced, window=250, use_c=True)

            # If new distance is less than the current, set current segments as new best pair and update the min_dist
            if d < min_dist:
                min_dist = d
                # Identifies segments on the their matching indices
                best_pair = (indices[i], indices[j])

    # Return calculated closest pair
    return best_pair, min_dist

# -------- 3. Apply Maximum Subarray (Kadane’s Algorithm) to Each Time Series ------------------------------------------

# Perform Kadane's algorithm to find the section of a segment with the largest cumulative increase
def kadane(segment):
    max_sum = float('-inf')
    current_sum = 0 # Tracks cumulative sum to find max
    start = 0 # Start of subarray
    end = 0 # End up subarray
    current_start = 0 # Tracks start of current sum

    # Check normalized values in a time-series to see if add to a cumulative sum or not
    s_index = 0
    for val in segment:
        current_sum += val
        # If current_sum is more than max_sum, current_sum becomes new max_sum
        if current_sum > max_sum:
            max_sum = current_sum
            start = current_start
            end = s_index
        # if current_sum goes negative, reset the sum to 0 and sum start to the next array value.
        if current_sum < 0:
            current_sum = 0
            current_start = s_index + 1
        s_index += 1

    # Return the maximum cumulative increase with the start & end index in the segment
    return max_sum, start, end

# Extract features from each segment in a cluster
def extract_kadane_features(cluster):
    features = []
    # For each segment, find largest cumulative increase
    for segment in cluster:
        max_sum, start, end = kadane(segment)
        features.append([max_sum, (start / len(segment)) * 10, (end / len(segment)) * 10])
    # Return array of cluster segments Kadane's results
    return np.array(features)

# -------- Step 4: Generate Reports and Visualizations -----------------------------------------------------------------

# Visualize n number of segments in a given array of clusters
def plot_cluster_examples(clusters, n):
    cmap = plt.get_cmap("tab20")

    # Controls segment plot colors
    colors = cmap(np.linspace(0, 1, 20))
    c = 0

    c_index = 0
    # Create a new plot for each cluster in array
    for (cluster, indices) in clusters:
        plt.figure(figsize=(16,8))
        plt.title(f'Cluster {c_index} - {len(cluster)} Segments')

        # Controls width for overlaying segments to help with visualization
        width = 1.2
        width_dec = 0.8 / (n)

        c_index += 1
        # Plot the first n segments in a cluster
        for segment in cluster[:n]:
            c += 1
            color = colors[c % 20]
            width -= width_dec
            plt.plot(segment, color=color, linewidth=width)

        # Display Plot
        plt.xlabel('Time Index')
        plt.ylabel('Normalized Value')
        plt.show()

# Visualize m number of segments in n number of clusters from array of clusters in one plot
def plot_all_clusters(clusters, n, m):
    plt.figure(figsize=(16,8))
    cmap = plt.get_cmap("tab20")
    colors = cmap(np.linspace(0, 1, len(clusters)))

    # Controls width for overlaying segments to help with visualization
    width = 1.2
    width_dec = 0.8 / (m * n)

    # Storing the mean segment of each segment
    mean_segments = []

    i = 0
    # Iterate through first n clusters in given clusters array
    for (cluster, indices) in clusters[:n]:
        i += 1
        color = colors[i % 20]
        width -= width_dec
        # Plot first m segments of current cluster
        for segment in cluster[:m]:
            width -= width_dec
            plt.plot(segment, color=color,linewidth=width, alpha=0.1)

        # Calculate and add mean segment of mean_segments
        mean_segment = np.mean(cluster, axis=0)
        mean_segments.append(mean_segment)

    # Find the mean of segments mean of each cluster and overlay on top
    mean_of_clusters = np.mean(mean_segments, axis=0)
    plt.plot(mean_of_clusters, color='purple', label="Mean of Clusters", linewidth=0.3)

    # Display Plot
    plt.title(f'Overlay of First {m} ABP Clusters')
    plt.xlabel('Time Index')
    plt.ylabel('Normalized Value')
    plt.legend()
    plt.show()

# Visualize the closest pair of an input cluster
def plot_closest_pair(cluster, indices):
    plt.figure(figsize=(16,8))

    # Find closest pair for given cluster
    pair, _ = closest_pair(cluster, indices)
    i, j = pair
    label1 = f'Segment {i}'
    label2 = f'Segment {j}'

    # Plot the closest pairs
    plt.plot(cluster[indices.index(i)], label='Segment Overlap', color='purple', linewidth=0.9)
    plt.plot(cluster[indices.index(i)], label=label1, color='red')
    plt.plot(cluster[indices.index(j)], label=label2, color='blue' )

    # Overlay first segment again with alpha to create purple representing the overlap
    plt.plot(cluster[indices.index(i)], color='red', linewidth=0.5, alpha=0.5)

    # Display plot
    plt.title(f'Closest Pair in Cluster - Seg: {i} & Seg: {j}')
    plt.xlabel('Time Index')
    plt.ylabel('Normalized Value')
    plt.legend()
    plt.show()

# Visualize results of kadane's algorithm on segment
def plot_kadane_interval(segment):
    # Calculate Kadane's algorithm for a given segment
    _, start, end = kadane(segment)

    # Plot segment and highlight subarray
    plt.figure(figsize=(16, 8))
    plt.plot(segment, label='Segment', color='blue', linewidth=0.5, alpha=0.5)
    plt.axvspan(start, end, color='red', alpha=0.5, label='Max Subarray Interval', linewidth=0.2)

    # Display plot
    plt.title('Kadane Max Interval Highlight')
    plt.xlabel('Time Index')
    plt.ylabel('Normalized Value')
    plt.legend()
    plt.show()

# -------- Toy Example -------------------------------------------------------------------------------------------------

# Toy Example to Verify Algorithms
def toy_example():
    print("\n---- Toy Example Verification ----")

    # Simple sample time-series
    segments= [
        np.array([1, 2, 3, 3, 8]),
        np.array([2, 2, 4 ,2 ,6]),
        np.array([-1, -2, -8, -4, -5]),
        np.array([-2, -1, -4, -5, -6])
    ]
    indices = [0, 1, 2, 3]

    # Normalize segments
    segments = [normalize_segment(segment) for segment in segments]

    # Testing Divide-and-Conquer Clustering
    print("\n---- 1. Divide-and-Conquer Clustering ----")
    clusters = cluster_segments(segments, indices, threshold=0.9, min_size=2)
    print(" - Num of Clusters:", len(clusters))

    # Testing Closest Pair
    print("\n---- 2. Closest Pair for Clusters ----")
    c_index = 0
    for (cluster, index) in clusters:
        # Find closest pair for each toy cluster
        c_index += 1
        pair, dist = closest_pair(cluster, index)
        print(f" - Cluster {c_index}: Closest pair {pair} with DTW distance {dist:.3f}")

    # Testing Kadane's Algorithm
    print("\n---- 3. Kadane's Algorithm ----")
    test_segment = np.array([-2, 3, 5, -1, 4, -5])
    max_sum, start, end = kadane(test_segment)
    print(" - Input:", test_segment)
    print(f" - Max Subarray Sum: {max_sum}, Start: {start}, End: {end}")

    # Test Visualizations
    plot_cluster_examples(clusters, 2)

    print("\n")

# -------- Main Method -------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    print()

    # Run Toy Example
    toy_example()

    # Divider for main execution
    print("--------------------------------------------------------------------------------------------------------\n")

    # Read PulseDB subset from it's file location
    with h5py.File(path, 'r') as f:

        # Extract first 1000 segments from PulseDB subset
        subset = f['Subset']
        signals = subset['Signals'][:1000]
        num_segments_raw = signals.shape[0]

        print("---- 1. Divide-and-Conquer Clustering of Time Series Segments ----")
        print("\n\tClustering by ABP")
        print("\t----------------------------------------------------")

        # Extract ABP Signals & Z-Score Normalize Them
        abp_all = np.array(signals[:, 2, :])
        abp_all = np.array([normalize_segment(segment) for segment in abp_all])

        # Define Threshold and Minimum Size for Clustering
        abp_threshold = 0.8
        abp_min_size = 5

        # Generate Indices for Segments
        segment_indices = list(range(len(abp_all)))

        # Cluster Segments Based on Threshold and Minimum Size
        abp_clusters = cluster_segments(abp_all, segment_indices, abp_threshold, abp_min_size)
        print("\t - Number of ABP Clusters:", len(abp_clusters))

        print("\n---- 2. Closest Pair of Time Series Within Clusters ----")
        print("\n\tComputing Closest Pair for Clusters")
        print("\t----------------------------------------------------")

        # Calculate Closest Pairs for Each Cluster
        closest_pairs = []
        c_index = 0
        for (cluster, indices) in abp_clusters:
            c_index += 1
            pair, dist = closest_pair(cluster, indices)
            closest_pairs.append(pair)
            print(f"\t - Cluster {c_index}: Closest pair (Segment {pair[0]}, Segment {pair[1]}) with a distance of {dist:.3f}")


        print("\n---- 3. Apply Maximum Subarray (Kadane’s Algorithm) to Each Time Series ----")
        print("\n\tKadane's Algorithm")
        print("\t----------------------------------------------------")

        # Run Kadane's Algorithm on All Segments
        kadane_features = extract_kadane_features(abp_all)
        print("\n\t - Kadane Features Shape:", kadane_features.shape)

        print("\n\tCluster Reasoning")
        print("\t----------------------------------------------------")

        # Outputs Kadane Features for Each Cluster to Show the Cluster Reasoning via the Average Features
        c_index = 0
        for (cluster, indices) in abp_clusters:
            c_index += 1
            cluster_features = kadane_features[indices]  # extract features of segments in cluster
            avg_max_sum = np.mean(cluster_features[:, 0])
            avg_start = np.mean(cluster_features[:, 1])
            avg_end = np.mean(cluster_features[:, 2])
            print(f"\t - Cluster {c_index}: avg_max_sum={avg_max_sum:.3f}, avg_start={avg_start:.2f}s, avg_end={avg_end:.2f}s")

        print("\n\tHighlighted Events")
        print("\t----------------------------------------------------")

        # Find Significant Events in 95th Percentile of the Kadane Features
        event_threshold = np.percentile(kadane_features[:, 0], 95)
        event_indices = np.where(kadane_features[:, 0] >= event_threshold)[0]
        event_features = kadane_features[event_indices]

        # Output Found Significant Events
        print("\tFound ", len(event_indices), "Significant Events:")
        for e_index, feature in zip(event_indices, event_features):
            print(f"\t - Segment {e_index}: max_change={feature[0]:.3f}, start={feature[1]:.2f}s, end={feature[2]:.2f}s")

        print("\n---- Step 4: Generate Reports and Visualizations ----")
        print("\n\tVisualizing First 3 Clusters Individually")
        print("\t----------------------------------------------------")

        # Visualize 3 Different Clusters
        visual_clusters = abp_clusters[:3]
        plot_cluster_examples(visual_clusters, 7)

        print("\n\tVisualizing First 10 Clusters Together")
        print("\t----------------------------------------------------")

        # Visualize 10 Clusters Overlayed
        visual_clusters = abp_clusters[:10]
        plot_all_clusters(visual_clusters, 5, 5)

        print("\n\tVisualizing 3 Sets of  Closest Pair")
        print("\t----------------------------------------------------")

        # Visualize 3 Different Closest Pairs
        v_cluster, v_indices = abp_clusters[20]
        plot_closest_pair(v_cluster, v_indices)
        v_cluster, v_indices = abp_clusters[45]
        plot_closest_pair(v_cluster, v_indices)
        v_cluster, v_indices = abp_clusters[1]
        plot_closest_pair(v_cluster, v_indices)

        print("\n\tVisualizing Maximum Subarray Intervals for 3 Segments")
        print("\t----------------------------------------------------")

        # Visualize 3 Different Maximum Subarray Intervals
        seg1 = abp_all[10]
        plot_kadane_interval(seg1)
        seg2 = abp_all[563]
        plot_kadane_interval(seg2)
        seg3 = abp_all[986]
        plot_kadane_interval(seg3)

