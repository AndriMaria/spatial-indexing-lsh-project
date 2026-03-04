import nltk
nltk.download('stopwords')
nltk.download('wordnet')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy.spatial import KDTree
from sklearn.feature_extraction.text import TfidfVectorizer
from datasketch import MinHash, MinHashLSH
from rtree import index

# 1) Φόρτωση και Προεπεξεργασία Δεδομένων

file_path = r'C:...'
data = pd.read_csv(file_path).dropna()
data.reset_index(drop=True, inplace=True)
N = len(data)

# Κωδικοποίηση κατηγορηματικών στηλών και κανονικοποίηση αριθμητικών
data['loc_country_encoded'] = data['loc_country'].astype('category').cat.codes
data['roast_encoded'] = data['roast'].astype('category').cat.codes
data['100g_USD_normalized'] = (data['100g_USD'] - data['100g_USD'].min()) / (data['100g_USD'].max() - data['100g_USD'].min())
data['rating_normalized'] = (data['rating'] - data['rating'].min()) / (data['rating'].max() - data['rating'].min())

# Συνένωση περιγραφικών πεδίων
data['combine'] = data['desc_1'].astype(str) + ' ' + data['desc_2'].astype(str) + ' ' + data['desc_3'].astype(str)

# Συνάρτηση προεπεξεργασίας κειμένου
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    tokens = text.split()
    tokens = [WordNetLemmatizer().lemmatize(word) for word in tokens if word not in stopwords.words('english')]
    return tokens

# Δημιουργία λεξικού με τα tokens για κάθε έγγραφο (για υπολογισμό του Jaccard similarity)
doc_tokens = {}
for i in range(N):
    tokens = preprocess_text(data.iloc[i]['combine'])
    doc_tokens[i] = set(tokens)


# 2) Κατασκευή KD-Tree και Query


start = time.time()
kd_tree_data = data[['100g_USD_normalized', 'rating_normalized', 'loc_country_encoded', 'roast_encoded']].values
kd_tree = KDTree(kd_tree_data)
kd_tree_build_time = time.time() - start
print(f"KD-Tree Build Time: {kd_tree_build_time:.4f} seconds")

# Εκτέλεση query στο KD-Tree (παράδειγμα: εύρεση σημείων κοντά στο center)
center_point = [0.5, 0.5, 10, 2]
radius = 5.25
kd_tree_indices = kd_tree.query_ball_point(center_point, r=radius)
print(f"KD-Tree found {len(kd_tree_indices)} points within radius {radius} of {center_point}")


# 3) Κατασκευή Octree και Query


class OctreeNode:
    def __init__(self, bounds):
        self.bounds = bounds  # [x_min, y_min, z_min, x_max, y_max, z_max]
        self.points = []
        self.children = None

    def insert(self, point, index):
        x, y, z = point
        x_min, y_min, z_min, x_max, y_max, z_max = self.bounds
        if not (x_min <= x <= x_max and y_min <= y <= y_max and z_min <= z <= z_max):
            return False
        if len(self.points) < 4 and self.children is None:
            self.points.append((point, index))
            return True
        if self.children is None:
            self.subdivide()
        for child in self.children:
            if child.insert(point, index):
                return True
        return False

    def subdivide(self):
        x_min, y_min, z_min, x_max, y_max, z_max = self.bounds
        x_mid = (x_min + x_max) / 2
        y_mid = (y_min + y_max) / 2
        z_mid = (z_min + z_max) / 2
        self.children = [
            OctreeNode([x_min, y_min, z_min, x_mid, y_mid, z_mid]),
            OctreeNode([x_mid, y_min, z_min, x_max, y_mid, z_mid]),
            OctreeNode([x_min, y_mid, z_min, x_mid, y_max, z_mid]),
            OctreeNode([x_mid, y_mid, z_min, x_max, y_max, z_mid]),
            OctreeNode([x_min, y_min, z_mid, x_mid, y_mid, z_max]),
            OctreeNode([x_mid, y_min, z_mid, x_max, y_mid, z_max]),
            OctreeNode([x_min, y_mid, z_mid, x_mid, y_max, z_max]),
            OctreeNode([x_mid, y_mid, z_mid, x_max, y_max, z_max])
        ]

    def query(self, bounds, found=None):
        if found is None:
            found = []
        if not self.intersects(bounds):
            return found
        for point, index in self.points:
            if self.contains(point, bounds):
                found.append((point, index))
        if self.children:
            for child in self.children:
                child.query(bounds, found)
        return found

    def contains(self, point, bounds):
        x, y, z = point
        x_min, y_min, z_min, x_max, y_max, z_max = bounds
        return (x_min <= x <= x_max and y_min <= y <= y_max and z_min <= z <= z_max)

    def intersects(self, bounds):
        x_min, y_min, z_min, x_max, y_max, z_max = self.bounds
        b_x_min, b_y_min, b_z_min, b_x_max, b_y_max, b_z_max = bounds
        return not (b_x_max < x_min or b_x_min > x_max or b_y_max < y_min or b_y_min > y_max or b_z_max < z_min or b_z_min > z_max)

# Κατασκευή Octree χρησιμοποιώντας 3D δεδομένα (100g_USD_normalized, rating_normalized, loc_country_encoded)
octree_data = data[['100g_USD_normalized', 'rating_normalized', 'loc_country_encoded']].values
octree_bounds = [0, 0, 0, 1, 1, data['loc_country_encoded'].max()]
octree = OctreeNode(octree_bounds)
start = time.time()
for idx, point in enumerate(octree_data):
    octree.insert(point, idx)
octree_build_time = time.time() - start
print(f"Octree Build Time: {octree_build_time:.4f} seconds")

# Query Octree (χρήση πλήρους εύρους)
query_bounds = [0, 0, 0, 1, 1, data['loc_country_encoded'].max()]
octree_candidates = octree.query(query_bounds)

# Συνάρτηση φιλτραρίσματος για το 'roast'
def filter_by_roast(candidates, target_roast, data):
    filtered = []
    for point, idx in candidates:
        if data.iloc[idx]['roast_encoded'] == target_roast:
            filtered.append(idx)
    return filtered

target_roast = 2
filtered_octree_results = filter_by_roast(octree_candidates, target_roast, data)
print(f"Octree found {len(filtered_octree_results)} points matching roast={target_roast}")


# 4) Κατασκευή LSH για ομοιότητα κειμένου


tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(data['combine'])

lsh_threshold = 0.7
lsh = MinHashLSH(threshold=lsh_threshold, num_perm=128)
minhashes = {}

for i in range(N):
    tokens = preprocess_text(data.iloc[i]['combine'])
    m = MinHash(num_perm=128)
    for token in tokens:
        m.update(token.encode('utf8'))
    minhashes[i] = m
    lsh.insert(i, m)
print("LSH engine built successfully")


# 5) Κατασκευή R-Tree (4D) και Query


p = index.Property()
p.dimension = 4
rtree_4d = index.Index(properties=p)
start = time.time()
for i in range(N):
    row = data.iloc[i]
    x = row['100g_USD_normalized']
    y = row['rating_normalized']
    z = row['loc_country_encoded']
    w = row['roast_encoded']
    rtree_4d.insert(i, (x, y, z, w, x, y, z, w))
rtree_build_time = time.time() - start
print(f"R-Tree Build Time: {rtree_build_time:.4f} seconds")

rtree_query_bounds = (0, 0, 0, 0, 1, 1, data['loc_country_encoded'].max(), data['roast_encoded'].max())
rtree_candidates = list(rtree_4d.intersection(rtree_query_bounds))
filtered_rtree_results = [idx for idx in rtree_candidates if data.iloc[idx]['roast_encoded'] == target_roast]
print(f"R-Tree found {len(filtered_rtree_results)} points matching roast={target_roast}")


# 6) Κατασκευή 3D Range Tree και Query


class RangeTreeNode3D:
    def __init__(self, points, dimension):
        self.points = points  # λίστα με ((x,y,z), idx)
        self.left = None
        self.right = None
        self.median_value = None
        self.dimension = dimension
        self.associated = None

def build_range_tree_3d(points, dimension=0, max_dim=3):
    if not points:
        return None
    if dimension >= max_dim or len(points) <= 1:
        return RangeTreeNode3D(points, dimension)
    points.sort(key=lambda x: x[0][dimension])
    mid = len(points) // 2
    median_val = points[mid][0][dimension]
    node = RangeTreeNode3D(points, dimension)
    node.median_value = median_val
    left_points = points[:mid]
    right_points = points[mid+1:]
    node.left = build_range_tree_3d(left_points, dimension, max_dim)
    node.right = build_range_tree_3d(right_points, dimension, max_dim)
    if dimension + 1 < max_dim:
        node.associated = build_range_tree_3d(points, dimension+1, max_dim)
    return node

def in_3d_box(pt, box3d):
    (x, y, z) = pt
    (xm, xM), (ym, yM), (zm, zM) = box3d
    return (xm <= x <= xM) and (ym <= y <= yM) and (zm <= z <= zM)

def range_query_3d(node, query_box, dimension=0, max_dim=3):
    if node is None:
        return []
    if dimension >= max_dim - 1 or (node.left is None and node.right is None):
        result = []
        for (pt, idx) in node.points:
            if in_3d_box(pt, query_box):
                result.append((pt, idx))
        return result
    d_min, d_max = query_box[dimension]
    median_val = node.median_value
    results = []
    if median_val is None or median_val >= d_min:
        results += range_query_3d(node.left, query_box, dimension, max_dim)
    if median_val is None or median_val <= d_max:
        results += range_query_3d(node.right, query_box, dimension, max_dim)
    refined = [ (pt, idx) for (pt, idx) in results if in_3d_box(pt, query_box) ]
    return refined

range_3d_points = []
for i, row in data.iterrows():
    pt = (row['100g_USD_normalized'], row['rating_normalized'], row['loc_country_encoded'])
    range_3d_points.append((pt, i))
start = time.time()
range_tree_3d_root = build_range_tree_3d(range_3d_points, dimension=0, max_dim=3)
range_tree_build_time = time.time() - start
print(f"3D Range Tree Build Time: {range_tree_build_time:.6f} seconds")

range_tree_query = [(0, 1), (0, 1), (0, data['loc_country_encoded'].max())]
range_tree_candidates = range_query_3d(range_tree_3d_root, range_tree_query)

def filter_by_roast_range_tree(candidates, target_roast, data):
    return [idx for (pt, idx) in candidates if data.iloc[idx]['roast_encoded'] == target_roast]

filtered_range_tree_results = filter_by_roast_range_tree(range_tree_candidates, target_roast, data)
print(f"3D Range Tree found {len(filtered_range_tree_results)} points matching roast={target_roast}")


# 7) Ορισμός συναρτήσεων για Precision, Recall και Jaccard Similarity


MAX_RESULTS = 10

def get_top_results(similar_docs, max_results=MAX_RESULTS):
    return similar_docs[:max_results]

def is_relevant(doc_idx, target_roast, data):
    return data.iloc[doc_idx]['roast_encoded'] == target_roast

def calculate_overall_precision_recall(results, target_roast, data):
    total_retrieved = 0
    total_relevant_retrieved = 0
    ground_truth = set(data.index[data['roast_encoded'] == target_roast].tolist())
    for query_idx, similar_docs in results.items():
        top_results = get_top_results(similar_docs, MAX_RESULTS)
        total_retrieved += len(top_results)
        relevant_count = sum(1 for doc in top_results if is_relevant(doc, target_roast, data))
        total_relevant_retrieved += relevant_count
    overall_precision = total_relevant_retrieved / total_retrieved if total_retrieved > 0 else 0
    overall_recall = total_relevant_retrieved / len(ground_truth) if len(ground_truth) > 0 else 0
    return overall_precision, overall_recall

def jaccard_similarity(set1, set2):
    if not set1 and not set2:
        return 1.0
    return len(set1.intersection(set2)) / len(set1.union(set2))


# 8) Εκτέλεση LSH Queries και μέτρηση Precision/Recall για κάθε μέθοδο


# LSH queries στα αποτελέσματα του KD-Tree
print("\n--- LSH on KD-Tree Results ---")
kdtree_results = {}
start_KDTree_lsh = time.time()
for idx in kd_tree_indices:
    query_minhash = minhashes[idx]
    similar_docs = lsh.query(query_minhash)
    kdtree_results[idx] = similar_docs
    query_jaccard = {doc: jaccard_similarity(doc_tokens[idx], doc_tokens[doc]) for doc in similar_docs}
    print(f"KD-Tree doc {idx} => LSH similar: {similar_docs} with Jaccard: {query_jaccard}")
KDTree_lsh_time = time.time() - start_KDTree_lsh
print(f"Time for KD-Tree based LSH queries: {KDTree_lsh_time:.4f} seconds")
overall_precision_kdtree, overall_recall_kdtree = calculate_overall_precision_recall(kdtree_results, target_roast, data)

# LSH queries στα αποτελέσματα του Octree
print("\n--- LSH on Octree Results ---")
octree_results = {}
start_OctTree_lsh = time.time()
for idx in filtered_octree_results:
    query_minhash = minhashes[idx]
    similar_docs = lsh.query(query_minhash)
    octree_results[idx] = similar_docs
    query_jaccard = {doc: jaccard_similarity(doc_tokens[idx], doc_tokens[doc]) for doc in similar_docs}
    print(f"Octree doc {idx} => LSH similar: {similar_docs} with Jaccard: {query_jaccard}")
OctTree_lsh_time = time.time() - start_OctTree_lsh
print(f"Time for Octree based LSH queries: {OctTree_lsh_time:.4f} seconds")
overall_precision_octree, overall_recall_octree = calculate_overall_precision_recall(octree_results, target_roast, data)

# LSH queries στα αποτελέσματα του R-Tree
print("\n--- LSH on R-Tree Results ---")
rtree_results = {}
start_Rtree_lsh = time.time()
for idx in filtered_rtree_results:
    query_minhash = minhashes[idx]
    similar_docs = lsh.query(query_minhash)
    rtree_results[idx] = similar_docs
    query_jaccard = {doc: jaccard_similarity(doc_tokens[idx], doc_tokens[doc]) for doc in similar_docs}
    print(f"R-Tree doc {idx} => LSH similar: {similar_docs} with Jaccard: {query_jaccard}")
Rtree_lsh_time = time.time() - start_Rtree_lsh
print(f"Time for R-Tree based LSH queries: {Rtree_lsh_time:.4f} seconds")
overall_precision_rtree, overall_recall_rtree = calculate_overall_precision_recall(rtree_results, target_roast, data)

# LSH queries στα αποτελέσματα του 3D Range Tree
print("\n--- LSH on Range Tree Results ---")
range_tree_results = {}
start_range_lsh = time.time()
for idx in filtered_range_tree_results:
    query_minhash = minhashes[idx]
    similar_docs = lsh.query(query_minhash)
    range_tree_results[idx] = similar_docs
    query_jaccard = {doc: jaccard_similarity(doc_tokens[idx], doc_tokens[doc]) for doc in similar_docs}
    print(f"Range Tree doc {idx} => LSH similar: {similar_docs} with Jaccard: {query_jaccard}")
range_lsh_time = time.time() - start_range_lsh
print(f"Time for Range Tree based LSH queries: {range_lsh_time:.4f} seconds")
overall_precision_range_tree, overall_recall_range_tree = calculate_overall_precision_recall(range_tree_results, target_roast, data)


# 9) Οπτικοποίηση και Σύγκριση Αποτελεσμάτων


methods = ['KD-Tree', 'Octree', 'R-Tree', 'Range Tree']
query_times = [KDTree_lsh_time, OctTree_lsh_time, Rtree_lsh_time, range_lsh_time]
precision_scores = [overall_precision_kdtree, overall_precision_octree, overall_precision_rtree, overall_precision_range_tree]
recall_scores = [overall_recall_kdtree, overall_recall_octree, overall_recall_rtree, overall_recall_range_tree]
f1_scores = [2 * (p * r) / (p + r) if (p + r) > 0 else 0 for p, r in zip(precision_scores, recall_scores)]

x = np.arange(len(methods))
width = 0.25

fig, ax = plt.subplots(1, 2, figsize=(14,6))

# Διάγραμμα για Precision, Recall και F1 Score
ax[0].bar(x - width, precision_scores, width, label='Precision')
ax[0].bar(x, recall_scores, width, label='Recall')
ax[0].bar(x + width, f1_scores, width, label='F1 Score')
ax[0].set_xticks(x)
ax[0].set_xticklabels(methods)
ax[0].set_ylabel('Αξίες')
ax[0].set_title('Precision, Recall & F1 Score Ανά Μέθοδο')
ax[0].legend()

# Διάγραμμα για χρόνους LSH queries
ax[1].bar(methods, query_times, color='orange')
ax[1].set_ylabel('Χρόνος (sec)')
ax[1].set_title('Χρόνοι LSH Queries Ανά Μέθοδο')

plt.tight_layout()
plt.show()

# Εμφάνιση συνοπτικών αποτελεσμάτων
print("\n=== Σύνοψη Αποτελεσμάτων ===")
for i, method in enumerate(methods):
    print(f"{method}: Time = {query_times[i]:.4f} sec, Precision = {precision_scores[i]:.2f}, Recall = {recall_scores[i]:.2f}, F1 = {f1_scores[i]:.2f}")

best_method_index = np.argmax(f1_scores)
print(f"\nΗ βέλτιστη λύση είναι: {methods[best_method_index]}")