# app.py
from flask import Flask, request, jsonify, send_from_directory
import numpy as np
from kmeans import kmeans, initialize_centroids, assign_clusters, update_centroids
import os
import json

app = Flask(__name__)

# Store the dataset globally (reset with each new dataset request)
data = None
centroids = None
clusters = None
manual_centroids_storage = None  # Variable to store manual centroids

@app.route('/')
def index():
    return send_from_directory(os.getcwd(), 'index.html')

# Route to generate a new dataset
@app.route('/new-dataset', methods=['GET'])
def generate_dataset():
    global data, centroids, clusters, manual_centroids_storage
    data = np.random.uniform(-10, 10, size=(300, 2)).tolist()
    centroids = None
    clusters = None
    manual_centroids_storage = None  # Reset manual centroids
    return jsonify({'data': data})

# Route to set manual centroids
@app.route('/set-manual-centroids', methods=['POST'])
def set_manual_centroids():
    global manual_centroids_storage, centroids, clusters
    k = int(request.form.get('k', 3))
    method = request.form.get('method', 'manual')
    centroids_json = request.form.get('centroids', '[]')

    try:
        centroids_list = json.loads(centroids_json)
        if len(centroids_list) != k:
            return jsonify({'success': False, 'error': f'Number of centroids provided ({len(centroids_list)}) does not match k ({k}).'}), 400
        manual_centroids_storage = np.array(centroids_list)
        centroids = manual_centroids_storage.copy()
        clusters = assign_clusters(np.array(data), centroids)
        return jsonify({'success': True})
    except json.JSONDecodeError:
        return jsonify({'success': False, 'error': 'Invalid JSON format for centroids.'}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# Route to step through KMeans
@app.route('/step', methods=['POST'])
def step_kmeans():
    global data, centroids, clusters, manual_centroids_storage
    if data is None:
        return jsonify({'error': 'No dataset found. Please generate a dataset first.'}), 400

    k = int(request.form['k'])
    method = request.form['method']
    
    # Initialize centroids
    if centroids is None:
        if method == 'manual':
            if manual_centroids_storage is None:
                return jsonify({'error': 'Manual centroids not set. Please set manual centroids first.'}), 400
            centroids = manual_centroids_storage.copy()
        else:
            centroids = initialize_centroids(np.array(data), k, method)

    # Perform one step of KMeans
    clusters = assign_clusters(np.array(data), centroids)
    new_centroids = update_centroids(np.array(data), clusters, k)

    # Check for convergence
    if np.allclose(centroids, new_centroids):
        centroids = new_centroids
        return jsonify({
            'converged': True,
            'centroids': new_centroids.tolist(),
            'clusters': clusters.tolist()
        })

    # Update centroids for the next step
    centroids = new_centroids

    return jsonify({
        'converged': False,
        'centroids': centroids.tolist(),
        'clusters': clusters.tolist()
    })

# Route to run KMeans to convergence
@app.route('/convergence', methods=['POST'])
def run_kmeans_convergence():
    global data, centroids, clusters, manual_centroids_storage
    if data is None:
        return jsonify({'error': 'No dataset found. Please generate a dataset first.'}), 400

    k = int(request.form['k'])
    method = request.form['method']

    # Initialize centroids
    if centroids is None:
        if method == 'manual':
            if manual_centroids_storage is None:
                return jsonify({'error': 'Manual centroids not set. Please set manual centroids first.'}), 400
            initial_centroids = manual_centroids_storage.copy()
        else:
            initial_centroids = initialize_centroids(np.array(data), k, method)
    else:
        initial_centroids = centroids.copy()

    # Run the KMeans algorithm to completion
    final_centroids, final_clusters = kmeans(
        np.array(data),
        k,
        method=method,
        initial_centroids=initial_centroids if method == 'manual' else None
    )

    centroids = final_centroids
    clusters = final_clusters

    return jsonify({
        'centroids': final_centroids.tolist(),
        'clusters': final_clusters.tolist()
    })

# Serve static files such as CSS or JavaScript if needed
@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(os.getcwd(), path)

# Update the app.run() to listen on port 3000
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
