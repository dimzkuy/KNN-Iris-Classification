from knn_utils import (
    load_dataset, cross_validation, normalize_features,
    euclidean_distance, manhattan_distance, minkowski_distance, cosine_distance
)

def comprehensive_comparison():
    """Compare different configurations comprehensively"""
    filename = 'dataset/IRIS.csv'
    data = load_dataset(filename)
    
    distance_functions = {
        'Euclidean': euclidean_distance,
        'Manhattan': manhattan_distance,
        'Minkowski': minkowski_distance,
        'Cosine': cosine_distance
    }
    
    k_values = [1, 3, 5, 7, 9, 11]
    
    print("="*80)
    print("COMPREHENSIVE KNN MODEL COMPARISON")
    print("="*80)
    
    results = {}
    
    # Test without normalization
    print("\n--- WITHOUT NORMALIZATION ---")
    for dist_name, dist_func in distance_functions.items():
        print(f"\n{dist_name} Distance:")
        results[f"{dist_name}_raw"] = {}
        
        for k in k_values:
            accuracy = cross_validation(data.copy(), k_folds=5, k_neighbors=k, distance_func=dist_func)
            results[f"{dist_name}_raw"][k] = accuracy
            print(f"K={k}: {accuracy:.2f}%")
    
    # Test with normalization
    print("\n--- WITH NORMALIZATION ---")
    normalized_data, _ = normalize_features(data.copy())
    
    for dist_name, dist_func in distance_functions.items():
        print(f"\n{dist_name} Distance (Normalized):")
        results[f"{dist_name}_norm"] = {}
        
        for k in k_values:
            accuracy = cross_validation(normalized_data.copy(), k_folds=5, k_neighbors=k, distance_func=dist_func)
            results[f"{dist_name}_norm"][k] = accuracy
            print(f"K={k}: {accuracy:.2f}%")
    
    # Find best configuration
    print("\n" + "="*80)
    print("BEST CONFIGURATIONS SUMMARY")
    print("="*80)
    
    best_overall = 0
    best_config = None
    
    for config_name, k_results in results.items():
        best_k = max(k_results, key=k_results.get)
        best_acc = k_results[best_k]
        print(f"{config_name:<20}: K={best_k}, Accuracy={best_acc:.2f}%")
        
        if best_acc > best_overall:
            best_overall = best_acc
            best_config = (config_name, best_k)
    
    print(f"\nOVERALL BEST: {best_config[0]} with K={best_config[1]}, Accuracy={best_overall:.2f}%")

if __name__ == '__main__':
    comprehensive_comparison()
