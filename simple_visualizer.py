from knn_utils import load_dataset
import math

def create_ascii_histogram(data, bins=10):
    """Create ASCII histogram of feature values"""
    if not data:
        return
    
    feature_names = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
    
    for feature_idx in range(4):
        print(f"\n{feature_names[feature_idx]} Distribution:")
        print("-" * 40)
        
        values = [sample[0][feature_idx] for sample in data]
        min_val, max_val = min(values), max(values)
        
        if min_val == max_val:
            print("All values are the same")
            continue
        
        bin_width = (max_val - min_val) / bins
        bin_counts = [0] * bins
        
        for value in values:
            bin_idx = min(int((value - min_val) / bin_width), bins - 1)
            bin_counts[bin_idx] += 1
        
        max_count = max(bin_counts)
        scale = 50 / max_count if max_count > 0 else 1
        
        for i, count in enumerate(bin_counts):
            bin_start = min_val + i * bin_width
            bin_end = min_val + (i + 1) * bin_width
            bar_length = int(count * scale)
            bar = "#" * bar_length
            print(f"{bin_start:5.1f}-{bin_end:5.1f}: {bar} ({count})")

def show_class_distribution(data):
    """Show distribution of classes in the dataset"""
    class_counts = {}
    for _, label in data:
        class_counts[label] = class_counts.get(label, 0) + 1
    
    print("\nClass Distribution:")
    print("-" * 30)
    total = len(data)
    max_count = max(class_counts.values())
    scale = 30 / max_count
    
    for class_name, count in sorted(class_counts.items()):
        percentage = (count / total) * 100
        bar_length = int(count * scale)
        bar = "#" * bar_length
        print(f"{class_name:15}: {bar} {count} ({percentage:.1f}%)")

def show_feature_statistics(data):
    """Show basic statistics for each feature"""
    feature_names = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
    
    print("\nFeature Statistics:")
    print("-" * 60)
    print(f"{'Feature':<15} {'Min':<8} {'Max':<8} {'Mean':<8} {'Std':<8}")
    print("-" * 60)
    
    for feature_idx in range(4):
        values = [sample[0][feature_idx] for sample in data]
        min_val = min(values)
        max_val = max(values)
        mean_val = sum(values) / len(values)
        
        # Calculate standard deviation
        variance = sum((x - mean_val) ** 2 for x in values) / len(values)
        std_val = math.sqrt(variance)
        
        print(f"{feature_names[feature_idx]:<15} {min_val:<8.2f} {max_val:<8.2f} {mean_val:<8.2f} {std_val:<8.2f}")

def visualize_data():
    """Main visualization function"""
    filename = 'dataset/IRIS.csv'
    data = load_dataset(filename)
    
    print("="*60)
    print("IRIS Dataset Visualization")
    print("="*60)
    
    show_class_distribution(data)
    show_feature_statistics(data)
    create_ascii_histogram(data)

if __name__ == '__main__':
    visualize_data()
