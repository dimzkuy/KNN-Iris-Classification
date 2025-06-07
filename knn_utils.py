# Implementasi KNN tanpa menggunakan library eksternal

# ===== IMPLEMENTASI MANUAL UNTUK FUNGSI MATEMATIS =====

def manual_sqrt(x):
    """Implementasi manual akar kuadrat menggunakan metode Newton-Raphson"""
    if x == 0:
        return 0
    if x < 0:
        return None  # Tidak valid untuk bilangan negatif
    
    # Tebakan awal
    guess = x / 2
    
    # Iterasi Newton-Raphson: x_new = (x_old + n/x_old) / 2
    for _ in range(50):  # Maksimal 50 iterasi
        new_guess = (guess + x / guess) / 2
        # Cek konvergensi dengan presisi 6 digit
        if abs(new_guess - guess) < 0.000001:
            break
        guess = new_guess
    
    return guess

def manual_power(base, exp):
    """Implementasi manual untuk pangkat"""
    if exp == 0:
        return 1
    if exp == 1:
        return base
    
    result = 1
    for _ in range(abs(int(exp))):
        result *= base
    
    return result if exp > 0 else 1 / result

def manual_abs(x):
    """Implementasi manual untuk nilai absolut"""
    return x if x >= 0 else -x

# ===== IMPLEMENTASI MANUAL UNTUK SHUFFLE TANPA RANDOM =====

def manual_shuffle_inplace(data):
    """Shuffle data secara in-place menggunakan Linear Congruential Generator"""
    # Seed berdasarkan hash dari panjang data
    seed = 12345 + len(data) * 31
    n = len(data)
    
    for i in range(n - 1, 0, -1):
        # Linear Congruential Generator
        seed = (seed * 1103515245 + 12345) % 2147483647
        j = seed % (i + 1)
        
        # Tukar elemen
        data[i], data[j] = data[j], data[i]
    
    return data

# ===== IMPLEMENTASI MANUAL UNTUK MEMBACA FILE CSV =====

def load_dataset(filename):
    """
    Memuat dataset dari file CSV tanpa menggunakan library csv
    Format yang diharapkan: sepal_length,sepal_width,petal_length,petal_width,species
    """
    dataset = []
    
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
            
            # Skip header (baris pertama)
            for line_num, line in enumerate(lines[1:], 2):
                line = line.strip()
                if not line:  # Skip baris kosong
                    continue
                    
                # Split manual berdasarkan koma
                parts = line.split(',')
                
                if len(parts) >= 5:  # Pastikan ada minimal 5 kolom
                    try:
                        # Ekstrak fitur numerik (4 kolom pertama)
                        features = []
                        for i in range(4):
                            # Konversi string ke float secara manual
                            features.append(string_to_float(parts[i].strip()))
                        
                        # Ekstrak label (kolom terakhir)
                        label = parts[4].strip()
                        
                        dataset.append((features, label))
                        
                    except ValueError as e:
                        print(f"Error pada baris {line_num}: {line}")
                        print(f"Detail error: {e}")
                        continue
                else:
                    print(f"Baris {line_num} tidak memiliki format yang benar: {line}")
    
    except FileNotFoundError:
        print(f"Error: File {filename} tidak ditemukan!")
        return []
    except Exception as e:
        print(f"Error membaca file: {e}")
        return []
    
    print(f"Berhasil memuat {len(dataset)} sampel dari {filename}")
    return dataset

def string_to_float(s):
    """Konversi string ke float tanpa menggunakan float() built-in"""
    s = s.strip()
    
    # Handle tanda negatif
    negative = False
    if s.startswith('-'):
        negative = True
        s = s[1:]
    elif s.startswith('+'):
        s = s[1:]
    
    # Split berdasarkan titik desimal
    if '.' in s:
        integer_part, decimal_part = s.split('.', 1)
    else:
        integer_part = s
        decimal_part = '0'
    
    # Konversi bagian integer
    integer_value = 0
    for char in integer_part:
        if '0' <= char <= '9':
            integer_value = integer_value * 10 + (ord(char) - ord('0'))
        else:
            raise ValueError(f"Karakter tidak valid: {char}")
    
    # Konversi bagian desimal
    decimal_value = 0
    decimal_place = 1
    for char in decimal_part:
        if '0' <= char <= '9':
            decimal_place *= 10
            decimal_value = decimal_value * 10 + (ord(char) - ord('0'))
        else:
            raise ValueError(f"Karakter tidak valid: {char}")
    
    result = integer_value + (decimal_value / decimal_place)
    return -result if negative else result

# ===== IMPLEMENTASI FUNGSI JARAK TANPA MATH LIBRARY =====

def euclidean_distance(a, b):
    """Menghitung jarak Euclidean tanpa library math"""
    if len(a) != len(b):
        raise ValueError("Dimensi vektor harus sama")
    
    sum_squares = 0
    for i in range(len(a)):
        diff = a[i] - b[i]
        sum_squares += diff * diff
    
    return manual_sqrt(sum_squares)

def manhattan_distance(a, b):
    """Menghitung jarak Manhattan tanpa library"""
    if len(a) != len(b):
        raise ValueError("Dimensi vektor harus sama")
    
    total_distance = 0
    for i in range(len(a)):
        total_distance += manual_abs(a[i] - b[i])
    
    return total_distance

def minkowski_distance(a, b, p=3):
    """Menghitung jarak Minkowski tanpa library"""
    if len(a) != len(b):
        raise ValueError("Dimensi vektor harus sama")
    
    sum_powered = 0
    for i in range(len(a)):
        diff = manual_abs(a[i] - b[i])
        sum_powered += manual_power(diff, p)
    
    # Akar pangkat p (implementasi sederhana untuk p=3)
    if p == 3:
        # Implementasi akar pangkat 3 dengan metode Newton
        if sum_powered == 0:
            return 0
        
        x = sum_powered / 3  # Tebakan awal
        for _ in range(20):
            x_new = (2 * x + sum_powered / (x * x)) / 3
            if manual_abs(x_new - x) < 0.000001:
                break
            x = x_new
        return x
    else:
        # Untuk p lainnya, gunakan pendekatan sederhana
        return manual_power(sum_powered, 1.0/p)

def cosine_distance(a, b):
    """Menghitung jarak Cosine tanpa library"""
    if len(a) != len(b):
        raise ValueError("Dimensi vektor harus sama")
    
    # Hitung dot product
    dot_product = 0
    for i in range(len(a)):
        dot_product += a[i] * b[i]
    
    # Hitung magnitude a
    magnitude_a = 0
    for val in a:
        magnitude_a += val * val
    magnitude_a = manual_sqrt(magnitude_a)
    
    # Hitung magnitude b
    magnitude_b = 0
    for val in b:
        magnitude_b += val * val
    magnitude_b = manual_sqrt(magnitude_b)
    
    # Hindari pembagian dengan nol
    if magnitude_a == 0 or magnitude_b == 0:
        return 1.0  # Maksimal distance
    
    # Cosine similarity
    cosine_sim = dot_product / (magnitude_a * magnitude_b)
    
    # Cosine distance = 1 - cosine similarity
    return 1.0 - cosine_sim

# ===== IMPLEMENTASI SAVE/LOAD TANPA JSON LIBRARY =====

def save_model_params(params, filename):
    """Simpan parameter model tanpa menggunakan library json"""
    try:
        with open(filename, 'w') as file:
            file.write("{\n")
            
            param_items = list(params.items())
            for i, (key, value) in enumerate(param_items):
                # Format berdasarkan tipe data
                if isinstance(value, str):
                    file.write(f'  "{key}": "{value}"')
                elif isinstance(value, (int, float)):
                    file.write(f'  "{key}": {value}')
                else:
                    file.write(f'  "{key}": "{str(value)}"')
                
                # Tambah koma kecuali item terakhir
                if i < len(param_items) - 1:
                    file.write(",")
                file.write("\n")
            
            file.write("}")
        
        print(f"Parameter berhasil disimpan ke {filename}")
        return True
        
    except Exception as e:
        print(f"Error menyimpan parameter: {e}")
        return False

def load_model_params(filename):
    """Muat parameter model tanpa menggunakan library json"""
    try:
        with open(filename, 'r') as file:
            content = file.read().strip()
        
        # Parsing manual sederhana untuk format JSON
        params = {}
        
        # Hapus kurung kurawal
        content = content.strip('{}')
        
        # Split berdasarkan koma (parsing sederhana)
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if ':' in line:
                # Split berdasarkan titik dua pertama
                key_part, value_part = line.split(':', 1)
                
                # Bersihkan key (hapus tanda kutip dan whitespace)
                key = key_part.strip().strip('"').strip("'")
                
                # Bersihkan value
                value = value_part.strip().rstrip(',').strip()
                
                # Konversi value berdasarkan format
                if value.startswith('"') and value.endswith('"'):
                    # String value
                    params[key] = value[1:-1]
                elif '.' in value:
                    # Float value
                    try:
                        params[key] = string_to_float(value)
                    except:
                        params[key] = value
                else:
                    # Integer value
                    try:
                        params[key] = int(value)
                    except:
                        params[key] = value
        
        print(f"Parameter berhasil dimuat dari {filename}")
        return params
        
    except FileNotFoundError:
        print(f"File {filename} tidak ditemukan!")
        return None
    except Exception as e:
        print(f"Error memuat parameter: {e}")
        return None

# ===== FUNGSI UTILITAS KNN =====

def normalize_features(dataset):
    """Normalisasi fitur ke rentang 0-1"""
    if not dataset:
        return dataset
    
    # Cari nilai minimum dan maksimum untuk setiap fitur
    num_features = len(dataset[0][0])
    min_vals = [float('inf')] * num_features
    max_vals = [float('-inf')] * num_features
    
    for features, _ in dataset:
        for i, val in enumerate(features):
            min_vals[i] = min(min_vals[i], val)
            max_vals[i] = max(max_vals[i], val)
    
    # Lakukan normalisasi
    normalized_dataset = []
    for features, label in dataset:
        normalized_features = []
        for i, val in enumerate(features):
            if max_vals[i] - min_vals[i] != 0:
                normalized_val = (val - min_vals[i]) / (max_vals[i] - min_vals[i])
            else:
                normalized_val = 0
            normalized_features.append(normalized_val)
        normalized_dataset.append((normalized_features, label))
    
    return normalized_dataset, (min_vals, max_vals)

def get_neighbors(training_data, test_instance, k, distance_func=euclidean_distance):
    """Mendapatkan k tetangga terdekat dari instance uji"""
    distances = []
    for train_instance in training_data:
        dist = distance_func(test_instance, train_instance[0])
        distances.append((train_instance, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = [distances[i][0] for i in range(k)]
    return neighbors

def predict_classification(neighbors):
    """Memprediksi klasifikasi berdasarkan voting mayoritas tetangga"""
    class_votes = {}
    for neighbor in neighbors:
        label = neighbor[1]
        if label not in class_votes:
            class_votes[label] = 0
        class_votes[label] += 1
    return max(class_votes, key=class_votes.get)

def calculate_accuracy(test_data, predictions):
    """Menghitung akurasi prediksi dalam persentase"""
    correct = 0
    for i in range(len(test_data)):
        if test_data[i][1] == predictions[i]:
            correct += 1
    return (correct / len(test_data)) * 100

def create_confusion_matrix(test_data, predictions):
    """Membuat confusion matrix untuk analisis performa detail"""
    # Dapatkan label unik
    labels = list(set([item[1] for item in test_data] + predictions))
    labels.sort()
    
    # Inisialisasi matrix
    matrix = {}
    for actual in labels:
        matrix[actual] = {}
        for predicted in labels:
            matrix[actual][predicted] = 0
    
    # Isi matrix
    for i in range(len(test_data)):
        actual = test_data[i][1]
        predicted = predictions[i]
        matrix[actual][predicted] += 1
    
    return matrix, labels

def print_confusion_matrix(matrix, labels):
    """Menampilkan confusion matrix dalam format yang mudah dibaca"""
    print("\nMatriks Konfusi:")
    print("Aktual\\Prediksi", end="")
    for label in labels:
        print(f"\t{label[:10]}", end="")
    print()
    
    for actual in labels:
        print(f"{actual[:15]}", end="")
        for predicted in labels:
            print(f"\t{matrix[actual][predicted]}", end="")
        print()

def calculate_precision_recall(matrix, labels):
    """Menghitung precision dan recall untuk setiap kelas"""
    metrics = {}
    for label in labels:
        # True positives
        tp = matrix[label][label]
        
        # False positives (diprediksi sebagai kelas ini tapi sebenarnya kelas lain)
        fp = sum(matrix[other][label] for other in labels if other != label)
        
        # False negatives (sebenarnya kelas ini tapi diprediksi sebagai kelas lain)
        fn = sum(matrix[label][other] for other in labels if other != label)
        
        # Hitung precision dan recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[label] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }
    
    return metrics

def cross_validation(data, k_folds=5, k_neighbors=3, distance_func=euclidean_distance):
    """Melakukan validasi silang k-fold tanpa random library"""
    # Shuffle data menggunakan implementasi manual
    data_copy = data.copy()
    manual_shuffle_inplace(data_copy)
    
    fold_size = len(data_copy) // k_folds
    accuracies = []
    
    for i in range(k_folds):
        # Bagi data menjadi set latih dan validasi
        start_idx = i * fold_size
        end_idx = start_idx + fold_size if i < k_folds - 1 else len(data_copy)
        
        validation_set = data_copy[start_idx:end_idx]
        train_set = data_copy[:start_idx] + data_copy[end_idx:]
        
        # Buat prediksi
        predictions = []
        for test_instance in validation_set:
            neighbors = get_neighbors(train_set, test_instance[0], k_neighbors, distance_func)
            result = predict_classification(neighbors)
            predictions.append(result)
        
        # Hitung akurasi
        accuracy = calculate_accuracy(validation_set, predictions)
        accuracies.append(accuracy)
        print(f"Fold {i+1}: {accuracy:.2f}%")
    
    avg_accuracy = sum(accuracies) / len(accuracies)
    print(f"Rata-rata Akurasi CV: {avg_accuracy:.2f}%")
    return avg_accuracy

def find_optimal_k(data, max_k=20, k_folds=5, distance_func=euclidean_distance):
    """Mencari nilai K optimal menggunakan validasi silang"""
    print("Mencari nilai K optimal...")
    best_k = 1
    best_accuracy = 0
    
    k_values = range(1, min(max_k + 1, len(data) // k_folds))
    
    for k in k_values:
        print(f"\nMenguji K = {k}")
        accuracy = cross_validation(data, k_folds, k, distance_func)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k
    
    print(f"\nK Optimal: {best_k} dengan akurasi: {best_accuracy:.2f}%")
    return best_k, best_accuracy

def evaluate_distance_metrics(data, k_neighbors=3, k_folds=5):
    """Membandingkan berbagai metrik jarak"""
    distance_functions = {
        'Euclidean': euclidean_distance,
        'Manhattan': manhattan_distance,
        'Minkowski': minkowski_distance,
        'Cosine': cosine_distance
    }
    
    results = {}
    print("Mengevaluasi berbagai metrik jarak...")
    
    for name, func in distance_functions.items():
        print(f"\nMenguji jarak {name}:")
        accuracy = cross_validation(data, k_folds, k_neighbors, func)
        results[name] = accuracy
    
    # Cari metrik jarak terbaik
    best_metric = max(results, key=results.get)
    print(f"\nMetrik jarak terbaik: {best_metric} ({results[best_metric]:.2f}%)")
    
    return results, best_metric
