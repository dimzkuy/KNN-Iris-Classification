from knn_utils import (
    load_dataset, get_neighbors, predict_classification, calculate_accuracy,
    normalize_features, create_confusion_matrix, print_confusion_matrix,
    calculate_precision_recall, cross_validation, find_optimal_k,
    save_model_params, load_model_params, evaluate_distance_metrics,
    euclidean_distance, manhattan_distance, minkowski_distance, cosine_distance,
    string_to_float
)

# Implementasi manual untuk seed tanpa library
def get_current_time_seed():
    """Mendapatkan seed waktu manual tanpa library"""
    # Gunakan hash dari string untuk mendapatkan seed pseudo-random
    seed_string = "manual_seed_generator_2024"
    seed = 12345  # Seed awal
    for char in seed_string:
        seed = (seed * 31 + ord(char)) % 2147483647
    return seed

# Implementasi manual untuk shuffle tanpa library apapun
def manual_shuffle(data):
    """Shuffle manual menggunakan Linear Congruential Generator"""
    # Gunakan seed manual
    seed = get_current_time_seed()
    
    data_copy = data.copy()
    n = len(data_copy)
    
    for i in range(n - 1, 0, -1):
        # Linear Congruential Generator untuk angka pseudo-random
        seed = (seed * 1103515245 + 12345) % 2147483647
        j = seed % (i + 1)
        
        # Tukar elemen
        data_copy[i], data_copy[j] = data_copy[j], data_copy[i]
    
    return data_copy

# Fungsi untuk menjalankan tes KNN dasar dengan split train/test
def basic_knn_test(data, k=3, use_normalization=False, distance_func=euclidean_distance):
    """
    Tes KNN dasar dengan pembagian train/test
    
    Algoritma KNN:
    1. Normalisasi data (opsional) - ubah rentang fitur ke 0-1
    2. Bagi data menjadi training (70%) dan testing (30%)
    3. Untuk setiap sampel test:
       - Hitung jarak ke semua sampel training
       - Ambil k tetangga terdekat berdasarkan jarak
       - Lakukan voting mayoritas untuk klasifikasi
    4. Evaluasi hasil dengan confusion matrix dan metrik
    """
    print(f"\n=== Tes KNN Dasar (K={k}) ===")
    
    # Langkah 1: Normalisasi fitur jika diminta
    if use_normalization:
        data, norm_params = normalize_features(data)
        print("Fitur dinormalisasi ke rentang 0-1")
    
    # Langkah 2: Shuffle data secara manual (tanpa random library)
    shuffled_data = manual_shuffle(data)
    
    # Langkah 3: Pembagian data - 70% untuk training, 30% untuk testing
    split = int(0.7 * len(shuffled_data))
    train_set = shuffled_data[:split]
    test_set = shuffled_data[split:]
    
    print(f"Set pelatihan: {len(train_set)} sampel")
    print(f"Set pengujian: {len(test_set)} sampel")
    
    # Langkah 4: Klasifikasi menggunakan KNN
    predictions = []
    for i, test_instance in enumerate(test_set):
        # Temukan k tetangga terdekat
        neighbors = get_neighbors(train_set, test_instance[0], k, distance_func)
        # Prediksi berdasarkan voting mayoritas
        result = predict_classification(neighbors)
        predictions.append(result)
        
        # Progress indicator (setiap 10 prediksi)
        if (i + 1) % 10 == 0:
            print(f"Memproses... {i + 1}/{len(test_set)} sampel")
    
    # Langkah 5: Evaluasi hasil
    accuracy = calculate_accuracy(test_set, predictions)
    print(f"Akurasi: {accuracy:.2f}%")
    
    # Tampilkan confusion matrix
    matrix, labels = create_confusion_matrix(test_set, predictions)
    print_confusion_matrix(matrix, labels)
    
    # Hitung precision, recall, F1-score untuk setiap kelas
    metrics = calculate_precision_recall(matrix, labels)
    print("\nMetrik evaluasi per kelas:")
    for label, metric in metrics.items():
        print(f"{label}:")
        print(f"  Presisi: {metric['precision']:.3f}")
        print(f"  Recall: {metric['recall']:.3f}")
        print(f"  Skor-F1: {metric['f1_score']:.3f}")
    
    return accuracy, test_set, predictions

# Fungsi untuk validasi apakah string adalah angka (tanpa isdigit)
def is_number_string(s):
    """Validasi manual apakah string adalah angka"""
    if not s:
        return False
    
    # Cek apakah semua karakter adalah digit
    for char in s:
        if not ('0' <= char <= '9'):
            return False
    return True

# Fungsi untuk input angka dengan validasi manual
def get_integer_input(prompt, default=None):
    """Input integer dengan validasi manual (tanpa library)"""
    while True:
        try:
            user_input = input(prompt).strip()
            if not user_input and default is not None:
                return default
            
            # Validasi manual apakah string adalah angka
            if is_number_string(user_input):
                return int(user_input)
            else:
                print("Error: Masukkan angka yang valid!")
        except:
            print("Error: Input tidak valid!")

def get_float_input(prompt):
    """Input float dengan validasi manual tanpa built-in float()"""
    while True:
        try:
            user_input = input(prompt).strip()
            # Gunakan implementasi manual string ke float
            result = string_to_float(user_input)
            return result
        except:
            print("Error: Masukkan angka desimal yang valid!")

# Fungsi untuk menampilkan menu interaktif
def interactive_menu():
    """
    Menu interaktif untuk operasi KNN
    Menyediakan berbagai opsi testing dan evaluasi model
    """
    filename = 'dataset/IRIS.csv'
    data = load_dataset(filename)
    print(f"Berhasil memuat {len(data)} sampel dari {filename}")
    
    # Daftar fungsi jarak yang tersedia
    distance_functions = {
        '1': ('Euclidean', euclidean_distance),
        '2': ('Manhattan', manhattan_distance), 
        '3': ('Minkowski', minkowski_distance),
        '4': ('Cosine', cosine_distance)
    }
    
    while True:
        # Tampilan menu utama
        print("\n" + "="*55)
        print("    SISTEM KLASIFIKASI IRIS MENGGUNAKAN KNN")
        print("="*55)
        print("1. Tes KNN Dasar dengan Train/Test Split")
        print("2. Validasi Silang (Cross Validation)")
        print("3. Pencarian K Optimal")
        print("4. Perbandingan Metrik Jarak")
        print("5. Tes dengan Normalisasi Fitur")
        print("6. Simpan Konfigurasi Model Terbaik")
        print("7. Muat Konfigurasi Model Tersimpan")
        print("8. Keluar dari Program")
        print("="*55)
        
        choice = input("Pilih menu (1-8): ").strip()
        
        if choice == '1':
            print("\n--- Tes KNN Dasar ---")
            k = get_integer_input("Masukkan nilai K (default 3): ", 3)
            
            print("\nPilih metrik jarak:")
            for key, (name, _) in distance_functions.items():
                print(f"  {key}. {name}")
            
            dist_choice = input("Pilih metrik jarak (1-4, default 1): ").strip() or "1"
            dist_name, dist_func = distance_functions.get(dist_choice, distance_functions['1'])
            
            print(f"\nMenjalankan KNN dengan K={k} dan jarak {dist_name}")
            basic_knn_test(data.copy(), k, False, dist_func)
        
        elif choice == '2':
            print("\n--- Validasi Silang ---")
            k = get_integer_input("Masukkan nilai K (default 3): ", 3)
            folds = get_integer_input("Masukkan jumlah fold (default 5): ", 5)
            print(f"\nMenjalankan {folds}-fold cross validation dengan K={k}")
            cross_validation(data.copy(), folds, k)
        
        elif choice == '3':
            print("\n--- Pencarian K Optimal ---")
            max_k = get_integer_input("Masukkan K maksimum untuk diuji (default 20): ", 20)
            folds = get_integer_input("Masukkan jumlah fold (default 5): ", 5)
            print(f"\nMencari K optimal dari 1 hingga {max_k}")
            find_optimal_k(data.copy(), max_k, folds)
        
        elif choice == '4':
            print("\n--- Perbandingan Metrik Jarak ---")
            k = get_integer_input("Masukkan nilai K (default 3): ", 3)
            folds = get_integer_input("Masukkan jumlah fold (default 5): ", 5)
            print(f"\nMembandingkan semua metrik jarak dengan K={k}")
            evaluate_distance_metrics(data.copy(), k, folds)
        
        elif choice == '5':
            print("\n--- Tes dengan Normalisasi ---")
            k = get_integer_input("Masukkan nilai K (default 3): ", 3)
            print(f"\nMenjalankan KNN dengan normalisasi fitur (K={k})")
            basic_knn_test(data.copy(), k, True)
        
        elif choice == '6':
            print("\n--- Simpan Konfigurasi Model ---")
            k = get_integer_input("Masukkan nilai K untuk disimpan: ")
            distance_metric = input("Masukkan nama metrik jarak: ")
            accuracy = get_float_input("Masukkan akurasi yang dicapai (%): ")
            
            # Buat dictionary parameter
            params = {
                'k': k,
                'metrik_jarak': distance_metric,
                'akurasi': accuracy,
                'ukuran_dataset': len(data)
            }
            
            filename = input("Nama file penyimpanan (default model_params.json): ") or "model_params.json"
            save_model_params(params, filename)
        
        elif choice == '7':
            print("\n--- Muat Konfigurasi Model ---")
            filename = input("Nama file yang akan dimuat (default model_params.json): ") or "model_params.json"
            params = load_model_params(filename)
            if params:
                print("\nKonfigurasi model yang dimuat:")
                print("-" * 30)
                for key, value in params.items():
                    print(f"  {key}: {value}")
                print("-" * 30)
        
        elif choice == '8':
            print("\n" + "="*55)
            print("    Terima kasih telah menggunakan program KNN!")
            print("="*55)
            break
        
        else:
            print("\nError: Pilihan tidak valid. Silakan pilih angka 1-8.")

# Fungsi utama program
def main():
    """
    Fungsi utama program KNN Iris
    Menjalankan sistem klasifikasi dengan menu interaktif
    """
    print("Inisialisasi program klasifikasi Iris menggunakan K-Nearest Neighbors")
    print("Program ini mengimplementasikan algoritma KNN tanpa library eksternal")
    
    # Jalankan menu interaktif
    interactive_menu()

# Entry point program
if __name__ == '__main__':
    main()
