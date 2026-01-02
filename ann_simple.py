import numpy as np

# Fungsi aktivasi Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Turunan dari fungsi Sigmoid (untuk backpropagation)
def sigmoid_derivative(x):
    return x * (1 - x)

class SimpleANN:
    def __init__(self, input_size, hidden_size, output_size):
        # Inisialisasi bobot (weights) secara acak
        self.weights_input_hidden = np.random.uniform(-1, 1, (input_size, hidden_size))
        self.weights_hidden_output = np.random.uniform(-1, 1, (hidden_size, output_size))
        
        # Inisialisasi bias
        self.bias_hidden = np.random.uniform(-1, 1, (1, hidden_size))
        self.bias_output = np.random.uniform(-1, 1, (1, output_size))

    def forward(self, X):
        # Forward pass (perambatan maju)
        
        # Hitung input ke hidden layer
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        # Aktivasi hidden layer
        self.hidden_output = sigmoid(self.hidden_input)
        
        # Hitung input ke output layer
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        # Aktivasi output layer (prediksi akhir)
        self.final_output = sigmoid(self.final_input)
        
        return self.final_output

    def backward(self, X, y, output, learning_rate):
        # Backward pass (perambatan mundur / training)
        # Menghitung seberapa besar kita harus mengubah bobot berdasarkan error
        
        # Hitung error (Target - Output Aktual)
        error = y - output
        
        # Hitung delta untuk output layer (Gradien)
        d_output = error * sigmoid_derivative(output)
        
        # Hitung error kontribusi dari hidden layer
        error_hidden_layer = d_output.dot(self.weights_hidden_output.T)
        
        # Hitung delta untuk hidden layer
        d_hidden_layer = error_hidden_layer * sigmoid_derivative(self.hidden_output)
        
        # Update bobot dan bias
        self.weights_hidden_output += self.hidden_output.T.dot(d_output) * learning_rate
        self.bias_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate
        
        self.weights_input_hidden += X.T.dot(d_hidden_layer) * learning_rate
        self.bias_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

    def train(self, X, y, epochs, learning_rate):
        print(f"Mulai training untuk {epochs} epoch...")
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output, learning_rate)
            
            if (epoch + 1) % 1000 == 0:
                loss = np.mean(np.square(y - output)) # Mean Squared Error
                print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

if __name__ == "__main__":
    # --- Contoh Penggunaan: XOR Problem ---
    
    # 1. Siapkan Data Latih
    # Input: [A, B] pola logika XOR
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])

    # Target Output: [A XOR B] (0 jika sama, 1 jika beda)
    y = np.array([[0],
                  [1],
                  [1],
                  [0]])

    print("Data Input (X):")
    print(X)
    print("\nTarget Output (y):")
    print(y)
    print("-" * 30)

    # 2. Inisialisasi ANN
    # Input layer: 2 neuron (sesuai jumlah fitur input)
    # Hidden layer: 4 neuron (bisa disesuaikan)
    # Output layer: 1 neuron (hasil prediksi 0 atau 1)
    ann = SimpleANN(input_size=2, hidden_size=4, output_size=1)

    # 3. Lakukan Training
    # Learning rate 0.1, diulang 10.000 kali
    ann.train(X, y, epochs=10000, learning_rate=0.1)

    # 4. Uji Coba / Prediksi
    print("\n--- Hasil Akhir Prediksi ---")
    prediction = ann.forward(X)
    
    for i in range(len(X)):
        input_data = X[i]
        actual_target = y[i][0]
        predicted_value = prediction[i][0]
        result_label = 1 if predicted_value > 0.5 else 0
        
        print(f"Input: {input_data} -> Prediksi Mentah: {predicted_value:.4f} -> Label: {result_label} (Target: {actual_target})")
