#include <iostream>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <random>
#include <fstream>
#include <string>

enum class Pattern {
    smooth,
    turbulent,
    periodic,
    noisy,
};

// Smooth pattern: simulates temperature field with gentle gradients (273K-373K range)
template<typename T>
T pattern_smooth(std::size_t r, std::size_t c) {
    double x = static_cast<double>(r) * 0.01;
    double y = static_cast<double>(c) * 0.01;
    double value = 300.0 + 50.0 * std::sin(x) * std::cos(y) + 20.0 * std::exp(-0.001 * (x*x + y*y));
    return static_cast<T>(value);
}

// Turbulent pattern: simulates turbulent flow with multi-scale variations
template<typename T>
T pattern_turbulent(std::size_t r, std::size_t c) {
    double x = static_cast<double>(r);
    double y = static_cast<double>(c);
    double large_scale = 100.0 * std::sin(x * 0.01) * std::cos(y * 0.01);
    double medium_scale = 30.0 * std::sin(x * 0.05) * std::cos(y * 0.05);
    double small_scale = 10.0 * std::sin(x * 0.2) * std::cos(y * 0.2);
    double value = 50.0 + large_scale + medium_scale + small_scale;
    return static_cast<T>(value);
}

// Periodic pattern: simulates wave-like oscillations
template<typename T>
T pattern_periodic(std::size_t r, std::size_t c) {
    double x = static_cast<double>(r) * 0.1;
    double y = static_cast<double>(c) * 0.1;
    double value = 100.0 + 50.0 * std::sin(x) + 30.0 * std::cos(y) + 20.0 * std::sin(x + y);
    return static_cast<T>(value);
}

// Noisy pattern: simulates measurement data with Gaussian noise
template<typename T>
T pattern_noisy(std::size_t r, std::size_t c) {
    (void)r; (void)c;  // Parameters unused for random noise
    static std::mt19937 gen(12345);  // Fixed seed for reproducibility
    static std::normal_distribution<double> dist(500.0, 50.0);  // mean=500, stddev=50
    return static_cast<T>(dist(gen));
}

template<typename T, typename PatternFunc>
void fill_pattern_2d(std::vector<std::vector<T>>& data,
                     std::size_t rows,
                     std::size_t elems_per_row,
                     PatternFunc pattern)
{
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < elems_per_row; ++c) {
            data[r][c] = pattern(r, c);
        }
    }
}

template<typename T>
void generateData(std::size_t DatasetSizeinMB,
                  std::size_t RowSizeinMB,
                  Pattern pattern_type,
                  const std::string& output_file = ""
                  )
{
    std::size_t bytes_per_row = RowSizeinMB * 1024u * 1024u;
    std::size_t elems_per_row = bytes_per_row / sizeof(T);
    std::size_t rows = DatasetSizeinMB / RowSizeinMB;

    // Create 2D matrix structure
    std::vector<std::vector<T>> data(rows, std::vector<T>(elems_per_row));

    // Select pattern function based on enum
    switch (pattern_type) {
        case Pattern::smooth:
            fill_pattern_2d<T>(data, rows, elems_per_row, pattern_smooth<T>);
            break;
        case Pattern::turbulent:
            fill_pattern_2d<T>(data, rows, elems_per_row, pattern_turbulent<T>);
            break;
        case Pattern::periodic:
            fill_pattern_2d<T>(data, rows, elems_per_row, pattern_periodic<T>);
            break;
        case Pattern::noisy:
            fill_pattern_2d<T>(data, rows, elems_per_row, pattern_noisy<T>);
            break;
    }
    
    // Save to file if output file is specified
    if (!output_file.empty()) {
        std::ofstream file(output_file, std::ios::binary);
        
        // Write data row by row in contiguous binary format
        for (std::size_t r = 0; r < rows; ++r) {
            file.write(reinterpret_cast<const char*>(data[r].data()), 
                        elems_per_row * sizeof(T));
        }
        file.close();
    }

}

int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <dataset_size_in_mb> <row_size_in_mb> <data_type> <pattern_type> [binary]\n";
        std::cerr << "Supported data types: float, double, int32, int64\n";
        std::cerr << "Supported patterns: smooth, turbulent, periodic, noisy\n";
        std::cerr << "\nExamples:\n";
        return 1;
    }
    std::size_t dataset_size_in_mb = std::stoul(argv[1]);
    std::size_t row_size_in_mb = std::stoul(argv[2]);
    std::string data_type = argv[3];
    std::string pattern_type = argv[4];
    
    // Parse pattern type
    Pattern pattern = Pattern::smooth;
    if (pattern_type == "smooth") {
        pattern = Pattern::smooth;
    } else if (pattern_type == "turbulent") {
        pattern = Pattern::turbulent;
    } else if (pattern_type == "periodic") {
        pattern = Pattern::periodic;
    } else if (pattern_type == "noisy") {
        pattern = Pattern::noisy;
    } else {
        std::cerr << "Unknown pattern type: " << pattern_type << "\n";
        return 1;
    }
    
    
    std::string filename = pattern_type + "_pattern.bin" ;
    
    if (data_type == "float") {
        generateData<float>(dataset_size_in_mb, row_size_in_mb, pattern, filename);
    } else if (data_type == "double") {
        generateData<double>(dataset_size_in_mb, row_size_in_mb, pattern, filename);
    } else if (data_type == "int32") {
        generateData<int32_t>(dataset_size_in_mb, row_size_in_mb, pattern, filename);
    } else if (data_type == "int64") {
        generateData<int64_t>(dataset_size_in_mb, row_size_in_mb, pattern, filename);
    } else {
        std::cerr << "Unknown data type: " << data_type << "\n";
        std::cerr << "Supported types: float, double, int32, int64\n";
        return 1;
    }
    
    return 0;
}
