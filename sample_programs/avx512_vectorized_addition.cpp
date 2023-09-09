#include <iostream>
#include <immintrin.h>

void vectorAdd(const float* a, const float* b, float* c, int size) {
    const int simdWidth = 16; // Number of elements per AVX-512 vector

    for (int i = 0; i < size; i += simdWidth) {
        __m512 va = _mm512_load_ps(&a[i]);
        __m512 vb = _mm512_load_ps(&b[i]);
        __m512 vc = _mm512_add_ps(va, vb);
        _mm512_store_ps(&c[i], vc);
    }
}

int main() {
    constexpr int size = 16; // Size of the vector
    alignas(64) float a[size];
    alignas(64) float b[size];
    alignas(64) float c[size];

    // Initialize input arrays
    for (int i = 0; i < size; i++) {
        a[i] = i;
        b[i] = i * 2;
    }

    vectorAdd(a, b, c, size);

    // Output the result
    for (int i = 0; i < size; i++) {
        std::cout << "c[" << i << "] = " << c[i] << std::endl;
    }

    return 0;
}
