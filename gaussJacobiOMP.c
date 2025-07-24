// Implementação do Algoritmo de Gauss-Jacobi
// clang-format off

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define EPSILON 1.0e-15
#define MAX_K 1000000

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

double jacobiSequential(double *A, double *b, int n) {
    double *x = (double *)calloc(n, sizeof(double));
    double *phi = (double *)calloc(n, sizeof(double));

    double *dA = (double *)malloc(n * n * sizeof(double)); // dA = D^(-1)C
    double *db = (double *)malloc(n * sizeof(double));     // db = D^(-1)b

    clock_t t;
    double dt;

    for (int i = 0; i < n; ++i) {
        double d = A[i * n + i];

        for (int j = 0; j < n; ++j)
            dA[i * n + j] = i == j ? 0.0 : (A[i * n + j] / d);

        db[i] = b[i] / d;
    }

    double sum, e; // e = max(|x(k+1) - x(k)|)
    t = clock();
    for (int k = 0;; ++k) {
        e = 0.0;
        for (int i = 0; i < n; ++i) {
            sum = 0.0;

            for (int j = 0; j < n; ++j)
                sum += dA[i * n + j] * x[j];

            phi[i] = db[i] - sum;
            e = MAX(e, fabs(phi[i] - x[i]));
        }

        if (e <= EPSILON || k >= MAX_K)
            break;

        for (int i = 0; i < n; ++i)
            x[i] = phi[i];
    }
    t = clock() - t;
    dt = (double)t / CLOCKS_PER_SEC;

    free(x);
    free(phi);
    free(dA);
    free(db);

    return dt;
}

double jacobiParallel(double *A, double *b, int n, int max_threads) {
    const int NUM_THREADS = MIN(n, max_threads);

    double *x = (double *)calloc(n, sizeof(double));
    double *phi = (double *)calloc(n, sizeof(double));

    double *dA = (double *)malloc(n * n * sizeof(double)); // dA = D^(-1)C
    double *db = (double *)malloc(n * sizeof(double));     // db = D^(-1)b

    double dt;

    #pragma omp parallel for num_threads(NUM_THREADS)
    for (int i = 0; i < n; ++i) {
        double d = A[i * n + i];

        for (int j = 0; j < n; ++j)
            dA[i * n + j] = i == j ? 0.0 : (A[i * n + j] / d);

        db[i] = b[i] / d;
    }

    double e; // e = max(|x(k+1) - x(k)|)
    dt = omp_get_wtime();
    for (int k = 0;; ++k) {
        e = 0.0;
        #pragma omp parallel for num_threads(NUM_THREADS)
        for (int i = 0; i < n; ++i) {
            double sum = 0.0;

            for (int j = 0; j < n; ++j)
                sum += dA[i * n + j] * x[j];

            phi[i] = db[i] - sum;
            #pragma omp critical
            e = MAX(e, fabs(phi[i] - x[i]));
        }

        if (e <= EPSILON || k >= MAX_K)
            break;

        #pragma omp parallel for num_threads(NUM_THREADS)
        for (int i = 0; i < n; ++i)
            x[i] = phi[i];
    }
    dt = omp_get_wtime() - dt;

    free(x);
    free(phi);
    free(dA);
    free(db);

    return dt;
}

double seidelSequential(double *A, double *b, int n) {
    double *x = (double *)calloc(n, sizeof(double));
    double *phi = (double *)calloc(n, sizeof(double));

    double *dA = (double *)malloc(n * n * sizeof(double)); // dA = D^(-1)C
    double *db = (double *)malloc(n * sizeof(double));     // db = D^(-1)b

    clock_t t;
    double dt;

    for (int i = 0; i < n; ++i) {
        double d = A[i * n + i];

        for (int j = 0; j < n; ++j)
            dA[i * n + j] = i == j ? 0.0 : (A[i * n + j] / d);

        db[i] = b[i] / d;
    }

    double sum, e; // e = max(|x(k+1) - x(k)|)
    t = clock();
    for (int k = 0;; ++k) {
        e = 0.0;
        for (int i = 0; i < n; ++i) {
            sum = 0.0;

            for (int j = 0; j < i; ++j)
                sum += dA[i * n + j] * phi[j];

            for (int j = i + 1; j < n; ++j)
                sum += dA[i * n + j] * x[j];

            phi[i] = db[i] - sum;
            e = MAX(e, fabs(phi[i] - x[i]));
        }

        if (e <= EPSILON || k >= MAX_K)
            break;

        for (int i = 0; i < n; i++)
            x[i] = phi[i];
    }
    t = clock() - t;
    dt = (double)t / CLOCKS_PER_SEC;

    free(x);
    free(phi);
    free(dA);
    free(db);

    return dt;
}

double *randA(int n, double density) {
    srand(time(NULL));

    double *A = (double *)malloc(n * n * sizeof(double));
    int d = (int)(trunc(n * density)); // Número de elementos > 0.0 por linha

    for (int i = 0; i < n; ++i) {
        double sum = 0.0;

        for (int j = 0; j < n - 1; ++j) {
            int _j = j + (j >= i);
            A[i * n + _j] = j < (d - 1) ? (rand() % n) + 1.0 : 0.0;
            sum += A[i * n + _j];
        }

        // Shuffling
        int k;
        double tmp;
        for (int j = n - 2; j >= 0; --j) {
            int _j = j + (j >= i);
            k = rand() % (j + 1);
            k += (k >= i);
            tmp = A[i * n + _j];
            A[i * n + _j] = A[i * n + k];
            A[i * n + k] = tmp;
        }

        A[i * n + i] = sum + (rand() % n) + 1.0;
    }

    return A;
}

double *randb(int n) {
    srand(time(NULL));

    double *b = (double *)malloc(n * sizeof(double));

    for (int i = 0; i < n; ++i)
        b[i] = (rand() % n);

    return b;
}

int main(int argc, char *argv[]) {
    int max_threads = omp_get_max_threads();
    int size = 4;
    double density = 1;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--max-threads"))
            max_threads = atoi(argv[i + 1]);
        if (!strcmp(argv[i], "--size"))
            size = atoi(argv[i + 1]);
        if (!strcmp(argv[i], "--density"))
            density = atof(argv[i + 1]);
    }

    printf("Tipo de Matriz: %s (Densidade = %.f%%)\n"
           "Tamanho: %d\n"
           "Nº Máximo de Threads/Processos: %d\n"
           "\n",
           density == 1 ? "Densa" : "Esparsa", density * 100, size, max_threads);

    double *A = randA(size, density);
    double *b = randb(size);

    printf("%19s %12s\n", "Algoritmo", "Tempo (s)");
    printf("%19s %12f\n", "Jacobi (Sequencial)", jacobiSequential(A, b, size));
    printf("%19s %12f\n", "Jacobi (Paralelo)", jacobiParallel(A, b, size, max_threads));
    printf("%19s %12f\n", "Seidel (Sequencial)", seidelSequential(A, b, size));

    free(A);
    free(b);

    return 0;
}
