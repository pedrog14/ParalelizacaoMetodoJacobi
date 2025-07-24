# Computação Paralela - Método de Gauss-Jacobi

Implementação em C do algoritmo sequencial e paralelo (utilizando a biblioteca [OpenMP](https://www.openmp.org/)) do método de Jacobi para resolução de sistemas lineares + método de Gauss-Seidel sequencial para comparação de tempo.

## Compilando...

    gcc ./gaussJacobiOMP.c -lm -fopenmp -O3

## Parâmetros de execução

    --size Int            # Define o tamanho do sistema a ser resolvido
    --density [0.0 - 1.0] # Define a densidade da matriz A
    --max-threads Int     # Define o número máximo de processos/threads utilizados pelo método paralelizado
