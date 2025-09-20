#include <iostream>
#include <vector>
#include <complex>
#include <random>
#include <Eigen/Dense>
#include <iomanip>
#include <chrono>
#include <map>
#include <Eigen/SparseCore>

// #include <Eigen/Sparse>
// using namespace std;
using namespace Eigen;

using Complex = std::complex<double>;
using ComplexMatrix = MatrixXcd;
using RealMatrix = MatrixXd;
using ComplexVector = VectorXcd;
using RealVector = VectorXd;
using ComplexTensor3 = std::vector<std::vector<std::vector<Complex>>>;
// static constexpr int count_of_threads = 4;
static constexpr int count_of_iter = 1;//10;
// static constexpr double gamma = 0.0001;
const double gamma_ = 0.1;
static std::map<std::string, ComplexMatrix> map_with_simmetrical_matrix;
static std::map<std::string, ComplexMatrix> map_with_antisymmetr;
static std::map<std::string, ComplexMatrix> map_with_diag;



void normalizeDiagonal(Eigen::MatrixXd& A, double target_rms = 1.0) {
    int n = A.rows();
    double sum_squares = 0.0;
    for (int i = 0; i < n; ++i)
        sum_squares += A(i, i) * A(i, i);

    double current_rms = std::sqrt(sum_squares / n);
    if (current_rms == 0) return;  // избежим деления на 0

    double scale = target_rms / current_rms;

    // Округляем масштабированные значения обратно в целые числа
    for (int i = 0; i < n; ++i)
        A(i, i) = static_cast<int>(std::round(A(i, i) * scale));

    // Повторно корректируем последний элемент, чтобы след был снова ноль
    int trace = A.diagonal().head(n - 1).sum();
    A(n - 1, n - 1) = -trace;
}
Eigen::MatrixXd generateSymmetricZeroTraceMatrix(int n, int rng) {
    // Инициализация генератора случайных чисел
    std::mt19937 gen(rng);
    std::uniform_int_distribution<> dist(-1.0, 1.0);

    // Генерация случайной матрицы
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(n, n);
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            double value = dist(gen);
            A(i, j) = value;
            A(j, i) = value;
        }
    }

    // Диагональные элементы сначала случайные
    for (int i = 0; i < n; ++i) {
        A(i, i) = dist(gen);
    }

    // Корректируем диагональ, чтобы след стал равен нулю
    double trace = A.trace();
    double correction = trace / n;
    for (int i = 0; i < n; ++i) {
        A(i, i) -= correction;
    }
    // normalizeDiagonal(A);
    return A;
}

// === Генерация базиса SU(N) ===
std::vector<ComplexMatrix> generate_su_basis(int N) {
    std::vector<ComplexMatrix> basis;

    // Симметричные S^(j,k)
    for (int j = 0; j < N; ++j) {
        for (int k = j + 1; k < N; ++k) {
            ComplexMatrix S = ComplexMatrix::Zero(N, N);
            S(j, k) = S(k, j) = 1.0 / sqrt(2.0);
            basis.push_back(S);
        }
    }

    // Антисимметричные J^(j,k)
    for (int j = 0; j < N; ++j) {
        for (int k = j + 1; k < N; ++k) {
            ComplexMatrix J = ComplexMatrix::Zero(N, N);
            J(j, k) = Complex(0, -1.0 / sqrt(2.0));
            J(k, j) = Complex(0, 1.0 / sqrt(2.0));
            basis.push_back(J);
        }
    }

    // Диагональные D^l
    for (int l = 1; l < N; ++l) {
        ComplexMatrix D = ComplexMatrix::Zero(N, N);
        double norm = sqrt(l * (l + 1.0));
        for (int k = 0; k < l; ++k)
            D(k, k) = Complex(1.0 / norm, 0.0);
        D(l, l) = Complex(-l / norm, 0.0);
        basis.push_back(D);
    }

    return basis;
}

// === Разложение матрицы A по базису: v_i = Tr(A * F_i) ===
ComplexVector decompose(const ComplexMatrix& A, const std::vector<ComplexMatrix>& basis) {
    int M = basis.size();
    ComplexVector v(M);
    int i = 0;
    // данную функцию не имеет смысла распараллеливать, тут всего один короткий цикл по перемножению маленьких матриц. На создание thread_ов будет больше потрачено времени, чем получим от multi_thread
    for (i = 0; i < M; ++i) {
        v[i] = (A * basis[i]).trace();
    }

    return v;
}


void compute_Fmns_Zmns(const std::vector<ComplexMatrix>& basis, ComplexTensor3& f_tensor, ComplexTensor3& z_tensor) {
    int M = basis.size();
    f_tensor.resize(M, std::vector<std::vector<Complex>>(M, std::vector<Complex>(M, Complex(0.0, 0.0))));
    z_tensor.resize(M, std::vector<std::vector<Complex>>(M, std::vector<Complex>(M, Complex(0.0, 0.0))));

    // ComplexMatrix commutator = ComplexMatrix::Zero(M, M);
    // ComplexMatrix anticommutator = ComplexMatrix::Zero(M, M);


    // for (int m = 0; m < M; ++m) {
    //     for (int n = 0; n < M; ++n) {
    //         commutator(m,n) = basis[m] * basis[n] - basis[n] * basis[m];
    //     }
    // }


    std::vector<std::vector<ComplexMatrix>> comm(M, std::vector<ComplexMatrix>(M, ComplexMatrix::Zero(M, M)));
    std::vector<std::vector<ComplexMatrix>> anti(M, std::vector<ComplexMatrix>(M, ComplexMatrix::Zero(M, M)));
    int m = 0, n = 0, s = 0;

    // #pragma omp parallel for private (m, n) shared (comm, anti, basis)
    // for (m = 0; m < M; ++m) {
    //     for (n = 0; n < M; ++n) {
    //         comm[m][n] = basis[m] * basis[n] - basis[n] * basis[m];
    //         anti[m][n] = basis[m] * basis[n] + basis[n] * basis[m];
    //     }
    // }

	// for (m = 0; m < M; ++m) {
	// 	for (n = m; n < M; ++n) {  // Вычисляем только верхний треугольник
	// 		// Антисимметричная часть (comm)
	// 		comm[m][n] = basis[m] * basis[n]; - basis[n] * basis[m];  // = bm_bn - bm_bn = 0, если m == n
	// 		comm[n][m] = -comm[m][n];                 // Нижний треугольник
	// 		// Симметричная часть (anti)
	// 		anti[m][n] = basis[m] * basis[n]; + basis[n] * basis[m];  // = 2 * bm_bn, если m != n
	// 		anti[n][m] = anti[m][n];                   // Нижний треугольник
	// 	}
	// }
	// Eigen::setNbThreads(2);

	// #pragma omp parallel for schedule(dynamic)
	// for (int m = 0; m < M; ++m) {
    //     for (int n = m; n < M; ++n) {  // Вычисляем только верхний треугольник

    //         MatrixXcd bm_bn = basis[m] * basis[n];
    //         MatrixXcd bn_bm = basis[n] * basis[m];

    //         comm[m][n] = bm_bn - bn_bm;
    //         comm[n][m] = -comm[m][n];  // Нижний треугольник

    //         anti[m][n] = bm_bn + bn_bm;
    //         anti[n][m] = anti[m][n];   // Нижний треугольник
    //     }
    // }

    short int i = 0;
    short int j = 1;
    short int k = 0;
    short int l = 1;
    const int N_ = basis[0].cols();
    std::vector<std::vector<ComplexMatrix>> comm_p(M, std::vector<ComplexMatrix>(M, ComplexMatrix::Zero(M, M)));
    std::vector<std::vector<ComplexMatrix>> anti_p(M, std::vector<ComplexMatrix>(M, ComplexMatrix::Zero(M, M)));
    std::string hash_Sij;
    std::string hash_Skl;
    const int size_ = N_ * (N_ - 1) / 2;

    for (int q = 0; q < M; q++) {
        for(int w = 0; w < M; w++) {
            comm_p[q][w] = ComplexMatrix::Zero(N_, N_);
            anti_p[q][w] = ComplexMatrix::Zero(N_, N_);
        }
    }

	for (int m = 0; m < size_; ++m) {
        for (int n = 0; n < size_; ++n) {

            // MatrixXcd bm_bn = basis[m] * basis[n];
            // MatrixXcd bn_bm = basis[n] * basis[m];

            // comm[m][n] = bm_bn - bn_bm;
            // std::cout << 

            // std::cout << " S1 = \n \n" << basis[m] << "\n \n S2 = " <<  basis[n] << " \n \n";
            if (i != k && j != l || i == k && j == l){
                // comm_p[m][n] = ComplexMatrix::Zero(N_, N_);

                if( j != l && j == k) {
                    std::string hash = "J_" + std::to_string(i) +'_'+ std::to_string(l);
                    comm_p[m][n] =  map_with_antisymmetr[hash] * Complex(0.0, 1.0) * (1.0/sqrt(2.0));
                } 

                else if (i == l) {
                    std::string hash = "J_" + std::to_string(k) +'_'+ std::to_string(j);
                    comm_p[m][n] =  (-1) * map_with_antisymmetr[hash] * Complex(0.0, 1.0) * (1.0/sqrt(2.0));
                }

            }
            else if (i != k && j == l) {
                std::string hash = "J_" + std::to_string(i) +'_'+ std::to_string(k);
                comm_p[m][n] =  map_with_antisymmetr[hash] * Complex(0.0, 1.0) * (1.0/sqrt(2.0));
                // std::cout << " i != k && j == l " << " | i = " << i << " k = " << k << " j = " << j << " l = " << l << " hash = " << hash << std::endl;
                
                if ( k < i ) {
                    std::string hash = "J_" + std::to_string(k) +'_'+ std::to_string(i);
                    comm_p[m][n] =  (-1) * map_with_antisymmetr[hash] * Complex(0.0, 1.0) * (1.0/sqrt(2.0));

                }
            }
            else if (i == k && j != l) {
                std::string hash = "J_" + std::to_string(j) +'_'+ std::to_string(l);
                comm_p[m][n] = map_with_antisymmetr[hash]* Complex(0.0, 1.0) * (1.0/sqrt(2.0));
                // std::cout << " i == k && j != l " << " | i = " << i << " k = " << k << " j = " << j << " l = " << l << " hash = " << hash << std::endl;
                if ( l < j) {
                    std::string hash = "J_" + std::to_string(l) +'_'+ std::to_string(j);
                    comm_p[m][n] = (-1) * map_with_antisymmetr[hash]* Complex(0.0, 1.0) * (1.0/sqrt(2.0));;
                }
            }
            // std::cout << "def comm =  \n \n" << comm[m][n] << "\n\n";
            // std::cout << "fast comm = \n \n" << comm_p[m][n] << "\n\n";
            // if (!(comm[m][n] == comm_p[m][n])) {
            //     // std::cout << "  = "  << (comm[m][n] - comm_p[m][n])<<"\n\n" << std::endl;
            //     std::cout << "All is shit, exit" << std::endl;
            //     std::cout << "def comm =  \n \n" << comm[m][n] << "\n\n";
            //     std::cout << "fast comm = \n \n" << comm_p[m][n] << "\n\n";
            //     // std::runtime_error("comm != fast comm");
            //     std::exit(-1);
            // }
            l++;

            if (l == N_) {
                k++;
                l = k + 1;
            }
            // std::cout << "_______________________________________________________________________________________________" << std::endl;
        }

        k = 0;
        l = 1;
        j++;

        if (j == N_) {
            i++;
            j = i+1;
        }

        // std::cout << "_______________________________________________________________________________________________" << std::endl;
    }

    ////////////////////////////////////////////////[J,J]////////////////////////////////////////////////////////
    i = 0;
    j = 1;
    k = 0;
    l = 1;
    for (int m = size_; m < 2*size_; ++m) {
        for (int n = size_; n < 2*size_; ++n) {

            if (i != k && j != l || i == k && j == l){
                // comm_p[m][n] = ComplexMatrix::Zero(N_, N_);

                if( j != l && j == k) {
                    std::string hash = "J_" + std::to_string(i) +'_'+ std::to_string(l);
                    comm_p[m][n] =  (-1) * map_with_antisymmetr[hash] * Complex(0.0, 1.0) * (1.0/sqrt(2.0));

                } 

                else if (i == l) {
                    std::string hash = "J_" + std::to_string(k) +'_'+ std::to_string(j);
                    comm_p[m][n] =  map_with_antisymmetr[hash] * Complex(0.0, 1.0) * (1.0/sqrt(2.0));
                }
                // std::cout << "def comm =  \n \n" << comm[m][n] << "\n\n";
                // std::cout << "fast comm = \n \n" << comm_p[m][n] << "\n\n";
            }
            else if (i != k && j == l) {
                std::string hash = "J_" + std::to_string(i) +'_'+ std::to_string(k);
                comm_p[m][n] = map_with_antisymmetr[hash] * Complex(0.0, 1.0) * (1.0/sqrt(2.0));
                // std::cout << " i != k && j == l " << " | i = " << i << " k = " << k << " j = " << j << " l = " << l << " hash = " << hash << std::endl;
                
                if ( k < i ) {
                    std::string hash = "J_" + std::to_string(k) +'_'+ std::to_string(i);
                    comm_p[m][n] =  (-1) * map_with_antisymmetr[hash] * Complex(0.0, 1.0) * (1.0/sqrt(2.0));
                }
                // std::cout << "def comm =  \n \n" << comm[m][n] << "\n\n";
                // std::cout << "fast comm = \n \n" << comm_p[m][n] << "\n\n";
            }
            else if (i == k && j != l) {
                std::string hash = "J_" + std::to_string(j) +'_'+ std::to_string(l);
                comm_p[m][n] =  map_with_antisymmetr[hash]* Complex(0.0, 1.0) * (1.0/sqrt(2.0));
                // std::cout << " i == k && j != l " << " | i = " << i << " k = " << k << " j = " << j << " l = " << l << " hash = " << hash << std::endl;
                if ( l < j) {
                    std::string hash = "J_" + std::to_string(l) +'_'+ std::to_string(j);
                    comm_p[m][n] = (-1) * map_with_antisymmetr[hash]* Complex(0.0, 1.0) * (1.0/sqrt(2.0));;
                }
                // std::cout << "def comm =  \n \n" << comm[m][n] << "\n\n";
                // std::cout << "fast comm = \n \n" << comm_p[m][n] << "\n\n";
            }

            // std::cout  << " m = " << m << " n = " << n << " | i = " << i << " j = " << j << " k = " << k << " l = " << l  << " hash = " << hash << std::endl;
            // std::cout << "def comm =  \n \n" << comm[m][n] << "\n\n";
            // std::cout << "fast comm = \n \n" << comm_p[m][n] << "\n\n";
            // if (!(comm[m][n] == comm_p[m][n])) {
            //     // std::cout << "  = "  << (comm[m][n] - comm_p[m][n])<<"\n\n" << std::endl;
            //     std::cout << "All is shit, exit" << std::endl;
            //     std::cout << "def comm =  \n \n" << comm[m][n] << "\n\n";
            //     std::cout << "fast comm = \n \n" << comm_p[m][n] << "\n\n";
            //     // std::runtime_error("comm != fast comm");
            //     std::exit(-1);
            // }
            l++;

            if (l == N_) {
                k++;
                l = k + 1;
            }
            // std::cout << "_______________________________________________________________________________________________" << std::endl;
        }

        k = 0;
        l = 1;
        j++;

        if (j == N_) {
            i++;
            j = i+1;
        }

        // std::cout << "_______________________________________________________________________________________________" << std::endl;
    }

        ////////////////////////////////////////////////[S,J]////////////////////////////////////////////////////////
    i = 0;
    j = 1;
    k = 0;
    l = 1;
    for (int m = 0; m < size_; ++m) {
        for (int n = size_; n < 2*size_; ++n) {
            comm_p[m][n] = ComplexMatrix::Zero(N_, N_);
            if (i != k && j != l){
                
                // comm_p[m][n] = ComplexMatrix::Zero(N_, N_);
                // std::cout  << " m = " << m << " n = " << n << " | i = " << i << " j = " << j << " k = " << k << " l = " << l  << " ZERO = " << std::endl;

                if( j != l && j == k) {
                    // std::cout << "j != l && j == k" << std::endl;
                    std::string hash = "S_" + std::to_string(i) +'_'+ std::to_string(l);
                    // std::cout  << " m = " << m << " n = " << n << " | i = " << i << " j = " << j << " k = " << k << " l = " << l  << " hash = " << hash << std::endl;
                    comm_p[m][n] =  (-1) * map_with_simmetrical_matrix[hash] * Complex(0.0, 1.0) * (1.0/sqrt(2.0));

                } 

                else if (i == l) {
                    // std::cout << "j != l && j == k && i == l" << std::endl;
                    std::string hash = "S_" + std::to_string(k) +'_'+ std::to_string(j);
                    comm_p[m][n] =  map_with_simmetrical_matrix[hash] * Complex(0.0, 1.0) * (1.0/sqrt(2.0));
                    // std::cout << "def comm =  \n \n" << comm[m][n] << "\n\n";
                    // std::cout << "fast comm = \n \n" << comm_p[m][n] << "\n\n";
                }

                // std::cout << "def comm =  \n \n" << comm[m][n] << "\n\n";
                // std::cout << "fast comm = \n \n" << comm_p[m][n] << "\n\n";
            }
            else if (i == k && j != l) {
                // std::cout << "i == k && j != l" << std::endl;
                std::string hash = "S_" + std::to_string(j) +'_'+ std::to_string(l);
                comm_p[m][n] =  (-1)*map_with_simmetrical_matrix[hash]* Complex(0.0, 1.0) * (1.0/sqrt(2.0));
                // std::cout << " i == k && j != l " << " | i = " << i << " k = " << k << " j = " << j << " l = " << l << " hash = " << hash << std::endl;
                if ( l < j) {
                    // std::cout << "i == k && j != l && l < j" << std::endl;
                    std::string hash = "S_" + std::to_string(l) +'_'+ std::to_string(j);
                    comm_p[m][n] = (-1)*map_with_simmetrical_matrix[hash]* Complex(0.0, 1.0) * (1.0/sqrt(2.0));;
                }
                // std::cout << "def comm =  \n \n" << comm[m][n] << "\n\n";
                // std::cout << "fast comm = \n \n" << comm_p[m][n] << "\n\n";
            }
            else if (i != k && j == l) {
                // std::cout << "i != k && j == l" << std::endl;
                std::string hash = "S_" + std::to_string(i) +'_'+ std::to_string(k);
                comm_p[m][n] = map_with_simmetrical_matrix[hash] * Complex(0.0, 1.0) * (1.0/sqrt(2.0));
                // std::cout << " i != k && j == l " << " | i = " << i << " k = " << k << " j = " << j << " l = " << l << " hash = " << hash << std::endl;
                
                if ( k < i ) {
                    // std::cout << "i != k && j == l && k < l" << std::endl;
                    std::string hash = "S_" + std::to_string(k) +'_'+ std::to_string(i);
                    comm_p[m][n] =   map_with_simmetrical_matrix[hash] * Complex(0.0, 1.0) * (1.0/sqrt(2.0));
                }
                // std::cout << "def comm =  \n \n" << comm[m][n] << "\n\n";
                // std::cout << "fast comm = \n \n" << comm_p[m][n] << "\n\n";
            }
            else if (i == k && j == l) {
                std::string hash1 = "J_" + std::to_string(i) +'_'+ std::to_string(k);
                std::string hash2 = "J_" + std::to_string(j) +'_'+ std::to_string(l);
                ComplexMatrix temp1 = map_with_antisymmetr[hash1] * Complex(0.0, 1.0/sqrt(2.0));
                ComplexMatrix temp2 = map_with_antisymmetr[hash2] * Complex(0.0, -1.0/sqrt(2.0));
                
                // comm_p[m][n] = temp1 * Complex(0.0, -1.0/sqrt(2.0));
                // if (l == N_ - 1) {
                //     temp2 = temp2 * Complex(1.0, 0.0);
                // }
                // std::cout  << " \n \n \n " << temp2<< "\n \n \n " << std::endl;
                // std::cout << "HASH = " <<hash2 << std::endl;

                // ComplexMatrix temp3 = temp1 + temp2;
                comm_p[m][n] = temp1 + temp2;

                // comm_p[m][n] =  (map_with_antisymmetr[hash1] * (1.0/sqrt(2.0)) + map_with_antisymmetr[hash2] * (1.0/sqrt(2.0)));
                // std::cout << " i == k && j == l " << " | i = " << i << " k = " << k << " j = " << j << " l = " << l << " hash = " << hash1 << " " << hash2 << std::endl;
                // std::cout << "def comm =  \n \n" << comm[m][n] << "\n\n";
                // std::cout << "fast comm = \n \n" << comm_p[m][n] << "\n\n";
            }

            // std::cout  << " m = " << m << " n = " << n << " | i = " << i << " j = " << j << " k = " << k << " l = " << l  << std::endl;
            // std::cout << "def comm =  \n \n" << comm[m][n] << "\n\n";
            // std::cout << "fast comm = \n \n" << comm_p[m][n] << "\n\n";
            // std::cout << "is similar  = "  << (comm[m][n] == comm_p[m][n])<<"\n\n" << std::endl;
            // if (!(comm[m][n] == comm_p[m][n])) {
            //     // std::cout << "  = "  << (comm[m][n] - comm_p[m][n])<<"\n\n" << std::endl;
            //     std::cout << "All is shit, exit" << std::endl;
            //     std::cout << "def comm =  \n \n" << comm[m][n] << "\n\n";
            //     std::cout << "fast comm = \n \n" << comm_p[m][n] << "\n\n";
            //     // std::runtime_error("comm != fast comm");
            //     std::exit(-1);
            // }

            l++;

            if (l == N_) {
                k++;
                l = k + 1;
            }
            // std::cout << "_______________________________________________________________________________________________" << std::endl;
        }

        k = 0;
        l = 1;
        j++;

        if (j == N_) {
            i++;
            j = i+1;
        }

        // std::cout << "_______________________________________________________________________________________________" << std::endl;
    }

    //////////////////////////////////////////////////////[S_ij, D_l] ////////////////////////////////////////////////////////
    i = 0;
    j = 1;
    l = 0;
    for (int m = 0; m < size_; ++m) {
        for (int n  = 2*size_; n < M; ++n) {
            // comm_p[m][n] = ComplexMatrix::Zero(N_,N_);
            // std::cout << "i = " << i << " j = " << j << " l = " << l << std::endl;
            if (j - 1 > l) {
                // std::cout << "i = " << i << " j = " << j << " l = " << l << std::endl;
                std::string hash = "J_" + std::to_string(i) +'_'+ std::to_string(j);

                comm_p[m][n] = Complex(0.0, 1.0) * (-1.0/sqrt(    (l+1) * (l+2)   ))* map_with_antisymmetr[hash];

                if (i > l) {
                    comm_p[m][n] *= (-1);
                }

                if(l+2 < i + 1 && l+2 < j+1) {
                    comm_p[m][n] = ComplexMatrix::Zero(N_, N_);
                }

                if ((i + 1) == (l+2)) {
                    comm_p[m][n]*=(l+1);
                }
            }
            else if (j - 1 == l) {
                // std::cout << "i = " << i << " j = " << j << " l = " << l << std::endl;
                std::string hash = "J_" + std::to_string(i) +'_'+ std::to_string(j);
                comm_p[m][n] = Complex(0.0, 1.0) * static_cast<double>((l+2)) *(-1.0/sqrt(    (l+1) * (l+2)   ))* map_with_antisymmetr[hash];

                // std::cout << "basis[m] = \n\n" << basis[m] << " \n\n basis[m] = \n\n" << basis[n] << std::endl;
                // std::cout << "\n\n def comm =  \n \n" << comm[m][n] << "\n\n";
                // std::cout << "fast comm = \n \n" << comm_p[m][n] << "\n\n";
            }

                // std::cout << "basis[m] = \n\n" << basis[m] << " \n\n basis[m] = \n\n" << basis[n] << std::endl;
                // std::cout << "\n\n def comm =  \n \n" << comm[m][n] << "\n\n";
                // std::cout << "fast comm = \n \n" << comm_p[m][n] << "\n\n";
                // if (!(comm[m][n] == comm_p[m][n])) {
                //     // std::cout << "  = "  << (comm[m][n] - comm_p[m][n])<<"\n\n" << std::endl;
                //     std::cout << "All is shit, exit" << std::endl;
                //     std::cout << "def comm =  \n \n" << comm[m][n] << "\n\n";
                //     std::cout << "fast comm = \n \n" << comm_p[m][n] << "\n\n";
                //     // std::runtime_error("comm != fast comm");
                //     std::exit(-1);
                // }

            l++;
            // std::cout << "_______________________________________________________________________________________________+++" << std::endl;
        }

        l = 0;
        j++;

        if (j == N_) {
            i++;
            j = i+1;
        }
        // std::cout << "_______________________________________________________________________________________________" << std::endl;
    }
///////////////////////////////////////////////////////[J,D]////////////////////////////////////////////////////////////////////
    i = 0;
    j = 1;
    l = 0;
    for (int m = size_; m < 2*size_; ++m) {
        for (int n  = 2*size_; n < M; ++n) {
            // comm_p[m][n] = ComplexMatrix::Zero(N_,N_);
            // std::cout << "i = " << i << " j = " << j << " l = " << l << std::endl;
            if (j - 1 > l) {
                // std::cout << "i = " << i << " j = " << j << " l = " << l << std::endl;
                std::string hash = "S_" + std::to_string(i) +'_'+ std::to_string(j);

                comm_p[m][n] = Complex(0.0, 1.0) * (1.0/sqrt(    (l+1) * (l+2)   ))* map_with_simmetrical_matrix[hash];

                if (i > l) {
                    comm_p[m][n] *= (-1);
                }

                if(l+2 < i + 1 && l+2 < j+1) {
                    comm_p[m][n] = ComplexMatrix::Zero(N_, N_);
                }

                if ((i + 1) == (l+2)) {
                    comm_p[m][n]*=(l+1);
                }


                // std::cout << "basis[m] = \n\n" << basis[m] << " \n\n basis[m] = \n\n" << basis[n] << std::endl;
                // std::cout << "\n\n def comm =  \n \n" << comm[m][n] << "\n\n";
                // std::cout << "fast comm = \n \n" << comm_p[m][n] << "\n\n";
            }
            else if (j - 1 == l) {
                // std::cout << "i = " << i << " j = " << j << " l = " << l << std::endl;
                std::string hash = "S_" + std::to_string(i) +'_'+ std::to_string(j);
                comm_p[m][n] = Complex(0.0, 1.0) * static_cast<double>((l+2)) *(1.0/sqrt(    (l+1) * (l+2)   ))* map_with_simmetrical_matrix[hash];

                // std::cout << "basis[m] = \n\n" << basis[m] << " \n\n basis[m] = \n\n" << basis[n] << std::endl;
                // std::cout << "\n\n def comm =  \n \n" << comm[m][n] << "\n\n";
                // std::cout << "fast comm = \n \n" << comm_p[m][n] << "\n\n";
            }

                // std::cout << "basis[m] = \n\n" << basis[m] << " \n\n basis[m] = \n\n" << basis[n] << std::endl;
                // std::cout << "\n\n def comm =  \n \n" << comm[m][n] << "\n\n";
                // std::cout << "fast comm = \n \n" << comm_p[m][n] << "\n\n";
                // if (!(comm[m][n] == comm_p[m][n])) {
                //     // std::cout << "  = "  << (comm[m][n] - comm_p[m][n])<<"\n\n" << std::endl;
                //     std::cout << "All is shit, exit" << std::endl;
                //     std::cout << "def comm =  \n \n" << comm[m][n] << "\n\n";
                //     std::cout << "fast comm = \n \n" << comm_p[m][n] << "\n\n";
                //     // std::runtime_error("comm != fast comm");
                //     std::exit(-1);
                // }

            l++;
            // std::cout << "_______________________________________________________________________________________________+++" << std::endl;
        }

        l = 0;
        j++;

        if (j == N_) {
            i++;
            j = i+1;
        }
        // std::cout << "_______________________________________________________________________________________________" << std::endl;
    }


        ////////////////////////////////////////////////{S,S}////////////////////////////////////////////////////////
                i = 0;
    j = 1;
    k = 0;
    l = 1;
    for (int m = 0; m < size_; ++m) {
        for (int n = 0; n < size_; ++n) {
            // anti_p[m][n] = ComplexMatrix::Zero(N_, N_);

            if (i != k && j != l) {
                
                // anti_p[m][n] = ComplexMatrix::Zero(N_, N_);
                // std::cout  << " m = " << m << " n = " << n << " | i = " << i << " j = " << j << " k = " << k << " l = " << l  << " ZERO = " << std::endl;

                if( j != l && j == k) {
                    std::string hash = "S_" + std::to_string(i) +'_'+ std::to_string(l);
                    // std::cout  << " m = " << m << " n = " << n << " | i = " << i << " j = " << j << " k = " << k << " l = " << l  << " hash = " << hash << std::endl;
                    anti_p[m][n] =  map_with_simmetrical_matrix[hash] * (1.0/sqrt(2.0));

                // std::cout << "def anti =  \n \n" << anti[m][n] << "\n\n";
                // std::cout << "fast anti_p = \n \n" << anti_p[m][n] << "\n\n";
                // if (!(anti[m][n] == anti_p[m][n])) {
                //     // std::cout << "  = "  << (comm[m][n] - comm_p[m][n])<<"\n\n" << std::endl;
                //     std::cout << "All is shit, exit" << std::endl;
                //     std::cout << "def anti =  \n \n" << anti[m][n] << "\n\n";
                //     std::cout << "fast anti_p = \n \n" << anti_p[m][n] << "\n\n";
                //     // std::runtime_error("comm != fast comm");
                //     std::exit(-1);
                // }

                } 

                else if (i == l) {
                    std::string hash = "S_" + std::to_string(k) +'_'+ std::to_string(j);
                    anti_p[m][n] = map_with_simmetrical_matrix[hash] * /*Complex(0.0, 1.0) */ (1.0/sqrt(2.0));

                    // std::cout << "def anti =  \n \n" << anti[m][n] << "\n\n";
                    // std::cout << "fast anti_p = \n \n" << anti_p[m][n] << "\n\n";
                    // if (!(anti[m][n] == anti_p[m][n])) {
                    //     // std::cout << "  = "  << (comm[m][n] - comm_p[m][n])<<"\n\n" << std::endl;
                    //     std::cout << "All is shit, exit" << std::endl;
                    //     std::cout << "def anti =  \n \n" << anti[m][n] << "\n\n";
                    //     std::cout << "fast anti_p = \n \n" << anti_p[m][n] << "\n\n";
                    //     // std::runtime_error("comm != fast comm");
                    //     std::exit(-1);
                    // }
                }

                // std::cout << "def anti =  \n \n" << anti[m][n] << "\n\n";
                // std::cout << "fast anti_p = \n \n" << anti_p[m][n] << "\n\n";
                // if (!(anti[m][n] == anti_p[m][n])) {
                //     // std::cout << "  = "  << (comm[m][n] - comm_p[m][n])<<"\n\n" << std::endl;
                //     std::cout << "All is shit, exit" << std::endl;
                //     std::cout << "def anti =  \n \n" << anti[m][n] << "\n\n";
                //     std::cout << "fast anti_p = \n \n" << anti_p[m][n] << "\n\n";
                //     // std::runtime_error("comm != fast comm");
                //     std::exit(-1);
                // }

            }
            else if (i == k && j != l) {
                std::string hash = "S_" + std::to_string(j) +'_'+ std::to_string(l);
                anti_p[m][n] =  map_with_simmetrical_matrix[hash] * (1.0/sqrt(2.0));
                // std::cout << " i == k && j != l " << " | i = " << i << " k = " << k << " j = " << j << " l = " << l << " hash = " << hash << std::endl;
                if ( l < j) {
                    std::string hash = "S_" + std::to_string(l) +'_'+ std::to_string(j);
                    anti_p[m][n] = map_with_simmetrical_matrix[hash] * (1.0/sqrt(2.0));;
                }
                // std::cout << "def anti =  \n \n" << anti[m][n] << "\n\n";
                // std::cout << "fast anti_p = \n \n" << anti_p[m][n] << "\n\n";
                // if (!(anti[m][n] == anti_p[m][n])) {
                //     // std::cout << "  = "  << (comm[m][n] - comm_p[m][n])<<"\n\n" << std::endl;
                //     std::cout << "All is shit, exit" << std::endl;
                //     std::cout << "def anti =  \n \n" << anti[m][n] << "\n\n";
                //     std::cout << "fast anti_p = \n \n" << anti_p[m][n] << "\n\n";
                //     // std::runtime_error("comm != fast comm");
                //     std::exit(-1);
                // }
            }
            else if (i != k && j == l) {
                std::string hash = "S_" + std::to_string(i) +'_'+ std::to_string(k);
                anti_p[m][n] = map_with_simmetrical_matrix[hash] * (1.0/sqrt(2.0));
                // std::cout << " i != k && j == l " << " | i = " << i << " k = " << k << " j = " << j << " l = " << l << " hash = " << hash << std::endl;
                
                if ( k < i ) {
                    std::string hash = "S_" + std::to_string(k) +'_'+ std::to_string(i);
                    anti_p[m][n] = map_with_simmetrical_matrix[hash] * (1.0/sqrt(2.0));
                }
                // std::cout << "def anti =  \n \n" << anti[m][n] << "\n\n";
                // std::cout << "fast anti_p = \n \n" << anti_p[m][n] << "\n\n";
                // if (!(anti[m][n] == anti_p[m][n])) {
                //     // std::cout << "  = "  << (comm[m][n] - comm_p[m][n])<<"\n\n" << std::endl;
                //     std::cout << "All is shit, exit" << std::endl;
                //     std::cout << "def anti =  \n \n" << anti[m][n] << "\n\n";
                //     std::cout << "fast anti_p = \n \n" << anti_p[m][n] << "\n\n";
                //     // std::runtime_error("comm != fast comm");
                //     std::exit(-1);
                // }
            }
            else if (i == k && j == l) {
                std::string hash1 = "S_" + std::to_string(i) +'_'+ std::to_string(k);
                std::string hash2 = "S_" + std::to_string(j) +'_'+ std::to_string(l);
                ComplexMatrix temp1 = map_with_simmetrical_matrix[hash1] * (1.0/sqrt(2.0));
                ComplexMatrix temp2 = map_with_simmetrical_matrix[hash2] * (1.0/sqrt(2.0));
                
                // comm_p[m][n] = temp1 * Complex(0.0, -1.0/sqrt(2.0));
                // if (l == N_ - 1) {
                //     temp2 = temp2 * Complex(1.0, 0.0);
                // }
                // std::cout  << " \n \n \n " << temp2<< "\n \n \n " << std::endl;
                // std::cout << "HASH = " <<hash2 << std::endl;

                // ComplexMatrix temp3 = temp1 + temp2;
                anti_p[m][n] = temp1 + temp2;

                // comm_p[m][n] =  (map_with_antisymmetr[hash1] * (1.0/sqrt(2.0)) + map_with_antisymmetr[hash2] * (1.0/sqrt(2.0)));
                // std::cout << " i == k && j == l " << " | i = " << i << " k = " << k << " j = " << j << " l = " << l << " hash = " << hash1 << " " << hash2 << std::endl;

                // std::cout << "def anti =  \n \n" << anti[m][n] << "\n\n";
                // std::cout << "fast anti_p = \n \n" << anti_p[m][n] << "\n\n";
                // if (!(anti[m][n] == anti_p[m][n])) {
                //     // std::cout << "  = "  << (comm[m][n] - comm_p[m][n])<<"\n\n" << std::endl;
                //     std::cout << "All is shit, exit" << std::endl;
                //     std::cout << "def anti =  \n \n" << anti[m][n] << "\n\n";
                //     std::cout << "fast anti_p = \n \n" << anti_p[m][n] << "\n\n";
                //     // std::runtime_error("comm != fast comm");
                //     std::exit(-1);
                // }
            }
                // std::cout << "def anti =  \n \n" << anti[m][n] << "\n\n";
                // std::cout << "fast anti_p = \n \n" << anti_p[m][n] << "\n\n";
                // if (!(anti[m][n] == anti_p[m][n])) {
                //     // std::cout << "  = "  << (comm[m][n] - comm_p[m][n])<<"\n\n" << std::endl;
                //     std::cout << "All is shit, exit" << std::endl;
                //     std::cout << "def anti =  \n \n" << anti[m][n] << "\n\n";
                //     std::cout << "fast anti_p = \n \n" << anti_p[m][n] << "\n\n";
                //     // std::runtime_error("comm != fast comm");
                //     std::exit(-1);
                // }
            l++;

            if (l == N_) {
                k++;
                l = k + 1;
            }
            // std::cout << "_______________________________________________________________________________________________" << std::endl;
        }

        k = 0;
        l = 1;
        j++;

        if (j == N_) {
            i++;
            j = i+1;
        }

        // std::cout << "_______________________________________________________________________________________________" << std::endl;
    }
    // i = 0;
    // j = 1;
    // k = 0;
    // l = 1;
    // for (int m = 0; m < size_; ++m) {
    //     for (int n = 0; n < size_; ++n) {
    //         anti_p[m][n] = ComplexMatrix::Zero(N_, N_);

    //         if (i != k && j != l){
                
    //             anti_p[m][n] = ComplexMatrix::Zero(N_, N_);
    //             // std::cout  << " m = " << m << " n = " << n << " | i = " << i << " j = " << j << " k = " << k << " l = " << l  << " ZERO = " << std::endl;

    //             if( j != l && j == k) {
    //                 std::string hash = "J_" + std::to_string(i) +'_'+ std::to_string(l);
    //                 // std::cout  << " m = " << m << " n = " << n << " | i = " << i << " j = " << j << " k = " << k << " l = " << l  << " hash = " << hash << std::endl;
    //                 anti_p[m][n] =  map_with_antisymmetr[hash] * /*Complex(0.0, 1.0) */ (1.0/sqrt(2.0));

    //             } 

    //             else if (i == l) {
    //                 std::string hash = "J_" + std::to_string(k) +'_'+ std::to_string(j);
    //                 anti_p[m][n] = map_with_antisymmetr[hash] * /*Complex(0.0, 1.0) */ (1.0/sqrt(2.0));
    //             }

    //         }

    //         else if (i == k && j != l) {
    //             std::string hash = "J_" + std::to_string(j) +'_'+ std::to_string(l);
    //             anti_p[m][n] =  map_with_antisymmetr[hash] * (-1.0/sqrt(2.0));
    //             // std::cout << " i == k && j != l " << " | i = " << i << " k = " << k << " j = " << j << " l = " << l << " hash = " << hash << std::endl;
    //             if ( l < j) {
    //                 std::string hash = "J_" + std::to_string(l) +'_'+ std::to_string(j);
    //                 anti_p[m][n] = map_with_antisymmetr[hash] * (1.0/sqrt(2.0));;
    //             }
    //         }

    //         else if (i != k && j == l) {
    //             std::string hash = "J_" + std::to_string(i) +'_'+ std::to_string(k);
    //             anti_p[m][n] = map_with_antisymmetr[hash] * (1.0/sqrt(2.0));
    //             // std::cout << " i != k && j == l " << " | i = " << i << " k = " << k << " j = " << j << " l = " << l << " hash = " << hash << std::endl;
                
    //             if ( k < i ) {
    //                 std::string hash = "J_" + std::to_string(k) +'_'+ std::to_string(i);
    //                 anti_p[m][n] =  map_with_antisymmetr[hash] * (-1.0/sqrt(2.0));
    //             }
    //         }

    //         l++;

    //         if (l == N_) {
    //             k++;
    //             l = k + 1;
    //         }
    //         // std::cout << "_______________________________________________________________________________________________" << std::endl;
    //     }

    //     k = 0;
    //     l = 1;
    //     j++;

    //     if (j == N_) {
    //         i++;
    //         j = i+1;
    //     }

    //     // std::cout << "_______________________________________________________________________________________________" << std::endl;
    // }
    ////////////////////////////////////////////////{S,J}////////////////////////////////////////////////////////
    i = 0;
    j = 1;
    k = 0;
    l = 1;
    for (int m = 0; m < size_; ++m) {
        for (int n = size_; n < 2*size_; ++n) {
            // anti_p[m][n] = ComplexMatrix::Zero(N_, N_);
            // MatrixXcd bm_bn = basis[m] * basis[n];
            // MatrixXcd bn_bm = basis[n] * basis[m];

            // comm[m][n] = bm_bn - bn_bm;
            // std::cout << 

            // std::cout << " S1 = \n \n" << basis[m] << "\n \n S2 = " <<  basis[n] << " \n \n";
            if (i != k && j != l || i == k && j == l) {
                // std::cout << " i != k && j == l " << " | i = " << i << " k = " << k << " j = " << j << " l = " << l << std::endl;
                // anti_p[m][n] = ComplexMatrix::Zero(N_, N_);

                if( j != l && j == k) {
                    std::string hash = "J_" + std::to_string(i) +'_'+ std::to_string(l);
                    anti_p[m][n] =  map_with_antisymmetr[hash]  * (1.0/sqrt(2.0));
                } 

                else if (i == l) {
                    std::string hash = "J_" + std::to_string(k) +'_'+ std::to_string(j);
                    anti_p[m][n] =   map_with_antisymmetr[hash] * (1.0/sqrt(2.0));
                }
            }
            else if (i != k && j == l) {
                std::string hash = "J_" + std::to_string(i) +'_'+ std::to_string(k);
                anti_p[m][n] =  map_with_antisymmetr[hash] * (-1.0/sqrt(2.0));
                // std::cout << " i != k && j == l " << " | i = " << i << " k = " << k << " j = " << j << " l = " << l << " hash = " << hash << std::endl;
                
                if ( k < i ) {
                    std::string hash = "J_" + std::to_string(k) +'_'+ std::to_string(i);
                    anti_p[m][n] =  map_with_antisymmetr[hash]  * (1.0/sqrt(2.0));
                }
                // std::cout << "def anti =  \n \n" << anti[m][n] << "\n\n";
                // std::cout << "fast anti_p = \n \n" << anti_p[m][n] << "\n\n";
                // if (!(anti[m][n] == anti_p[m][n])) {
                //     // std::cout << "  = "  << (comm[m][n] - comm_p[m][n])<<"\n\n" << std::endl;
                //     std::cout << "All is shit, exit" << std::endl;
                //     std::cout << "def anti =  \n \n" << anti[m][n] << "\n\n";
                //     std::cout << "fast anti_p = \n \n" << anti_p[m][n] << "\n\n";
                //     // std::runtime_error("comm != fast comm");
                //     std::exit(-1);
                // }
            }
            else if (i == k && j != l) {
                std::string hash = "J_" + std::to_string(j) +'_'+ std::to_string(l);
                anti_p[m][n] = map_with_antisymmetr[hash] * (1.0/sqrt(2.0));
                // std::cout << " i == k && j != l " << " | i = " << i << " k = " << k << " j = " << j << " l = " << l << " hash = " << hash << std::endl;
                if ( l < j) {
                    std::string hash = "J_" + std::to_string(l) +'_'+ std::to_string(j);
                    anti_p[m][n] =  map_with_antisymmetr[hash] * (-1.0/sqrt(2.0));;
                }

                // std::cout << "def anti =  \n \n" << anti[m][n] << "\n\n";
                // std::cout << "fast anti_p = \n \n" << anti_p[m][n] << "\n\n";
                // if (!(anti[m][n] == anti_p[m][n])) {
                //     // std::cout << "  = "  << (comm[m][n] - comm_p[m][n])<<"\n\n" << std::endl;
                //     std::cout << "All is shit, exit" << std::endl;
                //     std::cout << "def anti =  \n \n" << anti[m][n] << "\n\n";
                //     std::cout << "fast anti_p = \n \n" << anti_p[m][n] << "\n\n";
                //     // std::runtime_error("comm != fast comm");
                //     std::exit(-1);
                // }
            }
                // std::cout << "def anti =  \n \n" << anti[m][n] << "\n\n";
                // std::cout << "fast anti_p = \n \n" << anti_p[m][n] << "\n\n";
                // if (!(anti[m][n] == anti_p[m][n])) {
                //     // std::cout << "  = "  << (comm[m][n] - comm_p[m][n])<<"\n\n" << std::endl;
                //     std::cout << "All is shit, exit" << std::endl;
                //     std::cout << "def anti =  \n \n" << anti[m][n] << "\n\n";
                //     std::cout << "fast anti_p = \n \n" << anti_p[m][n] << "\n\n";
                //     // std::runtime_error("comm != fast comm");
                //     std::exit(-1);
                // }
            l++;

            if (l == N_) {
                k++;
                l = k + 1;
            }
            // std::cout << "_______________________________________________________________________________________________" << std::endl;
        }

        k = 0;
        l = 1;
        j++;

        if (j == N_) {
            i++;
            j = i+1;
        }

        // std::cout << "_______________________________________________________________________________________________" << std::endl;
    }
    ////////////////////////////////////////////////////////// {J,J} ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    i = 0;
    j = 1;
    k = 0;
    l = 1;
    for (int m = size_; m < 2*size_; ++m) {
        for (int n = size_; n < 2*size_; ++n) {
            // anti_p[m][n] = ComplexMatrix::Zero(N_, N_);

            if (i != k && j != l){
                
                // anti_p[m][n] = ComplexMatrix::Zero(N_, N_);
                // std::cout  << " m = " << m << " n = " << n << " | i = " << i << " j = " << j << " k = " << k << " l = " << l  << " ZERO = " << std::endl;

                if( j != l && j == k) {
                    std::string hash = "S_" + std::to_string(i) +'_'+ std::to_string(l);
                    // std::cout  << " m = " << m << " n = " << n << " | i = " << i << " j = " << j << " k = " << k << " l = " << l  << " hash = " << hash << std::endl;
                    anti_p[m][n] =  (-1) * map_with_simmetrical_matrix[hash] * (1.0/sqrt(2.0));

                } 

                else if (i == l) {
                    std::string hash = "S_" + std::to_string(k) +'_'+ std::to_string(j);
                    anti_p[m][n] = (-1) * map_with_simmetrical_matrix[hash] * /*Complex(0.0, 1.0) */ (1.0/sqrt(2.0));
                }

                // std::cout << "def anti =  \n \n" << anti[m][n] << "\n\n";
                // std::cout << "fast anti_p = \n \n" << anti_p[m][n] << "\n\n";
                // if (!(anti[m][n] == anti_p[m][n])) {
                //     // std::cout << "  = "  << (comm[m][n] - comm_p[m][n])<<"\n\n" << std::endl;
                //     std::cout << "All is shit, exit" << std::endl;
                //     std::cout << "def anti =  \n \n" << anti[m][n] << "\n\n";
                //     std::cout << "fast anti_p = \n \n" << anti_p[m][n] << "\n\n";
                //     // std::runtime_error("comm != fast comm");
                //     std::exit(-1);
                // }

            }
            else if (i == k && j != l) {
                std::string hash = "S_" + std::to_string(j) +'_'+ std::to_string(l);
                anti_p[m][n] =  map_with_simmetrical_matrix[hash] * (1.0/sqrt(2.0));
                // std::cout << " i == k && j != l " << " | i = " << i << " k = " << k << " j = " << j << " l = " << l << " hash = " << hash << std::endl;
                if ( l < j) {
                    std::string hash = "S_" + std::to_string(l) +'_'+ std::to_string(j);
                    anti_p[m][n] = map_with_simmetrical_matrix[hash] * (1.0/sqrt(2.0));;
                }
                // std::cout << "def anti =  \n \n" << anti[m][n] << "\n\n";
                // std::cout << "fast anti_p = \n \n" << anti_p[m][n] << "\n\n";
                // if (!(anti[m][n] == anti_p[m][n])) {
                //     // std::cout << "  = "  << (comm[m][n] - comm_p[m][n])<<"\n\n" << std::endl;
                //     std::cout << "All is shit, exit" << std::endl;
                //     std::cout << "def anti =  \n \n" << anti[m][n] << "\n\n";
                //     std::cout << "fast anti_p = \n \n" << anti_p[m][n] << "\n\n";
                //     // std::runtime_error("comm != fast comm");
                //     std::exit(-1);
                // }
            }
            else if (i != k && j == l) {
                std::string hash = "S_" + std::to_string(i) +'_'+ std::to_string(k);
                anti_p[m][n] = map_with_simmetrical_matrix[hash] * (1.0/sqrt(2.0));
                // std::cout << " i != k && j == l " << " | i = " << i << " k = " << k << " j = " << j << " l = " << l << " hash = " << hash << std::endl;
                
                if ( k < i ) {
                    std::string hash = "S_" + std::to_string(k) +'_'+ std::to_string(i);
                    anti_p[m][n] = map_with_simmetrical_matrix[hash] * (1.0/sqrt(2.0));
                }
                // std::cout << "def anti =  \n \n" << anti[m][n] << "\n\n";
                // std::cout << "fast anti_p = \n \n" << anti_p[m][n] << "\n\n";
                // if (!(anti[m][n] == anti_p[m][n])) {
                //     // std::cout << "  = "  << (comm[m][n] - comm_p[m][n])<<"\n\n" << std::endl;
                //     std::cout << "All is shit, exit" << std::endl;
                //     std::cout << "def anti =  \n \n" << anti[m][n] << "\n\n";
                //     std::cout << "fast anti_p = \n \n" << anti_p[m][n] << "\n\n";
                //     // std::runtime_error("comm != fast comm");
                //     std::exit(-1);
                // }
            }
            else if (i == k && j == l) {
                std::string hash1 = "S_" + std::to_string(i) +'_'+ std::to_string(k);
                std::string hash2 = "S_" + std::to_string(j) +'_'+ std::to_string(l);
                ComplexMatrix temp1 = map_with_simmetrical_matrix[hash1] * (1.0/sqrt(2.0));
                ComplexMatrix temp2 = map_with_simmetrical_matrix[hash2] * (1.0/sqrt(2.0));
                
                // comm_p[m][n] = temp1 * Complex(0.0, -1.0/sqrt(2.0));
                // if (l == N_ - 1) {
                //     temp2 = temp2 * Complex(1.0, 0.0);
                // }
                // std::cout  << " \n \n \n " << temp2<< "\n \n \n " << std::endl;
                // std::cout << "HASH = " <<hash2 << std::endl;

                // ComplexMatrix temp3 = temp1 + temp2;
                anti_p[m][n] = temp1 + temp2;

                // comm_p[m][n] =  (map_with_antisymmetr[hash1] * (1.0/sqrt(2.0)) + map_with_antisymmetr[hash2] * (1.0/sqrt(2.0)));
                // std::cout << " i == k && j == l " << " | i = " << i << " k = " << k << " j = " << j << " l = " << l << " hash = " << hash1 << " " << hash2 << std::endl;

                // std::cout << "def anti =  \n \n" << anti[m][n] << "\n\n";
                // std::cout << "fast anti_p = \n \n" << anti_p[m][n] << "\n\n";
                // if (!(anti[m][n] == anti_p[m][n])) {
                //     // std::cout << "  = "  << (comm[m][n] - comm_p[m][n])<<"\n\n" << std::endl;
                //     std::cout << "All is shit, exit" << std::endl;
                //     std::cout << "def anti =  \n \n" << anti[m][n] << "\n\n";
                //     std::cout << "fast anti_p = \n \n" << anti_p[m][n] << "\n\n";
                //     // std::runtime_error("comm != fast comm");
                //     std::exit(-1);
                // }
            }
                // std::cout << "def anti =  \n \n" << anti[m][n] << "\n\n";
                // std::cout << "fast anti_p = \n \n" << anti_p[m][n] << "\n\n";
                // if (!(anti[m][n] == anti_p[m][n])) {
                //     // std::cout << "  = "  << (comm[m][n] - comm_p[m][n])<<"\n\n" << std::endl;
                //     std::cout << "All is shit, exit" << std::endl;
                //     std::cout << "def anti =  \n \n" << anti[m][n] << "\n\n";
                //     std::cout << "fast anti_p = \n \n" << anti_p[m][n] << "\n\n";
                //     // std::runtime_error("comm != fast comm");
                //     std::exit(-1);
                // }
            l++;

            if (l == N_) {
                k++;
                l = k + 1;
            }
            // std::cout << "_______________________________________________________________________________________________" << std::endl;
        }

        k = 0;
        l = 1;
        j++;

        if (j == N_) {
            i++;
            j = i+1;
        }

        // std::cout << "_______________________________________________________________________________________________" << std::endl;
    }
//////////////////////////////////////////////////////////////// { S , D } /////////////////////////////////////////////////////////////////////

    i = 0;
    j = 1;
    l = 0;
    for (int m = 0; m < size_; ++m) {
        for (int n  = 2*size_; n < M; ++n) {
            // anti_p[m][n] = ComplexMatrix::Zero(N_,N_);
            // std::cout << "i = " << i << " j = " << j << " l = " << l << std::endl;
                if (j < l + 1) {
                    // std::string hash = "S_" + std::to_string(k) +'_'+ std::to_string(i);
                    std::string hash = "S_" + std::to_string(i) +'_'+ std::to_string(j);
                    // std::cout << "hash = " << hash << std::endl;
                    anti_p[m][n] = (2.0/sqrt(    (l+1) * (l+2)   ))* map_with_simmetrical_matrix[hash];
                    
                    // std::cout << "basis[m] = \n\n" << basis[m] << " \n\n basis[m] = \n\n" << basis[n] << std::endl;
                    // std::cout << "\n\n def comm =  \n \n" << anti[m][n] << "\n\n";
                    // std::cout << "fast comm = \n \n" << anti_p[m][n] << "\n\n";
                    // if (!(anti[m][n] == anti_p[m][n])) {
                    //     // std::cout << "  = "  << (comm[m][n] - comm_p[m][n])<<"\n\n" << std::endl;
                    //     std::cout << "All is shit, exit" << std::endl;
                    //     std::cout << "def comm =  \n \n" << comm[m][n] << "\n\n";
                    //     std::cout << "fast comm = \n \n" << comm_p[m][n] << "\n\n";
                    //     // std::runtime_error("comm != fast comm");
                    //     std::exit(-1);
                    // }
                }
                else if (l+1 == j && l != 0) {
                    std::string hash = "S_" + std::to_string(i) +'_'+ std::to_string(j);
                    // std::cout << "hash = " << hash << std::endl;
                    anti_p[m][n] = (-l)*(1.0/sqrt(    (l+1) * (l+2)   ))* map_with_simmetrical_matrix[hash];

                    // std::cout << "basis[m] = \n\n" << basis[m] << " \n\n basis[m] = \n\n" << basis[n] << std::endl;
                    // std::cout << "\n\n def comm =  \n \n" << anti[m][n] << "\n\n";
                    // std::cout << "fast comm = \n \n" << anti_p[m][n] << "\n\n";
                }
                else if (i < l+1 && j > l+1) {
                    std::string hash = "S_" + std::to_string(i) +'_'+ std::to_string(j);
                    // std::cout << "hash = " << hash << std::endl;
                    anti_p[m][n] = (1.0/sqrt(    (l+1) * (l+2)   ))* map_with_simmetrical_matrix[hash];

                    // std::cout << "basis[m] = \n\n" << basis[m] << " \n\n basis[m] = \n\n" << basis[n] << std::endl;
                    // std::cout << "\n\n def anti =  \n \n" << anti[m][n] << "\n\n";
                    // std::cout << "fast anti_p = \n \n" << anti_p[m][n] << "\n\n";
                }
                else if ( i == l+1) {
                    std::string hash = "S_" + std::to_string(i) +'_'+ std::to_string(j);
                    // std::cout << "hash = " << hash << std::endl;
                    anti_p[m][n] = (-1.0) * static_cast<double>((l+1))* (1.0/sqrt(    (l+1) * (l+2)   ))* map_with_simmetrical_matrix[hash];
                    // std::cout << "basis[m] = \n\n" << basis[m] << " \n\n basis[m] = \n\n" << basis[n] << std::endl;
                    // std::cout << "\n\n def anti =  \n \n" << anti[m][n] << "\n\n";
                    // std::cout << "fast anti_p = \n \n" << anti_p[m][n] << "\n\n";
                }
                    // std::cout << "basis[m] = \n\n" << basis[m] << " \n\n basis[m] = \n\n" << basis[n] << std::endl;
                    // std::cout << "\n\n def anti =  \n \n" << anti[m][n] << "\n\n";
                    // std::cout << "fast anti_p = \n \n" << anti_p[m][n] << "\n\n";
                // if (!(comm[m][n] == comm_p[m][n])) {
                //     // std::cout << "  = "  << (comm[m][n] - comm_p[m][n])<<"\n\n" << std::endl;
                //     std::cout << "All is shit, exit" << std::endl;
                //     std::cout << "def comm =  \n \n" << comm[m][n] << "\n\n";
                //     std::cout << "fast comm = \n \n" << comm_p[m][n] << "\n\n";
                //     // std::runtime_error("comm != fast comm");
                //     std::exit(-1);
                // }

            l++;
            // std::cout << "_______________________________________________________________________________________________+++" << std::endl;
        }

        l = 0;
        j++;

        if (j == N_) {
            i++;
            j = i+1;
        }
        // std::cout << "_______________________________________________________________________________________________" << std::endl;
    }

//////////////////////////////////////////////////////////////// { J , D } /////////////////////////////////////////////////////////////////////

    i = 0;
    j = 1;
    l = 0;
    for (int m = size_; m < 2*size_; ++m) {
        for (int n  = 2*size_; n < M; ++n) {
            // anti_p[m][n] = ComplexMatrix::Zero(N_,N_);
            // std::cout << "i = " << i << " j = " << j << " l = " << l << std::endl;
                if (j < l + 1) {
                    // std::string hash = "S_" + std::to_string(k) +'_'+ std::to_string(i);
                    std::string hash = "J_" + std::to_string(i) +'_'+ std::to_string(j);
                    // std::cout << "hash = " << hash << std::endl;
                    anti_p[m][n] = (2.0/sqrt(    (l+1) * (l+2)   ))* map_with_antisymmetr[hash];
                    
                    // std::cout << "basis[m] = \n\n" << basis[m] << " \n\n basis[m] = \n\n" << basis[n] << std::endl;
                    // std::cout << "\n\n def comm =  \n \n" << anti[m][n] << "\n\n";
                    // std::cout << "fast comm = \n \n" << anti_p[m][n] << "\n\n";
                    // if (!(anti[m][n] == anti_p[m][n])) {
                    //     // std::cout << "  = "  << (comm[m][n] - comm_p[m][n])<<"\n\n" << std::endl;
                    //     std::cout << "All is shit, exit" << std::endl;
                    //     std::cout << "def comm =  \n \n" << comm[m][n] << "\n\n";
                    //     std::cout << "fast comm = \n \n" << comm_p[m][n] << "\n\n";
                    //     // std::runtime_error("comm != fast comm");
                    //     std::exit(-1);
                    // }
                }
                else if (l+1 == j && l != 0) {
                    std::string hash = "J_" + std::to_string(i) +'_'+ std::to_string(j);
                    // std::cout << "hash = " << hash << std::endl;
                    anti_p[m][n] = (-l)*(1.0/sqrt(    (l+1) * (l+2)   ))* map_with_antisymmetr[hash];

                    // std::cout << "basis[m] = \n\n" << basis[m] << " \n\n basis[m] = \n\n" << basis[n] << std::endl;
                    // std::cout << "\n\n def comm =  \n \n" << anti[m][n] << "\n\n";
                    // std::cout << "fast comm = \n \n" << anti_p[m][n] << "\n\n";
                }
                else if (i < l+1 && j > l+1) {
                    std::string hash = "J_" + std::to_string(i) +'_'+ std::to_string(j);
                    // std::cout << "hash = " << hash << std::endl;
                    anti_p[m][n] = (1.0/sqrt(    (l+1) * (l+2)   ))* map_with_antisymmetr[hash];

                    // std::cout << "basis[m] = \n\n" << basis[m] << " \n\n basis[m] = \n\n" << basis[n] << std::endl;
                    // std::cout << "\n\n def anti =  \n \n" << anti[m][n] << "\n\n";
                    // std::cout << "fast anti_p = \n \n" << anti_p[m][n] << "\n\n";
                }
                else if ( i == l+1) {
                    std::string hash = "J_" + std::to_string(i) +'_'+ std::to_string(j);
                    // std::cout << "hash = " << hash << std::endl;
                    anti_p[m][n] = (-1.0) * static_cast<double>((l+1))* (1.0/sqrt(    (l+1) * (l+2)   ))* map_with_antisymmetr[hash];
                    // std::cout << "basis[m] = \n\n" << basis[m] << " \n\n basis[m] = \n\n" << basis[n] << std::endl;
                    // std::cout << "\n\n def anti =  \n \n" << anti[m][n] << "\n\n";
                    // std::cout << "fast anti_p = \n \n" << anti_p[m][n] << "\n\n";
                }
                    // std::cout << "basis[m] = \n\n" << basis[m] << " \n\n basis[m] = \n\n" << basis[n] << std::endl;
                    // std::cout << "\n\n def anti =  \n \n" << anti[m][n] << "\n\n";
                    // std::cout << "fast anti_p = \n \n" << anti_p[m][n] << "\n\n";
                // if (!(comm[m][n] == comm_p[m][n])) {
                //     // std::cout << "  = "  << (comm[m][n] - comm_p[m][n])<<"\n\n" << std::endl;
                //     std::cout << "All is shit, exit" << std::endl;
                //     std::cout << "def comm =  \n \n" << comm[m][n] << "\n\n";
                //     std::cout << "fast comm = \n \n" << comm_p[m][n] << "\n\n";
                //     // std::runtime_error("comm != fast comm");
                //     std::exit(-1);
                // }

            l++;
            // std::cout << "_______________________________________________________________________________________________+++" << std::endl;
        }

        l = 0;
        j++;

        if (j == N_) {
            i++;
            j = i+1;
        }
        // std::cout << "_______________________________________________________________________________________________" << std::endl;
    }

//////////////////////////////////////////////////////////////// {D, D} //////////////////////////////////////////////////////////////////////////////////
    l = 1;
    int m_D = 1;
    for (int m = 2*size_; m < M; ++m) {
        for (int n  = 2*size_; n < M; ++n) {
            // anti_p[m][n] = ComplexMatrix::Zero(N_,N_);
            // std::cout << "i = " << i << " j = " << j << " l = " << l << std::endl;
            // std::cout << "m = " << m_D << " l = " << l << std::endl;
            
            // if ( m > l) {
            //     std::string hash = "D_" + std::to_string(l+1);
            //     std::cout << "i = " << i << " j = " << j << " l = " << l << "hash = " << hash << std::endl;
            //     anti_p[m][n] = static_cast<double>(l+1) * map_with_diag[hash];
            //     std::cout << "basis[m] = \n\n" << basis[m] << " \n\n basis[m] = \n\n" << basis[n] << std::endl;
            //     std::cout << "\n\n def comm =  \n \n" << anti_p[m][n] << "\n\n";
            //     std::cout << "fast comm = \n \n" << anti_p[m][n] << "\n\n";
            // }
            if (l > m_D) {
                std::string hash = "D_" + std::to_string(m_D);
                // std::cout << "i = " << i << " j = " << j << " l = " << l << "hash = " << hash << std::endl; static_cast<double>(m_D)
                anti_p[m][n] = map_with_diag[hash] * (1.0/sqrt(    (l) * (l+1)   )) * 2.0;

                // std::cout << "basis[m] = \n\n" << basis[m] << " \n\n basis[m] = \n\n" << basis[n] << std::endl;
                // std::cout << "\n\n def anti =  \n \n" << anti[m][n] << "\n\n";
                // std::cout << "fast anti_p = \n \n" << anti_p[m][n] << "\n\n";
            }
            else if (l < m_D) {
                std::string hash = "D_" + std::to_string(l);
                // std::cout << "i = " << i << " j = " << j << " l = " << l << "hash = " << hash << std::endl; static_cast<double>(m_D)
                anti_p[m][n] = map_with_diag[hash] * (1.0/sqrt(    (m_D) * (m_D+1)   )) * 2.0;

                // std::cout << "basis[m] = \n\n" << basis[m] << " \n\n basis[m] = \n\n" << basis[n] << std::endl;
                // std::cout << "\n\n def anti =  \n \n" << anti[m][n] << "\n\n";
                // std::cout << "fast anti_p = \n \n" << anti_p[m][n] << "\n\n";
            }
            else if ( m_D == l) {
                std::string hash = "D_" + std::to_string(l) + "sqr";
                anti_p[m][n] = 2.0 *  map_with_diag[hash];

            }
                // std::cout << "basis[m] = \n\n" << basis[m] << " \n\n basis[m] = \n\n" << basis[n] << std::endl;
                // std::cout << "\n\n def anti =  \n \n" << anti[m][n] << "\n\n";
                // std::cout << "fast anti_p = \n \n" << anti_p[m][n] << "\n\n";
                // if (!(anti[m][n] == anti_p[m][n])) {
                //     // std::cout << "  = "  << (comm[m][n] - comm_p[m][n])<<"\n\n" << std::endl;
                //     std::cout << "All is shit, exit" << std::endl;
                //     std::cout << "def anti =  \n \n" << anti[m][n] << "\n\n";
                //     std::cout << "fast anti_p = \n \n" << anti_p[m][n] << "\n\n";
                //     // std::runtime_error("comm != fast comm");
                //     std::exit(-1);
                // }
            l++;
            // std::cout << "_______________________________________________________________________________________________+++" << std::endl;
        }

        l = 1;

        m_D++;

        // if (j == N_) {
        //     i++;
        //     j = i+1;
        // }
        // std::cout << "_______________________________________________________________________________________________" << std::endl;
    }

	#pragma omp parallel for schedule(dynamic)
	for (int m = 0; m < M; ++m) {
        for (int n = m; n < M; ++n) {
            // ComplexMatrix anti_sqr =anti[m][n] * anti[m][n];
            // ComplexMatrix anti_fast_sqr =anti_p[m][n] * anti_p[m][n];
            // ComplexMatrix zero = ComplexMatrix::Zero(N_, N_);
            // if (anti_sqr - anti_fast_sqr != zero) {
            //     std::cout << "m = " << m << " n = " << n << "\n" <<anti_sqr - anti_fast_sqr << "\n\n" << std::endl;

            //     std::cout << "def comm =  \n \n" << comm[m][n] << "\n\n";
            //     std::cout << "fast comm = \n \n" << comm_p[m][n] << "\n\n";
            //     std::cout << "_______________________________________________________________________________________________" << std::endl;
            // }

            comm_p[n][m] = -comm_p[m][n];
            anti_p[n][m] = anti_p[m][n];
        }
        // std::cout << "***********************************************************************************************" << std::endl;
    }

    // #pragma omp parallel for private (m, n, s) shared (comm, anti, basis, f_tensor, z_tensor)
	// #pragma omp parallel for collapse(2) schedule(dynamic)
	Complex f_mns;
	Complex z_mns;

	Complex minus_i = Complex(0.0, -1.0);
	Complex plus_i = Complex(0.0, 1.0);

	#pragma omp parallel for schedule(dynamic)
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < M; ++n) {

            // const ComplexMatrix commutator = comm[m][n];
            const ComplexMatrix commutator = comm_p[m][n];
			const bool isCommutatorZero = commutator.isZero();
            const ComplexMatrix anti_commutator = anti_p[m][n];
			const bool isAntiCommutatorZero = anti_commutator.isZero();

			if(!isCommutatorZero && !isAntiCommutatorZero) {
				for (s = 0; s < M; ++s) {
					Complex f_mns = minus_i * (basis[s] * commutator).trace();
					Complex z_mns = f_mns + plus_i * (basis[s] * anti_commutator).trace();

					f_tensor[m][n][s] = f_mns;
					z_tensor[m][n][s] = z_mns;
				}
			} else if (!isCommutatorZero && isAntiCommutatorZero) {
				for (s = 0; s < M; ++s) {
					Complex f_mns = minus_i * (basis[s] * commutator).trace();
					
					f_tensor[m][n][s] = f_mns;
				}
			} else if (isCommutatorZero && !isAntiCommutatorZero) {
				for (s = 0; s < M; ++s) {
					Complex z_mns = plus_i * (basis[s] * anti_commutator).trace();

					z_tensor[m][n][s] = z_mns;
				}
			}
        }
    }
}

// void filling_tensors_using_formula(const std::vector<ComplexMatrix>& basis, ComplexTensor3& f_tensor, ComplexTensor3& z_tensor) {
//     int M = basis.size();
//     f_tensor.resize(M, std::vector<std::vector<Complex>>(M, std::vector<Complex>(M, Complex(0.0, 0.0))));
//     z_tensor.resize(M, std::vector<std::vector<Complex>>(M, std::vector<Complex>(M, Complex(0.0, 0.0))));


// }

inline ComplexMatrix compute_Q(const ComplexVector& h, const ComplexTensor3& f) {
    int M = h.size();
    ComplexMatrix Q = ComplexMatrix::Zero(M, M);

    for (int s = 0; s < M; ++s) {
        for (int n = 0; n < M; ++n) {
            for (int m = 0; m < M; ++m) {
                Q(s, n) += (h(m) * f[m][n][s]).real();
            }
        }
    }

    return Q;
}

inline ComplexMatrix compute_Q_(const ComplexVector& h, const ComplexTensor3& f) {
    const int M = h.size();
    ComplexMatrix Q = ComplexMatrix::Zero(M, M);


    #pragma omp parallel
    {
        ComplexMatrix Q_private = ComplexMatrix::Zero(M, M);
        
        #pragma omp for nowait
        for (int s = 0; s < M; ++s) {
            for (int n = 0; n < M; ++n) {
                double sum_real = 0.0;
                
                // Векторизованный цикл без редукции
                for (int m = 0; m < M; ++m) {
                    const auto product = h(m) * f[m][n][s];
                    sum_real += product.real();
                }
                
                Q_private(s,n) = sum_real;
            }
        }
        
        #pragma omp critical
        Q += Q_private;
    }

    return Q;
}

// K_s = (i / N) * sum_{m,n} l_m * conj(l_n) * f_{mns}
ComplexVector compute_K_vector(const ComplexVector& l, const ComplexVector& l_conj, const ComplexTensor3& f_tensor, int N) {
    int M = l.size();
    ComplexVector K = ComplexVector::Zero(M);
    Complex filler(0.0, 1.0);
    Complex N_complex(N, 0.0);
    for (int s = 0; s < M; ++s) {
        Complex sum (0.0, 0.0);
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < M; ++n) {
                sum += ((l[m] * l_conj[n] * f_tensor[m][n][s]) * filler);
            }
        }
        K[s] = sum / N_complex; // (i / N) * sum
    }
    // matrixK[i] += real(fillerMatrixK[i][m] * filler) / 3.0; // check matrixK[5] and add i/N
    return K;
}

ComplexMatrix compute_R_matrix(const double gamma, const int size, const ComplexVector& l, const ComplexVector& lConj, ComplexTensor3& Fmns, ComplexTensor3& Zmns) {
    std::complex<double> fillerMatrixEone[l.size()][l.size()] = { 0 };
    for (int i = 0; i < l.size(); ++i) {
        if (real(l[i]) != 0 || imag(l[i]) != 0) {
            for (int m = 0; m < l.size(); ++m) {
                for (int n = 0; n < l.size(); ++n) {
                    fillerMatrixEone[m][n] += l[i] * Zmns[i][m][n];
                }
            }
        }
    }

    std::complex<double> fillerMatrixEtwo[l.size()][l.size()] = { 0 };
    for (int i = 0; i < l.size(); ++i) {
        if (real(l[i]) != 0 || imag(l[i]) != 0) {
            for (int m = 0; m < l.size(); ++m) {
                for (int n = 0; n < l.size(); ++n) {
                    fillerMatrixEtwo[m][n] += lConj[i] * Fmns[i][m][n];
                }
            }
        }
    }

    // double matrixR[size * size][size * size] = { 0 };
    ComplexMatrix matrixR = ComplexMatrix::Zero(l.size(), l.size());
    for (int i = 0; i < l.size(); ++i) {
        for (int j = 0; j < l.size(); ++j) {
            for (int u = 0; u < l.size(); ++u) {
                // matrixR(j,i) += -gamma * real(fillerMatrixEone[i][u] * fillerMatrixEtwo[j][u]);
                matrixR(j,i) += gamma * real(fillerMatrixEone[i][u] * fillerMatrixEtwo[j][u]);
            }
        }
    }
    matrixR *= 0.5;
    return matrixR;
}

inline ComplexMatrix compute_R(const ComplexVector& l, const ComplexTensor3& f, const ComplexTensor3& z, const std::complex<double>& gamma_) {
    int M = l.size();
    ComplexMatrix R = ComplexMatrix::Zero(M, M);
    const ComplexVector l_conj = l.conjugate();
    double g = 0.1;
	const Complex zero(0.0, 0.0);
	double l_real = 0.0;
	double l_imag = 0.0;











    for (int s = 0; s < M; ++s) {
        for (int n = 0; n < M; ++n) {
            for (int j = 0; j < M; ++j) {

				l_real = l[j].real();
				l_imag = l[j].imag();

				if ( std::abs(l_real) < 1e-15 && std::abs(l_imag) < 1e-15 ) {
					for (int k = 0; k < M; ++k) {
						if(std::abs(l_conj[k].real()) < 1e-15 && std::abs(l_conj[k].imag()) < 1e-15) {
							for (int l_idx = 0; l_idx < M; ++l_idx) {
								// R(s, n) += gamma_ * l(j) * std::conj(l(k)) * (  (z[j][l_idx][n]) * f[k][l_idx][s] + std::conj(z[k][l_idx][n]) * f[j][l_idx][s] );
								// if (l[j] != )
								R(s, n) += l[j] * l_conj[k] * (  (z[j][l_idx][n]) * f[k][l_idx][s] + std::conj(z[k][l_idx][n]) * f[j][l_idx][s] );
								// R(s, n) += l(j) * std::conj(l(n)) * (  (z[j][l_idx][n]) * f[k][l_idx][s] + std::conj(z[k][l_idx][n]) * f[j][l_idx][s] );
							}
						}
					}
				}
            }
        }
    }

    // std::cout << " R bef = " << std::endl << R << std::endl;
    R *= -0.5 * g;
    // std::cout << " R aft = " << std::endl << R << std::endl;
    // R *= -0.25 * gamma_;
    // R *= -0.25;

    return R;
}



inline ComplexMatrix compute_R_(const ComplexVector& l, const ComplexTensor3& f, const ComplexTensor3& z, const std::complex<double>& gamma_) {
    const int M = l.size();
    ComplexMatrix R = ComplexMatrix::Zero(M, M);
    const ComplexVector l_conj = l.conjugate();
    const double g = 0.1;

    // Создаем временные матрицы для каждого потока
    std::vector<ComplexMatrix> thread_results(omp_get_max_threads(), ComplexMatrix::Zero(M, M));

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        ComplexMatrix& R_local = thread_results[tid];
        
        #pragma omp for collapse(2) schedule(dynamic) nowait
        for (int s = 0; s < M; ++s) {
            for (int n = 0; n < M; ++n) {
                for (int j = 0; j < M; ++j) {
                    const double l_real = l[j].real();
                    const double l_imag = l[j].imag();

                    if (std::abs(l_real) < 1e-15 && std::abs(l_imag) < 1e-15) {
                        for (int k = 0; k < M; ++k) {
                            const double l_conj_real = l_conj[k].real();
                            const double l_conj_imag = l_conj[k].imag();
                            
                            if (std::abs(l_conj_real) < 1e-15 && std::abs(l_conj_imag) < 1e-15) {
                                for (int l_idx = 0; l_idx < M; ++l_idx) {
                                    R_local(s,n) += l[j] * l_conj[k] * (z[j][l_idx][n] * f[k][l_idx][s] + std::conj(z[k][l_idx][n]) * f[j][l_idx][s]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    // Собираем результаты от всех потоков
    for (const auto& local_R : thread_results) {
        R += local_R;
    }

    R *= -0.5 * g;
    return R;
}



inline ComplexMatrix compute_R_optimized(const ComplexVector& l, ComplexTensor3& f, ComplexTensor3& z, const std::complex<double>& gamma_) {
    const int M = l.size();
    ComplexMatrix R = ComplexMatrix::Zero(M, M);
    const ComplexVector l_conj = l.conjugate();
    const double g = 0.1;
    
    std::vector<Eigen::SparseMatrix<std::complex<double>>> FcomplexTensorCRS(M);
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < M; j++) {
            for(int k = 0; k < M; k++) {
                
            }
        }
    }

    // for (int i = 0; i < M; i++) {
    //     for(int j = 0; j < M; j++) {
    //         for(int k = 0; k < M; k++) {
    //             f[i][j][k] = l_conj[i] * f[i][j][k];
    //             z[i][j][k] = l[i] * z[i][j][k];
    //         }
    //     }
    // }



    // Создаем временные матрицы для каждого потока
    std::vector<ComplexMatrix> thread_results(omp_get_max_threads(), ComplexMatrix::Zero(M, M));

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        ComplexMatrix& R_local = thread_results[tid];
        
        #pragma omp for collapse(2) schedule(dynamic) nowait
        for (int s = 0; s < M; ++s) {
            for (int n = 0; n < M; ++n) {
                for (int j = 0; j < M; ++j) {
                    const double l_real = l[j].real();
                    const double l_imag = l[j].imag();

                    if (std::abs(l_real) < 1e-15 && std::abs(l_imag) < 1e-15) {
                        for (int k = 0; k < M; ++k) {
                            const double l_conj_real = l_conj[k].real();
                            const double l_conj_imag = l_conj[k].imag();
                            
                            if (std::abs(l_conj_real) < 1e-15 && std::abs(l_conj_imag) < 1e-15) {
                                for (int l_idx = 0; l_idx < M; ++l_idx) {
                                    R_local(s,n) += l[j] * l_conj[k] * (z[j][l_idx][n] * f[k][l_idx][s] + std::conj(z[k][l_idx][n]) * f[j][l_idx][s]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    // Собираем результаты от всех потоков
    for (const auto& local_R : thread_results) {
        R += local_R;
    }

    R *= -0.5 * g;
    return R;
}







ComplexMatrix to_complex_matrix(const MatrixXd& real_matrix) {
    return real_matrix.cast<std::complex<double>>();
}


inline const ComplexMatrix get_b1_conj_transp(const int N) {
    ComplexMatrix b = ComplexMatrix::Zero(N, N);
    for (int q = 1; q <= N - 1; q++) {
        b(q - 1, q) = static_cast<std::complex<double>>((sqrt(q)));
    }

    return b;
}

inline const ComplexMatrix get_b2_conj_transp(const int N) {
    ComplexMatrix b = ComplexMatrix::Zero(N, N);
    for (int q = 1; q <= N - 1; q++) {
        b(q - 1, q) = static_cast<std::complex<double>>((sqrt(N - q)));
    }

    return b;
}

inline const ComplexMatrix get_b1(const int N) {
    ComplexMatrix b = get_b1_conj_transp(N);;
    return b.adjoint();
}

inline const ComplexMatrix get_b2(const int N) {
    ComplexMatrix b = get_b2_conj_transp(N);;
    return b.adjoint();
}

const ComplexMatrix get_L(const int N) {
    ComplexMatrix b1 = get_b1(N);
    ComplexMatrix b2 = get_b2(N);

    ComplexMatrix b1_conj_transp = get_b1_conj_transp(N);
    ComplexMatrix b2_conj_transp = get_b2_conj_transp(N);

    static ComplexMatrix res = (b1_conj_transp + b2_conj_transp)*(b1 + b2);

    return res;
}


const ComplexMatrix get_L_D (const int N, const ComplexMatrix& rho, const ComplexMatrix& L_) {

    ComplexMatrix L = L_;
    ComplexMatrix L_conj_transp = L.adjoint();

    ComplexMatrix L_rho_L_conj_trans = ((L * rho) * L_conj_transp);
    ComplexMatrix L_conj_trans_L_rho = ((L_conj_transp * L) * rho);
    ComplexMatrix rho_L_conj_trans_L = ((rho * L_conj_transp) * L);

    
    
    ComplexMatrix L_D = (0.1 / (N-1)) * (L_rho_L_conj_trans - 0.5 * (L_conj_trans_L_rho + rho_L_conj_trans_L));
    // return L_D;

    // ComplexMatrix zero = ComplexMatrix::Zero(N,N);
    return (L_D * 0);

}

const ComplexMatrix funcInit(const int N, const double t, const ComplexMatrix& H, const ComplexMatrix& rho, const ComplexMatrix L_) {
    ComplexMatrix commutator_H_rho = H * rho - rho * H;
    commutator_H_rho = commutator_H_rho * std::complex<double>(0.0, -1.0);
    return (commutator_H_rho + get_L_D(N, rho, L_));
    // return commutator_H_rho;
}

ComplexMatrix rk4init(const int N, double t, const ComplexMatrix& H, /*const ComplexMatrix L_D,*/ const ComplexMatrix& rho, const double dt, const ComplexMatrix L_) {

    const int size_matrix = N * N;
    ComplexMatrix k1(N,N);
    ComplexMatrix k2(N,N);
    ComplexMatrix k3(N,N);
    ComplexMatrix k4(N,N);

    ComplexMatrix temp_rho(N,N);

    double h = dt;

    k1 = funcInit(N, t, H, rho, L_);

    // for (int i = 0; i < size_matrix; i++) {
    //     temp_rho(i,0) = rho(i,0) + (h / 2) * k1(i,0);
    temp_rho = rho + k1 * static_cast<double>((h / 2));
    // }


    k2 = funcInit(N, t + h / 2, H, temp_rho, L_);

    // for (int i = 0; i < size_matrix; i++) {
    //     temp_rho(i,0) = rho(i,0) + (h / 2) * k2(i,0);
    temp_rho = rho + k2 * static_cast<double>((h / 2));
    // }

    k3 = funcInit(N, t + h / 2, H, temp_rho, L_);

    // for (int i = 0; i < size_matrix; i++) {
    //     temp_rho(i,0) = rho(i,0) + h * k3(i,0);
    temp_rho = rho + k3 * static_cast<double>(h);
    // }

    k4 = funcInit(N, t + h, H, temp_rho, L_);

    // for (int i = 0; i < size_matrix; i++) {
    //     std::complex<double> const_two(2.0 , 0.0);
    //     std::complex<double> const__h_div_6(h/6 , 0.0);
    //     result(i,0) = rho(i,0) + static_cast<double>(h/6) * (k1(i,0) + 2.0 * k2(i,0) + 2.0 * k3(i,0) + k4(i,0));
    auto result = rho + static_cast<double>(h/6) * (k1 + k2 * 2.0 + k3 * 2.0 + k4);
    // }

    return result;
}



// Функция нормализации следа матрицы
// Изменяет элементы главной диагонали так, что trace(matrix) становится равным 1
void normalizeTrace(Eigen::MatrixXcd &matrix) {

    int n = matrix.rows();
    // Вычисляем текущий след матрицы (сумма диагональных элементов)
    std::complex<double> currentTrace = matrix.trace();
    
    // Вычисляем сдвиг, который необходимо прибавить к каждому диагональному элементу:
    // Для новых диагональных элементов A(i,i)_new = A(i,i)_old + delta, 
    // тогда новый след будет: trace_new = trace_old + n * delta,
    // а условие trace_new == 1 даст: delta = (1 - trace_old) / n.
    std::complex<double> delta = (std::complex<double>(1.0, 0.0) - currentTrace) / static_cast<double>(n);
    
    // Модифицируем диагональные элементы
    for (int i = 0; i < n; i++) {
        matrix(i, i) += delta;
    }
}


ComplexMatrix getErmmitMatrix(const int N) {
        // Создаем две матрицы случайных вещественных чисел для действительной и мнимой частей
        Eigen::MatrixXd realPart = Eigen::MatrixXd::Random(N, N);
        Eigen::MatrixXd imagPart = Eigen::MatrixXd::Random(N, N);
    
        // Составляем матрицу комплексных чисел A из случайных вещественных и мнимых частей
        Eigen::MatrixXcd A(N, N);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A(i, j) = std::complex<double>(realPart(i, j), imagPart(i, j));
            }
        }
    
        // Построение гермитовой матрицы H из матрицы A:
        // Берем сумму A и её аджойнт (сопряжённо-транспонированную матрицу) и делим на 2.
        Eigen::MatrixXcd H = (A + A.adjoint()) / 2.0;
        normalizeTrace(H);
    return H;
}

ComplexMatrix ComputeInitialSys(const int N, const ComplexMatrix& H, const ComplexMatrix& rho, const double dt, const ComplexMatrix L_, const double t_start, const double t_end) { 

    ComplexMatrix rho_res = rho;

    for (double t = t_start; t < t_end; t += dt) {
        rho_res = rk4init(N, 3.14, H, rho_res, dt, L_);
    }
    return rho_res;
}

inline ComplexVector func(double t, const ComplexVector& vectorV, const ComplexMatrix& matrixQ, const ComplexMatrix& matrixR, const ComplexVector& matrixK) {
    ComplexVector out_vec = ( (matrixQ + matrixR) * vectorV + matrixK );
    return out_vec;
}


inline ComplexVector deriv_for_real(const int N, const ComplexVector& vectorV, const ComplexMatrix& matrixQ, const ComplexMatrix& matrixR, const ComplexVector& vectorK) {
    ComplexVector out_vec = ( (matrixQ + matrixR) * vectorV + vectorK );
    return out_vec;
}

ComplexVector rk4_step_real_system(const int N,const double curr_time, const double dt, const ComplexVector& vectorV, const ComplexMatrix& matrixQ, const ComplexMatrix& matrixR, const ComplexVector& vectorK) {
    
    auto k1 = deriv_for_real(N, vectorV, matrixQ, matrixR, vectorK);
    auto k2 = deriv_for_real(N, vectorV + (k1 * (dt/2)  ), matrixQ, matrixR, vectorK);
    auto k3 = deriv_for_real(N, vectorV + (k2 * (dt/2)  ), matrixQ, matrixR, vectorK);
    auto k4 = deriv_for_real(N, vectorV + (k3 * dt      ), matrixQ, matrixR, vectorK);

    ComplexVector result = vectorV + (k1 + k2*2+ k3*2 + k4) * (dt/6);
    return result;
}

ComplexVector rk4_for_real_system_NEW(const int N, const double t_start, const double t_end, const double dt, const ComplexVector& vectorV, const ComplexMatrix& matrixQ, const ComplexMatrix& matrixR, const ComplexVector& vectorK) {

    ComplexVector tempV = vectorV;
    const size_t steps = static_cast<const size_t>(std::round(  (t_end - t_start) / dt  ));
    double currr_time = 0.0;

    for (size_t q = 0; q < steps; q++) {
        tempV = rk4_step_real_system(N, currr_time, dt, tempV, matrixQ, matrixR, vectorK);
        currr_time += dt;
    }

    return tempV; 
}



void print_matrix (const ComplexMatrix& matrix) {
    for (int m = 0; m < 3 * 3 - 1 ; ++m) {
        for (int n = 0; n < 3 * 3 - 1; ++n) {
            std::cout << std::setw(10) << matrix(m,n) << ' ';
        }
        std::cout << std::endl;
    }
}

void print_tensor(const int size, const ComplexTensor3& tensor) {

    for(int i = 0; i < size; i++) {
        std::cout << "layer = " << i << std::endl;
        for(int j = 0; j < size; j++) {

            for(int k = 0; k < size; k++) {
                std::cout << tensor[i][j][k] << "        ";
            }
            std::cout << std::endl;
        }
        std::cout<< "---------------------------------------------------------------------------------------------------------------" <<std::endl;
    }
    
}


ComplexMatrix get_L_ODE(const int N, const ComplexMatrix& H, const ComplexMatrix& rho, const double dt, const ComplexMatrix L_, const double t_start, const double t_end) {

    ComplexMatrix L_D = get_L_D(N, rho, L_);
    // ComplexMatrix L = get_L(N);
    // std::cout << " L = " << std::endl << L << std::endl;
    // std::cout << "L_D = " << L_D << std::endl;
    // std::cout << 
    std::vector<ComplexMatrix> SU_N_BASIS =  generate_su_basis(N);
    // std::cout << "basis = " << std::endl;
    // for(int q = 0; q < N*N -1; q++) {
    //     std::cout << SU_N_BASIS[q] << std::endl <<'\n';
    // }


    std::complex<double> zero(0.0, 0.0);
    ComplexVector h = decompose(H, SU_N_BASIS);
    // std::cout << "h = " << std::endl << h << std::endl;

    ComplexVector l = decompose(L_D, SU_N_BASIS);
    // ComplexVector l = decompose(L, SU_N_BASIS);

    // std::cout << "l = " << std::endl << l << std::endl;
    ComplexVector l_conj = l.conjugate();

    ComplexTensor3 f_tensor, z_tensor;
    // for(int i = 0; i < N*N-1; i++) 
    //     for(int j = 0; j < N*N-1; j++)
    //         for(int k = 0; k < N*N-1; k++) {
    //             f_tensor[i][j][k] = zero;
    //             z_tensor[i][j][k] = zero;
    //         }

    compute_Fmns_Zmns(SU_N_BASIS, f_tensor, z_tensor);
    // print_tensor(h.size() , z_tensor);

    ComplexMatrix Q = compute_Q_(h, f_tensor);
    // std::cout << Q << std::endl;
    ComplexVector K = compute_K_vector(l, l_conj, f_tensor, N);
    // std::cout<<"K = " <<K<<std::endl;
    ///////////////////////////////////////////////////////////////////////////////////////////////////
    //ComplexMatrix R = compute_R_matrix(-0.1, N, l, l_conj, f_tensor, z_tensor);
     ComplexMatrix R = compute_R_(l, f_tensor, z_tensor, std::complex<double>(0.1, 0.0));

    // std::cout << " R = "<< std::endl << R << std::endl;
    // std::cout << std::endl;
    // std::cout << " K = "<< std::endl << K << std::endl;
    // std::cout << std::endl;
    // std::cout << " Q = " << std::endl << Q << std::endl;


    // std::cout << " R = " << std::endl << R << std::endl;
    // std::cout << std::endl;
    // print(R);

    // ComplexVector gamma_vec = ComplexVector::Zero(1);
    // gamma_vec[0] = Complex(1.0, 0.0);
    // ComplexMatrix R = compute_R(l, f_tensor, z_tensor, gamma_vec);
/////////////////////////////////////////////////////////////////////////////////////////////////////
    ComplexVector initialVector = decompose(rho, SU_N_BASIS);
    // ComplexVector resultVector = ComplexVector::Zero(SU_N_BASIS.size());



    // for (double t = 0; t < M_PI; t += 0.01) {
    // // for (double t = 0; t < 0.000001; t += 0.0000001) { 
    //     resultVector = rk4_for_real_system(N, t, resultVector, Q, R, K);
    // }

    auto resultVector = rk4_for_real_system_NEW(N, t_start, t_end, dt, initialVector, Q, R, K);
    // std::cout << "result_vector" << resultVector << std::endl << std::endl;
    // return resultVector;
    ComplexMatrix rho_res = ComplexMatrix::Identity(N,N).real();

    rho_res = rho_res / N;

    for (int q = 0; q < l.size(); q++) {
        rho_res += resultVector[q] * SU_N_BASIS[q];
    }
    // std::cout << "res vector = " << resultVector << std::endl;
    return rho_res;
}

int main() {

    constexpr int N = 5;
    constexpr double dt = 0.000005;

    constexpr double t_start = 0.0;
    constexpr double t_end = M_PI;
    // int N = 0;
    // std::cin >> N;

    // ComplexVector e_i(N);
    // ComplexVector e_j = ComplexVector::Zero(N);
    // ComplexVector e_k = ComplexVector::Zero(N);
    // ComplexVector e_l(N);

    // int j  = 0, k = 1;
    // for(; j < k; j++) {
    //     e_j[j] = std::complex<double>(1.0, 0.0);
    //     for(k = j + 1; k < N; k++) {
    //         e_k[k] = std::complex<double>(1.0, 0.0);
    //         std::cout << e_j * e_k.transpose() + e_k * e_j.transpose() << std::endl <<std::endl;
    //         e_k[k] = std::complex<double>(0.0, 0.0);
    //     }
    //     e_j[j] = std::complex<double>(0.0, 0.0);
    // }

    VectorXcd e_j = VectorXcd::Zero(N);
    VectorXcd e_k = VectorXcd::Zero(N);
    // std::map<std::string, ComplexMatrix> map_with_simmetrical_matrix;
    for (int j = 0; j < N - 1; ++j) {
        e_j(j) = 1.0;  // Устанавливаем j-й элемент в 1
        
        for (int k = j + 1; k < N; ++k) {
            e_k(k) = 1.0;  // Устанавливаем k-й элемент в 1
            
            // Вычисляем e_j * e_k^T + e_k * e_j^T (внешнее произведение)
            ComplexMatrix temp = 1.0/sqrt(2.0) * (e_j * e_k.transpose() + e_k * e_j.transpose());
            std::string hash = "S_" + std::to_string(j) +'_'+ std::to_string(k);


            std::cout << hash << std::endl;
            map_with_simmetrical_matrix[hash] = temp;

            // std::cout << 1.0/sqrt(2.0) * (e_j * e_k.transpose() + e_k * e_j.transpose()) << "\n\n";
            // std::cout <<  "hash =  " << hash << " matrix =  " << map_with_simmetrical_matrix[hash]  << " \n \n " << std::endl;

            
            e_k(k) = 0.0;  // Сбрасываем k-й элемент обратно в 0
        }
        
        e_j(j) = 0.0;  // Сбрасываем j-й элемент обратно в 0
    }


    for (int j = 0; j < N; ++j) {
        e_j(j) = 1.0;  // Устанавливаем j-й элемент в 1
        std::string hash = "S_" + std::to_string(j) +'_'+ std::to_string(j);
        ComplexMatrix temp = 1.0/sqrt(2.0) * (2.0 * e_j * e_j.transpose());
        map_with_simmetrical_matrix[hash] = temp;
        // std::cout << " S = " << map_with_simmetrical_matrix[hash] << " \n \n " << std::endl;
        e_j(j) = 0.0;  // Сбрасываем j-й элемент обратно в 0
    }


    // std::map<std::string, ComplexMatrix> map_with_antisymmetr;
    const Complex i_complex(0.0, 1.0);
    for (int j = 0; j < N - 1; ++j) {
        e_j(j) = 1.0;  // Устанавливаем j-й элемент в 1
        
        for (int k = j + 1; k < N; ++k) {
            e_k(k) = 1.0;  // Устанавливаем k-й элемент в 1
            
            // Вычисляем e_j * e_k^T + e_k * e_j^T (внешнее произведение)
            ComplexMatrix temp = i_complex/sqrt(2.0) * ((-1)*e_j * e_k.transpose() + e_k * e_j.transpose());
            std::string hash = "J_" + std::to_string(j) +'_'+ std::to_string(k);


            std::cout << hash << std::endl;
            map_with_antisymmetr[hash] = temp;

            // std::cout << 1.0/sqrt(2.0) * (e_j * e_k.transpose() + e_k * e_j.transpose()) << "\n\n";
            // std::cout <<  "hash =  " << hash << " matrix anti=  " << map_with_antisymmetr[hash]  << " \n \n " << std::endl;

            
            e_k(k) = 0.0;  // Сбрасываем k-й элемент обратно в 0
        }
        
        e_j(j) = 0.0;  // Сбрасываем j-й элемент обратно в 0
    }

    for (int j = 0; j < N; ++j) {
        e_j(j) = 1.0;  // Устанавливаем j-й элемент в 1
        std::string hash = "J_" + std::to_string(j) +'_'+ std::to_string(j);
        ComplexMatrix temp = 1.0/sqrt(2.0) * (2.0 * e_j * e_j.transpose());
        map_with_antisymmetr[hash] = temp;
        // std::cout << " J = " << map_with_antisymmetr[hash] << " \n \n " << std::endl;
        e_j(j) = 0.0;  // Сбрасываем j-й элемент обратно в 0
    }

    // std::map<std::string, ComplexMatrix> map_with_diag;

    for (int l = 1; l < N; ++l) {
        ComplexMatrix D = ComplexMatrix::Zero(N, N);
        double norm = sqrt(l * (l + 1.0));
        for (int k = 0; k < l; ++k)
            D(k, k) = Complex(1.0 / norm, 0.0);
        D(l, l) = Complex(-l / norm, 0.0);
        std::string hash = "D_" + std::to_string(l);
        map_with_diag[hash] = D;

        hash = "D_" + std::to_string(l) + "sqr";
        map_with_diag[hash] = D*D;  // aka D^T * D, but D = diag(D1...DN) => D = D^T
        // basis.push_back(D);
    }

    ComplexMatrix H =  getErmmitMatrix(N);

    // for (int q = 0; q < N*N; q++) {
    //     H.data()[q] = std::complex<double>(0,0);
    // }

    // H.data()[0] = std::complex<double>(-8,0);
    // H.data()[1] = std::complex<double>(sqrt(3),0);
    // H.data()[3] = std::complex<double>(sqrt(3),0);
    // H.data()[4] = std::complex<double>(9.2,0);
    // H.data()[5] = std::complex<double>(2,0);
    // H.data()[7] = std::complex<double>(2,0);
    // H.data()[8] = std::complex<double>(-0.2,0);


    // std::cout << "H = " << std::endl;
    // std::cout<< H << std::endl;
    // std::cout << std::endl;
    
    ComplexMatrix rho =  getErmmitMatrix(N);

    std::cout << "start rho = " <<std::endl;
    std::cout << rho << std::endl;
    std::cout << rho.trace() << std::endl;

    ComplexMatrix rho_res = ComplexMatrix::Zero(N,N);
    ComplexMatrix rho_from_real = ComplexMatrix::Zero(N,N);

    std::cout << "rho_from_initial: " << std::endl;

    ComplexMatrix L = get_L(N);
	std::vector<int> n_threads = {1,2}; //{1, 2, 4, 6};

	for (int num = 0; num < n_threads.size(); num++) {
		Eigen::setNbThreads(n_threads[num]);
		const auto start_initial = std::chrono::high_resolution_clock::now();
		for (int q = 0; q < count_of_iter; q++)
			rho_res = ComputeInitialSys(N, H, rho, dt, L, t_start, t_end);
		const auto end_initial = std::chrono::high_resolution_clock::now();
		Eigen::setNbThreads(1);

		std::cout << rho_res << std::endl;
		std::cout << rho_res.trace() << std::endl;

		omp_set_num_threads(n_threads[num]);
		const auto start_real = std::chrono::high_resolution_clock::now();
		for (int q = 0; q < count_of_iter; q++) 
			rho_from_real = get_L_ODE(N, H, rho, dt, L, t_start, t_end);
		const auto end_real = std::chrono::high_resolution_clock::now();

		std::cout << "rho_from_real = " << std::endl;
		std::cout << rho_from_real << std::endl;
		std::cout << "rho_from_real trace = " << rho_from_real.trace() << std::endl;
		
		//////////////////////////////////////////////////////////////////////////////////////////
		// std::cout << get_L(3) << std::endl;
		// ComplexMatrix rho = ComplexMatrix::Zero(N,N);
		// for (int q = 0; q < N; q++) {
		//     rho(q, q) = std::complex<double>(1,0);
		// }
		// std::cout << rho << std::endl;
		// std::cout << std::endl;
		// std::cout << get_L_D(3, rho) << std::endl;

		ComplexMatrix diff = ComplexMatrix::Zero(N, N);

		for(int q = 0; q < N; q++) {
			for (int w = 0; w < N; w++){
				diff(q,w) = rho_res(q,w) - rho_from_real(q,w);
			}
		}

		std::cout << "diff = " << std::endl;
		std::cout << diff << std::endl;

		const double* matrix_diff_as_array = reinterpret_cast<const double*>(diff.data());
		double max = -1e+10;
		for(int q = 0; q < 2*N*N; q++) {
			if(max < std::abs(matrix_diff_as_array[q])) 
				max = std::abs(matrix_diff_as_array[q]);
		}
		std::cout << "max abs diff = " << max << std::endl;
		const std::chrono::duration<double> time_initial = end_initial - start_initial;
		const std::chrono::duration<double> time_real = end_real - start_real;
		std::cout << "n_threads = " << n_threads[num] << std::endl;
		std::cout << "time for initial = " << static_cast<float> (time_initial.count() / count_of_iter)  << std::endl;
		std::cout << "time for real = " << static_cast<float> ( time_real.count() / count_of_iter )<< std::endl;
	}

    return 0;
}
