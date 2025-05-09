#include <iostream>
#include <vector>
#include <complex>
#include <random>
#include <Eigen/Dense>
#include <iomanip>

using namespace std;
using namespace Eigen;

using Complex = std::complex<double>;
using ComplexMatrix = MatrixXcd;
using RealMatrix = MatrixXd;
using ComplexVector = VectorXcd;
using RealVector = VectorXd;
using ComplexTensor3 = std::vector<std::vector<std::vector<Complex>>>;

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
vector<ComplexMatrix> generate_su_basis(int N) {
    vector<ComplexMatrix> basis;

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
ComplexVector decompose(const ComplexMatrix& A, const vector<ComplexMatrix>& basis) {
    int M = basis.size();
    ComplexVector v(M);
    for (int i = 0; i < M; ++i) {
        v[i] = (A * basis[i]).trace();
        // if (v[i].imag() != 0) {
        //     v[i] = v[i] * std::complex<double>(0.0, -1.0);
        // }
    }
    return v;
}


ComplexVector decompose_with_conj(const ComplexMatrix& A, const vector<ComplexMatrix>& basis) {
    int M = basis.size();
    ComplexVector v(M);
    for (int i = 0; i < M; ++i) {
        v[i] = (A * basis[i]).trace();
        if (v[i].imag() != 0) {
            v[i] = v[i] * std::complex<double>(0.0, -1.0);
        }
    }
    return v;
}


void compute_Fmns_Zmns(const vector<ComplexMatrix>& basis, ComplexTensor3& f_tensor, ComplexTensor3& z_tensor) {
    int M = basis.size();
    f_tensor.resize(M, vector<vector<Complex>>(M, vector<Complex>(M, Complex(0.0, 0.0))));
    z_tensor.resize(M, vector<vector<Complex>>(M, vector<Complex>(M, Complex(0.0, 0.0))));

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < M; ++n) {
            ComplexMatrix commutator = basis[m] * basis[n] - basis[n] * basis[m];
            ComplexMatrix anticommutator = basis[m] * basis[n] + basis[n] * basis[m];

            for (int s = 0; s < M; ++s) {
                Complex f_mns = Complex(0.0, -1.0) * (basis[s] * commutator).trace();
                Complex z_mns = f_mns + Complex(0.0, 1.0) * (basis[s] * anticommutator).trace();
                f_tensor[m][n][s] = f_mns;
                z_tensor[m][n][s] = z_mns;
            }
        }
    }
}

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

ComplexMatrix compute_R_matrix (const double gamma, const int size, const ComplexVector& l, const ComplexVector& lConj, ComplexTensor3& Fmns, ComplexTensor3& Zmns) {
    std::complex<double> fillerMatrixEone[l.size()][l.size()] = { 0 };
    for (int i = 0; i < l.size(); ++i) {
        if (real(l[i]) != 0 || imag(l[i]) != 0) {
            for (int m = 0; m < l.size(); ++m) {
                for (int n = 0; n < l.size(); ++n) {
                    fillerMatrixEone[m][n] += l[i] * Zmns[i][m][n]; // checked
                }
            }
        }
    }

    std::complex<double> fillerMatrixEtwo[l.size()][l.size()] = { 0 };
    for (int i = 0; i < l.size(); ++i) {
        if (real(l[i]) != 0 || imag(l[i]) != 0) {
            for (int m = 0; m < l.size(); ++m) {
                for (int n = 0; n < l.size(); ++n) {
                    fillerMatrixEtwo[m][n] += lConj[i] * Fmns[i][m][n]; // checked
                }
            }
        }
    }

    // double matrixR[size * size][size * size] = { 0 };
    ComplexMatrix matrixR = ComplexMatrix::Zero(l.size(), l.size());
    for (int i = 0; i < l.size(); ++i) {
        for (int j = 0; j < l.size(); ++j) {
            for (int u = 0; u < l.size(); ++u) {
                matrixR(j,i) += -gamma * real(fillerMatrixEone[i][u] * fillerMatrixEtwo[j][u]);
            }
        }
    }
    return matrixR;
}

inline ComplexMatrix compute_R(const ComplexVector& l, const ComplexTensor3& f, const ComplexTensor3& z, const ComplexVector& gamma) {
    int M = l.size();
    int P = gamma.size();
    ComplexMatrix R = ComplexMatrix::Zero(M, M);

    for (int s = 0; s < M; ++s) {
        for (int n = 0; n < M; ++n) {
            for (int p = 0; p < 1; ++p) {
                for (int j = 0; j < M; ++j) {
                    for (int k = 0; k < M; ++k) {
                        for (int l_idx = 0; l_idx < M; ++l_idx) {
                            R(s, n) += gamma[p] * l(j) * std::conj(l(n)) * (  (z[j][l_idx][n]) * f[k][l_idx][s] + std::conj(z[k][l_idx][n]) * f[j][l_idx][s]  );
                        }
                    }
                }
            }
        }
    }

    R *= -0.5;

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

    // ComplexMatrix L = get_L(N);
    ComplexMatrix L = L_;
    ComplexMatrix L_conj_transp = L.adjoint();

    ComplexMatrix L_rho_L_conj_trans = ((L * rho) * L_conj_transp);
    ComplexMatrix L_conj_trans_L_rho = ((L_conj_transp * L) * rho);
    ComplexMatrix rho_L_conj_trans_L = ((rho * L_conj_transp) * L);

    
    
    ComplexMatrix L_D = ((0.1) / (N-1)) * (L_rho_L_conj_trans - 0.5 * (L_conj_trans_L_rho + rho_L_conj_trans_L));
    ComplexMatrix zero = ComplexMatrix::Zero(N,N);
    return zero;
    // return L_D;
    // ComplexMatrix scobki_vnutr = (L_conj_transp * L * rho + rho * L_conj_transp * L);
    // ComplexMatrix scobki_vneshnie = L * rho * L_conj_transp - (static_cast<double>(1/2) * scobki_vnutr);
    // return ( (0.1 / (N - 1)) * scobki_vneshnie);
    // return static_cast<double>(0.1 / (N - 1)) * ( L * rho * L_conj_transp - (static_cast<double>(1/2)) * (L_conj_transp * L * rho +
                                                                    //   rho * L_conj_transp * L));
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



// ComplexMatrix rk4init_(const int N, double t, const ComplexMatrix& H, /*const ComplexMatrix L_D,*/ const ComplexMatrix& rho, const ComplexMatrix L_) {

//     ComplexMatrix result(N,N);
//     const int size_matrix = N * N;
//     ComplexMatrix k1(N,N);
//     ComplexMatrix k2(N,N);
//     ComplexMatrix k3(N,N);
//     ComplexMatrix k4(N,N);

//     ComplexMatrix temp_rho(N,N);

//     double h = 0.0001;

//     k1 = funcInit(N, t, H, rho, L_);

//     // for (int i = 0; i < size_matrix; i++) {
//     //     temp_rho(i,0) = rho(i,0) + (h / 2) * k1(i,0);
//     temp_rho = rho + static_cast<double>((h / 2)) * k1;
//     // }


//     k2 = funcInit(N, t + h / 2, H, rho, L_);

//     // for (int i = 0; i < size_matrix; i++) {
//     //     temp_rho(i,0) = rho(i,0) + (h / 2) * k2(i,0);
//     temp_rho = rho + static_cast<double>((h / 2)) * k2;
//     // }

//     k3 = funcInit(N, t + h / 2, H, rho, L_);

//     // for (int i = 0; i < size_matrix; i++) {
//     //     temp_rho(i,0) = rho(i,0) + h * k3(i,0);
//     temp_rho = rho + static_cast<double>(h) * k3;
//     // }

//     k4 = funcInit(N, t + h, H, rho, L_);

//     // for (int i = 0; i < size_matrix; i++) {
//     //     std::complex<double> const_two(2.0 , 0.0);
//     //     std::complex<double> const__h_div_6(h/6 , 0.0);
//     //     result(i,0) = rho(i,0) + static_cast<double>(h/6) * (k1(i,0) + 2.0 * k2(i,0) + 2.0 * k3(i,0) + k4(i,0));
//     result = rho + static_cast<double>(h/6) * (k1 + static_cast<double>(2.0) * k2 + static_cast<double>(2.0) * k3 + k4);
//     // }

//     return result;
// }

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

ComplexMatrix ComputeInitialSys(const int N, const ComplexMatrix& H, const ComplexMatrix& rho, const double dt, const ComplexMatrix L_) { 

    // ComplexMatrix result = rk4init(N, 3.14, H, rho);
    ComplexMatrix rho_res = rho;

    for (double t = 0; t < M_PI; t += dt) {
    // for (double t = 0; t < 0.000001; t += 0.0000001) { 
        rho_res = rk4init(N, 3.14, H, rho_res, dt, L_);
    }
    return rho_res;
}

inline ComplexVector func(double t, const ComplexVector& vectorV, const ComplexMatrix& matrixQ, const ComplexMatrix& matrixR, const ComplexVector& matrixK) {
    ComplexVector out_vec = ( (matrixQ + matrixR) * vectorV + matrixK );
    return out_vec;
}

// inline ComplexVector rk4_for_real_system(int N, double t, const ComplexVector& vectorV, const ComplexMatrix& matrixQ, const ComplexMatrix& matrixR, const ComplexVector& matrixK) {
//     // std::vector<double> result(N * N - 1);

//     // double h = 0.000001;
//     double h = 0.0001;
//     // ComplexVector k1(N * N - 1);
//     // ComplexVector k1shift(N * N - 1);
//     // ComplexVector k2shift(N * N - 1);
//     // ComplexVector k3shift(N * N - 1);

//     // k1 = func(N, t, matrixQ, matrixR, vectorV, matrixK);
//     ComplexVector k1 = func(t, vectorV, matrixQ, matrixR, matrixK);

//     // for (int i = 0; i < N * N - 1; ++i) {
//     //     k1shift[i] = vectorV[i] + (h / 2) * k1[i];
//     // }
//     ComplexVector k1shift = vectorV + (h / 2) * k1;


//     // auto k2 = func(N, t + h / 2, matrixQOne, matrixQTwo, matrixR, k1shift, matrixK);
//     ComplexVector k2 = func(t + h / 2, vectorV, matrixQ, matrixR, matrixK);

//     // for (int i = 0; i < N * N - 1; ++i) {
//     //     k2shift[i] = vectorV[i] + (h / 2) * k2[i];
//     // }
//     ComplexVector k2shift = vectorV + (h / 2) * k2;

//     // auto k3 = func(N, t + h / 2, matrixQOne, matrixQTwo, matrixR, k2shift, matrixK);
//     ComplexVector k3 = func(t + h / 2, vectorV, matrixQ, matrixR, matrixK);

//     // for (int i = 0; i < N * N - 1; ++i) {
//     //     k3shift[i] = vectorV[i] + h * k3[i];
//     // }
//     ComplexVector k3shift = vectorV + h * k3;

//     // auto k4 = func(N, t + h, matrixQOne, matrixQTwo, matrixR, k3shift, matrixK);
//     ComplexVector k4 = func(t + h / 2, vectorV, matrixQ, matrixR, matrixK);

//     // for (int i = 0; i < N * N - 1; ++i) {
//     //     result[i] = vectorV[i] + h / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
//     // }
//     ComplexVector result = vectorV + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4);
//     return result;
// }



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


ComplexMatrix get_L_ODE(const int N, const ComplexMatrix& H, const ComplexMatrix& rho, const double dt, const ComplexMatrix L_) {

    ComplexMatrix L_D = get_L_D(N, rho, L_);
    vector<ComplexMatrix> SU_N_BASIS =  generate_su_basis(N);
    // for(int q = 0; q < N*N -1; q++) {
    //     std::cout << SU_N_BASIS[q] << std::endl <<'\n';
    // }

    ComplexVector h = decompose(H, SU_N_BASIS);
    std::cout << "h = " << std::endl << h << std::endl;
    ComplexVector l = decompose(L_D, SU_N_BASIS);
    std::cout << "l = " << std::endl << l << std::endl;
    ComplexVector l_conj = l.conjugate();

    ComplexTensor3 f_tensor, z_tensor;
    compute_Fmns_Zmns(SU_N_BASIS, f_tensor, z_tensor);
    print_tensor(h.size() ,z_tensor);

    ComplexMatrix Q = compute_Q(h, f_tensor);
    // std::cout << Q << std::endl;
    ComplexVector K = compute_K_vector(l, l_conj, f_tensor, N);
    std::cout<<"K = " <<K<<std::endl;
    ///////////////////////////////////////////////////////////////////////////////////////////////////
    ComplexMatrix R = compute_R_matrix(0.1, N, l, l_conj, f_tensor, z_tensor);

    // std::cout << " R = "<< std::endl << R << std::endl;
    // std::cout << std::endl;
    // std::cout << " K = "<< std::endl << K << std::endl;
    // std::cout << std::endl;
    std::cout << " Q = " << std::endl << Q << std::endl;


    std::cout << " R = " << std::endl << R << std::endl;
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

    auto resultVector = rk4_for_real_system_NEW(N, 0.0, M_PI, dt, initialVector, Q, R, K);
    std::cout << "result_vector" << resultVector << std::endl << std::endl;
    // return resultVector;
    ComplexMatrix rho_res = ComplexMatrix::Identity(N,N).real();

    rho_res = rho_res / N;

    for (int q = 0; q < l.size(); q++) {
        rho_res += resultVector[q] * SU_N_BASIS[q];
    }
    std::cout << "res vector = " << resultVector << std::endl;
    return rho_res;
}

int main() {

    constexpr int N = 3;
    constexpr double dt = 0.0000001;
    // int N = 0;
    // std::cin >> N;


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


    std::cout << "H = " << std::endl;
    std::cout<< H << std::endl;
    std::cout << std::endl;
    
    ComplexMatrix rho =  getErmmitMatrix(N);
    std::cout << "start rho = " <<std::endl;
    std::cout << rho << std::endl;
    std::cout << rho.trace() << std::endl;

    ComplexMatrix L = get_L(N);
    // std::cout << "########################################################################" <<std::endl;
    // // std::cout<<get_L_D(3,rho);
    // std::cout << funcInit(3, 0.0, H, rho) << std::endl;

    // std::cout << "########################################################################" <<std::endl;

    std::cout << "rho_from_initial: " << std::endl;
    ComplexMatrix rho_res = ComputeInitialSys(N, H, rho, dt, L);
    std::cout << rho_res << std::endl;
    std::cout << rho_res.trace() << std::endl;

    ComplexMatrix rho_from_real = get_L_ODE(N, H, rho, dt, L);
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
    return 0;
}
