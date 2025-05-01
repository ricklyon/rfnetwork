#include <stdio.h>
#include <cstdint>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <cstring>
#include <complex>
#include <random>
#include <math.h>
#include <iostream>
#include <thread>

#include "connect.h"

#include "Eigen/Dense"

using Eigen::MatrixXd;

typedef Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> MatrixType;

int cascade_noise_data(
    char * m1, char * m2, char * c1, char * c2, char * out, int flen, int m1_a, int m2_a
)
{   
    int m_b = m1_a + m2_a;

    int itemsize = sizeof(std::complex<double>);
    // size of each frequency matrix in m1, m2
    int m1_size = (m_b * m1_a * itemsize);
    int m2_size = (m_b * m2_a * itemsize);
    // size of c1, c2 frequency matrix
    int c1_size = (m1_a * m1_a * itemsize);
    int c2_size = (m2_a * m2_a * itemsize);

    int out_size = (m_b * m_b * itemsize);

    // temporary working matrix
    // Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> T1(m1_a, m1_a);
    // Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> T2(m2_a, m2_a);


    for (int f = 0; f < flen; f++)
    {
        MatrixType M1 ((std::complex<double> *) (m1 + (f * m1_size)), m_b, m1_a);
        MatrixType M2 ((std::complex<double> *) (m2 + (f * m2_size)), m_b, m2_a);
        MatrixType C1 ((std::complex<double> *) (c1 + (f * c1_size)), m1_a, m1_a);
        MatrixType C2 ((std::complex<double> *) (c2 + (f * c2_size)), m2_a, m2_a);
        MatrixType OUT ((std::complex<double> *) (out + (f * out_size)), m_b, m_b);

        std::complex<double> t1;
        std::complex<double> t2;

        for (int i = 0; i < m_b; i++)
        {
            for (int j = 0; j < m_b; j++)
            {   
                t1 = (M1.row(i).transpose() * M1.row(j).conjugate()).cwiseProduct(C1).sum();
                t2 = (M2.row(i).transpose() * M2.row(j).conjugate()).cwiseProduct(C2).sum();

                OUT(i, j) = t1 + t2;
                // T1 = (M1.row(i).transpose() * M1.row(j).conjugate());
                // T2 = (M2.row(i).transpose() * M2.row(j).conjugate());

                // T1 = T1.cwiseProduct(C1);
                // T2 = T2.cwiseProduct(C2);

                // OUT(i, j) = T1.sum() + T2.sum();
            }
        }
    }

    return 0;
}

int cascade_self_noise_data(
    char * m1, char * c1, char * out, int flen, int m1_len
)
{
    int itemsize = sizeof(std::complex<double>);
    // size of each frequency matrix in m1, m2
    int m_size = (m1_len * m1_len * itemsize);


    for (int f = 0; f < flen; f++)
    {
        MatrixType M1 ((std::complex<double> *) (m1 + (f * m_size)), m1_len, m1_len);
        MatrixType C1 ((std::complex<double> *) (c1 + (f * m_size)), m1_len, m1_len);
        MatrixType OUT ((std::complex<double> *) (out + (f * m_size)), m1_len, m1_len);

        for (int i = 0; i < m1_len; i++)
        {
            for (int j = 0; j < m1_len; j++)
            {   
                OUT(i, j)  = (M1.row(i).transpose() * M1.row(j).conjugate()).cwiseProduct(C1).sum();
            }
        }
    }

    return 0;
}