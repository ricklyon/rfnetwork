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

typedef Eigen::Map<Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> MatrixIntType;

int connection_matrix(
    char * s1,
    char * s2,
    char * connections,
    char * probes,
    char * m1,
    char * m2,
    char * row_order,
    int f_len, int s1_b, int s1_a, int s2_b, int s2_a, int n_connections
)
{
    int m_b = s1_b + s2_b;
    int m2_a = s1_a + s2_a - (2 * n_connections);

    int itemsize = sizeof(std::complex<double>);
    // size of each frequency matrix in m1
    int m1_size = (m_b * m_b * itemsize);
    // size of each frequency matrix in m2
    int m2_size = (m_b * m2_a * itemsize);

    // size of each frequency matrix in s1
    int s1_size = (s1_b * s1_a * itemsize);
    // size of each frequency matrix in s2
    int s2_size = (s2_b * s2_a * itemsize);

    int p1, p2;

    MatrixIntType CONN ((int *) connections, n_connections, 2);
    std::cout << CONN << "\n";

    for (int f = 0; f < f_len; f++)
    {
        MatrixType M1 ((std::complex<double> *) (m1 + (f * m1_size)), m_b, m_b);
        MatrixType M2 ((std::complex<double> *) (m2 + (f * m2_size)), m_b, m2_a);
        
        MatrixType S1 ((std::complex<double> *) (s1 + (f * s1_size)), s1_b, s1_a);
        MatrixType S2 ((std::complex<double> *) (s2 + (f * s2_size)), s2_b, s2_a);

        // create first matrix, starts with just the identity matrix
        M1.setIdentity(); 

        // move columns from s1 and s2 to m1 based on the connections
        for (int n = 0; n < n_connections; n++)
        {
            p1 = CONN(n, 0);
            p2 = CONN(n, 1);

            // if p1 or p2 are greater than the external number of ports, they are internal ports that cannot be 
            // connected.
            if ((p1 > s1_a) || (p2 > s2_a))
            {
                throw std::runtime_error("Connection ports must be less than the number of external ports of s1, s2.");
            }

            // populate connected columns from s1/s2 into M1
            M1.col(s1_b + p2 - 1).segment(0, s1_b) = -S1.col(p1 - 1);
            M1.col(p1 - 1).segment(s1_b, s2_b) = -S2.col(p2 - 1);

            // take the inverse
            M1 = M1.inverse();
        }
        
        
        // populate columns in M2 with the unconnected columns from S1
        int m2_col = 0;
        for (int i = 0; i < s1_a; i++)
        {   
            // if port number does not appear in the connections matrix for component 1, add the S1 column
            // to M2.
            if ((CONN.col(0).array() != (i + 1)).all())
            {
                M2.col(m2_col).segment(0, s1_b) = S1.col(i);
                m2_col++;
            }
        }
        
        std::cout<< "S2 " << CONN.col(1) << "\n";
        // populate columns in M2 with the unconnected columns from S2
        for (int i = 0; i < s2_a; i++)
        {
            // if port number does not appear in the connections matrix for component 2, add the S2 column
            // to M2.
            if ((CONN.col(1).array() != (i + 1)).all())
            {
                M2.col(m2_col).segment(s1_b, s2_b) = S2.col(i);
                m2_col++;
            }
        }
    }

    return 0;
}

int cascade_ndata(
    char * m1, char * m2, char * c1, char * c2, char * out, int f_len, int m1_a, int m2_a
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


    for (int f = 0; f < f_len; f++)
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
            }
        }
    }

    return 0;
}

int cascade_self_ndata(
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