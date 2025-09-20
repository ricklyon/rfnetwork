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

#define MAX_THREADS 20

using Eigen::MatrixXd;

typedef Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> MatrixType;

typedef Eigen::Map<Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> MatrixIntType;


int connect_other(
    char * s1,
    char * s2,
    char * c1,
    char * c2,
    char * connections,
    char * probes,
    char * row_order,
    char * cas_s,
    char * cas_n,
    int n_row, int f_len, int s1_b, int s1_a, int s2_b, int s2_a, int n_connections, int n_threads
)
{

    int m_b = s1_b + s2_b;
    int m2_a = s1_a + s2_a - (2 * n_connections);

    int itemsize = sizeof(std::complex<double>);

    // size of each frequency matrix in s1
    int s1_size = (s1_b * s1_a * itemsize);
    // size of each frequency matrix in s2
    int s2_size = (s2_b * s2_a * itemsize);
    // size of each frequency matrix in cascaded sdata
    int cas_s_size = (n_row * m2_a * itemsize);
    int cas_n_size = (m2_a * m2_a * itemsize);

    // size of c1, c2 frequency matrix
    int c1_size = (s1_a * s1_a * itemsize);
    int c2_size = (s2_a * s2_a * itemsize);

    int p1, p2;

    MatrixIntType ROW_ORDER ((int *) row_order, 1, n_row);

    cascaded_row_order(
        connections,
        probes,
        row_order,
        n_row, s1_b, s1_a, s2_b, s2_a, n_connections
    );

    // create permutation matrix that will reorder the rows of M1 after taking its inverse
    Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> P(n_row, m_b);
    P.setConstant(0);
    
    for (int r = 0; r < n_row; r++)
    {
        P(r, ROW_ORDER(0, r)) = 1;
    }

    // divide frequency vector into batches for each thread
    if (f_len < n_threads)
    {
        n_threads = 1;
    }
    else if (n_threads > MAX_THREADS)
    {
        n_threads = MAX_THREADS;
    }

    // minimum number of frequencies in each batch
    int batch_len = f_len / n_threads;
    // remainder
    int batch_rm = f_len % n_threads;

    std::thread threads[MAX_THREADS];
    int f_idx = 0; // frequency index of current batch
    int f_blen; // length of current batch

    char * p_ptr = (char *) P.data();

    char * c1_p = NULL;
    char * c2_p = NULL; 
    char * cas_n_p = NULL;

    for (int t = 0; t < n_threads; t++)
    {   

        f_blen = batch_len;

        // add an extra frequency to the thread if the remainder is still nonzero
        if (batch_rm > 0)
        {
            f_blen++;
            batch_rm--;
        }
        
        if (cas_n != NULL)
        {
            c1_p = c1 + (f_idx * c1_size);
            c2_p = c2 + (f_idx * c2_size);
            cas_n_p = cas_n + (f_idx * cas_n_size);
        }

        threads[t] = std::thread(
            connect_other_th,
            s1 + (f_idx * s1_size), 
            s2 + (f_idx * s2_size), 
            c1_p, c2_p,
            connections, 
            p_ptr, 
            cas_s + (f_idx * cas_s_size), 
            cas_n_p,
            n_row, f_blen, s1_b, s1_a, s2_b, s2_a, n_connections
        );
        
        // increment the frequency index for the next batch
        f_idx += f_blen;
    }

    int err = 0;
    // wait for all threads to complete
    for (int t = 0; t < n_threads; t++)
    {
        threads[t].join();
    }

    return 0;
}

int connect_other_th(
    char * s1,
    char * s2,
    char * c1,
    char * c2,
    char * connections,
    char * permutation_m,
    char * cas_s,
    char * cas_n,
    int n_row, int f_len, int s1_b, int s1_a, int s2_b, int s2_a, int n_connections
)
{

    int m_b = s1_b + s2_b;
    int m2_a = s1_a + s2_a - (2 * n_connections);

    int itemsize = sizeof(std::complex<double>);

    // size of each frequency matrix in s1
    int s1_size = (s1_b * s1_a * itemsize);
    // size of each frequency matrix in s2
    int s2_size = (s2_b * s2_a * itemsize);
    // size of each frequency matrix in cascaded sdata
    int cas_s_size = (n_row * m2_a * itemsize);
    int cas_n_size = (m2_a * m2_a * itemsize);

    // size of c1, c2 frequency matrix
    int c1_size = (s1_a * s1_a * itemsize);
    int c2_size = (s2_a * s2_a * itemsize);

    int p1, p2;

    MatrixIntType CONN ((int *) connections, n_connections, 2);

    Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> M1(m_b, m_b);
    Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> M2(m_b, m2_a);

    // create permutation matrix that will reorder the rows of M1 after taking its inverse
    MatrixType P ((std::complex<double> *) (permutation_m), n_row, m_b);

    // M matrix with probe rows and columns removed, and row-ordered
    Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> M_EXT(m2_a, m2_a);

    std::complex<double> t1;
    std::complex<double> t2;

    for (int f = 0; f < f_len; f++)
    {   
        MatrixType S1 ((std::complex<double> *) (s1 + (f * s1_size)), s1_b, s1_a);
        MatrixType S2 ((std::complex<double> *) (s2 + (f * s2_size)), s2_b, s2_a);
        MatrixType CAS_SDATA ((std::complex<double> *) (cas_s + (f * cas_s_size)), n_row, m2_a);

        // create first matrix, starts with just the identity matrix
        M1.setIdentity(); 
        M2.setConstant(0);

        // move columns from s1 and s2 to m1 based on the connections
        for (int n = 0; n < n_connections; n++)
        {
            p1 = CONN(n, 0);
            p2 = CONN(n, 1);

            // populate connected columns from s1/s2 into M1
            M1.col(s1_b + p2 - 1).segment(0, s1_b) = -S1.col(p1 - 1);
            M1.col(p1 - 1).segment(s1_b, s2_b) = -S2.col(p2 - 1);
        }

        // take the inverse
        M1 = M1.inverse();
    
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

        CAS_SDATA = P * M1 * M2;

        int m_b = s1_a + s2_a;

        if (cas_n != NULL)
        {
            MatrixType C1 ((std::complex<double> *) (c1 + (f * c1_size)), s1_a, s1_a);
            MatrixType C2 ((std::complex<double> *) (c2 + (f * c2_size)), s2_a, s2_a);
            MatrixType CAS_NDATA ((std::complex<double> *) (cas_n + (f * cas_n_size)), m2_a, m2_a);

            // put the probe rows last, noise cascades only use the external rows, up to m2_a.
            M_EXT = P * M1;

            // after row-ordering, the M_EXT matrix will have the rows for external ports placed first, followed
            // by the probes. Noise cascades do not support probes, so only build the cascade matrix for the external
            // ports.
            for (int i = 0; i < m2_a; i++)
            {
                for (int j = 0; j < m2_a; j++)
                {
                    t1 = (M_EXT.row(i).segment(0, s1_a).transpose() * M_EXT.row(j).segment(0, s1_a).conjugate()).cwiseProduct(C1).sum();
                    t2 = (M_EXT.row(i).segment(s1_b, s2_a).transpose() * M_EXT.row(j).segment(s1_b, s2_a).conjugate()).cwiseProduct(C2).sum();

                    CAS_NDATA(i, j) = t1 + t2;
                }
            }
        }
    }

    return 0;
}

int connect_self(
    char * s1,
    char * c1,
    char * connections,
    char * probes,
    char * row_order,
    char * cas_s,
    char * cas_n,
    int n_row, int f_len, int s1_b, int s1_a, int n_connections, int n_threads
)
{
    
    int m2_a = s1_a - (2 * n_connections);

    int itemsize = sizeof(std::complex<double>);

    // size of each frequency matrix in s1
    int s1_size = (s1_b * s1_a * itemsize);

    // size of each frequency matrix in cascaded sdata
    int cas_s_size = (n_row * m2_a * itemsize);
    int cas_n_size = (m2_a * m2_a * itemsize);

    // size of c1, c2 frequency matrix
    int c1_size = (s1_a * s1_a * itemsize);

    MatrixIntType ROW_ORDER ((int *) row_order, 1, n_row);
        
    self_cascaded_row_order(
        connections,
        probes,
        row_order,
        n_row, s1_b, s1_a, n_connections
    );

    // create permutation matrix that will reorder the rows of M1 after taking its inverse
    Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> P(n_row, s1_b);
    P.setConstant(0);
    
    for (int r = 0; r < n_row; r++)
    {
        P(r, ROW_ORDER(0, r)) = 1;
    }

    // divide frequency vector into batches for each thread
    if (f_len < n_threads)
    {
        n_threads = 1;
    }
    else if (n_threads > MAX_THREADS)
    {
        n_threads = MAX_THREADS;
    }

    // minimum number of frequencies in each batch
    int batch_len = f_len / n_threads;
    // remainder
    int batch_rm = f_len % n_threads;

    std::thread threads[MAX_THREADS];
    int f_idx = 0; // frequency index of current batch
    int f_blen; // length of current batch

    char * p_ptr = (char *) P.data();

    char * c1_p = NULL;
    char * cas_n_p = NULL;

    for (int t = 0; t < n_threads; t++)
    {   
        f_blen = batch_len;

        // add an extra frequency to the thread if the remainder is still nonzero
        if (batch_rm > 0)
        {
            f_blen++;
            batch_rm--;
        }

        if (cas_n != NULL)
        {
            c1_p = c1 + (f_idx * c1_size);
            cas_n_p = cas_n + (f_idx * cas_n_size);
        }

        threads[t] = std::thread(
            connect_self_th,
            s1 + (f_idx * s1_size), 
            c1_p, 
            connections, 
            p_ptr, 
            cas_s + (f_idx * cas_s_size), 
            cas_n_p,
            n_row, f_blen, s1_b, s1_a, n_connections
        );

        // increment the frequency index for the next batch
        f_idx += f_blen;
    }

    int err = 0;
    // wait for all threads to complete
    for (int t = 0; t < n_threads; t++)
    {
        threads[t].join();
    }

    return 0;
}

int connect_self_th(
    char * s1,
    char * c1,
    char * connections,
    char * permutation_m,
    char * cas_s,
    char * cas_n,
    int n_row, int f_len, int s1_b, int s1_a, int n_connections
)
{

    int m2_a = s1_a - (2 * n_connections);

    int itemsize = sizeof(std::complex<double>);

    // size of each frequency matrix in s1
    int s1_size = (s1_b * s1_a * itemsize);

    // size of each frequency matrix in cascaded sdata
    int cas_s_size = (n_row * m2_a * itemsize);
    int cas_n_size = (m2_a * m2_a * itemsize);

    // size of c1, c2 frequency matrix
    int c1_size = (s1_a * s1_a * itemsize);

    int p1, p2;

    Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> M1(s1_b, s1_b);
    Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> M2(s1_b, m2_a);

    MatrixIntType CONN ((int *) connections, n_connections, 2);

    // create permutation matrix that will reorder the rows of M1 after taking its inverse
    MatrixType P ((std::complex<double> *) (permutation_m), n_row, s1_b);
    // M matrix with probe rows and columns removed, and row-ordered
    Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> M_EXT(m2_a, m2_a);

    std::complex<double> t1;
    std::complex<double> t2;


    for (int f = 0; f < f_len; f++)
    {   
        MatrixType S1 ((std::complex<double> *) (s1 + (f * s1_size)), s1_b, s1_a);
        MatrixType CAS_SDATA ((std::complex<double> *) (cas_s + (f * cas_s_size)), n_row, m2_a);

        // create first matrix, starts with just the identity matrix
        M1.setIdentity(); 
        M2.setConstant(0);

        // move columns from s1 and s2 to m1 based on the connections
        // # move columns from m2 to m1 based on the connections
        // m1[..., p2-1] = m1[..., :, p2-1] - m2[..., :s1_b, p1-1]
        // m1[..., p1-1] = m1[..., :, p1-1] - m2[..., :s1_b, p2-1]

        for (int n = 0; n < n_connections; n++)
        {
            p1 = CONN(n, 0);
            p2 = CONN(n, 1);

            // populate connected columns from s1/s2 into M1
            M1.col(p2 - 1) -= S1.col(p1 - 1);
            M1.col(p1 - 1) -= S1.col(p2 - 1);
        }

        // take the inverse
        M1 = M1.inverse();
    
        // populate columns in M2 with the unconnected columns from S1
        int m2_col = 0;
        for (int i = 0; i < s1_a; i++)
        {   
            // if port number does not appear in the connections matrix for component 1, add the S1 column
            // to M2.
            if (((CONN.col(0).array() != (i + 1)).all()) && ((CONN.col(1).array() != (i + 1)).all()) )
            {
                M2.col(m2_col) = S1.col(i);
                m2_col++;
            }
        }

        CAS_SDATA = P * M1 * M2;

        if (cas_n != NULL)
        {
            MatrixType C1 ((std::complex<double> *) (c1 + (f * c1_size)), s1_a, s1_a);
            MatrixType CAS_NDATA ((std::complex<double> *) (cas_n + (f * cas_n_size)), m2_a, m2_a);

            M_EXT = P * M1;

            // after row-ordering, the M_EXT matrix will have the rows for external ports placed first, followed
            // by the probes. Noise cascades do not support probes, so only build the cascade matrix for the external
            // ports.
            for (int i = 0; i < m2_a; i++)
            {
                for (int j = 0; j < m2_a; j++)
                {
                    CAS_NDATA(i, j)  = (M_EXT.row(i).segment(0, s1_a).transpose() * M_EXT.row(j).segment(0, s1_a).conjugate()).cwiseProduct(C1).sum();
                }
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


int cascaded_row_order(
    char * connections,
    char * probes,
    char * row_order,
    int n_row, int s1_b, int s1_a, int s2_b, int s2_a, int n_connections
)
{
    int m_b = s1_b + s2_b;
    int m2_a = s1_a + s2_a - (2 * n_connections);

    // number of existing probes on component data
    int s1_probe_n = s1_b - s1_a;
    int s2_probe_n = s2_b - s2_a;

    int p1, p2;

    MatrixIntType CONN ((int *) connections, n_connections, 2);
    MatrixIntType PROBES ((int *) probes, n_connections, 2);
    MatrixIntType ROW_ORDER ((int *) row_order, 1, n_row);

    // Walk through each row of the first component and place unconnected rows that are external ports
    int ext_r = 0;
    for (int r = 0; r < s1_a; r++)
    {
        // row is an external port (not a connected port)
        if ((CONN.col(0).array() != (r + 1)).all())
        {
            ROW_ORDER(0, ext_r) = r;
            ext_r++;
        }
    }

    // Walk through each row of the second component and place unconnected rows
    for (int r = 0; r < s2_a; r++)
    {
        // row is an external port (not a connected port)
        if ((CONN.col(1).array() != (r + 1)).all())
        {
            ROW_ORDER(0, ext_r) = r + s1_b;
            ext_r++;
        }
    }

    // Put the existing probes after the external port rows.
    int pb_r = 0;
    for (int r = s1_a; r < s1_b; r++)
    {
        ROW_ORDER(0, m2_a + pb_r) = r;
        pb_r++;
    }
    // place probes for the second component after the first component
    for (int r = s2_a; r < s2_b; r++)
    {
        ROW_ORDER(0, m2_a + pb_r) = r + s1_b;
        pb_r++;
    }

    // place connected rows that are assigned as probes from the first component
    for (int n = 0; n < n_connections; n++)
    {
        p1 = CONN(n, 0);

        // if p1 or p2 are greater than the external number of ports, they are internal ports that cannot be 
        // connected.
        if (p1 > s1_a)
        {
            throw std::runtime_error("Connection ports must be less than the number of external ports of s1.");
        }

        // connection from component 1 is assigned as a probe, add to the row assignment list.
        if (PROBES(n, 0))
        {
            ROW_ORDER(0, m2_a + pb_r) = p1 - 1;
            pb_r++;
        }
    
    }

    // place connected rows that are assigned as probes from the second component
    for (int n = 0; n < n_connections; n++)
    {
        p2 = CONN(n, 1);
    
        // if p1 or p2 are greater than the external number of ports, they are internal ports that cannot be 
        // connected.
        if (p2 > s2_a)
        {
            throw std::runtime_error("Connection ports must be less than the number of external ports of s2.");
        }
        
        // connection from component 2 is assigned as a probe, add to the row assignment list.
        if (PROBES(n, 1))
        {   
            ROW_ORDER(0, m2_a + pb_r) = s1_b + (p2 - 1);
            pb_r++;
        }
    }

    if ((pb_r + ext_r) != n_row)
    {
        throw std::runtime_error("cascaded_row_order: Row order vector has invalid number of rows.");
    }

    return 0;
}


int self_cascaded_row_order(
    char * connections,
    char * probes,
    char * row_order,
    int n_row, int s1_b, int s1_a, int n_connections
)
{
    int m_b = s1_b;
    int m2_a = s1_a - (2 * n_connections);

    // number of existing probes on component data
    int s1_probe_n = s1_b - s1_a;

    int p1, p2;

    MatrixIntType CONN ((int *) connections, n_connections, 2);
    MatrixIntType PROBES ((int *) probes, n_connections, 2);
    MatrixIntType ROW_ORDER ((int *) row_order, 1, n_row);

    // Walk through each row of the first component and place unconnected rows that are external ports
    int ext_r = 0;
    for (int r = 0; r < s1_a; r++)
    {
        // row is an external port (not a connected port)
        if (((CONN.col(0).array() != (r + 1)).all()) && ((CONN.col(1).array() != (r + 1)).all()) )
        {
            ROW_ORDER(0, ext_r) = r;
            ext_r++;
        }
    }

    // Put the existing probes after the external port rows.
    int pb_r = 0;
    for (int r = s1_a; r < s1_b; r++)
    {
        ROW_ORDER(0, m2_a + pb_r) = r;
        pb_r++;
    }

    // place connected rows that are assigned as probes from the first component
    for (int n = 0; n < n_connections; n++)
    {
        p1 = CONN(n, 0);
        p2 = CONN(n, 1);

        // if p1 or p2 are greater than the external number of ports, they are internal ports that cannot be 
        // connected.
        if ((p1 > s1_a) || ( p2 > s1_a))
        {
            throw std::runtime_error("Connection ports must be less than the number of external ports of s1.");
        }

        // connection from component 1 is assigned as a probe, add to the row assignment list.
        if (PROBES(n, 0))
        {
            ROW_ORDER(0, m2_a + pb_r) = p1 - 1;
            pb_r++;
        }

        if (PROBES(n, 1))
        {
            ROW_ORDER(0, m2_a + pb_r) = p2 - 1;
            pb_r++;
        }
    
    }

    if ((pb_r + ext_r) != n_row)
    {
        throw std::runtime_error("self_cascaded_row_order: Row order vector has invalid number of rows.");
    }

    return 0;
}