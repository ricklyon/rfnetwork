#ifndef CONNECT_H
#define CONNECT_H

int cascade_ndata(
    char * m1, char * m2, char * c1, char * c2, char * out, int flen, int m1_len, int m2_len
);

int cascade_self_ndata(
    char * m1, char * c1, char * out, int flen, int m1_len
);

int connect_other(
    char * s1,
    char * s2,
    char * c1,
    char * c2,
    char * connections,
    char * probes,
    char * row_order,
    char * cas_sdata,
    char * cas_ndata,
    int n_row, int f_len, int s1_b, int s1_a, int s2_b, int s2_a, int n_connections, int n_threads
);

int connect_self(
    char * s1,
    char * c1,
    char * connections,
    char * probes,
    char * row_order,
    char * cas_s,
    char * cas_n,
    int n_row, int f_len, int s1_b, int s1_a, int n_connections, int n_threads
);

int cascaded_row_order(
    char * connections,
    char * probes,
    char * row_order,
    int n_row, int s1_b, int s1_a, int s2_b, int s2_a, int n_connections
);

int self_cascaded_row_order(
    char * connections,
    char * probes,
    char * row_order,
    int n_row, int s1_b, int s1_a, int n_connections
);

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
);

int connect_self_th(
    char * s1,
    char * c1,
    char * connections,
    char * permutation_m,
    char * cas_s,
    char * cas_n,
    int n_row, int f_len, int s1_b, int s1_a, int n_connections
);

#endif /* CONNECT_H */