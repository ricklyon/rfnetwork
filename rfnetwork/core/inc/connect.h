#ifndef CONNECT_H
#define CONNECT_H

int cascade_ndata(
    char * m1, char * m2, char * c1, char * c2, char * out, int flen, int m1_len, int m2_len
);

int cascade_self_ndata(
    char * m1, char * c1, char * out, int flen, int m1_len
);

int connection_matrix(
    char * s1,
    char * s2,
    char * connections,
    char * probes,
    char * m1,
    char * m2,
    char * row_order,
    int f_len, int s1_b, int s1_a, int s2_b, int s2_a, int n_connections
);

#endif /* CONNECT_H */