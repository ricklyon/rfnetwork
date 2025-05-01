#ifndef CONNECT_H
#define CONNECT_H

int cascade_noise_data(
    char * m1, char * m2, char * c1, char * c2, char * out, int flen, int m1_len, int m2_len
);

int cascade_self_noise_data(
    char * m1, char * c1, char * out, int flen, int m1_len
);

#endif /* CONNECT_H */