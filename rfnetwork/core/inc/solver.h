#ifndef SOLVER_H
#define SOLVER_H

#include <string>
#include <vector>

#define N_FIELDS 18
#define N_COEFF 24

const char* FIELD_NAMES[]  = {
    "ex_y", "ex_z", "ex", 
    "ey_z", "ey_x", "ey", 
    "ez_x", "ez_y", "ez", 
    "hx_y", "hx_z", "hx", 
    "hy_z", "hy_x", "hy", 
    "hz_x", "hz_y", "hz", 
};

const char* COEFF_NAMES[] = {
    "Ca_ex_y", "Ca_ex_z", "Cb_ex_y", "Cb_ex_z",
    "Ca_ey_z", "Ca_ey_x", "Cb_ey_z", "Cb_ey_x",
    "Ca_ez_x", "Ca_ez_y", "Cb_ez_x", "Cb_ez_y",
    "Da_hx_y", "Da_hx_z", "Db_hx_y", "Db_hx_z",
    "Da_hy_z", "Da_hy_x", "Db_hy_z", "Db_hy_x",
    "Da_hz_x", "Da_hz_y", "Db_hz_x", "Db_hz_y",
};


struct Port {
    int *  idx;
    float * Vs_a;
    float * src;
    float * v_probe;
};


#endif /* SOLVER_H */