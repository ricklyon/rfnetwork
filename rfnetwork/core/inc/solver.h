#ifndef SOLVER_H
#define SOLVER_H

#include <string>
#include <vector>

#define N_FIELDS 18
#define N_COEFF 24

extern const char* FIELD_NAMES[];
extern const char* COEFF_NAMES[];

struct Port {
    int *  idx;
    float * Vs_a;
    float * src;
    float * v_probe;
};

struct SolverConfig {
    float* field[N_FIELDS] = {nullptr}; 
    float* coeff[N_COEFF] = {nullptr}; 
    int Nx;
    int Ny;
    int Nz;
    int Nt;
};

int solver_run(SolverConfig * sc);

int solver_update_ex(SolverConfig * sc, int x_start, int x_stop);

int solver_coeff_index(const char* value);

int solver_field_index(const char* value);

#endif /* SOLVER_H */