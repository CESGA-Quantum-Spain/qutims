#pragma once

#include <cmath>
#include <numeric>


class base {
    public:
        int nT; 
        int nE;
        int nM;
        int nL;
        int nx;
        int shots;
        int rseed;
        int NE;
        int NM;

        explicit base(int nT, int nE, int nM, int nL, int nx, int shots=0, int rseed=0) : nT{nT}, nE{nE}, nM{nM}, nL{nL}, nx{nx}, shots{shots}, rseed{rseed}, NE{(int)pow(2, nE)}, NM{(int)pow(2,nM)}
        {
            rseed = rand(); 
        }

        //virtual int encode_Nparams();


};