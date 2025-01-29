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

        base(int nT, int nE, int nM, int nL, int nx, int shots=0, int rseed=0) : nT{nT}, nE{nE}, nM{nM}, nL{nL}, nx{nx}, shots{shots}, rseed{rseed}
        {
            rseed = rand(); 
        }
};