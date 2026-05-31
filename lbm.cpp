#include <iostream>
#include <vector>
#include <utility> // Required for std::pair

using namespace std;

const int Nx = 400;
const int Ny = 400;

double f[Nx][Ny][9];
double f_new[Nx][Ny][9];


double rho[Nx][Ny];
double ux[Nx][Ny];
double uy[Nx][Ny];

const vector<double> weights = {
    4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0
};

const vector<int> opp = {0, 3, 4, 1, 2, 7, 8, 5, 6};


const vector<pair<int, int>> eis = {
    {0,0}, {1,0}, {0,1}, {-1,0}, {0,-1},
    {1,1}, {-1,1}, {-1,-1}, {1,-1}
};


const double cs2 = 1.0/3.0;
const double tau   = 0.6;
const double rho0  = 1.0;
const double uw    = 0.1;
const int R = Ny/10;

bool solid[Nx][Ny];

void computeEquilibrium(double rho, double ux, double uy, double feq[9]){
    double u2 = ux*ux + uy*uy;
    for(int i = 0; i < 9; i++){
        double eiu = eis[i].first*ux + eis[i].second*uy;
        feq[i] = weights[i]*rho*(1.0 + eiu/cs2 + eiu*eiu/(2.0*cs2*cs2) - u2/(2.0*cs2));
    }
}


void initialise(){
    for(int x = 0; x < Nx; x++){
        for(int y = 0; y < Ny; y++){
            int dx = x - Nx/4;
            int dy = y - Ny/2 - 5;
            solid[x][y] = (dx*dx + dy*dy < R*R);

            computeEquilibrium(rho0, uw, 0.0, f[x][y]);
            for(int i = 0; i < 9; i++){
                f[x][y][i] += ((double)rand() / RAND_MAX - 0.5) * 0.01;
            }
        }
    }
}

void computeMacros(){
    for(int x = 0; x < Nx; x++){
        for(int y = 0; y < Ny; y++){
            if(solid[x][y]){
                rho[x][y] = 0;
                ux[x][y] = 0;
                uy[x][y] = 0;
            }
            else{
                double r = 0.0, mx = 0.0, my = 0.0;
                for(int i = 0; i < 9; i++){
                    r  += f[x][y][i];
                    mx += eis[i].first  * f[x][y][i];
                    my += eis[i].second * f[x][y][i];
                }
                rho[x][y] = r;
                ux[x][y] = mx/r;
                uy[x][y] = my/r;
            }
        }
    }
}

void collide(){
    for(int x = 0; x < Nx; x++){
        for(int y = 0; y < Ny; y++){
            if(solid[x][y]) continue;
            double feq[9];
            computeEquilibrium(rho[x][y], ux[x][y], uy[x][y], feq);
            for(int i = 0; i < 9; i++){
                f[x][y][i] -= (1.0/tau) * (f[x][y][i] - feq[i]);
            }
        }
    }
}

void stream(){
    for (int i = 0; i < Nx; ++i)
        for (int j = 0; j < Ny; ++j)
            for (int k = 0; k < 9; ++k)
                f_new[i][j][k] = 0.0;

    for(int x = 0; x < Nx; x++){
        for(int y = 0; y < Ny; y++){
            if(solid[x][y]) continue;
            for(int i = 0; i < 9; i++){
                int xnew = x + eis[i].first;
                int ynew = y + eis[i].second;
                bool outOfBounds = (xnew < 0 || xnew >= Nx || ynew < 0 || ynew >= Ny);
                if(outOfBounds || solid[xnew][ynew]){
                    f_new[x][y][opp[i]] = f[x][y][i];
                } else {
                    f_new[xnew][ynew][i] = f[x][y][i];
                }
            }
        }
    }
    for (int i = 0; i < Nx; ++i)
        for (int j = 0; j < Ny; ++j)
            for (int k = 0; k < 9; ++k)
                f[i][j][k] = f_new[i][j][k];
}

void applyBoundaries() {
    for (int y = 0; y < Ny; y++) {
        if (solid[0][y]) continue;

        double ux_in = uw;
        double uy_in = 0.0;

        double rho_in = (1.0 / (1.0 - ux_in)) * (f[0][y][0] + f[0][y][2] + f[0][y][4] +
        2.0 * (f[0][y][3] + f[0][y][6] + f[0][y][7]));

        f[0][y][1] = f[0][y][3] + (2.0 / 3.0) * rho_in * ux_in;

        f[0][y][5] = f[0][y][7] - 0.5 * (f[0][y][2] - f[0][y][4]) +
        (1.0 / 6.0) * rho_in * ux_in + 0.5 * rho_in * uy_in;

        f[0][y][8] = f[0][y][6] + 0.5 * (f[0][y][2] - f[0][y][4]) +
        (1.0 / 6.0) * rho_in * ux_in - 0.5 * rho_in * uy_in;
    }

    for (int y = 0; y < Ny; y++) {
        for (int i = 0; i < 9; i++) {
            f[Nx - 1][y][i] = f[Nx - 2][y][i];
        }
    }
}

#include <fstream>
#include <string>

void saveVTK(int step) {
    // Create a sequential filename: lbm_0.vtk, lbm_500.vtk, etc.
    std::string filename = "lbm_" + std::to_string(step) + ".vtk";
    std::ofstream file(filename);

    // 1. VTK Header
    file << "# vtk DataFile Version 3.0\n";
    file << "LBM Fluid Flow\n";
    file << "ASCII\n";
    file << "DATASET STRUCTURED_POINTS\n";
    file << "DIMENSIONS " << Nx << " " << Ny << " 1\n";
    file << "ORIGIN 0 0 0\n";
    file << "SPACING 1 1 1\n";

    int numPoints = Nx * Ny;
    file << "POINT_DATA " << numPoints << "\n";

    // 2. Write Velocity Vectors
    // VTK expects X to increase fastest, so X is the inner loop.
    file << "VECTORS velocity double\n";
    for (int y = 0; y < Ny; y++) {
        for (int x = 0; x < Nx; x++) {
            if (solid[x][y]) {
                file << "0.0 0.0 0.0\n"; // Inside cylinder
            } else {
                file << ux[x][y] << " " << uy[x][y] << " 0.0\n"; // 3D vector, Z=0
            }
        }
    }

    // 3. Write Density (Scalar)
    file << "SCALARS density double 1\n";
    file << "LOOKUP_TABLE default\n";
    for (int y = 0; y < Ny; y++) {
        for (int x = 0; x < Nx; x++) {
            file << rho[x][y] << "\n";
        }
    }

    // 4. Write Solid Mask (Scalar)
    file << "SCALARS solid_mask int 1\n";
    file << "LOOKUP_TABLE default\n";
    for (int y = 0; y < Ny; y++) {
        for (int x = 0; x < Nx; x++) {
            file << solid[x][y] << "\n";
        }
    }

    file.close();
}

int main() {
    // 1. Setup the initial state
    initialise();
    computeMacros();

    int max_steps = 10000; // Number of iterations

    // 2. Main simulation loop
    for (int step = 0; step < max_steps; step++) {

        collide();
        stream();
        applyBoundaries();
        computeMacros();

        if (step % 50 == 0) {
            cout << "Step: " << step << " / " << max_steps << " completed." << endl;
            saveVTK(step);
        }
    }

    cout << "Simulation finished!" << endl;
    return 0;
}
