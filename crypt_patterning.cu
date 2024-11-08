
#include <curand_kernel.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <time.h>

#include "../yalla/include/dtypes.cuh"
#include "../yalla/include/inits.cuh"
#include "../yalla/include/polarity.cuh"
#include "../yalla/include/property.cuh"
#include "../yalla/include/solvers.cuh"
#include "../yalla/include/utils.cuh"
#include "../yalla/include/vtk.cuh"

// Generic simulation parameters
const auto r_max = 1;
const auto r_eq = 0.8;
const auto prolif_rate = 0.04f;
const auto n_0 = 2500;//5000;
const auto n_max = 1000000;
const auto dt = 0.1;
const auto force_modifier = 0.2f; // This controls the ratio between forces and friction

const auto real_time = 1000.0f;//1000.0f;
// const auto relax_time = 100.0f;
auto n_time_steps = int(real_time/dt);
auto skip_step = n_time_steps/100;

// Model parameters
const auto k_diff = 1.0f;
const auto k_stem = 1.0f;
// const auto t_diff = 4.0f; // differentiation time-scale
const auto k_w_d = 0.2f;
const auto k_b_d = 0.5f;
// const auto k_w_deg = 0.05;
const auto k_b_deg = 0.5;
const auto k_pol = 0.0f;//5.0f;

// const auto paneth_ratio = 0.05f;
const auto stem_cell_ratio = 0.8f;

const auto k_het = 2.0f;
// const auto Ft = 3.0f;
const auto compression_ratio = 0.3f;

const auto polarity_half_life = 1.0;
const auto polarity_update_amplitude = 2*M_PI;

// const auto r_seed = 1.2f;
// const auto r_pattern = 2.55f;
// const auto l_pattern = 9.24f;

std::string output_label = "crypt_simulation";
// std::string output_path = "/g/sharpeba/members/Miquel_Marin/crypt_patterning_output/crypt_patterning_sweep_time_scales_15-06-21/";
// std::string output_path = "/home/miquel/ownCloud/crypt_patterning_simulation/output/";
// std::string output_path = "/g/sharpe-hd/marin/crypt_patterning_spatial_scales_28-10-21/";
// std::string output_path = "/g/sharpe/members/Miquel_Marin/crypt_patterning_output/crypt_patterning_spatial_scales_12-01-21/";
// std::string output_path = "/g/sharpe/members/Miquel_Marin/crypt_patterning_output/crypt_patterning_sweep_time_scales_28-01-22/";
// std::string output_path = "/g/sharpe/members/Miquel_Marin/crypt_patterning_output/crypt_patterning_small_sweep_w_patterns_07-02-22/";
// std::string output_path = "/home/miquel/data/crypt_simulation_output/crypt_patterning_small_sweep_w_patterns_30-08-22/";
std::string output_path = "output";


MAKE_PT(crypt_cell, w, b, theta, phi, diff);

// command line parameters that need to be passed inside the solver device methods
#define N_PARAMS 3
__device__ float* d_solver_params;

__device__ float* d_cell_cycle;
__device__ int* d_epi_nbs;
__device__ bool* d_is_paneth;
__device__ float* d_compression;
__device__ bool* d_is_pattern;
__device__ int* d_n_homotypic;
__device__ int* d_n_heterotypic;


__device__ crypt_cell relaxation_force(
    crypt_cell Xi, crypt_cell r, float dist, int i, int j)
{
    crypt_cell dF{0};
    if (i == 0 or j == 0) return dF; // ghost node
    if (i == j) return dF;

    if (dist > r_max) return dF;

    d_epi_nbs[i]+=1;

    auto F = force_modifier * (fmaxf(r_eq - dist, 0) - fmaxf(dist - r_eq, 0));
    dF.x = r.x * F / dist;
    dF.y = r.y * F / dist;
    // dF.z = r.z * F / dist;

    return dF;
}

__device__ crypt_cell force(
    crypt_cell Xi, crypt_cell r, float dist, int i, int j)
{
    crypt_cell dF{0};
    if (i == 0 or j == 0) return dF; // ghost node
    if (i == j){
        auto effective_w = Xi.w + float(d_is_pattern[i]);
        dF.w = d_is_paneth[i] - d_solver_params[2]*Xi.w;
        dF.b = (effective_w <= 0.05) - k_b_deg*Xi.b ;
        // dF.b = (Xi.diff >= 1.0) - k_b_deg*Xi.b ;

        // dF.b = 1.0 * (Xi.diff>= 0.2f) - 0.1*Xi.b;
        auto pos_inc = k_diff * Xi.b;
        auto neg_inc = k_stem * effective_w;

        if(Xi.diff > 1.0){
            pos_inc = 0.0f;
            neg_inc = 0.0f;
        } else if(Xi.diff < 0.0)
            neg_inc = 0.0f;

        auto t_diff = d_solver_params[0];
        dF.diff = (pos_inc - neg_inc)/t_diff;

        return dF;
    }

    if (dist > r_max) return dF;

    auto is_homotypic = false;
    if(abs(r.diff)<0.5)
        is_homotypic = true;
    auto k_adh = 1.0;
    auto k_rep = 1.0;
    if(!is_homotypic)
        k_rep = k_het;

    auto F = force_modifier * (k_rep*fmaxf(r_eq - dist, 0) - k_adh*fmaxf(dist - r_eq, 0));

    dF.x = r.x * F / dist;
    dF.y = r.y * F / dist;
    // dF.z = r.z * F / dist;

    dF.w = -k_w_d*r.w;
    dF.b = -k_b_d*r.b;

    d_epi_nbs[i] += 1;
    if(is_homotypic)
        d_n_homotypic[i] += 1;
    else
        d_n_heterotypic[i] += 1;

    // Biasing polarity vector via heterospecific contacts (eph-ephrin like repulsion)
    auto r_hat = pt_to_pol(-r, dist);
    // auto diff_j = Xi.diff - r.diff;
    dF -= !is_homotypic * k_pol*unidirectional_polarization_force(Xi, r_hat);

    if(r_eq > dist)
        d_compression[i] += r_eq - dist;


    return dF;
}


__global__ void proliferate(float rate, int n_cells, curandState* d_state,
    crypt_cell* d_X, float3* d_old_v, int* d_n_cells)
{
    D_ASSERT(n_cells * rate <= n_max);
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_cells) return;  // Dividing new cells is problematic!
    if (i == 0) return;  // ghost node

    if(d_X[i].diff >= 1.0) return;

    if(d_compression[i]/float(d_epi_nbs[i]) > compression_ratio*r_eq) return;

    if(!d_is_paneth[i])
        d_cell_cycle[i] += rate * dt;


    if (d_cell_cycle[i] < 1.f) return;

    auto n = atomicAdd(d_n_cells, 1);
    // auto theta = acosf(2. * curand_uniform(&d_state[i]) - 1);
    auto phi = curand_uniform(&d_state[i]) * 2 * M_PI;
    // d_X[n].x = d_X[i].x + r_eq / 4 * sinf(theta) * cosf(phi);
    // d_X[n].y = d_X[i].y + r_eq / 4 * sinf(theta) * sinf(phi);
    // d_X[n].z = d_X[i].z + r_eq / 4 * cosf(theta);
    d_X[n].x = d_X[i].x + 0.25*r_eq * cosf(phi);
    d_X[n].y = d_X[i].y + 0.25*r_eq * sinf(phi);
    d_X[n].z = 0.0;
    d_old_v[n] = d_old_v[i];

    d_X[i].w = 0.5*d_X[i].w;
    d_X[i].b = 0.5*d_X[i].b;
    d_X[n].w = d_X[i].w;
    d_X[n].b = d_X[i].b;
    d_X[n].diff = d_X[i].diff;

    d_X[n].theta = d_X[i].theta;
    d_X[n].phi = curand_uniform(&d_state[i]) * 2 * M_PI;

    d_is_paneth[n] = d_is_paneth[i];
    d_compression[n] = d_compression[i];
    d_is_paneth[n] = false;

    d_cell_cycle[i] = -0.25 + 0.5*curand_uniform(&d_state[i]);
    d_cell_cycle[n] = -0.25 + 0.5*curand_uniform(&d_state[n]);

}


__global__ void set_up_cell_cycle(int* d_n_cells, crypt_cell* d_X, curandState* d_state)
{
    auto i = blockIdx.x * blockDim.x + threadIdx.x;

    d_cell_cycle[i] = curand_uniform(&d_state[i]);
}


__global__ void update_pattern(const int n_cells, crypt_cell* d_X, float r_pattern, float l_pattern)
{
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i>= n_cells) return;

    d_is_pattern[i] = false;

    auto px = d_X[i].x;
    auto py = d_X[i].y;

    auto center0_x = floor(px / l_pattern) * l_pattern;
    auto center1_x = ceil(px / l_pattern) * l_pattern;

    auto center0_y= floor(py / l_pattern) * l_pattern;
    auto center1_y = ceil(py / l_pattern) * l_pattern;
    // printf("i %i x %f x0 %f x1 %f\n",i, d_X[i].x, center0_x, center1_x);

    auto in_pattern = false;
    // distance from center 0-0
    auto dist = sqrt(pow(px - center0_x, 2) + pow(py - center0_y, 2));
    if(dist < r_pattern)
        in_pattern = true;

    // distance from center 0-1
    dist = sqrt(pow(px - center0_x, 2) + pow(py - center1_y, 2));
    if(dist < r_pattern)
        in_pattern = true;

    // distance from center 1-0
    dist = sqrt(pow(px - center1_x, 2) + pow(py - center0_y, 2));
    if(dist < r_pattern)
        in_pattern = true;

    // distance from center 1-1
    dist = sqrt(pow(px - center1_x, 2) + pow(py - center1_y, 2));
    if(dist < r_pattern)
        in_pattern = true;

    if(in_pattern){
        d_is_pattern[i] = true;
        // d_X[i].w -= w_diff*(d_X[i].w - 1.0);
    }
}

// Implementation of cell motility ********************************************

__global__ void update_polarities(const int n_cells, crypt_cell* d_X,
    float prob_update, curandState* d_state)
{
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i>= n_cells) return;

    if (curand_uniform(&d_state[i]) < prob_update){
        d_X[i].phi += polarity_update_amplitude * (curand_uniform(&d_state[i]) - 0.5);
        if (d_X[i].phi < 0.0)
            d_X[i].phi = 2 * M_PI + d_X[i].phi;
        else if(d_X[i].phi > 2*M_PI)
            d_X[i].phi = d_X[i].phi - 2 * M_PI;

    }

}

template<typename Pt>
using Traction_force = void(const Pt* __restrict__ d_X, const int i,
    Pt* d_dX);

template<typename Pt>
__device__ void constant_force_on_vector(const Pt* __restrict__ d_X, const int i,
    Pt* d_dX)
{
        if(i == 0) return;
        if(d_X[i].diff >= 1.0) return;
        auto Ft = d_solver_params[1];
        auto phi = d_X[i].phi;
        d_dX[i].x += cosf(phi) * force_modifier * Ft;
        d_dX[i].y += sinf(phi) * force_modifier * Ft;
}

template<typename Pt, Traction_force<Pt> force>
__global__ void traction(const Pt* __restrict__ d_X, Pt* d_dX,
    int n_max)
{
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_max) return;

    force(d_X, i, d_dX);
}

template<typename Pt, Traction_force<Pt> force>
void traction_forces(const int n, const Pt* __restrict__ d_X, Pt* d_dX)
{
    traction<Pt, force><<<(n + 32 - 1) / 32, 32>>>(
        d_X, d_dX, n);
}

//*****************************************************************************

int main(int argc, const char* argv[])
{
    // Prepare initial state
    Solution<crypt_cell, David_Gabriel_solver> cells{n_max, 50, 1.0f};

    Property<float> solver_params{N_PARAMS, "solver_params"};
    cudaMemcpyToSymbol(
        d_solver_params, &solver_params.d_prop, sizeof(d_solver_params));


    auto there_is_pattern = std::stoi(argv[1]) == 1;  // command line argument: 0 for no-pattern, 1 for pattern

    auto Ft = std::stof(argv[2]);  // stem cell traction force
    auto t_diff = std::stof(argv[3]);  // cell differentiation time scale
    auto r_seed = std::stof(argv[4]);  // mean cell-cell distance at t0
    auto k_w_deg = std::stof(argv[5]);  // degradation rate of Wnt
    auto paneth_ratio = std::stof(argv[6]);  // Initial ratio of Paneth cells
    auto r_pattern = std::stof(argv[7]);  // radius of Wnt patterns
    auto l_pattern = std::stof(argv[8]);  // radius of Wnt patterns


    std::string replicate = argv[9];

    solver_params.h_prop[0] = t_diff;
    solver_params.h_prop[1] = Ft;
    solver_params.h_prop[2] = k_w_deg;
    solver_params.copy_to_device();

    // Polarity vector used for cell motility randomly changes orientation
    // with exponential probability defined by a half-life parameter
    auto pol_update_probability = log(2)*dt/polarity_half_life;

    if(!there_is_pattern)
        output_label += "_no_pattern";
    else
        output_label += "_w_pattern";

    output_label = output_label + "_Ft_" + argv[2] +
                "_t-diff_" + argv[3] +
                "_r-seed_" + argv[4] +
                "_k-w-deg_" + argv[5] +
                "_paneth-ratio_" + argv[6] +
                "_r-pattern_" + argv[7] +
                "_l-pattern_" + argv[8] +
                "_rep_" + argv[9];

    // output_label += "_Ft_"+std::to_string(Ft).substr(0,5);

    std::cout<<output_label<<std::endl;


    *cells.h_n = n_0;
    random_disk(r_seed, cells);


    Property<float> cell_cycle{n_max,"cell_cycle"};
    cudaMemcpyToSymbol(d_cell_cycle, &cell_cycle.d_prop, sizeof(d_cell_cycle));

    Property<bool> is_paneth{n_max,"is_paneth"};
    cudaMemcpyToSymbol(d_is_paneth, &is_paneth.d_prop, sizeof(d_is_paneth));

    Property<bool> is_pattern{n_max,"is_pattern"};
    cudaMemcpyToSymbol(d_is_pattern, &is_pattern.d_prop, sizeof(d_is_pattern));


    Property<float> compression{n_max,"compression"};
    cudaMemcpyToSymbol(d_compression, &compression.d_prop, sizeof(d_compression));


    cells.h_X[0].x = 0.0f;
    cells.h_X[0].y = 0.0f;
    cells.h_X[0].z = 0.0f;
    is_paneth.h_prop[0] = false;
    cells.h_X[0].w = 0.0f;
    cells.h_X[0].b = 0.0f;
    cells.h_X[0].diff = 0.0f;
    cell_cycle.h_prop[0] = 0.0f;
    cells.set_fixed(0);

    for (auto i = 1; i < n_0; i++) {
        auto temp = cells.h_X[i].z;
        cells.h_X[i].z = cells.h_X[i].x;
        cells.h_X[i].x = temp;
        cell_cycle.h_prop[i] = rand() / (RAND_MAX + 1.);

        cells.h_X[i].theta = 0.5*M_PI;
        cells.h_X[i].phi = rand() / (RAND_MAX + 1.) * 2 * M_PI;

        cells.h_X[i].diff = 0.0;
        cells.h_X[i].w = 0.0f;
        cells.h_X[i].b = 0.0f;

        is_paneth.h_prop[i] = false;
        auto dice = rand() / (RAND_MAX + 1.);
        if (dice < paneth_ratio)
            is_paneth.h_prop[i] = true;
        else if (dice < stem_cell_ratio)
            cells.h_X[i].diff = 0.0;
        else
            cells.h_X[i].diff = 1.0;

    }
    cells.copy_to_device();
    cell_cycle.copy_to_device();
    is_paneth.copy_to_device();

    Property<int> n_epi_nbs{n_max,"n_epi_nbs"};
    cudaMemcpyToSymbol(d_epi_nbs, &n_epi_nbs.d_prop, sizeof(d_epi_nbs));

    Property<int> n_homotypic{n_max,"n_homotypic"};
    cudaMemcpyToSymbol(d_n_homotypic, &n_homotypic.d_prop, sizeof(d_n_homotypic));

    Property<int> n_heterotypic{n_max,"n_heterotypic"};
    cudaMemcpyToSymbol(d_n_heterotypic, &n_heterotypic.d_prop, sizeof(d_n_heterotypic));

    auto traction = [&](const int n, const crypt_cell* __restrict__ d_X, crypt_cell* d_dX) {
        thrust::fill(thrust::device, n_epi_nbs.d_prop,
            n_epi_nbs.d_prop + cells.get_d_n(), 0);
        thrust::fill(thrust::device, compression.d_prop,
            compression.d_prop + cells.get_d_n(), 0);

        thrust::fill(thrust::device, n_homotypic.d_prop,
            n_homotypic.d_prop + cells.get_d_n(), 0);
        thrust::fill(thrust::device, n_heterotypic.d_prop,
            n_heterotypic.d_prop + cells.get_d_n(), 0);

        return traction_forces<crypt_cell, constant_force_on_vector>(n, d_X, d_dX);
    };

    // // Relaxation phase
    // for (auto time_step = 0; time_step <= int(relax_time/dt); time_step++)
    //     cells.take_step<relaxation_force, friction_on_background>(dt, reset_nbs);


    curandState* d_state;
    cudaMalloc(&d_state, n_max * sizeof(curandState));
    auto seed = time(NULL);
    setup_rand_states<<<(n_max + 128 - 1) / 128, 128>>>(n_max, seed, d_state);

    set_up_cell_cycle<<<(cells.get_d_n() + 128 - 1) / 128, 128>>>(cells.d_n, cells.d_X, d_state);

    // Simulate growth
    Vtk_output output{output_label, output_path, true};
    for (auto time_step = 0; time_step <= n_time_steps; time_step++) {

        if(there_is_pattern)
            update_pattern<<<(cells.get_d_n() + 128 - 1) / 128, 128>>>(
                cells.get_d_n(), cells.d_X, r_pattern, l_pattern);

        update_polarities<<<(cells.get_d_n() + 128 - 1) / 128, 128>>>(
            cells.get_d_n(), cells.d_X, pol_update_probability, d_state);

        cells.take_step<force, friction_on_background>(dt, traction);
        proliferate<<<(cells.get_d_n() + 128 - 1) / 128, 128>>>(
            prolif_rate, cells.get_d_n(), d_state,
            cells.d_X, cells.d_old_v, cells.d_n);

        if(time_step % skip_step == 0){
            cudaDeviceSynchronize();
            cells.copy_to_host();
            is_paneth.copy_to_host();
            is_pattern.copy_to_host();
            compression.copy_to_host();
            n_homotypic.copy_to_host();
            n_heterotypic.copy_to_host();


            n_epi_nbs.copy_to_host();
            output.write_positions(cells);
            output.write_polarity(cells);
            output.write_field(cells, "w", &crypt_cell::w);
            output.write_field(cells, "b", &crypt_cell::b);
            output.write_field(cells, "diff", &crypt_cell::diff);
            output.write_property(n_epi_nbs);
            output.write_property(is_paneth);
            output.write_property(is_pattern);
            output.write_property(compression);
            output.write_property(n_homotypic);
            output.write_property(n_heterotypic);


        }

    }

    return 0;
}
