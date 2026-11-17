/* See zang_policy.h. Generic dense MLP: ELU on hidden layers, linear output. */
#include "zang_policy.h"

#include <math.h>

#if (ZP_HAS_IMU && ZP_OBS_DIM != 56) || (!ZP_HAS_IMU && ZP_OBS_DIM != 50) || ZP_ACT_DIM != 15
#error "policy_weights.h does not match the observation/action layout in zang_rl/obs.py"
#endif

static float s_a[ZP_MAX_WIDTH];
static float s_b[ZP_MAX_WIDTH];

static void dense(const float *w, const float *b, const float *x, float *y, int n_in, int n_out, int elu)
{
    for (int o = 0; o < n_out; ++o) {
        const float *row = w + o * n_in;
        float acc = b[o];
        for (int i = 0; i < n_in; ++i) {
            acc += row[i] * x[i];
        }
        y[o] = (elu && acc < 0.0f) ? (expf(acc) - 1.0f) : acc;
    }
}

void zang_policy_run(const float *obs, float *action)
{
    for (int i = 0; i < ZP_OBS_DIM; ++i) {
        s_a[i] = (obs[i] - zp_obs_mean[i]) * zp_obs_inv_std[i];
    }
    float *in = s_a, *out = s_b;
    for (int l = 0; l < ZP_N_LAYERS; ++l) {
        const int last = (l == ZP_N_LAYERS - 1);
        dense(zp_w[l], zp_b[l], in, last ? action : out, zp_layer_in[l], zp_layer_out[l], !last);
        float *t = in;
        in = out;
        out = t;
    }
    for (int i = 0; i < ZP_ACT_DIM; ++i) {
#if ZP_HAS_ACTION_CLIP
        action[i] = action[i] > ZP_ACTION_CLIP ? ZP_ACTION_CLIP : (action[i] < -ZP_ACTION_CLIP ? -ZP_ACTION_CLIP : action[i]);
#endif
        action[i] *= ZP_ACTION_GAIN;
    }
}

float zang_policy_target(int i, float action)
{
    return zp_home_q[i] + ZP_ACTION_SCALE * action;
}
