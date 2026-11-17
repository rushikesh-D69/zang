/* Host check: reads N x ZANG_POLICY_OBS_DIM float32 observations from argv[1], writes N x 15 float32 actions to argv[2].
 * export_policy.py compiles this with gcc against the generated policy_weights.h and compares with onnxruntime. */
#include <stdio.h>
#include <stdlib.h>

#include "zang_policy.h"

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: %s obs.f32 act.f32\n", argv[0]);
        return 2;
    }
    FILE *fi = fopen(argv[1], "rb"), *fo = fopen(argv[2], "wb");
    if (!fi || !fo) {
        perror("open");
        return 1;
    }
    float obs[ZANG_POLICY_OBS_DIM], act[ZANG_POLICY_ACT_DIM];
    while (fread(obs, sizeof(float), ZANG_POLICY_OBS_DIM, fi) == ZANG_POLICY_OBS_DIM) {
        zang_policy_run(obs, act);
        fwrite(act, sizeof(float), ZANG_POLICY_ACT_DIM, fo);
    }
    fclose(fi);
    fclose(fo);
    return 0;
}
