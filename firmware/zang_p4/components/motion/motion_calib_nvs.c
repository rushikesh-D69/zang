/* motion_calib_nvs.c - calibration table persistence (NVS namespace "zang_calib"). */
#include <string.h>
#include "esp_check.h"
#include "esp_log.h"
#include "motion.h"
#include "nvs.h"

static const char *TAG = "calib";
#define NS      "zang_calib"
#define KEY     "table"
#define VERSION 1

typedef struct {
    uint16_t version;
    uint16_t n_joints;
    char robot_sha[16]; /* first 16 hex chars of ROBOT_JOINTS_SOURCE_SHA256 when saved (informational) */
    motion_calib_joint_t j[MOTION_N];
} calib_blob_t;

esp_err_t motion_calib_load(motion_calib_t *c)
{
    motion_calib_defaults(c);
    nvs_handle_t h;
    esp_err_t err = nvs_open(NS, NVS_READONLY, &h);
    if (err != ESP_OK) return err; /* ESP_ERR_NVS_NOT_FOUND on a fresh robot */
    calib_blob_t blob;
    size_t len = sizeof(blob);
    err = nvs_get_blob(h, KEY, &blob, &len);
    nvs_close(h);
    if (err != ESP_OK) return err;
    if (len != sizeof(blob) || blob.version != VERSION || blob.n_joints != MOTION_N) {
        ESP_LOGW(TAG, "stored calibration has another layout (v%u, %u joints): ignored", blob.version,
                 blob.n_joints);
        return ESP_ERR_INVALID_VERSION;
    }
    memcpy(c->j, blob.j, sizeof(c->j));
    c->from_nvs = true;
    if (strncmp(blob.robot_sha, ROBOT_JOINTS_SOURCE_SHA256, sizeof(blob.robot_sha)) != 0) {
        ESP_LOGW(TAG, "calibration was saved with another robot.json; check joint order/limits");
    }
    return ESP_OK;
}

esp_err_t motion_calib_store(const motion_calib_t *c)
{
    calib_blob_t blob = {.version = VERSION, .n_joints = MOTION_N};
    memcpy(blob.robot_sha, ROBOT_JOINTS_SOURCE_SHA256, sizeof(blob.robot_sha));
    memcpy(blob.j, c->j, sizeof(blob.j));
    nvs_handle_t h;
    ESP_RETURN_ON_ERROR(nvs_open(NS, NVS_READWRITE, &h), TAG, "open");
    esp_err_t err = nvs_set_blob(h, KEY, &blob, sizeof(blob));
    if (err == ESP_OK) err = nvs_commit(h);
    nvs_close(h);
    return err;
}
