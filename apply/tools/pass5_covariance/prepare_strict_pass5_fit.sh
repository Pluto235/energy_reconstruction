#!/usr/bin/env bash
set -euo pipefail

RUN_DIR=${RUN_DIR:-/home/lhaaso/liushijie/energy/pass5_crab_v6_125d_covariance}
FIT_DIR="$RUN_DIR/sorted_gti_fit"
MAP="$RUN_DIR/pass5_v6_sorted_gti_map.root"
PASS5_BASE=/home/lhaaso/xishaoqiang/lhaaso/data/pass5/z50
OFFICIAL_CRAB=/home/lhaaso/liushijie/energy/wcda_crab_sed_pass5_20260616_104941

export GROUP=lhaaso
export GROUPNAME=lhaaso
set +u
source /afs/ihep.ac.cn/users/x/xishaoqiang/.bashrc_everyone
set -u

if [[ ! -s "$MAP" ]]; then
    echo "Missing strict merged map: $MAP" >&2
    exit 2
fi

mkdir -p "$FIT_DIR"
cd "$FIT_DIR"

LIVE_DAYS=$(
    root -l -b -q -e \
        "TFile f(\"$MAP\"); auto t=(TTree*)f.Get(\"bkg_header\"); if(!t)t=(TTree*)f.Get(\"map_header\"); if(!t)t=(TTree*)f.Get(\"Map_header\"); if(!t){fprintf(stderr,\"Missing bkg/map header\\n\"); gSystem->Exit(4);} double x=0; t->SetBranchAddress(\"Ltime\",&x); t->GetEntry(0); printf(\"STRICT_LIVE_DAYS %.12f\\n\",x/86400.);" \
        | awk '/STRICT_LIVE_DAYS/ {print $2}'
)
if [[ -z "$LIVE_DAYS" ]]; then
    echo "Could not read strict Pass5 live time from $MAP" >&2
    exit 3
fi
echo "$LIVE_DAYS" > strict_pass5_live_days.txt

python3 - "$MAP" "$FIT_DIR/data_config.yaml" "$FIT_DIR/data.root" "$LIVE_DAYS" <<'PY'
import sys
import yaml

source_map, output, data_root, live_days = sys.argv[1:]
payload = {
    "time_array_run": [["2022-01-01", "2022-06-30"]],
    "zenith_max": 50,
    "cut_std": 95,
    "live_time": [float(live_days)],
    "data_save_path": data_root,
    "data_read_path": [source_map],
}
with open(output, "w", encoding="utf-8") as handle:
    yaml.safe_dump(payload, handle, sort_keys=False)
PY
read_hsc_allsky data_config.yaml

python3 - "$PASS5_BASE/WCDA/irfs.yaml" "$FIT_DIR/irfs.yaml" "$FIT_DIR/irfs.root" "$LIVE_DAYS" <<'PY'
import sys
import yaml

source, output, irfs_map, live_days = sys.argv[1:]
with open(source, encoding="utf-8") as handle:
    payload = yaml.safe_load(handle)
payload["time_array_run"] = [["2022-01-01", "2022-06-30"]]
payload["live_time"] = [float(live_days)]
payload["irfs_model"]["irfs_map"] = irfs_map
with open(output, "w", encoding="utf-8") as handle:
    yaml.safe_dump(payload, handle, sort_keys=False)
PY
gtirfs irfs.yaml

cp "$OFFICIAL_CRAB/bg.yaml" bg.yaml
python3 - "$FIT_DIR/bg.yaml" "$FIT_DIR/data.root" <<'PY'
import sys
import yaml

path, data_root = sys.argv[1:]
with open(path, encoding="utf-8") as handle:
    payload = yaml.safe_load(handle)
payload["selection"]["all_sky_map"] = data_root
payload["selection"]["roi_map"] = "roi_ccube.root"
with open(path, "w", encoding="utf-8") as handle:
    yaml.safe_dump(payload, handle, sort_keys=False)
PY

gtselect bg.yaml
gtsrcmap bg.yaml

cp bg.yaml src_stage1.yaml
tune_yaml src_stage1.yaml add_source roi 7
gtsrcmap src_stage1.yaml
tune_yaml src_stage1.yaml set_source free_all_norm
gtsrcmap src_stage1.yaml
gtlike src_stage1.yaml fit_stage1.yaml

cp fit_stage1.yaml input_stage2.yaml
tune_yaml input_stage2.yaml set_source free_all_norm
tune_yaml input_stage2.yaml rm_source ts_cut 4
gtlike input_stage2.yaml fit_stage2.yaml

cp fit_stage2.yaml covariance_input.yaml
tune_yaml covariance_input.yaml set_source free_all_norm
tune_yaml covariance_input.yaml set_source free_one_sed J0534+2200
python3 - "$FIT_DIR/covariance_input.yaml" "$FIT_DIR/pass5_pivot_transform.json" <<'PY'
import json
import math
import sys
import yaml

path, audit_path = sys.argv[1:]
with open(path, encoding="utf-8") as handle:
    payload = yaml.safe_load(handle)
sed = payload["source_dict"]["J0534+2200"]["sed_model"]
old_pivot = float(sed["E_0"])
new_pivot = 3.0
norm = float(sed["norm"][0])
scale = float(sed["norm"][3])
alpha = float(sed["index1"][0])
beta = float(sed["index2"][0])
log_ratio = math.log(new_pivot / old_pivot)
physical_norm = norm * scale
new_physical_norm = physical_norm * math.exp(-alpha * log_ratio - beta * log_ratio**2)
new_alpha = alpha + 2.0 * beta * log_ratio
sed["norm"][0] = new_physical_norm / scale
sed["index1"][0] = new_alpha
sed["E_0"] = new_pivot
with open(path, "w", encoding="utf-8") as handle:
    yaml.safe_dump(payload, handle, sort_keys=False)
audit = {
    "model": sed["sed_type"],
    "old_pivot_tev": old_pivot,
    "new_pivot_tev": new_pivot,
    "old_parameters": {"physical_norm": physical_norm, "alpha": alpha, "beta": beta},
    "new_parameters": {"physical_norm": new_physical_norm, "alpha": new_alpha, "beta": beta},
    "transformation": "natural-log LogPar exact pivot change",
}
with open(audit_path, "w", encoding="utf-8") as handle:
    json.dump(audit, handle, indent=2)
    handle.write("\n")
PY
"$RUN_DIR/bin/gtlike_cov" covariance_input.yaml covariance_fit.yaml

echo "STRICT_PASS5_FIT_COMPLETE $FIT_DIR/covariance_fit.yaml"
