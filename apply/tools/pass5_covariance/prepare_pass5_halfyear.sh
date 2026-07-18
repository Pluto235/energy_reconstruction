#!/usr/bin/env bash
set -euo pipefail

RUN_DIR=${RUN_DIR:-/home/lhaaso/liushijie/energy/pass5_crab_2022H1_covariance}
PASS5_BASE=/home/lhaaso/xishaoqiang/lhaaso/data/pass5/z50
OFFICIAL_CRAB=/home/lhaaso/liushijie/energy/wcda_crab_sed_pass5_20260616_104941
DI_DIR=/home/lhaaso/hushicong/8_All_sky_survey/script/DI_mask_v1.1.1_nbins_v2
MAP_BASE=/eos/user/h/hushicong/WCDA/8_All_sky_survey/data/Cod_FullArray/DI_mask/WCDA_v0/PincOpt_7bins_nq03_ge200_test_v2_timebin1_4hours_bkg10hours_le2000_bkgJnow/2022
LIVE_DAYS=166.451442205

export GROUP=lhaaso
export GROUPNAME=lhaaso
source /afs/ihep.ac.cn/users/x/xishaoqiang/.bashrc_everyone

mkdir -p "$RUN_DIR" "$RUN_DIR/logs" "$RUN_DIR/profiles"
cd "$RUN_DIR"

cat > halfyear_monthly_maps.list <<EOF
$MAP_BASE/202201_bkgJ2000.root
$MAP_BASE/202202_bkgJ2000.root
$MAP_BASE/202203_bkgJ2000.root
$MAP_BASE/202204_bkgJ2000.root
$MAP_BASE/202205_bkgJ2000.root
$MAP_BASE/202206_bkgJ2000.root
EOF

if [[ ! -s pass5_2022H1_map.root ]]; then
  "$DI_DIR/DI_Merge" halfyear_monthly_maps.list hon hbkg hoff step2 pass5_2022H1_map.root
fi

cat > data_config.yaml <<EOF
time_array_run:
  - [2022-01-01, 2022-06-30]
zenith_max: 50
cut_std: 95
live_time: [$LIVE_DAYS]
data_save_path: data.root
data_read_path:
  - $RUN_DIR/pass5_2022H1_map.root
EOF

read_hsc_allsky data_config.yaml

python3 - "$PASS5_BASE/WCDA/irfs.yaml" "$RUN_DIR/irfs.yaml" "$RUN_DIR/irfs.root" "$LIVE_DAYS" <<'PY'
import sys
import yaml

source, output, irfs_map, live_days = sys.argv[1:]
with open(source, encoding="utf-8") as handle:
    data = yaml.safe_load(handle)
data["time_array_run"] = [["2022-01-01", "2022-06-30"]]
data["live_time"] = [float(live_days)]
data["irfs_model"]["irfs_map"] = irfs_map
with open(output, "w", encoding="utf-8") as handle:
    yaml.safe_dump(data, handle, sort_keys=False)
PY

gtirfs irfs.yaml

cp "$OFFICIAL_CRAB/bg.yaml" bg.yaml
python3 - "$RUN_DIR/bg.yaml" "$RUN_DIR/data.root" <<'PY'
import sys
import yaml

path, data_root = sys.argv[1:]
with open(path, encoding="utf-8") as handle:
    data = yaml.safe_load(handle)
data["selection"]["all_sky_map"] = data_root
data["selection"]["roi_map"] = "roi_ccube.root"
with open(path, "w", encoding="utf-8") as handle:
    yaml.safe_dump(data, handle, sort_keys=False)
PY

gtselect bg.yaml
gtsrcmap bg.yaml

cp bg.yaml src_stage1.yaml
tune_yaml src_stage1.yaml add_source roi 7
gtsrcmap src_stage1.yaml
tune_yaml src_stage1.yaml set_source free_all_norm
gtsrcmap src_stage1.yaml
gtlike src_stage1.yaml fit_stage1.yaml

cp fit_stage1.yaml central_input.yaml
tune_yaml central_input.yaml set_source free_all_norm
tune_yaml central_input.yaml set_source free_one_sed J0534+2200
gtsrcmap central_input.yaml
gtlike central_input.yaml central_fit.yaml

python3 "$RUN_DIR/generate_profile_grid.py" \
  --fit-yaml central_fit.yaml \
  --output-dir profiles \
  --manifest profile_manifest.json \
  --step-scale 0.75

root -l -b -q -e 'TFile f("data.root"); auto h=(TH1*)f.Get("header"); std::cout << "DATA_LIVE_DAYS " << h->GetBinContent(1) << std::endl; gSystem->Exit(0);'
root -l -b -q -e 'TFile f("irfs.root"); auto h=(TH1*)f.Get("header"); std::cout << "IRFS_LIVE_DAYS " << h->GetBinContent(1) << std::endl; gSystem->Exit(0);'

