#!/bin/bash
# Closed-loop eval in a HyperDrone scene: grade.cpp flies, ext_depth_server.py supplies stereo.
# Usage: run_closed_loop.sh NAME GRADER CKPT MODE LENS SEEDS MISSIONS VOXEL_DIR OUTDIR [ENV...]
# Env: THREADS, RES (320), SERVER_ARGS (extra ext_depth_server.py options, e.g. --calib).
# MODE teacher, or LENS=preset: no rendering (teachers use true depth; preset = our noise model on the
# voxelized scene, for comparison with the real-stereo flights).
NAME=$1 GRADER=$2 CK=$3 MODE=$4 LENS=$5 SEEDS=$6 MDIR=$7 VDIR=$8 OUT=$9; shift 9
HERE=$(cd $(dirname $0) && pwd)
REPO=$(cd "$HERE/../.." && pwd)
RAPTOR=${RAPTOR_BIN:-$REPO/assets/raptor.bin}
T=${THREADS:-4}
export RLV2_INIT_SCAN=1 RLV2_INIT_SCAN_TURNS=0 RLV2_INIT_SCAN_SETTLE=1 RLV2_INIT_SCAN_CLIMB=0
mkdir -p $OUT
if [ $MODE = teacher ] || [ $LENS = preset ]; then
  env "$@" RLV2_THREADS=$T $GRADER $MDIR "$RAPTOR" $CK $MODE 1 $SEEDS $OUT/$NAME.csv > $OUT/$NAME.txt 2>/dev/null
else
  F=$(mktemp -d /tmp/extd.XXXX)
  python $HERE/ext_depth_server.py $F $T $VDIR --lens $LENS ${RES:+--res $RES} --metrics $OUT/${NAME}_rays.json --save $OUT/frames $SERVER_ARGS > $OUT/${NAME}_server.log 2>&1 &
  SP=$!
  while ! grep -q ready $OUT/${NAME}_server.log 2>/dev/null; do sleep 1; done
  env "$@" RLV2_EXT_DEPTH=$F RLV2_THREADS=$T RLV2_DEPTH_PRESET=realistic RLV2_PRESET_DESPECKLE=1 \
    $GRADER $MDIR "$RAPTOR" $CK $MODE 1 $SEEDS $OUT/$NAME.csv > $OUT/$NAME.txt 2>/dev/null
  wait $SP; rm -rf $F
fi
echo "$NAME $(grep ^all $OUT/$NAME.txt)"
