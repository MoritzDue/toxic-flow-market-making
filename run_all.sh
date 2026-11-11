#!/bin/sh
# Reproduces every result in results/ and figures/ (about 2 hours on 2 cores).
# For a quick low-precision pass:  PATH_SCALE=0.05 sh run_all.sh
set -e
for e in e0_verification e1_main e2_switching_speed e3_severity e4_detection e5_param_misspec e6_risk_aversion e7_structural e8_strategic x1_bayes_strategic; do
    python -m experiments.$e
done
python -m experiments.make_figures
