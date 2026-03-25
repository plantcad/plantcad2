#!/bin/bash
RESULTS=../results
SCRIPT=eval_accuracy.py

PREFIXES=(GCF_000001405.26_GRCh38 GCF_000001635.27_GRCm39)
MODELS=(evo2_7b evo2_1b_base)

# junction_type  input_basename   fwd_length  rc_length  n_tokens
# rc_length differs when the junction is not centered (start/stop: fwd=4094, rc=4095)
JUNCTIONS=(
    "donor    _donor         4095 4095 2"
    "acceptor _acceptor      4095 4095 2"
    "start    _start_sites   4094 4095 3"
    "stop     _stop_sites    4094 4095 3"
)

for PREFIX in "${PREFIXES[@]}"; do
    for JUNCTION in "${JUNCTIONS[@]}"; do
        read -r JNAME INFIX FWD_LEN RC_LEN NTOKENS <<< "$JUNCTION"
        INPUT_TSV="${RESULTS}/${PREFIX}${INFIX}.tsv"

        for MODEL in "${MODELS[@]}"; do
            FWD_TSV="${RESULTS}/${PREFIX}${INFIX}_${MODEL}_ntokens_${NTOKENS}.tsv"
            REV_TSV="${RESULTS}/${PREFIX}${INFIX}_${MODEL}_rc_ntokens_${NTOKENS}.tsv"

            if [[ -f "$FWD_TSV" ]]; then
                echo "=== ${PREFIX} | ${JNAME} | ${MODEL} | forward ==="
                python "$SCRIPT" \
                    --input-tsv "$INPUT_TSV" \
                    --pred-tsv  "$FWD_TSV" \
                    --length "$FWD_LEN" --n-tokens "$NTOKENS"
                echo
            fi

            if [[ -f "$REV_TSV" ]]; then
                echo "=== ${PREFIX} | ${JNAME} | ${MODEL} | reverse ==="
                python "$SCRIPT" \
                    --input-tsv "$INPUT_TSV" \
                    --pred-tsv  "$REV_TSV" \
                    --length "$RC_LEN" --n-tokens "$NTOKENS" --reverse
                echo
            fi
        done
    done
done
