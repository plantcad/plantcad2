#!/usr/bin/env bash

set -euo pipefail

method_mid () {
  local WINDOW_SIZE=$1
  local PROXIMAL_CUTOFF=$2

  echo "Using midpoint method to generate windows" >&2

  if (( (WINDOW_SIZE % 2) != 0 )) || (( WINDOW_SIZE <= 1 )); then
    echo "Window size needs to be even and 2bp or more" >&2
    return 1
  else
    echo "Using window size of ${WINDOW_SIZE}bp" >&2
  fi

  TMP="tmp/data/${DATA_HASH}"
  mkdir -p "$TMP"

  for INTERVAL_FILE in "${INTERVALS}"/*.bed; do
    SPECIES=$(basename "${INTERVAL_FILE%.*}")
    IDX="${GENOMES}/${SPECIES}.fa.fai"
    echo "Processing ${SPECIES/_/ }" >&2
    # grab seqid, start, and end of positive peaks
    # expand peaks to WINDOW_SIZE, from midpoint
    awk 'NR==FNR{A[$1];next}($1 in A)' "$IDX" "${INTERVAL_FILE}" | \
      cut -f1-3 > "${TMP}/${SPECIES}.pos.filt.bed"
    awk -v OFS=$'\t' '{mid = int(($2+$3)/2); print $1, mid, mid + 1}' < "${TMP}/${SPECIES}.pos.filt.bed" | \
      bedtools slop -l $(( WINDOW_SIZE / 2 )) -r $(( (WINDOW_SIZE / 2) - 1 )) -g "$IDX" -i - | \
      awk '($3 - $2) == '$WINDOW_SIZE > "${TMP}/${SPECIES}.pos.${WINDOW_SIZE}.bed"
    bedtools sort -i "${TMP}/${SPECIES}.pos.${WINDOW_SIZE}.bed" -g "$IDX" > "${TMP}/${SPECIES}.pos.bed"

    # get the complement of the resized shorter and unresized longer open regions and make non-overlapping windows of WINDOW_SIZE
    awk '($3 - $2) > '$WINDOW_SIZE < "${TMP}/${SPECIES}.pos.filt.bed" > "${TMP}/${SPECIES}.pos.large.bed"
    bedtools sort -i <(cat "${TMP}/${SPECIES}.pos.${WINDOW_SIZE}.bed" "${TMP}/${SPECIES}.pos.large.bed") -g "$IDX" | \
      bedtools merge -i - | \
      bedtools complement -i - -g "$IDX" | \
      bedtools makewindows -b - -w $WINDOW_SIZE | \
      awk '($3 - $2) == '$WINDOW_SIZE | \
      cut -f1-3 > "${TMP}/${SPECIES}.neg.bed"

    # add targets/labels, trim windows on edges of chromosomes (for safety during random shifting), and combine into one file
    cat <(sed 's/$/\t1/' "${TMP}/${SPECIES}.pos.bed") <(sed 's/$/\t0/' "${TMP}/${SPECIES}.neg.bed") | \
      bedtools intersect -wa -f 1 -a - -b <(awk -v OFS=$'\t' '{if ($2 >= 1000) {print $1, 100, $2 - 100}}' "$IDX") | \
      bedtools sort -i - > "${TMP}/${SPECIES}.unlabeled.bed"

    # assign species intervals into genic, proximal, and distal if annotation provided
    if [ "$ANNOTATIONS" = "null" ] || [ "$PROXIMAL_CUTOFF" = "null"  ]; then
      ln -s "${SPECIES}.unlabeled.bed" "${TMP}/${SPECIES}.final.bed"
    else
      ## convert GFF3 into BED of genes
      grep $'\tgene\t' "${ANNOTATIONS}/${SPECIES}.gff3" | \
        awk -v 'OFS=\t' '{print $1, $4 - 1, $5}' | \
        sort -k1,1 -k2,2n > "${TMP}/${SPECIES}.genes.bed"
      ## extract genic intervals
      bedtools intersect -wa -u -f 0.5 -a "${TMP}/${SPECIES}.unlabeled.bed" -b "${TMP}/${SPECIES}.genes.bed" | \
        sed 's/$/\tgenic/' > "${TMP}/${SPECIES}.genic.bed"
      ## extract proximal intervals
      bedtools intersect -wa -u -f 0.5 -a "${TMP}/${SPECIES}.unlabeled.bed" -b <(bedtools flank -b $PROXIMAL_CUTOFF -g "$IDX" -i "${TMP}/${SPECIES}.genes.bed") | \
        bedtools subtract -f 1 -A -a - -b "${TMP}/${SPECIES}.genic.bed" | \
        sed 's/$/\tproximal/' > "${TMP}/${SPECIES}.proximal.bed"
      ## extract distal intervals
      bedtools slop -b $PROXIMAL_CUTOFF -g "$IDX" -i "${TMP}/${SPECIES}.genes.bed" | \
        bedtools sort -g "$IDX" -i - | \
        bedtools complement -g "$IDX" -i - | \
        bedtools intersect -wa -u -f 0.5 -a "${TMP}/${SPECIES}.unlabeled.bed" -b - | \
        bedtools subtract -f 1 -A -a - -b "${TMP}/${SPECIES}.genic.bed" | \
        bedtools subtract -f 1 -A -a - -b "${TMP}/${SPECIES}.proximal.bed" | \
        sed 's/$/\tdistal/' > "${TMP}/${SPECIES}.distal.bed"
      ## combine extracted intervals and sort
      cat "${TMP}/${SPECIES}."{genic,proximal,distal}'.bed' | \
        sort -k1,1 -k2,2n > "${TMP}/${SPECIES}.labeled.bed"
      ln -s "${SPECIES}.labeled.bed" "${TMP}/${SPECIES}.final.bed"
    fi
  done

  rm -f "$OUT_PATH"
  echo "Merging species files" >&2
  for SPECIES_FILE in "$TMP"/*.final.bed; do
    SPECIES=$(basename "${SPECIES_FILE%.*.*}")
    echo "Merging ${SPECIES/_/ }" >&2
    sed "s/\$/\t${SPECIES/_/ }/" "${SPECIES_FILE}" | \
      sort -k1,1 -k2,2n >> "$OUT_PATH"
  done

  # remove temp dir
  rm -rf "$TMP"

  return $?
}

if [ -z "${1:-}" ]; then
  echo "Usage: $0 config.json" >&2
  exit 1
fi

if [ ! -d "tmp/data" ]; then
  mkdir -p tmp/data
fi

PREPROCESSING_CONFIG=$1

INTERVALS=$(jq --raw-output '.paths.intervals' < "$PREPROCESSING_CONFIG")
GENOMES=$(jq --raw-output '.paths.genomes' < "$PREPROCESSING_CONFIG")
ANNOTATIONS=$(jq --raw-output '.paths.annotations' < "$PREPROCESSING_CONFIG")
DATA_HASH=$(jq '.paths, .preprocessing' < "$PREPROCESSING_CONFIG" | sha256sum | cut -f1 -d' ')
OUT_PATH="tmp/data/${DATA_HASH}.tsv"

if [ -f "$OUT_PATH" ]; then
  echo "Output file $OUT_PATH exists, exiting"
  exit 1
else
  method_mid \
    $(jq --raw-output '.preprocessing.window_size' < "$PREPROCESSING_CONFIG") \
    $(jq --raw-output '.preprocessing.proximal_cutoff' < "$PREPROCESSING_CONFIG")
fi
