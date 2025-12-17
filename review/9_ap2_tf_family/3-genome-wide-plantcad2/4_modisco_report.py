import os
import pickle
import h5py
import pandas
import tempfile
import subprocess
import sys
import matplotlib

matplotlib.use("pdf")
from matplotlib import pyplot as plt

import pandas as pd
import numpy as np

import logomaker

pd.options.display.max_colwidth = 500

def read_meme(filename):
    motifs = {}

    with open(filename, "r") as infile:
        motif, width, i = None, None, 0

        for line in infile:
            if motif is None:
                if line[:5] == "MOTIF":
                    motif = line.split()[1]
                else:
                    continue

            elif width is None:
                if line[:6] == "letter":
                    width = int(line.split()[5])
                    pwm = np.zeros((width, 4))

            elif i < width:
                pwm[i] = list(map(float, line.split()))
                i += 1

            else:
                motifs[motif] = pwm
                motif, width, i = None, None, 0

    return motifs


#def compute_per_position_ic(ppm, background, pseudocount):
#    alphabet_len = len(background)
#    ic = (
#        np.log((ppm + pseudocount) / (1 + pseudocount * alphabet_len)) / np.log(2)
#    ) * ppm - (np.log(background) * background / np.log(2))[None, :]
#    return np.sum(ic, axis=1)


def write_meme_file(ppm, bg, fname):
    f = open(fname, "w")
    f.write("MEME version 4\n\n")
    f.write("ALPHABET= ACGT\n\n")
    f.write("strands: + -\n\n")
    f.write("Background letter frequencies (from unknown source):\n")
    f.write("A %.3f C %.3f G %.3f T %.3f\n\n" % tuple(list(bg)))
    f.write("MOTIF 1 TEMP\n\n")
    f.write(
        "letter-probability matrix: alength= 4 w= %d nsites= 1 E= 0e+0\n" % ppm.shape[0]
    )
    for s in ppm:
        f.write("%.5f %.5f %.5f %.5f\n" % tuple(s))
    f.close()


def fetch_tomtom_matches(
    ppm,
    model_pred,
    motifs_db,
    background=[0.25, 0.25, 0.25, 0.25],
    tomtom_exec_path="tomtom",
    trim_threshold=0.3,
    trim_min_length=3,
):
    _, fname = tempfile.mkstemp()
    _, tomtom_fname = tempfile.mkstemp()

    # Use model predictions (hypothetical contribs)
    chosen_distribution = model_pred

    ic = logomaker.transform_matrix(
        pd.DataFrame(chosen_distribution),
        from_type="probability", to_type="information",
    ).values

    score = np.sum(ic, axis=1)
    trim_thresh = (np.max(score) * trim_threshold)
    pass_inds = np.where(score >= trim_thresh)[0]

    # Check if trimming removed everything
    if len(pass_inds) == 0:
        return []

    trimmed = chosen_distribution[np.min(pass_inds) : np.max(pass_inds) + 1]

    if trimmed is None:
        return []

    # Write the temporary MEME file
    write_meme_file(trimmed, background, fname)

    # Construct the command
    # Note: Added -redirect to ensure we catch stderr if needed,
    # but primarily we check return codes now.
    cmd = (
        "%s -no-ssc -oc . --verbosity 1 -text -min-overlap 5 -mi 1 -dist pearson -thresh 0.05 %s %s"
        % (tomtom_exec_path, fname, motifs_db)
    )

    # Execute with subprocess to capture errors
    try:
        # We run with shell=True to match your original logic,
        # but we capture output to write to file manually to avoid shell redirect issues
        process = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )

        # Write stdout to the file pandas expects
        with open(tomtom_fname, "w") as f:
            f.write(process.stdout)

    except subprocess.CalledProcessError as e:
        print(f"ERROR: Tomtom command failed.")
        print(f"Command run: {cmd}")
        print(f"Error output: {e.stderr}")
        return []

    # Check if file is empty before reading
    if os.stat(tomtom_fname).st_size == 0:
        print(f"WARNING: Tomtom produced empty output for pattern. (Stderr: {process.stderr})")
        return []

    try:
        # Read the file
        tomtom_results = pandas.read_csv(tomtom_fname, sep="\t", usecols=(1, 5))
    except Exception as e:
        print(f"Pandas read error on file content: {open(tomtom_fname).read()}")
        raise e
    finally:
        # Cleanup
        if os.path.exists(tomtom_fname):
            os.remove(tomtom_fname)
        if os.path.exists(fname):
            os.remove(fname)

    return tomtom_results


def run_tomtom(
    modisco_h5py,
    output_prefix,
    meme_motif_db,
    top_n_matches=3,
    tomtom_exec="tomtom",
    trim_threshold=0.3,
    trim_min_length=3,
    background=[0.25, 0.25, 0.25, 0.25],
):
    modisco_results = h5py.File(modisco_h5py, "r")

    tomtom_results = {"pattern": [], "num_seqlets": []}
    for i in range(top_n_matches):
        tomtom_results["match{}".format(i)] = []
        tomtom_results["qval{}".format(i)] = []

    for name in ["pos_patterns", "neg_patterns"]:
        if name not in modisco_results.keys():
            continue

        metacluster = modisco_results[name]
        key = lambda x: int(x[0].split("_")[-1])
        for pattern_name, pattern in sorted(metacluster.items(), key=key):
            ppm = np.array(pattern["sequence"][:])

            model_pred = pattern["hypothetical_contribs"][:] + 0.25

            num_seqlets = pattern["seqlets"]["n_seqlets"][:][0]
            tag = "{}.{}".format(name, pattern_name)

            r = fetch_tomtom_matches(
                ppm,
                model_pred,
                motifs_db=meme_motif_db,
                tomtom_exec_path=tomtom_exec,
                trim_threshold=trim_threshold,
                trim_min_length=trim_min_length,
                background=background,
            )

            tomtom_results["pattern"].append(tag)
            tomtom_results["num_seqlets"].append(num_seqlets)

            matches_found = 0

            # --- FIX: Check if 'r' is a DataFrame and not empty ---
            if isinstance(r, pandas.DataFrame) and not r.empty:
                # Take top N
                r_subset = r.iloc[:top_n_matches]

                # Iterate safely
                for _, row in r_subset.iterrows():
                    # Ensure we have data to unpack
                    if len(row) >= 2:
                        target = row.iloc[0] # First column (Target ID)
                        qval = row.iloc[1]   # Second column (q-value)

                        tomtom_results["match{}".format(matches_found)].append(target)
                        tomtom_results["qval{}".format(matches_found)].append(qval)
                        matches_found += 1

            # Fill remaining slots (or all slots if r was empty) with None
            for j in range(matches_found, top_n_matches):
                tomtom_results["match{}".format(j)].append(None)
                tomtom_results["qval{}".format(j)].append(None)

    modisco_results.close()
    return pandas.DataFrame(tomtom_results)


def path_to_image_html(path):
    return '<img src="' + path + '" width="240" >' if path != "" else ""


def _plot_weights(array, path, figsize=(10, 3), **kwargs):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    df = pandas.DataFrame(array, columns=["A", "C", "G", "T"])
    df.index.name = "pos"

    crp_logo = logomaker.Logo(df, ax=ax)
    crp_logo.style_spines(visible=False)
    plt.ylim(min(df.sum(axis=1).min(), 0), df.sum(axis=1).max())

    plt.savefig(path)
    plt.close()


def make_logo(match, logo_dir, motifs):
    if match == "NA":
        return

    #background = np.array([0.25, 0.25, 0.25, 0.25])
    ppm = motifs[match]
    #ic = compute_per_position_ic(ppm, background, 0.001)
    ic = logomaker.transform_matrix(
        pd.DataFrame(ppm),
        from_type="probability", to_type="information",
    ).values
    #_plot_weights(ppm * ic[:, None], path="{}/{}.png".format(logo_dir, match))
    _plot_weights(ic, path="{}/{}.png".format(logo_dir, match))


def create_modisco_logos(modisco_file, modisco_logo_dir, trim_threshold):
    results = h5py.File(modisco_file, "r")
    tags = []

    for name in ["pos_patterns", "neg_patterns"]:
        if name not in results.keys():
            continue

        metacluster = results[name]
        key = lambda x: int(x[0].split("_")[-1])
        for pattern_name, pattern in sorted(metacluster.items(), key=key):
            tag = "{}.{}".format(name, pattern_name)
            tags.append(tag)

            # print(pattern.keys())

            # print(pattern['contrib_scores'][:].mean(axis=1))
            # print("*")
            # print(pattern['hypothetical_contribs'][:].mean(axis=1))

            # for key in pattern.keys():
            # 	print(key)
            # 	print(pattern[key].shape)
            # raise Exception("debug")

            # cwm_fwd = np.array(pattern['contrib_scores'][:])

            ppm = pattern["sequence"][:]  # empirical frequencies
            model_pred = pattern["hypothetical_contribs"][:] + 0.25  # model probs

            # chosen_distribution = ppm
            chosen_distribution = model_pred

            cwm_fwd = logomaker.transform_matrix(
                pd.DataFrame(chosen_distribution),
                from_type="probability",
                to_type="information",
            ).values
            cwm_rev = cwm_fwd[::-1, ::-1]

            score_fwd = np.sum(np.abs(cwm_fwd), axis=1)
            score_rev = np.sum(np.abs(cwm_rev), axis=1)

            trim_thresh_fwd = np.max(score_fwd) * trim_threshold
            trim_thresh_rev = np.max(score_rev) * trim_threshold

            pass_inds_fwd = np.where(score_fwd >= trim_thresh_fwd)[0]
            pass_inds_rev = np.where(score_rev >= trim_thresh_rev)[0]

            start_fwd, end_fwd = max(np.min(pass_inds_fwd) - 4, 0), min(
                np.max(pass_inds_fwd) + 4 + 1, len(score_fwd) + 1
            )
            start_rev, end_rev = max(np.min(pass_inds_rev) - 4, 0), min(
                np.max(pass_inds_rev) + 4 + 1, len(score_rev) + 1
            )

            trimmed_cwm_fwd = cwm_fwd[start_fwd:end_fwd]
            trimmed_cwm_rev = cwm_rev[start_rev:end_rev]

            _plot_weights(
                trimmed_cwm_fwd, path="{}/{}.cwm.fwd.png".format(modisco_logo_dir, tag)
            )
            _plot_weights(
                trimmed_cwm_rev, path="{}/{}.cwm.rev.png".format(modisco_logo_dir, tag)
            )

    return tags


def report_motifs(
    modisco_h5py,
    output_dir,
    meme_motif_db,
    suffix="./",
    top_n_matches=3,
    trim_threshold=0.3,
    trim_min_length=3,
    background=[0.25, 0.25, 0.25, 0.25],
):

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    if not os.path.isdir(output_dir + "/trimmed_logos/"):
        os.mkdir(output_dir + "/trimmed_logos/")
    modisco_logo_dir = output_dir + "/trimmed_logos/"

    motifs = read_meme(meme_motif_db)
    names = create_modisco_logos(modisco_h5py, modisco_logo_dir, trim_threshold)

    tomtom_df = run_tomtom(
        modisco_h5py,
        output_dir,
        meme_motif_db,
        top_n_matches=top_n_matches,
        tomtom_exec="tomtom",
        trim_threshold=trim_threshold,
        trim_min_length=trim_min_length,
        background=background,
    )

    tomtom_df["modisco_cwm_fwd"] = [
        "{}trimmed_logos/{}.cwm.fwd.png".format(suffix, name) for name in names
    ]
    tomtom_df["modisco_cwm_rev"] = [
        "{}trimmed_logos/{}.cwm.rev.png".format(suffix, name) for name in names
    ]

    reordered_columns = ["pattern", "num_seqlets", "modisco_cwm_fwd", "modisco_cwm_rev"]
    for i in range(top_n_matches):
        name = "match{}".format(i)
        logos = []

        for index, row in tomtom_df.iterrows():
            if name in tomtom_df.columns:
                if pandas.isnull(row[name]):
                    logos.append("")
                else:
                    make_logo(row[name], output_dir, motifs)
                    logos.append("{}{}.png".format(suffix, row[name]))
            else:
                break

        tomtom_df["{}_logo".format(name)] = logos
        reordered_columns.extend([name, "qval{}".format(i), "{}_logo".format(name)])

    tomtom_df = tomtom_df[reordered_columns]
    print(tomtom_df)
    tomtom_df = tomtom_df.fillna("")
    print(tomtom_df)
    tomtom_df.to_html(
        open("{}/motifs.html".format(output_dir), "w"),
        escape=False,
        formatters=dict(
            modisco_cwm_fwd=path_to_image_html,
            modisco_cwm_rev=path_to_image_html,
            match0_logo=path_to_image_html,
            match1_logo=path_to_image_html,
            match2_logo=path_to_image_html,
        ),
        index=False,
    )

report_motifs(
    '../../../results/review/9_ap2_tf_family/genome_wide_gpn_modisco_results.h5',
    "../../../results/review/9_ap2_tf_family/genome_wide_gpn_modisco_results_plantTFDB/",
    suffix="",
    top_n_matches=1,
    meme_motif_db='Ath_TF_binding_motifs.meme',
    background=[0.3310, 0.1690, 0.1689, 0.3311]
)
