import os
import pickle
import h5py
import pandas
import tempfile
import matplotlib
import pandas as pd
import numpy as np
import logomaker
from matplotlib import pyplot as plt

matplotlib.use("pdf")

pd.options.display.max_colwidth = 500

# ==========================================
# 1. utils
# ==========================================

def read_meme(filename):
    """read meme motifs"""
    motifs = {}
    with open(filename, "r") as infile:
        motif, width, i = None, None, 0
        pwm = None
        
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

def write_meme_file(ppm, bg, fname):
    """将 PPM 写入临时 MEME 文件供 TomTom 使用"""
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

def _plot_weights(array, path, figsize=(10, 3), **kwargs):
    """使用 Logomaker 绘图"""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    df = pandas.DataFrame(array, columns=["A", "C", "G", "T"])
    df.index.name = "pos"

    crp_logo = logomaker.Logo(df, ax=ax)
    crp_logo.style_spines(visible=False)
    
    min_val = min(df.sum(axis=1).min(), 0)
    max_val = df.sum(axis=1).max()
    if max_val == 0: max_val = 1 
    plt.ylim(min_val, max_val)

    plt.savefig(path)
    plt.close()

def make_logo(match, logo_dir, motifs):
    """为匹配到的数据库 Motif 生成 Logo 图片"""
    if match == "NA" or match not in motifs:
        return

    ppm = motifs[match]
    ic = logomaker.transform_matrix(
        pd.DataFrame(ppm, columns=["A", "C", "G", "T"]),
        from_type="probability", to_type="information",
    ).values
    
    _plot_weights(ic, path="{}/{}.png".format(logo_dir, match))

def path_to_image_html(path):
    return '<img src="' + path + '" width="240" >' if path != "" else ""

# ==========================================
# 2. MoDisco and TomTom
# ==========================================

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

    chosen_distribution = ppm

    ic = logomaker.transform_matrix(
        pd.DataFrame(chosen_distribution, columns=["A", "C", "G", "T"]),
        from_type="probability", to_type="information",
    ).values

    score = np.sum(ic, axis=1)
    trim_thresh = np.max(score) * trim_threshold
    pass_inds = np.where(score >= trim_thresh)[0]
    
    if len(pass_inds) == 0:
        return pd.DataFrame() # 空结果

    trimmed = chosen_distribution[np.min(pass_inds) : np.max(pass_inds) + 1]

    write_meme_file(trimmed, background, fname)

    # 运行 TomTom
    cmd = (
        "%s -no-ssc -oc . --verbosity 1 -text -min-overlap 5 -mi 1 -dist pearson -thresh 0.05 %s %s > %s"
        % (tomtom_exec_path, fname, motifs_db, tomtom_fname)
    )

    os.system(cmd)
    
    # 检查结果文件是否为空
    if os.stat(tomtom_fname).st_size == 0:
        return pd.DataFrame()
        
    tomtom_results = pandas.read_csv(tomtom_fname, sep="\t", usecols=(1, 5))
    
    if os.path.exists(tomtom_fname): os.remove(tomtom_fname)
    if os.path.exists(fname): os.remove(fname)
    
    return tomtom_results

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

            ppm = np.array(pattern["sequence"][:]) 

            cwm_fwd = logomaker.transform_matrix(
                pd.DataFrame(ppm, columns=["A", "C", "G", "T"]),
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

            if len(pass_inds_fwd) > 0:
                start_fwd, end_fwd = max(np.min(pass_inds_fwd) - 4, 0), min(
                    np.max(pass_inds_fwd) + 5, len(score_fwd)
                )
                trimmed_cwm_fwd = cwm_fwd[start_fwd:end_fwd]
            else:
                trimmed_cwm_fwd = cwm_fwd

            if len(pass_inds_rev) > 0:
                start_rev, end_rev = max(np.min(pass_inds_rev) - 4, 0), min(
                    np.max(pass_inds_rev) + 5, len(score_rev)
                )
                trimmed_cwm_rev = cwm_rev[start_rev:end_rev]
            else:
                trimmed_cwm_rev = cwm_rev

            _plot_weights(
                trimmed_cwm_fwd, path="{}/{}.cwm.fwd.png".format(modisco_logo_dir, tag)
            )
            _plot_weights(
                trimmed_cwm_rev, path="{}/{}.cwm.rev.png".format(modisco_logo_dir, tag)
            )
    results.close()
    return tags

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
            model_pred = ppm 
            
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
            
            match_found = False
            if not r.empty:
                for i, (target, qval) in r.iloc[:top_n_matches].iterrows():
                    tomtom_results["match{}".format(i)].append(target)
                    tomtom_results["qval{}".format(i)].append(qval)
                
                for j in range(len(r), top_n_matches):
                    tomtom_results["match{}".format(j)].append(None)
                    tomtom_results["qval{}".format(j)].append(None)
            else:
                for j in range(top_n_matches):
                    tomtom_results["match{}".format(j)].append(None)
                    tomtom_results["qval{}".format(j)].append(None)

    modisco_results.close()
    return pandas.DataFrame(tomtom_results)

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
                val = row[name]
                if pandas.isnull(val) or val == "":
                    logos.append("")
                else:
                    make_logo(val, output_dir, motifs)
                    logos.append("{}{}.png".format(suffix, val))
            else:
                break

        tomtom_df["{}_logo".format(name)] = logos
        reordered_columns.extend([name, "qval{}".format(i), "{}_logo".format(name)])

    tomtom_df = tomtom_df[reordered_columns]
    tomtom_df = tomtom_df.fillna("")
    
    html_path = "{}/motifs.html".format(output_dir)
    print(f"Report saved to: {html_path}")
    
    tomtom_df.to_html(
        open(html_path, "w"),
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
    return tomtom_df

report_motifs(
    './evo2-context-2k_modisco_results.h5',
    "evo2-context-2k_modisco_results_JASPAR/",
    suffix="",
    top_n_matches=1,
    meme_motif_db='/workdir/jz963/modisco_test/maize/JASPAR2024_CORE_plants_non-redundant_pfms_meme.txt',
    background=[0.2930715, 0.2036401, 0.2075366, 0.2957519], 
)
