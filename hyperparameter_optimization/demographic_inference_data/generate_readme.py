import os
import sys
import importlib.util
import re


REPO = "https://bitbucket.org/noscode/demographic_inference_data/src/master/"

INITIAL_README = """# Data for demographic inference

This repo contains AFS data (simulated and real) for different kinds of
demographic inference, for example, it could be used for some algorithm
benchmark and so on.

Right now everything is for the python package
[moments](https://bitbucket.org/simongravel/moments/src/master).
"""

INFO = """
## Structure of data folders

### Name of folder

All data is located in its folder each of which has special name:

`{N}_{Descr}_{M}_{Origin}`

where `N` is the **number of populations**, `Descr` stands for **simple
description** of the demographic model, `M` is the **number of parameters**
in model and `Origin` is three letters of data origin. For example:

`2_DivMig_5_Sim` means AFS data for 2 populations that was simulated (`Sim`)
under the demographic history with divergence and migration that has in total
5 parameters.

For real data `Origin` contains the first three letters of the first author of the
corresponding paper. For example:

`2_YRI_CEU_6_Gut` means AFS data and demographic model for YRI and CEU
populations from the **Gut**enkunst et al., 2009 paper.

### Folder structure

Each directory contains several files:

* demographic_model.py - file with demographic model code. Code is written
for [moments](https://bitbucket.org/simongravel/moments/src/master).

* `main_script.py` - script with all information about AFS data. If data is
simulated then during script run it will be saved to `fs_data.fs`.

* `fs_data.fs` - AFS data in fs dadi format.

* `model_plot.png` - schematic plot of the demographic model.

* `fs_plot.png` or `fs_plot_projections.png` - plots of AFS.

File `main.py` is a valid file with python code and contains following
information about data:

| Variable | Description |
| --- | --- |
| `n_pop` | Number of populations |
| `par_labels` | Labels of model parameters |
| `max_ll` | Maximum log composite likelihood |
| `popt` | Optimal values parameters |
| `Nanc` | Size of ancestral population |
| `lower_bound` | Lower bound for parameters |
| `upper_bound` | Upper bound for parameters |
| `ns` | Sample sizes of AFS |

### Units

Note that all parameters values are in genetic units (relative to size
`Nanc` of ancestral population). Size of populations are relative to `Nanc`,
time is in `2 * Nanc` generations abd migrations are in `1 / (2 * Nanc)`
units. For more information see [moments manual](https://bitbucket.org/
simongravel/moments/src/master/doc/manual/manual.pdf) section `5.2 Units`.

"""

def load_module(dirname, filename):
    save_dir = os.path.abspath(".")
    os.chdir(os.path.abspath(dirname))
    sys.path.append(".")#dirname)
    module = importlib.import_module(
        os.path.join(dirname, filename).replace('/', '.').rstrip('.py'))
    if "demographic_model" in sys.modules:
        del sys.modules["demographic_model"]
    sys.path = sys.path[:-1]
    os.chdir(save_dir)
    return module

def generate_model_info(dirname, working_dir=None):
    if working_dir is None:
        working_dir = dirname
    s = f"## {dirname}\n\n"

    dem_model_file = 'demographic_model.py'
    sim_file = 'main_script.py'

    model = load_module(dirname, dem_model_file)
    model_description = load_module(dirname,
                                    dem_model_file).model_func.__doc__
    sim_info = load_module(dirname, sim_file)

    # Fast info
    s += "\n| Number of populations | Number of parameters "\
         "| Max log likelihood | Size of spectrum |\n"\
         "| --- | --- | --- | --- |\n"
    sp_size = "x".join([str(x) for x in sim_info.ns])
    s += f"| {sim_info.n_pop} | {len(sim_info.par_labels)} | "\
         f"{sim_info.max_ll:.3f} | {sp_size} |\n\n"

    # Description
    s += "\n### Model Description\n\n"
    par_labels = sim_info.par_labels
    values = {par_name: value for par_name, value in zip(par_labels,
                                                         sim_info.popt)}
    split_ind = model_description.rfind("\n\n")
    descr = re.sub("(&!\n)\n(&!\n)", "", model_description[:split_ind])
    # TODO regex for proper text creation
    descr = " ".join([x.strip() for x in re.split("\n ", descr)]).strip()
    s += descr + "\n"

    # Plots
    s += "\n### Plots\n\n"
    s += "Schematic model plot:\n\n"
    s += f'<img src="{working_dir}/model_plot.png" height="500" />\n\n'

    if sim_info.n_pop <= 3:
        s += "Simulated allele frequency spectrum:\n\n"
        s += f'<img src="{working_dir}/fs_plot.png" height="200" />\n\n'
    if sim_info.n_pop >= 3:
        s += "Simulated allele frequency spectrum (projections):\n\n"
        s += f'<img src="{working_dir}/fs_plot_projections.png" '
        if not sim_info.n_pop in [4, 5]:
            s += 'height="200" />\n\n'
        else:
            s += '/>\n\n'

    # Parameters
    s += "\n### Optimal parameter values\n\n"
    s += "| Parameter | Value | Description |\n"
    s += "| --- | --- | --- |\n"
    for line in model_description[split_ind+1:].split(":param "):
        if line.strip() == "":
            continue
        spl_ind = line.find(":")
        par_name = line[:spl_ind]
        par_descr = line[spl_ind + 1:].strip()
        par_descr = " ".join([x.strip() for x in par_descr.split("\n ")])
        s += f"| `{par_name}` | {values[par_name]:.3f} | {par_descr} |\n"
    s += "\n"
    return s

def valid_dirname(dirname):
    return (os.path.isdir(dirname) and
           not dirname.startswith("_") and
           not dirname.startswith("."))


dirnames = []
for dirname in os.listdir():
    if valid_dirname(dirname):
        dirnames.append(dirname)
dirnames = sorted(dirnames, key=lambda x: int(x.split("_")[-2]))
dirnames = sorted(dirnames, key=lambda x: int(x.split("_")[0]))
sim_dirs = []
real_dirs = []
for dirname in dirnames:
    if dirname.endswith("Sim"):
        sim_dirs.append(dirname)
    else:
        real_dirs.append(dirname)

def generate_toc():
    s = "# Available AFS data\n\n"
    if len(sim_dirs) > 0:
        s += "- [Simulated data](#simulated-data)\n"
        for dirname in sim_dirs:
            s += f"\t* [{dirname}](#{dirname})\n"
    if len(real_dirs) > 0:
        s += "- [Real data](#real-data)\n"
        for dirname in real_dirs:
            s += f"\t* [{dirname}](#{dirname})\n"
    s += "\n"
    return s

with open("README.md", "w") as f:
    f.write(INITIAL_README)
    f.write(generate_toc())
    f.write(INFO)
    f.write("# Simulated data\n\n")
    for data_dir in sim_dirs:
        info = generate_model_info(data_dir)
        f.write(info)
        with open(os.path.join(data_dir, "README.md"), "w") as loc_f:
            loc_f.write(info.replace(f"{data_dir}/", ""))
    f.write("# Real data\n\n")
    for data_dir in real_dirs:
        if data_dir.endswith("Sim"):
            continue
        info = generate_model_info(data_dir)
        f.write(info)
        with open(os.path.join(data_dir, "README.md"), "w") as loc_f:
            loc_f.write(info.replace(f"{data_dir}/", ""))
