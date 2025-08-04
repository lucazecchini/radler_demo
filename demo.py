import csv
import gc
import math
import os
import pandas as pd
import pickle as pkl
import plotly.express as px
import streamlit as st
import config
import radler
from utils import compute_distribution, detect_distribution, scroll_script


class Task:
    """
    The object describing the task at hand
    """

    def __init__(self, task_id, run_id, dataset, algorithm, mode, sample_attributes, groups, distribution_type, target_distribution,
                 sample_size=math.inf, blocker=None, matcher=None):

        self.task_id = task_id
        self.run_id = run_id

        # Dataset
        self.ds = dataset
        self.path_ds = config.er_features[self.ds]["path_ds"]

        # Blocking function (i.e., blocker)
        self.blocker = config.er_features[self.ds]["default_blocker"] if blocker is None else blocker
        self.path_candidates = config.er_features[self.ds]["blockers"][self.blocker]["path_candidates"]
        self.neighbors = None

        # Matching function (i.e., matcher)
        self.matcher = config.er_features[self.ds]["default_matcher"] if matcher is None else matcher
        self.path_gold = config.er_features[self.ds]["matchers"][self.matcher]["path_gold"]
        self.matches = None

        # Aggregation functions (i.e., aggregations)
        self.default_aggregation = config.er_features[self.ds]["default_aggregation"]
        self.aggregations = config.er_features[self.ds]["default_fusion"]

        # Attribute projection
        self.attributes = list(self.aggregations.keys())

        # Sampling
        self.algorithm = algorithm
        self.mode = mode
        self.sample_attributes = sample_attributes
        self.num_attributes = len(self.sample_attributes)
        self.groups = groups
        self.num_groups = len(self.groups)
        self.distribution_type = distribution_type
        self.target_distribution = compute_distribution(target_distribution)
        self.sample_size = sample_size
        self.stochastic_acceptance_size = config.stochastic_acceptance_size
        self.stochastic_acceptance_timeout = config.stochastic_acceptance_timeout

    def set_neighbors(self, neighbors):
        self.neighbors = neighbors

    def set_matches(self, matches):
        self.matches = matches


def tuple_to_string(t):
    s = str(t)[1:-1].replace("', ", ",\u00A0").replace("'", "").rstrip(",")
    return(s)


def string_to_tuple(s):
    t = tuple([x for x in s.split(",\u00A0")])
    return t


def load_matches(task, ds, id_mapping):

    reader = csv.reader(open(task.path_gold, mode="r"))
    next(reader)  # skip the header row

    task.matches = [set() for _ in range(len(ds["_id"]))]

    for row in reader:
        l_id, r_id = row
        (l_id, r_id) = (id_mapping[l_id], id_mapping[r_id])

        task.matches[l_id].add(r_id)
        task.matches[r_id].add(l_id)


def load_neighbors(task, ds, id_mapping):

    reader = csv.reader(open(task.path_candidates, mode="r"))
    next(reader)  # skip the header row

    task.neighbors = [{i} for i in range(len(ds["_id"]))]

    for row in reader:
        l_id, r_id = row
        (l_id, r_id) = (id_mapping[l_id], id_mapping[r_id])

        task.neighbors[l_id].add(r_id)
        task.neighbors[r_id].add(l_id)


def run_radler(task, dirty_data, placeholder):
    radler.run(task, dirty_data, verbose=False, demo=True, res_demo=placeholder)


def run_process():
    st.session_state.run_process = True
    st.session_state.stop_process = False
    # st.session_state.df = None
    st.session_state.selection = None


def stop_process():
    st.session_state.resume = True
    st.session_state.stop_process = True
    st.session_state.run_process = False
    st.session_state.update_metrics = True


def clear_result():
    st.session_state.df = None
    st.session_state.selection = None
    st.session_state.num_comparisons = 0
    st.session_state.resume = False


if "ds_name" not in st.session_state:
    st.session_state.ds_name = None

if "blocker" not in st.session_state:
    st.session_state.blocker = None

if "matcher" not in st.session_state:
    st.session_state.matcher = None

if "run_process" not in st.session_state:
    st.session_state.run_process = False

if "stop_process" not in st.session_state:
    st.session_state.stop_process = False

if "resume" not in st.session_state:
    st.session_state.resume = False

if "df" not in st.session_state:
    st.session_state.df = None

if "records" not in st.session_state:
    st.session_state.records = dict()

if "selection" not in st.session_state:
    st.session_state.selection = None

if "show_groups" not in st.session_state:
    st.session_state.show_groups = False

if "show_distro" not in st.session_state:
    st.session_state.show_distro = False

if "num_comparisons" not in st.session_state:
    st.session_state.num_comparisons = 0

if "update_metrics" not in st.session_state:
    st.session_state.update_metrics = False


st.title("🍋 RadlER")

# Select a dataset
ds_name = st.selectbox("Select a dataset", config.datasets)
if ds_name != st.session_state.ds_name:
    st.session_state.ds_name = ds_name
    st.session_state.blocker = None
    st.session_state.matcher = None
    st.session_state.df = None
    st.session_state.selection = None
    st.session_state.resume = False
    st.rerun()

# Visualize the dirty dataset as a dataframe
dirty_data = pd.read_csv(f"datasets/{ds_name}/dataset.csv")
id_mapping = dict(zip(dirty_data["_id"], range(len(dirty_data))))
dirty_data["_id"] = range(len(dirty_data))
dirty_data.columns = dirty_data.columns.str.lower()
for column in dirty_data.columns:
    if dirty_data[column].dtype == "object":
        dirty_data[column] = dirty_data[column].str.lower()
attributes = config.er_features[ds_name]["attributes"][1:]
st.dataframe(dirty_data[attributes], height=150)

if st.session_state.blocker is not None and st.session_state.matcher is not None:
    path_candidates = config.er_features[ds_name]["blockers"][st.session_state.blocker]["path_candidates"]
    num_comparison = len(pd.read_csv(path_candidates))
    est1, est2, est3 = st.columns(3)
    with est1:
        st.markdown("<div style='text-align: center; color: red'>📈 &nbsp;&nbsp; {} comparisons</div>"
                    .format("{:,}".format(num_comparison).replace(",", " ")), unsafe_allow_html=True)
    with est2:
        tpc = config.er_features[ds_name]["matchers"][st.session_state.matcher]["time_per_comparison"]
        st.markdown("<div style='text-align: center; color: red'>⏳ &nbsp;&nbsp; {} hours</div>"
                    .format(round((num_comparison * tpc) / 3600, 2)), unsafe_allow_html=True)
    with est3:
        cpc = config.er_features[ds_name]["matchers"][st.session_state.matcher]["cost_per_comparison"]
        st.markdown("<div style='text-align: center; color: red'>💸 &nbsp;&nbsp; {} USD</div>"
                    .format(round(num_comparison * cpc, 3)), unsafe_allow_html=True)

# Produce the clean dataset
clean_data = pd.read_csv(f"clean_datasets/{ds_name}.csv")
clean_data.columns = clean_data.columns.str.lower()
for column in clean_data.columns:
    if clean_data[column].dtype == "object":
        clean_data[column] = clean_data[column].str.lower()

# Visualize the RadlER functionalities
st.markdown("---")
# st.write("<p style='font-size: 20px;'>Get a clean sample with RadlER</p>", unsafe_allow_html=True)

# Organize RadlER functionalities within a sidebar
with st.sidebar:

    st.markdown("""
        <style>
        [data-testid=stVerticalBlock] {
            gap: 0.5rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # Select the sampling attributes
    st.write("<p style='font-size: 14px;'>Select sampling attributes</p>", unsafe_allow_html=True)
    string_attributes = config.string_attributes[ds_name]
    sample_attributes = st.multiselect("Select sampling attributes", [c for c in string_attributes if c != "_id"],
                                       default=config.sample_attributes[ds_name], label_visibility="collapsed")

    # Select the groups
    with st.container():
        st.write("<p style='font-size: 14px;'>Select groups</p>", unsafe_allow_html=True)
        group_select_modes = ["Most represented groups", "Groups of interest"]
        group_select_mode = st.selectbox("Select groups", group_select_modes, index=0, label_visibility="collapsed")

        # Select the k most represented groups (and support their visualization)
        if group_select_mode == "Most represented groups":
            sel_group_1, sel_group_2 = st.columns([8, 2])
            with sel_group_1:
                num_groups = st.number_input("Select the number of groups", min_value=1, max_value=10, step=1, value=2,
                                            label_visibility="collapsed")
                groups, clean_distribution, max_sample_size = detect_distribution(clean_data, sample_attributes,
                                                                                  distribution_type="demographic_parity",
                                                                                  value_filter=config.value_filter[ds_name],
                                                                                  min_support=0, top_groups=num_groups)
            with sel_group_2:
                if st.button("📋"):
                    st.session_state.show_groups = not st.session_state.show_groups

            if st.session_state.show_groups:
                group_info = [{sample_attributes[j]: groups[i][j] for j in range(len(sample_attributes))} for i in range(num_groups)]
                st.dataframe(pd.DataFrame(group_info))

        # Select some groups of interest
        else:
            st.session_state.show_groups = False
            top_groups, clean_distribution, max_sample_size = detect_distribution(clean_data, sample_attributes,
                                                                                  distribution_type="demographic_parity",
                                                                                  value_filter=config.value_filter[ds_name],
                                                                                  min_support=0)
            groups = st.multiselect("Select the groups", [tuple_to_string(g) for g in top_groups], label_visibility="collapsed")
            groups = [string_to_tuple(g) for g in groups]
            if len(groups) > 0:
                group_indices = [top_groups.index(g) for g in groups]
                clean_distribution = compute_distribution([clean_distribution[i] for i in group_indices])
            num_groups = len(groups)

        if num_groups == 0:
            st.session_state.show_distro = False

    # Select the target distribution
    with st.container():
        st.write("<p style='font-size: 14px;'>Select the target distribution</p>", unsafe_allow_html=True)
        sel_distro_1, sel_distro_2 = st.columns([8, 2])
        with sel_distro_1:
            distribution_types = ["Equal representation", "Demographic parity", "Custom distribution"]
            distribution_type = st.selectbox("Select the target distribution", distribution_types, index=0,
                                             label_visibility="collapsed")
        with sel_distro_2:
            if st.button("📊", disabled=(num_groups == 0)):
                st.session_state.show_distro = not st.session_state.show_distro

        if distribution_type == "Custom distribution":
            cols = [st.columns([2, 7], gap="small") for _ in range(num_groups)]
            custom_distribution = list()
            group_labels = [tuple_to_string(g) for g in groups]
            for item, (col1, col2) in zip(group_labels, cols):
                with col1:
                    val = st.number_input(label=item, min_value=1, value=1, step=1, format="%d", key=f"input_{item}",
                                          label_visibility="collapsed")
                with col2:
                    st.write(f"<p style='font-size: 14px;'>{item}</p>", unsafe_allow_html=True)
                custom_distribution.append(val)
            custom_distribution = compute_distribution([x if x >= 1 else 1 for x in custom_distribution])

        if distribution_type == "Equal representation":
            target_distribution = [1.0 / num_groups for _ in range(num_groups)]
        elif distribution_type == "Demographic parity":
            target_distribution = clean_distribution
        elif distribution_type == "Custom distribution":
            target_distribution = custom_distribution
        distribution_details = f"{distribution_type}: {" - ".join([str(round(x, 3)) for x in target_distribution])}" \
                               if distribution_type == "Custom distribution" else distribution_type

        if st.session_state.show_distro:
            fig = px.bar(pd.DataFrame({"Group": [", ".join([v for v in g]) for g in groups],
                                       "Value": [round(x, 3) for x in target_distribution]}), x="Group", y="Value")
            st.plotly_chart(fig)

    # Select the sample size
    st.write("<p style='font-size: 14px;'>Select the sample size</p>", unsafe_allow_html=True)
    sample_size = st.number_input("Select the sample size", min_value=1, step=1, value=10, label_visibility="collapsed")

    # Select the blocking and matching function
    with st.container():
        st.write("<p style='font-size: 14px;'>Select blocking and matching functions</p>", unsafe_allow_html=True)
        blocker = st.selectbox("Select the blocking function", list(config.er_features[ds_name]["blockers"].keys()),
                               index=0, label_visibility="collapsed")
        if blocker != st.session_state.blocker:
            st.session_state.blocker = blocker
            st.rerun()
        matcher = st.selectbox("Select the matching function",
                               list(config.er_features[ds_name]["matchers"].keys()), index=0, label_visibility="collapsed")
        if matcher != st.session_state.matcher:
            st.session_state.matcher = matcher
            st.rerun()

    # Activate or deactivate the weighting scheme
    weighting_scheme = st.checkbox("Weighting scheme", value=True)

    st.markdown("""
        <style>
        .invisible-space {
            height: 10px;
        }
        </style>
    """, unsafe_allow_html=True)
    st.markdown('<div class="invisible-space"></div>', unsafe_allow_html=True)

    # Buttons to manage the deduplication process
    col1, col2, col3 = st.columns(3)
    col1.button("Run", on_click=run_process, disabled=(st.session_state.run_process or num_groups == 0))
    col2.button("Stop", on_click=stop_process, disabled=(not st.session_state.run_process))
    col3.button("Clear", on_click=clear_result, disabled=(st.session_state.run_process or st.session_state.df is None))

entities = st.empty() if st.session_state.df is None \
           else (st.dataframe(st.session_state.df) if not st.session_state.stop_process \
                 else st.dataframe(st.session_state.df, on_select="rerun", selection_mode="single-row"))

# Run RadlER to produce the clean sample progressively
if st.session_state.run_process:
    records = st.empty()
    st.session_state.records = None
    st.markdown(scroll_script, unsafe_allow_html=True)
    if st.session_state.resume:
        task = pkl.load(open("checkpoints/status.pkl", "rb"))["task"]
        task.sample_size = sample_size
        check_value = {
            "ds": ds_name,
            "mode": "complete" if weighting_scheme else "unweighed",
            "sample_attributes": sample_attributes,
            "groups": groups,
            "distribution_type": distribution_details,
            "blocker": blocker,
            "matcher": matcher
        }
        print([getattr(task, k) for k, v in check_value.items()])
        print([v for k, v in check_value.items()])
        if any(getattr(task, k) != v for k, v in check_value.items()):
            st.session_state.resume = False
    if not st.session_state.resume:
        st.session_state.df = None
        task = Task(0, 0, ds_name, "radler", "complete" if weighting_scheme else "unweighed", sample_attributes, groups,
                    distribution_details, target_distribution, sample_size=sample_size, blocker=blocker, matcher=matcher)
        os.makedirs("checkpoints", exist_ok=True)
        load_neighbors(task, dirty_data, id_mapping)
        load_matches(task, dirty_data, id_mapping)
    run_radler(task, dirty_data, entities)
    st.markdown(scroll_script, unsafe_allow_html=True)
    st.session_state.update_metrics = True
    st.session_state.resume = True
    st.session_state.stop_process = True
    st.session_state.run_process = False
    st.rerun()

if st.session_state.update_metrics:
    if os.path.isfile("checkpoints/status.pkl"):
        status = pkl.load(open("checkpoints/status.pkl", "rb"))
        st.session_state.num_comparisons = status["num_comparisons"]
        st.session_state.records = status["records"]
        if st.session_state.df is None and len(status["entities"]) > 0:
            st.session_state.df = pd.DataFrame(status["entities"])
        if st.session_state.df is not None:
            if len(st.session_state.df) < len(status["entities"]):
                new_row = pd.DataFrame(status["entities"][-1:])
                st.session_state.df = pd.concat([st.session_state.df, new_row], ignore_index=True)
        del status
        gc.collect()
    st.session_state.update_metrics = False
    st.rerun()

if st.session_state.df is not None:
    st.session_state.selection = entities.selection.rows[0] if len(entities.selection.rows) > 0 else None

if st.session_state.df is not None and not st.session_state.run_process:
    exp1, exp2, exp3 = st.columns(3)
    with exp1:
        st.markdown("<div style='text-align: center; color: green'>📈 &nbsp;&nbsp; {} comparisons</div>"
                    .format("{:,}".format(st.session_state.num_comparisons).replace(",", " ")), unsafe_allow_html=True)
    with exp2:
        tpc = config.er_features[ds_name]["matchers"][st.session_state.matcher]["time_per_comparison"]
        time = st.session_state.num_comparisons * tpc
        st.markdown("<div style='text-align: center; color: green'>⏳ &nbsp;&nbsp; %s %s</div>" %
                    (round(time / (60 if time >= 60 else 1), 2),
                     "minutes" if time >= 60 else "seconds"), unsafe_allow_html=True)
    with exp3:
        cpc = config.er_features[ds_name]["matchers"][st.session_state.matcher]["cost_per_comparison"]
        st.markdown("<div style='text-align: center; color: green'>💸 &nbsp;&nbsp; {} USD</div>"
                    .format(round(st.session_state.num_comparisons * cpc, 3)), unsafe_allow_html=True)
    st.markdown('<div style="height: 30px;"></div>', unsafe_allow_html=True)

# header_before = "Select an entity to inspect its cluster of matching records"
# header_after = "Cluster of matching records for the selected entity"
# record_header = st.empty() if st.session_state.df is None or st.session_state.run_process \
#                 else (st.write("<p style='font-size: 15px;'>%s</p>" % (header_before), unsafe_allow_html=True)
#                     if st.session_state.selection is None \
#                     else st.write("<p style='font-size: 15px;'>%s</p>" % (header_after), unsafe_allow_html=True))

records = st.empty() if st.session_state.selection is None else st.dataframe(st.session_state.records[st.session_state.selection])

