import csv
import math
import pandas as pd
import plotly.express as px
import streamlit as st
import config
import radler
from utils import compute_distribution, detect_distribution


class Task:
    """
    The object describing the task at hand
    """

    def __init__(self, task_id, run_id, dataset, algorithm, mode, sample_attributes, groups, target_distribution,
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
        self.target_distribution = compute_distribution(target_distribution)
        self.sample_size = sample_size
        self.stochastic_acceptance_size = config.stochastic_acceptance_size
        self.stochastic_acceptance_timeout = config.stochastic_acceptance_timeout

    def set_neighbors(self, neighbors):
        self.neighbors = neighbors

    def set_matches(self, matches):
        self.matches = matches


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
    st.session_state.df = None


def stop_process():
    st.session_state.stop_process = True
    st.session_state.run_process = False


def clear_result():
    st.session_state.df = None


if "run_process" not in st.session_state:
    st.session_state.run_process = False

if "stop_process" not in st.session_state:
    st.session_state.stop_process = False

if "df" not in st.session_state:
    st.session_state.df = None

if "expanded_rows" not in st.session_state:
    st.session_state.expanded_rows = set()


st.title("RadlER")

# Select a dataset
ds_name = st.selectbox("Select a dataset", config.datasets)

# Visualize the dirty dataset as a dataframe
dirty_data = pd.read_csv(f"datasets/{ds_name}/dataset.csv")
id_mapping = dict(zip(dirty_data["_id"], range(len(dirty_data))))
dirty_data["_id"] = range(len(dirty_data))
dirty_data.columns = dirty_data.columns.str.lower()
for column in dirty_data.columns:
    if dirty_data[column].dtype == "object":
        dirty_data[column] = dirty_data[column].str.lower()
st.dataframe(dirty_data)

# Visualize the RadlER functionalities
st.markdown("---")
st.write("<p style='font-size: 20px;'>Get a clean sample with RadlER</p>", unsafe_allow_html=True)

# Select the sampling attributes
string_attributes = dirty_data.select_dtypes(include=["object"]).columns
sample_attributes = st.multiselect("Select sampling attributes", [c for c in string_attributes if c != "_id"],
                                   default=config.sample_attributes[ds_name])

# Select the number of groups
num_groups = st.number_input("Select the number of groups", min_value=1, max_value=10, step=1, value=2)
clean_data = pd.read_csv(f"clean_datasets/{ds_name}.csv")
groups, clean_distribution, max_sample_size = detect_distribution(clean_data, sample_attributes,
                                                                  distribution_type="demographic_parity",
                                                                  value_filter=config.value_filter[ds_name],
                                                                  min_support=0, top_groups=num_groups)

# Visualize the selected groups
with st.expander("Visualize the groups"):
    group_info = [{sample_attributes[j]: groups[i][j] for j in range(len(sample_attributes))} for i in range(num_groups)]
    st.dataframe(pd.DataFrame(group_info))

# Select the target distribution
distribution_types = ["equal_representation", "demographic_parity", "custom_distribution"]
distribution_type = st.selectbox("Select the target distribution", distribution_types, index=0)
if distribution_type == "equal_representation":
    target_distribution = [round(1.0 / num_groups, 3) for _ in range(num_groups)]
elif distribution_type == "demographic_parity":
    target_distribution = [round(i, 3) for i in clean_distribution]

# Visualize the selected target distribution
with st.expander("Visualize the target distribution"):
    # st.bar_chart(pd.Series(target_distribution))
    fig = px.bar(pd.DataFrame({"Group": [", ".join([v for v in g]) for g in groups], "Value": target_distribution}),
                               x="Group", y="Value")
    st.plotly_chart(fig)

# Select the sample size
sample_size = st.number_input("Select the sample size", min_value=1, step=1, value=10)

# Select the matching function
matcher = st.selectbox("Select the matching function", list(config.er_features[ds_name]["matchers"].keys()),
                       index=0)

# Buttons to manage the deduplication process
col1, col2, col3 = st.columns([1, 1, 1])
col1.button("Run", on_click=run_process, disabled=(st.session_state.run_process or st.session_state.df is not None))
col2.button("Stop", on_click=stop_process, disabled=(not st.session_state.run_process))
col3.button("Clear", on_click=clear_result, disabled=(st.session_state.run_process or st.session_state.df is None))

placeholder = st.empty() if st.session_state.df is None else st.dataframe(st.session_state.df)

# Run RadlER to produce the clean sample progressively
if st.session_state.run_process:
    task = Task(0, 0, ds_name, "radler", "complete", sample_attributes, groups, target_distribution,
                sample_size=sample_size, matcher=matcher)
    load_neighbors(task, dirty_data, id_mapping)
    load_matches(task, dirty_data, id_mapping)
    run_radler(task, dirty_data, placeholder)
    st.session_state.run_process = False


