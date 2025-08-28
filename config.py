datasets = ["alaska_cameras", "athletes", "beers", "nc_voters", "nyc_funding"]
distribution_types = ["equal_representation"]  # ["equal_representation", "demographic_parity"]
algorithms = ["radler"]  # ["radler", "cost", "benefit", "random"]
sample_sizes = [1]  # [i for i in range(1, 11)]
num_runs = 1  # 10
min_support = 0.01
top_groups = 10
stochastic_acceptance_size = 100_000
stochastic_acceptance_timeout = 0.1

df_lib = {
    "alaska_cameras": "pandas",
    "athletes": "pandas",
    "beers": "pandas",
    "nc_voters": "pandas",
    "nyc_funding": "pandas"
}

sample_attributes = {
    "alaska_cameras": ["brand"],
    "athletes": ["country"],
    "beers": ["style"],
    "nc_voters": ["sex", "race"],
    "nyc_funding": ["source"]
}

string_attributes = {
    "alaska_cameras": ["brand", "type"],
    "athletes": ["country", "sport"],
    "beers": ["style", "brewery", "city", "state"],
    "nc_voters": ["sex", "race", "birth_place", "city", "party"],
    "nyc_funding": ["source"]
}

value_filter = {
    "alaska_cameras": dict(),
    "athletes": dict(),
    "beers": dict(),
    "nc_voters": dict(),
    "nyc_funding": dict()
}

radler_modes = {
    "radler": "complete",
    "cost": "cost",
    "benefit": "benefit",
    "random": "unweighed"
}

path_dir_ds = "./datasets/"
file_ds = "dataset.csv"
file_gold = "matches.csv"
file_candidates = "blockers/candidates_"
file_blocks = "blockers/blocks_"

er_features = {
    "alaska_cameras": {
        "path_ds": path_dir_ds + "alaska_cameras/" + file_ds,
        "attributes": ["_id", "brand", "model", "type", "mp", "screen_size", "description", "price"],
        "default_aggregation": "vote",
        "default_fusion": {
            "brand": "vote",
            "model": "vote",
            "type": "vote",
            "mp": "vote",
            "screen_size": "vote",
            "price": "avg"
        },
        "blockers": {
            "SparkER (Meta-Blocking)": {
                "path_candidates": path_dir_ds + "alaska_cameras/" + file_candidates + "sparker.csv"
            },
            "SparkER (Meta-Blocking) - Unfair": {
                "path_candidates": path_dir_ds + "alaska_cameras/" + file_candidates + "unfair.csv"
            }
        },
        "default_blocker": "SparkER (Meta-Blocking)",
        "matchers": {
            "GPT-4o": {
                "path_gold": path_dir_ds + "alaska_cameras/" + file_gold,
                "time_per_comparison": 0.51,
                "cost_per_comparison": 0.00026
            },
            "Ditto": {
                "path_gold": path_dir_ds + "alaska_cameras/" + file_gold,
                "time_per_comparison": 0.0106,
                "cost_per_comparison": 0
            },
            "Ditto - Unfair": {
                "path_gold": path_dir_ds + "alaska_cameras/matchers/matches_unfair.csv",
                "time_per_comparison": 0.0106,
                "cost_per_comparison": 0
            }
        },
        "default_matcher": "GPT-4o"
    },
    "athletes": {
        "path_ds": path_dir_ds + "athletes/" + file_ds,
        "attributes": ["_id", "name", "sex", "age", "height", "weight", "country", "sport"],
        "default_aggregation": "vote",
        "default_fusion": {
            "name": "vote",
            "sex": "vote",
            "age": "vote",
            "height": "vote",
            "weight": "vote",
            "country": "vote",
            "sport": "vote"
        },
        "blockers": {
            "PyJedAI (Similarity Join)": {
                "path_candidates": path_dir_ds + "athletes/" + file_candidates + "pyjedai.csv"
            }
        },
        "default_blocker": "PyJedAI (Similarity Join)",
        "matchers": {
            "GPT-4o": {
                "path_gold": path_dir_ds + "athletes/" + file_gold,
                "time_per_comparison": 0.51,
                "cost_per_comparison": 0.00016
            },
            "Ditto": {
                "path_gold": path_dir_ds + "athletes/" + file_gold,
                "time_per_comparison": 0.0084,
                "cost_per_comparison": 0
            }
        },
        "default_matcher": "GPT-4o"
    },
    "beers": {
        "path_ds": path_dir_ds + "beers/" + file_ds,
        "attributes": ["_id", "name", "style", "ounces", "abv", "ibu", "brewery", "city", "state"],
        "default_aggregation": "vote",
        "default_fusion": {
            "name": "vote",
            "style": "vote",
            "ounces": "vote",
            "abv": "vote",
            "ibu": "vote",
            "brewery": "vote",
            "city": "vote",
            "state": "vote"
        },
        "blockers": {
            "PyJedAI (Similarity Join)": {
                "path_candidates": path_dir_ds + "beers/" + file_candidates + "pyjedai.csv"
            }
        },
        "default_blocker": "PyJedAI (Similarity Join)",
        "matchers": {
            "GPT-4o": {
                "path_gold": path_dir_ds + "beers/" + file_gold,
                "time_per_comparison": 0.51,
                "cost_per_comparison": 0.00026
            },
            "Ditto": {
                "path_gold": path_dir_ds + "beers/" + file_gold,
                "time_per_comparison": 0.0106,
                "cost_per_comparison": 0
            }
        },
        "default_matcher": "GPT-4o"
    },
    "nc_voters": {
        "path_ds": path_dir_ds + "nc_voters/" + file_ds,
        "attributes": ["_id", "first_name", "last_name", "sex", "race", "age", "birth_place", "city", "party"],
        "default_aggregation": "vote",
        "default_fusion": {
            "first_name": "vote",
            "last_name": "vote",
            "sex": "vote",
            "race": "vote",
            "age": "max",
            "birth_place": "vote",
            "city": "vote",
            "party": "vote"
        },
        "blockers": {
            "PyJedAI (Similarity Join)": {
                "path_candidates": path_dir_ds + "nc_voters/" + file_candidates + "pyjedai.csv",
            }
        },
        "default_blocker": "PyJedAI (Similarity Join)",
        "matchers": {
            "GPT-4o": {
                "path_gold": path_dir_ds + "nc_voters/" + file_gold,
                "time_per_comparison": 0.51,
                "cost_per_comparison": 0.00016
            },
            "Ditto": {
                "path_gold": path_dir_ds + "nc_voters/" + file_gold,
                "time_per_comparison": 0.0084,
                "cost_per_comparison": 0
            }
        },
        "default_matcher": "GPT-4o"
    },
    "nyc_funding" : {
        "path_ds": path_dir_ds + "nyc_funding/" + file_ds,
        "attributes": ["_id", "name", "address", "source", "year", "amount", "status"],
        "default_aggregation": "vote",
        "default_fusion": {
            "name": "vote",
            "address": "vote",
            "source": "vote",
            "amount": "sum"
        },
        "blockers": {
            "SparkER (Meta-Blocking)": {
                "path_candidates": path_dir_ds + "nyc_funding/" + file_candidates + "sparker.csv",
            }
        },
        "default_blocker": "SparkER (Meta-Blocking)",
        "matchers": {
            "GPT-4o": {
                "path_gold": path_dir_ds + "nyc_funding/" + file_gold,
                "time_per_comparison": 0.51,
                "cost_per_comparison": 0.00020
            },
            "Ditto": {
                "path_gold": path_dir_ds + "nyc_funding/" + file_gold,
                "time_per_comparison": 0.0088,
                "cost_per_comparison": 0
            }
        },
        "default_matcher": "GPT-4o"
    }
}
