import json
import pathlib
import torch

modelsFolderPath = pathlib.Path(__file__).parent / 'models'

with open(modelsFolderPath / 'scaling_stats.json', 'r', encoding='utf-8') as f:
    scaling_stats = json.load(f)



feature_names = ['yardline_100', 'ydstogo', 'score_differential', 'game_seconds_remaining']


#Set up the model for each decision type
model_GO = torch.nn.Sequential(
    torch.nn.Linear(4, 8),
    torch.nn.ReLU(),
    torch.nn.Linear(8, 1)
)

model_FieldGoal = torch.nn.Sequential(
    torch.nn.Linear(4, 8),
    torch.nn.ReLU(),
    torch.nn.Linear(8, 1)
)   

model_Punt = torch.nn.Sequential(
    torch.nn.Linear(4, 8),
    torch.nn.ReLU(),
    torch.nn.Linear(8, 1)
)

model_GO.load_state_dict(torch.load(modelsFolderPath / 'model_GO.pt', map_location=torch.device('cpu')))
model_FieldGoal.load_state_dict(torch.load(modelsFolderPath / 'model_FieldGoal.pt', map_location=torch.device('cpu')))
model_Punt.load_state_dict(torch.load(modelsFolderPath / 'model_Punt.pt', map_location=torch.device('cpu')))

model_GO.eval()
model_FieldGoal.eval()
model_Punt.eval()

labelToModel = {'GO' : model_GO, 'PUNT' : model_Punt, 'FIELD_GOAL' : model_FieldGoal}


def recommend(yardline_100, ydstogo, score_differential, game_seconds_remaining):
    featureToNum = {'yardline_100' : yardline_100, 'ydstogo' : ydstogo, 'score_differential' : score_differential, 'game_seconds_remaining' : game_seconds_remaining}
    EPAToResult = {'GO' : None, 'PUNT': None, 'FIELD_GOAL': None}
    for label, model in labelToModel.items():
        scaledFeatureToNum = {}
        mean = scaling_stats[label]["mean"]
        std = scaling_stats[label]["std"]
        minFeatureVal = scaling_stats[label]['min']
        maxFeatureVal = scaling_stats[label]['max']
        outOfRangeFeatureValues = []
        for feature in featureToNum:
            if featureToNum[feature] < minFeatureVal[feature] or featureToNum[feature] > maxFeatureVal[feature]:
                outOfRangeFeatureValues.append(featureToNum[feature])
            scaledFeatureToNum[feature] = (featureToNum[feature] - mean[feature]) / std[feature]

        ordered_values = [scaledFeatureToNum[name] for name in feature_names]
        featureTensor = torch.tensor(ordered_values).reshape(1,4)

        with torch.no_grad():
            predictedEPA = model(featureTensor).item()
            EPAToResult[label] = {'epa' : predictedEPA, 'out_of_range' : outOfRangeFeatureValues}

    EPAToResult['BEST'] = max(
        (item for item in EPAToResult.items() if len(item[1]['out_of_range']) == 0),
        key=lambda item: item[1]['epa']
    )[0]

    return EPAToResult


# ---------------------------------------------------------------------------
# Phase 2: win-probability-based decisions (see CLAUDE.md's confirmed design
# decisions #4). v2 (2026-09-23) adds a 5th, engineered input --
# `score_after_fg` = score_differential + 3, the margin right after an
# otherwise-successful field goal -- so the model has a direct signal for
# "does a FG even help here", instead of having to infer it indirectly from
# raw score_differential. Because the feature set differs from the EPA
# models' 4, this uses its own scaling_stats_wp.json (not the EPA one) and
# its own Linear(5, 8) architecture -- see backend/train_wp.py. Kept as
# separate models/function rather than replacing recommend(), so EPA and WP
# recommendations can be compared side by side.
# ---------------------------------------------------------------------------

with open(modelsFolderPath / 'scaling_stats_wp.json', 'r', encoding='utf-8') as f:
    scaling_stats_wp = json.load(f)

feature_names_wp = feature_names + ['score_after_fg']

model_GO_wp = torch.nn.Sequential(
    torch.nn.Linear(5, 8),
    torch.nn.ReLU(),
    torch.nn.Linear(8, 1)
)

model_FieldGoal_wp = torch.nn.Sequential(
    torch.nn.Linear(5, 8),
    torch.nn.ReLU(),
    torch.nn.Linear(8, 1)
)

model_Punt_wp = torch.nn.Sequential(
    torch.nn.Linear(5, 8),
    torch.nn.ReLU(),
    torch.nn.Linear(8, 1)
)

model_GO_wp.load_state_dict(torch.load(modelsFolderPath / 'model_GO_wp.pt', map_location=torch.device('cpu')))
model_FieldGoal_wp.load_state_dict(torch.load(modelsFolderPath / 'model_FieldGoal_wp.pt', map_location=torch.device('cpu')))
model_Punt_wp.load_state_dict(torch.load(modelsFolderPath / 'model_Punt_wp.pt', map_location=torch.device('cpu')))

model_GO_wp.eval()
model_FieldGoal_wp.eval()
model_Punt_wp.eval()

labelToModelWp = {'GO': model_GO_wp, 'PUNT': model_Punt_wp, 'FIELD_GOAL': model_FieldGoal_wp}


def recommend_wp(yardline_100, ydstogo, score_differential, game_seconds_remaining):
    featureToNum = {
        'yardline_100' : yardline_100,
        'ydstogo' : ydstogo,
        'score_differential' : score_differential,
        'game_seconds_remaining' : game_seconds_remaining,
        'score_after_fg' : score_differential + 3,
    }
    WPToResult = {'GO' : None, 'PUNT': None, 'FIELD_GOAL': None}
    for label, model in labelToModelWp.items():
        scaledFeatureToNum = {}
        mean = scaling_stats_wp[label]["mean"]
        std = scaling_stats_wp[label]["std"]
        minFeatureVal = scaling_stats_wp[label]['min']
        maxFeatureVal = scaling_stats_wp[label]['max']
        outOfRangeFeatureValues = []
        for feature in featureToNum:
            if featureToNum[feature] < minFeatureVal[feature] or featureToNum[feature] > maxFeatureVal[feature]:
                outOfRangeFeatureValues.append(featureToNum[feature])
            scaledFeatureToNum[feature] = (featureToNum[feature] - mean[feature]) / std[feature]

        ordered_values = [scaledFeatureToNum[name] for name in feature_names_wp]
        featureTensor = torch.tensor(ordered_values).reshape(1,5)

        with torch.no_grad():
            predictedWP = model(featureTensor).item()
            WPToResult[label] = {'wpa' : predictedWP, 'out_of_range' : outOfRangeFeatureValues}

    WPToResult['BEST'] = max(
        (item for item in WPToResult.items() if len(item[1]['out_of_range']) == 0),
        key=lambda item: item[1]['wpa']
    )[0]

    return WPToResult
