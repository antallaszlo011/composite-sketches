import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import json

# Precision, Recall, Accuracy, F1 score

# df = pd.DataFrame([['g1','Precision',10],['g1','Recall',12],['g1','Accuracy',13], ['g1', 'F1 score', 14],
#                    ['g2','Precision',8],['g2','Recall',10],['g2','Accuracy',12], ['g2', 'F1 score', 12]],
#                    columns=['group','column','val'])

# df.pivot("column", "group", "val").plot(kind='bar')

# plt.show()

font = {
        'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)

with open('final_statistics.json') as json_file:
    data = json.load(json_file)

    df = pd.DataFrame(columns=['Feature', 'Evaluation', 'Value'])

    for feature in data.keys():
        scores = data[feature]
        tp = scores['TP']
        fp = scores['FP']
        fn = scores['FN']
        tn = scores['TN']

        accuracy = (tp + tn) / (fp + fn + tp + tn)
        precision = tp / (fp + tp)
        recall = tp / (fn + tp)
        f1_score = 2 * (precision * recall) / (precision + recall)

        df = df.append({'Feature': feature, 'Evaluation': 'Accuracy', 'Value': accuracy}, ignore_index=True)
        df = df.append({'Feature': feature, 'Evaluation': 'Precision', 'Value': precision}, ignore_index=True)
        df = df.append({'Feature': feature, 'Evaluation': 'Recall', 'Value': recall}, ignore_index=True)
        df = df.append({'Feature': feature, 'Evaluation': 'F1_score', 'Value': f1_score}, ignore_index=True)

    df.pivot('Feature', 'Evaluation', 'Value').plot(kind='bar')

    plt.title('Evaluation of the feature extractor for 6 predefined feature')
    plt.legend(bbox_to_anchor=(0.93, 1.19), loc='upper left')
    plt.xticks(rotation=0)
    plt.ylabel('Evaluation value')
    plt.show()