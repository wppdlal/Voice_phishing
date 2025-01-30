from sklearn.metrics import accuracy_score, f1_score, recall_score

def metrics(y, pred):

    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred, average='macro')
    recall = recall_score(y, pred)

    return round(acc, 5), round(f1, 5), round(recall, 5)