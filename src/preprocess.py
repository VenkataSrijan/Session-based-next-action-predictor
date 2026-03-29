import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_data(path):
    df = pd.read_csv(path)
    df = df[['visitorid', 'itemid', 'timestamp']]
    df.columns = ['session_id', 'item_id', 'timestamp']
    df = df.sort_values(by=['session_id', 'timestamp'])
    return df


def create_sessions(df):
    sessions = df.groupby('session_id')['item_id'].apply(list)
    return sessions.tolist()


def filter_sessions(sessions):
    # Keep only sessions with more than 1 item
    return [s for s in sessions if len(s) > 1]


def create_sequences(sessions):
    inputs  = []
    targets = []

    for session in sessions:
        for i in range(1, len(session)):
            inputs.append(session[:i])
            targets.append(session[i])

    return inputs, targets


def encode_items(inputs, targets):
    le = LabelEncoder()

    all_items = [item for seq in inputs for item in seq] + list(targets)
    le.fit(all_items)

    # FIX: vectorized transform per sequence instead of item-by-item loop
    # Old: [[le.transform([item])[0] for item in seq] for seq in inputs]  ← very slow
    inputs_encoded  = [le.transform(seq).tolist() for seq in inputs]
    targets_encoded = le.transform(targets)

    return inputs_encoded, targets_encoded, le