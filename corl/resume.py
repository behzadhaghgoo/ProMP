import joblib

def load(pkl_file):
    with open(pkl_file, 'rb') as file:
         itr = joblib.load(file)

    print(itr)

if __name__ == '__main__':
    pkl_file = './data/trpo/test_pick_place_0_0.5488135039273248/itr_20.pkl'
    load(pkl_file)
