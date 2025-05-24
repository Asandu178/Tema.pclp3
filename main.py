import pandas as pd
import matplotlib as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
# librarie pentru a genera date false
from faker import Faker

fake = Faker()

def fake_nationality():
    nationality = Faker().country_code()
    return nationality

def random_gender():
    rng = np.random.rand()
    if rng < 0.5:
        return 'M'
    return 'F'

    
# citesc csv-ul intr-un dataframe
df = pd.read_csv("chess_games.csv")

# simplific modelul prins stergerea unor campuri fie redundante fie care complica prea mult
df = df.drop(columns=['opening_fullname', 'opening_variation', 'opening_moves', 'opening_response', 'white_id', 'black_id'])

# mapez in fct de castigator la 1, 2 sau 0
df['winner'] = df['winner'].map({'White': 1, 'Black': 2, 'Draw': 0})

# mapez in fct de conditia de castig

df['victory_status'] = df['victory_status'].map({'Mate' : 0, 'Resign' : 1, 'Out of Time' : 2, 'Draw' : 3})

# nu vreau sa folosesc combinatia de miscari pentru ca altfel as stii mereu cine castiga
# folosim in schimb numarul de mutari totale

df['num_moves'] = df['moves'].str.split().apply(len)

df = df.drop(columns=['moves'])

# gasesc lungimea dataframe-ului
length = len(df)

# generez coloanele ce vor contine
# informatii despre varsta, sexul si nationalitatea castigatorului
df['gender'] = 'Unknown'
df['age'] = 'Nan'
df['nationality'] = 'Unknown'

# generez in mod aleatoriu varstele castigatorilor
df['age'] = np.random.randint(14, 65, size=length)

# generez in mod aleatoriu nationalitatile castigatorilor
df['nationality'] = [fake_nationality() for _ in range(length)]

# generez in mod aleatoriu sexul castigatorilor
df['gender'] = [random_gender() for _ in range(length)]

# preprocesare

# aplic one-hot encoding pentru rated, opening_shortname, nationality, opening_code

df_encoded = pd.get_dummies(df, columns=['rated', 'opening_shortname', 'nationality', 'opening_code'], drop_first=True)

encoder = LabelEncoder()

# label-encoding pt sex
df_encoded['gender'] = encoder.fit_transform(df_encoded['gender'])

# normalizarea variabilelor numerice
scaler = MinMaxScaler()

df_encoded[['turns', 'white_rating', 'black_rating', 'age', 'num_moves']] = scaler.fit_transform(df_encoded[['turns', 'white_rating', 'black_rating', 'age', 'num_moves']])

# standardizarea variabilelor
scaler = StandardScaler()

df_encoded[['turns', 'white_rating', 'black_rating', 'age', 'num_moves']] = scaler.fit_transform(df_encoded[['turns', 'white_rating', 'black_rating', 'age', 'num_moves']])

print(df_encoded.head())
print(df_encoded.columns)
print(length)