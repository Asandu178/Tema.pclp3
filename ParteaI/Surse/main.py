import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
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
df = pd.read_csv("../chess_games.csv")

# simplific modelul prins stergerea unor campuri fie redundante fie care complica prea mult
df = df.drop(columns=['opening_fullname', 'opening_variation', 'opening_moves', 'opening_response', 'white_id', 'black_id', 'game_id'])

# mapez in fct de castigator la 1, 2 sau 0
df['winner'] = df['winner'].map({'White': 1, 'Black': 2, 'Draw': 0})

# mapez in fct de conditia de castig

df['victory_status'] = df['victory_status'].map({'Mate' : 0, 'Resign' : 1, 'Out of Time' : 2, 'Draw' : 3})

# nu vreau sa folosesc combinatia de miscari pentru ca altfel as stii mereu cine castiga

df = df.drop(columns=['moves'])

# din moment ce ar fi mai greu sa lucrez direct pe time_increment il sparg in 2
df[['initial_time', 'increment_sec']] = df['time_increment'].str.split('+', expand=True).astype(int)

df = df.drop(columns=['time_increment'])
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

# analiza datelor initiale

print(df.describe())

#               turns  victory_status        winner  white_rating  black_rating  initial_time  increment_sec           age
# count  20058.000000     20058.00000  20058.000000  20058.000000  20058.000000  20058.000000   20058.000000  20058.000000
# mean      60.465999         0.85876      1.406671   1596.631868   1588.831987     13.824110       5.249626     38.874813
# std       33.570585         0.74823      0.579682    291.253376    291.036126     17.160179      14.289591     14.751223
# min        1.000000         0.00000      0.000000    784.000000    789.000000      0.000000       0.000000     14.000000
# 25%       37.000000         0.00000      1.000000   1398.000000   1391.000000     10.000000       0.000000     26.000000
# 50%       55.000000         1.00000      1.000000   1567.000000   1562.000000     10.000000       0.000000     39.000000
# 75%       79.000000         1.00000      2.000000   1793.000000   1784.000000     15.000000       7.000000     52.000000
# max      349.000000         3.00000      2.000000   2700.000000   2723.000000    180.000000     180.000000     64.000000

# observatie, am analizat direct dataset-ul mare, pentru a incerca sa intuim aproximarile finale
# observam ca numarul mediu de mutari este 60, avem totusi si meciuri aberant de lungi(maxim 349)
# media pentru statutul de victory este 0.85 deci tendinta este ca un meci sa se termine fie in mat sau capitulare
# rating-ul pentru jucatori este in jur de 1590, exista o disperie mare(jucatori intre 784-2700)
# media timpului(minute) este de 13 minute
# acesta este incrementat dupa fiecare mutare cu in medie 5.2 secunde
# varsta medie a jucatorilor este de 38 de ani cu interval intre 14 si 64 de ani
# tendinta generala arata ca meciurile se termina intr-o victorie a jucatorului alt ca rezultat al mediei 1.4
# alb - 1, negru - 2, egal - 0

# construiesc histogramele pentru tipurile numerice
# caut in dataset coloanele cu valori numerice si formez o lista cu acestea
num_cols = df.select_dtypes(include=['number']).columns.tolist()

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    axes[i].hist(df[col], bins=30, color='skyblue', edgecolor='black')
    axes[i].set_title(f'Histograma pentru {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Number of games')

# sterg ploturile goale care apar
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig('../histograme.png')  # salvează figura în fișier

# caut in dataset coloanele cu valori categorice si formez o lista cu acestea
num_cols_categorice = df.select_dtypes(include=['object']).columns.tolist()

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))
axes = axes.flatten()

for i, col in enumerate(num_cols_categorice):
    # ma limitez la cele mai mari 10 valori din fiecare categorie(fiindca am multe nationalitati,opening-uri si coduri de incepere)
    counts = df[col].value_counts().head(10)
    axes[i].bar(counts.index, counts.values, color='skyblue')
    axes[i].set_title(f'Barplot pentru {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Numar de aparitii')
    # rotesc etichetele ca sa nu se suprapuna
    axes[i].tick_params(axis='x', rotation=45)

# sterg plot-urile goale care apar
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])


plt.tight_layout()

# salvez img
plt.savefig('../categorical_barplots.png')

num_cols = df.select_dtypes(include=['number']).columns

# calculăm matricea de corelații
corr_matrix = df[num_cols].corr()

# afișăm heatmap
plt.figure(figsize=(14,14))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title('Matricea de corelatii pentru variabilele numerice')
plt.savefig('../heatmap.png')

# din analiza heatmap-ului observam ca nu exista o legatura intre varsta si castigator,
# deoarece varsta este generata aleatoriu in model
# exista o legatura liniara moderata, de unde reiese faptul ca in general
# jucatorii sunt plasati in meciuri cu oponenti de un nivel similar
# observam ca exista o legatura slaba intre cine castiga si modul de castig

plt.figure(figsize=(8, 6))

# Scatter plot cu 3 clase: 0=draw, 1=white win, 2=black win
sns.scatterplot(
    data=df,
    x='white_rating',
    y='black_rating',
    hue='winner',
    palette='coolwarm',
    style='winner',
    markers={0: 'o', 1: 's', 2: 'X'},
    s=50
)

plt.title("Scatter Plot intre white_rating și black_rating (Winner)")
plt.xlabel('White Rating')
plt.ylabel('Black Rating')
plt.legend(title="Castigator", loc='best')
plt.tight_layout()

# Salveaz imaginea
plt.savefig("../Castigator.png")
# tendinta generala este ca la nivel inalt negru si alb castiga cam la fel de mult,la egalitate cu sansa de a se termina
# in egal meciul
# pentru nivelurile de joc sub 2000 rating, tendinta este ca meciurile sa se termina fie in win/lose, aproape niciodata in egal,
# unde in continuare alb la nivel egal de rating are un avantaj


plt.figure(figsize=(8, 6))

# Scatter plot cu 4 clase : Mat = 0, Resign = 1, Out of time = 2, Draw = 3
sns.scatterplot(
    data=df,
    x='white_rating',
    y='black_rating',
    hue='victory_status',
    palette='coolwarm',
    style='victory_status',
    markers={0: 'o', 1: 's', 2: 'X', 3: '*'},
    s=50,
    alpha=0.6
)

plt.title("Scatter Plot între white_rating și black_rating (Win_con)")
plt.xlabel('White Rating')
plt.ylabel('Black Rating')
plt.legend(title="Win_con", loc='best')
plt.tight_layout()

# Salveaz imaginea
plt.savefig("../Mod_de_castig.png")

# observam ca la nivel inalt majoritatea meciurilor se termina in resign
# sau egalitate,dar si pierderea prin lipsa de timp este frecventa,comparativ cu
# incheierea meciului prin mat
# in schimb pentru restul jucatorilor, majoritatea meciurilor se termina prin
# resign sau mat

plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x='white_rating', y='turns', hue='winner', palette='coolwarm')
plt.title("White rating vs numărul de mutări")
plt.savefig("../rating&turns.png")
# observam ca odata cu cresterea rating-ului, creste si numarul de mutari facute de jucator
# intuim cu usurinta ca acest lucru se datoreaza atat experientei jucatorului analizat,dar si a oponentului acestuia

# preprocesare

# aplic one-hot encoding pentru rated, opening_shortname, nationality, opening_code

df_encoded = pd.get_dummies(df, columns=['rated', 'opening_shortname', 'nationality', 'opening_code'], drop_first=True)

encoder = LabelEncoder()

# label-encoding pt sex
df_encoded['gender'] = encoder.fit_transform(df_encoded['gender'])

# standardizarea variabilelor
scaler = StandardScaler()

transform = scaler.fit_transform(df_encoded[['turns', 'white_rating', 'black_rating', 'age', 'initial_time', 'increment_sec']])

df_encoded[['turns', 'white_rating', 'black_rating', 'age', 'initial_time', 'increment_sec']] = transform

# separarea datelor

X = df_encoded.drop(columns=['winner', 'victory_status'])
y = df_encoded[['winner', 'victory_status']]

# impartim datele: 80% pentru antrenament, 20% pentru test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Setul de antrenament X:", X_train.shape)
print("Setul de test X:", X_test.shape)
print("Setul de antrenament y:", y_train.shape)
print("Setul de test y:", y_test.shape)

model_baza = LogisticRegression(max_iter=1000, random_state=42)
model_final = MultiOutputClassifier(model_baza)

model_final.fit(X_train, y_train)

y_pred = model_final.predict(X_test)

# Vedem primele 5 predicții
print("Predicții:", y_pred[:5])
# construiesc y_test_array de forma matriceala ca sa pot compara cele 2 output-uri corect
y_test_array = y_test.to_numpy()
print("Adevărate:", y_test_array[:5])

from sklearn.metrics import accuracy_score, confusion_matrix


cm_winner = confusion_matrix(y_test['winner'], y_pred[:, 0])
plt.figure(figsize=(6, 5))
sns.heatmap(cm_winner, annot=True, fmt='d', cmap='Blues')
plt.title("Matricea de confuzie pentru Winner")
plt.xlabel('Predicții')
plt.ylabel('Realitate')
plt.savefig("../Confusion_matrix_winner.png")

# pentru victory_status
cm_victory = confusion_matrix(y_test['victory_status'], y_pred[:, 1])
plt.figure(figsize=(6, 5))
sns.heatmap(cm_victory, annot=True, fmt='d', cmap='Blues')
plt.title("Matricea de confuzie pentru Victory Status")
plt.xlabel('Predicții')
plt.ylabel('Realitate')
plt.savefig("../Confusion_matrix_victory.png")

# calculez scorul de acuratete pt fiecare categorie in parte
acc_winner = accuracy_score(y_test['winner'], y_pred[:, 0])
acc_victory = accuracy_score(y_test['victory_status'], y_pred[:, 1])

print(f"Acuratețe winner: {acc_winner:.4f}")
print(f"Acuratețe victory_status: {acc_victory:.4f}")


# creez dataframe-urile pt train si test
train = pd.concat([X_train, y_train], axis=1)
test = pd.concat([X_test, y_test], axis=1)

# creez csv-urile

train.to_csv("../train_data.csv", index=False)
test.to_csv("../test_data.csv", index=False)


