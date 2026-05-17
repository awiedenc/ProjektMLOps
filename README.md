# Projekt MLOps

Projekt przedstawia kompletny pipeline MLOps do klasyfikacji medycznej dla trzech zbiorów danych: **diabetes**, **heart** oraz **breast_cancer**. W projekcie przygotowywane są dane, trenowane są modele i zapisywane. Eksperymenty logowane są w MLflow, a gotowe modele są udostępnione przez API oparte na FastAPI.

## Cel projektu

Celem projektu jest porównanie kilku klasycznych modeli uczenia maszynowego w zadaniu klasyfikacji medycznej oraz pokazanie pełnego przepływu MLOps: od przygotowania danych, przez trening i ewaluację, po udostępnienie predykcji przez API. W projekcie wykorzystywane są trzy zbiory danych oraz wspólny schemat przetwarzania, co ułatwia porównanie wyników między problemami.

## Struktura projektu

Przykładowy układ katalogów w projekcie:

```text
ProjektMlops/
├── api/                    # warstwa REST API (FastAPI)
├── data/
│   ├── raw/               # dane wejściowe
│   └── processed/         # dane po czyszczeniu
├── models/                # zapisane modele
├── src/                   # logika przygotowania danych i treningu
├── tests/                 # testy projektu
├── mlflow.db              # baza SQLite dla MLflow
├── run_all.py             # uruchomienie całego pipeline'u
└── requirements.txt       # zależności projektu
```

Najważniejsze pliki źródłowe używane w projekcie to moduł przygotowania danych, konfiguracja ścieżek i eksperymentów, konfiguracja datasetów, moduł treningowy oraz moduł ewaluacji.

## Wymagania środowiskowe

Projekt uruchamiany jest w środowisku Python z zależnościami instalowanymi z pliku `requirements.txt`. W rozmowie roboczej projektu wykorzystywane były między innymi: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `mlflow`, `fastapi`, `uvicorn`, `pydantic`, `joblib`, `matplotlib`, `seaborn`, `sqlalchemy`, `psycopg2-binary` oraz `mlflow-skinny`.

Rekomendowana wersja Pythona to 3.10 lub 3.11, ponieważ taki zestaw zwykle dobrze współpracuje z FastAPI, MLflow i bibliotekami scikit-learn. Dokumentacja MLflow zaleca obecnie przejście z file store na backend bazodanowy, dlatego w projekcie używany jest backend SQLite.

## Instalacja projektu

1. Otworzyć terminal w katalogu projektu.
2. Utworzyć i aktywować środowisko wirtualne.
3. Zainstalować zależności z pliku `requirements.txt`.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Po instalacji środowisko powinno umożliwiać zarówno trening modeli, jak i uruchomienie API oraz interfejsu MLflow.

## Przygotowanie danych i trening

Pipeline można uruchomić jednym poleceniem:

```bash
python run_all.py
```

Skrypt wykonuje dwa główne kroki: przygotowanie danych oraz trening modeli. W etapie przygotowania dane są czyszczone, kolumny są normalizowane, zbędne pola są usuwane, a zmienne docelowe są konwertowane do postaci binarnej zależnie od datasetu.

W etapie treningu projekt dzieli dane na zbiory train, validation i test, buduje preprocessing, trenuje kilka modeli i zapisuje metryki jakości. W kodzie treningowym wykorzystywane są pipeline'y scikit-learn oraz logowanie wyników do MLflow.

## Obsługiwane datasety

| Dataset | Zmienna docelowa | Uwagi |
|---------|------------------|-------|
| diabetes | `diabetes` | Dane binarne 0/1. |
| heart | `num` → `target` binarne | Wartości większe od 0 są mapowane do klasy pozytywnej.|
| breast_cancer | `diagnosis` | `B` mapowane na 0, `M` mapowane na 1. |

Takie ujednolicenie celu klasyfikacji pozwala porównywać modele w podobnym schemacie ewaluacyjnym. 

## Modele i ewaluacja

W projekcie trenowanych jest kilka modeli klasyfikacyjnych, między innymi logistic regression, random forest, gradient boosting oraz MLP, a konfiguracja treningu jest wspólna dla wszystkich trzech problemów. Modele są zapisywane w katalogu `models/`, a metryki walidacyjne i testowe są wypisywane w logach oraz rejestrowane w MLflow.

Do oceny używane są standardowe miary klasyfikacyjne: accuracy, precision, recall, F1 oraz ROC-AUC. Dzięki temu możliwa jest nie tylko ocena skuteczności ogólnej, ale również analiza jakości rozpoznawania klasy pozytywnej, co ma znaczenie w zadaniach medycznych.

## MLflow

Projekt korzysta z MLflow do śledzenia eksperymentów. Dokumentacja MLflow informuje, że filesystem tracking backend jest przestarzały i zaleca użycie backendu bazodanowego, na przykład `sqlite:///mlflow.db`.

Aby uruchomić interfejs MLflow lokalnie, należy użyć polecenia:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Po uruchomieniu interfejs jest zwykle dostępny pod adresem [http://127.0.0.1:5000](http://127.0.0.1:5000). W panelu można przeglądać eksperymenty, porównywać modele i analizować zapisane metryki.
## Uruchomienie API

Warstwa API została przygotowana w FastAPI i udostępnia endpointy predykcyjne dla modeli. Lokalny serwer można uruchomić poleceniem:

```bash
uvicorn api.main:app --reload
```

Po starcie aplikacji dokumentacja Swagger UI jest zwykle dostępna pod adresem [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs). Pozwala to testować endpointy bez pisania dodatkowego klienta. [web:392]

## Przykładowy przebieg testu API

Dla endpointu `/predict/diabetes` można wysłać dane pacjenta w formacie JSON z polami zgodnymi z modelem wejściowym API. Jeśli predykcja zwraca odpowiedź w rodzaju:

```json
{
  "prediction": 0,
  "probability": 0.12766490819590692
}
```

oznacza to, że model przewidział klasę negatywną, a prawdopodobieństwo klasy pozytywnej wynosi około 12.8%, czyli znajduje się poniżej domyślnego progu decyzyjnego 0.5 stosowanego w klasyfikatorach binarnych scikit-learn.

Najprostsza ścieżka weryfikacji projektu:

1. Zainstalować zależności z `requirements.txt`.
2. Uruchomić cały pipeline komendą `python run_all.py`.
3. Uruchomić MLflow UI komendą `mlflow ui --backend-store-uri sqlite:///mlflow.db`.
4. Uruchomić API komendą `uvicorn api.main:app --reload`.
5. Otworzyć dokumentację pod adresem `http://127.0.0.1:8000/docs` i wykonać testowy request predykcyjny.

Taki przebieg pozwala zobaczyć cały projekt od strony danych, treningu, monitorowania eksperymentów i udostępniania modelu.

## Przykładowe payloady do FastAPI

### `/predict/diabetes`
```json
{
  "gender": "Female",
  "age": 54.0,
  "hypertension": 0,
  "heart_disease": 0,
  "smoking_history": "never",
  "bmi": 27.4,
  "HbA1c_level": 6.1,
  "blood_glucose_level": 140
}
```

### `/predict/heart`

```json
{
  "age": 58,
  "sex": 1,
  "cp": 3,
  "trestbps": 140,
  "chol": 240,
  "fbs": 0,
  "restecg": 1,
  "thalch": 150,
  "exang": 0,
  "oldpeak": 1.0,
  "slope": 2,
  "ca": 0,
  "thal": 2
}
```

### `/predict/breast_cancer`

```json
{
  "radius_mean": 14.2,
  "texture_mean": 20.1,
  "perimeter_mean": 92.5,
  "area_mean": 654.0,
  "smoothness_mean": 0.097,
  "compactness_mean": 0.104,
  "concavity_mean": 0.082,
  "concave_points_mean": 0.053,
  "symmetry_mean": 0.181,
  "fractal_dimension_mean": 0.062,
  "radius_se": 0.42,
  "texture_se": 1.2,
  "perimeter_se": 2.9,
  "area_se": 38.0,
  "smoothness_se": 0.006,
  "compactness_se": 0.021,
  "concavity_se": 0.028,
  "concave_points_se": 0.011,
  "symmetry_se": 0.019,
  "fractal_dimension_se": 0.003,
  "radius_worst": 16.8,
  "texture_worst": 27.2,
  "perimeter_worst": 111.3,
  "area_worst": 880.0,
  "smoothness_worst": 0.132,
  "compactness_worst": 0.218,
  "concavity_worst": 0.241,
  "concave_points_worst": 0.114,
  "symmetry_worst": 0.290,
  "fractal_dimension_worst": 0.084
}
```

## Co znajduje się w logach po poprawnym uruchomieniu

Po wykonaniu `python run_all.py` w logach powinny pojawić się informacje o przygotowaniu trzech datasetów, liczebności zbiorów train/val/test, zapisaniu modeli do katalogu `models/` oraz metrykach walidacyjnych i testowych. Taki rezultat potwierdza, że pipeline działa end-to-end.
## Charakter projektu

Projekt ma charakter badawczo-inżynierski. Łączy element porównania modeli klasyfikacyjnych z elementem wdrożeniowym, ponieważ obejmuje zarówno trening i ewaluację, jak i warstwę API oraz rejestrowanie eksperymentów w MLflow. 

## Szybki start

```bash
pip install -r requirements.txt
python run_all.py
mlflow ui --backend-store-uri sqlite:///mlflow.db
uvicorn api.main:app --reload
```
